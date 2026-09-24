"""Measure next-token CE of the early-exit policy against full depth.

Loads the two-stage checkpoint exactly like test.py (fp32 backbone + fp32
readout), replays the backbone loop keeping full-sequence states, and reports
per-token cross entropy (nats) and perplexity under:
  * teacher      -- plain lm_head at full depth (the unmodified model)
  * full readout -- calibrated readout at depth 24 (speedup disabled)
  * policy       -- the deployed online selection rule (halt at first cost rise)
for both the deployed default weights and the trained loss weights, on
full-sequence CE and on next-token (generation-position) CE.

Windows come from the token store; the store was seen during readout training,
so the numbers describe deployed in-distribution quality.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_transformer import (  # noqa: E402
    MemoryTransformerConfig,
    MemoryTransformerForCausalLM,
    ensure_exit_readout,
    selection_exit_depth,
)
from pretrain_data import TokenStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("data/checkpoints/exit-readout-v1.pt"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/tokens_v2"))
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trained-weights", default="0.543,0.068,0.066,0.065,0.193,0.065")
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> MemoryTransformerForCausalLM:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = MemoryTransformerConfig(**payload["config"])
    model = MemoryTransformerForCausalLM(config)
    if any(name.startswith("exit_readout.") for name in payload["model"]):
        ensure_exit_readout(model, rank=int(payload.get("exit_readout_rank", config.exit_readout_rank)))
    model.load_state_dict(payload["model"])
    model.config.early_exit_enabled = True
    model.to(device)
    model.eval()
    return model


def rollout_full(model: MemoryTransformerForCausalLM, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Backbone replay keeping every depth's full-sequence normed states."""
    states, deltas = [], []
    prev_normed = None
    with torch.no_grad():
        x = model.embed_tokens(inputs)
        for layer in model.layers:
            x, _indices, _weights, _probabilities = layer(x, model.memory, attention_mask=None)
            normed = model.norm(x)
            states.append(normed)
            if prev_normed is None:
                deltas.append(x.new_zeros(x.shape[0]))
            else:
                difference = normed.float() - prev_normed
                denominator = normed.float().norm(dim=-1).clamp_min(1e-6)
                deltas.append((difference.norm(dim=-1) / denominator).mean(-1))
            prev_normed = normed.detach()
    return {
        "states": torch.stack(states),
        "deltas": torch.stack(deltas),
        "teacher_logits": F.linear(states[-1][:, -1, :], model.lm_head.weight),
    }


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    config = model.config
    num_depths = config.num_layers
    store = TokenStore(args.dataset_dir)
    samples = (store.total_tokens - 1) // args.sequence_length
    variants = {
        "deployed_default": tuple(float(v) for v in config.exit_selection_weights),
        "trained_pi": tuple(float(v) for v in args.trained_weights.split(",")),
    }
    seq_ce = torch.zeros(num_depths, dtype=torch.float64)
    last_ce = torch.zeros(num_depths, dtype=torch.float64)
    cost_terms: list[dict[str, np.ndarray]] = []
    teacher_seq, teacher_last, windows_done = [], [], 0
    with torch.no_grad():
        for start in range(0, args.windows, args.batch_size):
            size = min(args.batch_size, args.windows - start)
            rows_in, rows_out = [], []
            for offset in range(size):
                index = int((start + offset + 0.5) * samples / args.windows)
                window = store.read(index * args.sequence_length, args.sequence_length + 1)
                rows_in.append(window[:-1])
                rows_out.append(window[1:])
            inputs = torch.from_numpy(np.stack(rows_in)).to(device)
            labels = torch.from_numpy(np.stack(rows_out)).to(device)
            rollout = rollout_full(model, inputs)
            states = rollout["states"]
            deltas = rollout["deltas"]
            teacher_seq.extend(
                F.cross_entropy(
                    F.linear(states[-1][:, :-1].reshape(-1, states.shape[-1]), model.lm_head.weight),
                    labels[:, :-1].reshape(-1),
                    reduction="mean",
                ).repeat(size).tolist()
            )
            teacher_last.extend(
                F.cross_entropy(
                    F.linear(states[-1][:, -1, :], model.lm_head.weight),
                    labels[:, -1],
                    reduction="mean",
                ).repeat(size).tolist()
            )
            confidences, margins = [], []
            for depth in range(num_depths):
                logits = model.exit_readout.logits_at(depth, states[depth][:, :-1], model.lm_head.weight)
                seq_ce[depth] += float(F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels[:, :-1].reshape(-1), reduction="mean"
                )) * size
                last = model.exit_readout.logits_at(depth, states[depth][:, -1:], model.lm_head.weight)
                last_ce[depth] += float(F.cross_entropy(last[:, 0, :], labels[:, -1], reduction="mean")) * size
                probabilities = torch.softmax(last[:, 0, :].float(), dim=-1)
                top2 = torch.topk(probabilities, 2, dim=-1).values
                confidences.append(top2[:, 0].cpu().numpy())
                margins.append((top2[:, 0] - top2[:, 1]).cpu().numpy())
                del logits, last
            cost_terms.append(
                {
                    "confidence": np.stack(confidences, axis=1),
                    "margin": np.stack(margins, axis=1),
                    "delta": deltas.float().cpu().numpy().T,
                }
            )
            windows_done += size
            print(f"windows {windows_done}/{args.windows}", flush=True)
    teacher_seq_mean, teacher_last_mean = float(np.mean(teacher_seq)), float(np.mean(teacher_last))
    seq_mean = seq_ce / windows_done
    last_mean = last_ce / windows_done
    print(f"\nper-depth full-sequence CE (nats/token): {' '.join(f'{v:.4f}' for v in seq_mean.tolist())}")
    print(f"per-depth next-token CE (nats):         {' '.join(f'{v:.4f}' for v in last_mean.tolist())}")
    for label, curve, teacher in (("sequence", seq_mean, teacher_seq_mean), ("next-token", last_mean, teacher_last_mean)):
        print(f"\n== {label} CE ==")
        print(f"{'policy':>18} {'exit_depth':>10} {'ce_nats':>9} {'ppl':>8} {'delta_nats':>11} {'rel_%':>7}")
        print(f"{'teacher@full':>18} {num_depths:>10} {teacher:>9.4f} {np.exp(teacher):>8.3f} "
              f"{teacher - float(curve[-1]):>11.4f} {100 * (teacher - float(curve[-1])) / float(curve[-1]):>6.2f}")
        print(f"{'full_readout@24':>18} {num_depths:>10} {float(curve[-1]):>9.4f} {np.exp(float(curve[-1])):>8.3f} "
              f"{0.0:>11.4f} {0.0:>6.2f}")
        for name, weights in variants.items():
            config.exit_selection_weights = weights
            config.__post_init__()
            chosen, policy_rows = [], []
            for batch in cost_terms:
                costs = [
                    float(model._exit_selection_cost(
                        torch.from_numpy(batch["confidence"][:, d]),
                        torch.from_numpy(batch["margin"][:, d]),
                        torch.from_numpy(batch["delta"][:, d]),
                        d,
                    ))
                    for d in range(num_depths)
                ]
                depth, _ = selection_exit_depth(
                    costs, min(config.early_exit_min_depth, num_depths), num_depths
                )
                chosen.append(depth)
                policy_rows.append(float(curve[depth - 1]))
            policy_mean = float(np.mean(policy_rows))
            delta = policy_mean - float(curve[-1])
            print(f"{name:>18} {float(np.mean(chosen)):>10.2f} {policy_mean:>9.4f} {np.exp(policy_mean):>8.3f} "
                  f"{delta:>11.4f} {100 * delta / float(curve[-1]):>6.2f}   "
                  f"speedup={num_depths / float(np.mean(chosen)):.2f}x")


if __name__ == "__main__":
    main()
