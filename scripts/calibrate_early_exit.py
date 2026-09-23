"""Grid-search early-exit thresholds on held-out windows.

Phase-2 calibration: roll frozen full-depth trajectories, record per-depth
confidence / margin / state-delta / CE on the last token position, derive the
oracle exit depth under a quality budget, and grid-search (optionally refine)
the threshold tuple ``(confidence, margin, delta)`` for the fastest policy
whose relative CE degradation stays within ``--budget``.  The result is an
early-exit policy artifact consumed by ``test.py --early-exit-config``.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from memory_transformer import (
    MemoryTransformerConfig,
    MemoryTransformerForCausalLM,
    ensure_exit_readout,
)
from pretrain_data import TokenStore, token_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/tokens"))
    parser.add_argument("--output", type=Path, default=Path("early_exit_calibration.pt"))
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--windows", type=int, default=256, help="total windows to collect")
    parser.add_argument("--epsilon", type=float, default=0.05, help="oracle CE slack in nats")
    parser.add_argument("--budget", type=float, default=0.01, help="relative CE degradation cap")
    parser.add_argument("--min-depth", type=int, default=0, help="0 uses the checkpoint policy")
    parser.add_argument("--grid-confidence", default="0.5:0.99:0.025")
    parser.add_argument("--grid-margin", default="0.0,0.02,0.05,0.1,0.2,0.3")
    parser.add_argument("--grid-delta", default="0.01,0.02,0.05,0.1,0.2,0.5,1000.0")
    parser.add_argument("--refine-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def load_model(path: Path, device: torch.device, precision: str) -> MemoryTransformerForCausalLM:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = MemoryTransformerConfig(**payload["config"])
    model = MemoryTransformerForCausalLM(config)
    if any(name.startswith("exit_readout.") for name in payload["model"]):
        ensure_exit_readout(
            model, rank=int(payload.get("exit_readout_rank", config.exit_readout_rank))
        )
    model.load_state_dict(payload["model"])
    dtype = torch.bfloat16 if precision == "bf16" and device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype)
    if model.exit_readout is not None:
        model.exit_readout.float()
    model.eval()
    return model


def depth_logits(
    model: MemoryTransformerForCausalLM, states: torch.Tensor, depth_index: int
) -> torch.Tensor:
    """Calibrated (or plain) logits for last-position states at one depth."""
    head = model.lm_head.weight.to(states.dtype)
    if model.exit_readout is not None:
        return model.exit_readout.logits_at(depth_index, states, head)
    return torch.nn.functional.linear(states, head)


def collect_features(
    model: MemoryTransformerForCausalLM,
    store: TokenStore,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    depth = model.config.num_layers
    confidence_rows, margin_rows, delta_rows, ce_rows = [], [], [], []
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and args.precision == "bf16"
        else contextlib.nullcontext()
    )
    steps = math.ceil(args.windows / args.batch_size)
    with torch.no_grad():
        for inputs, labels in token_batches(
            store, args.sequence_length, args.batch_size, steps, seed=args.seed
        ):
            input_tensor = torch.from_numpy(inputs).to(device)
            target = torch.from_numpy(labels).to(device)[:, -1]
            with autocast:
                rollout = model.exit_rollout(input_tensor)
            states = rollout["states"]
            confidence_batch, margin_batch, ce_batch = [], [], []
            for depth_index in range(depth):
                logits = depth_logits(model, states[depth_index], depth_index)
                probabilities = torch.softmax(logits.float(), dim=-1)
                top2 = torch.topk(probabilities, 2, dim=-1).values
                confidence_batch.append(top2[:, 0])
                margin_batch.append(top2[:, 0] - top2[:, 1])
                ce_batch.append(
                    torch.nn.functional.cross_entropy(
                        logits.float(), target, reduction="none"
                    )
                )
            confidence_rows.append(torch.stack(confidence_batch, dim=1).cpu())
            margin_rows.append(torch.stack(margin_batch, dim=1).cpu())
            ce_rows.append(torch.stack(ce_batch, dim=1).cpu())
            delta_rows.append(rollout["deltas"].float().t().cpu())
    return {
        "confidence": torch.cat(confidence_rows)[: args.windows],
        "margin": torch.cat(margin_rows)[: args.windows],
        "delta": torch.cat(delta_rows)[: args.windows],
        "ce": torch.cat(ce_rows)[: args.windows],
    }


def oracle_depths(ce: torch.Tensor, epsilon: float, min_depth: int) -> torch.Tensor:
    depth = ce.shape[1]
    depths = torch.arange(1, depth + 1)
    allowed = (depths >= min_depth)[None, :] & ((ce - ce[:, -1:]) <= epsilon)
    found = allowed.any(dim=1)
    first = allowed.float().argmax(dim=1) + 1
    return torch.where(found, first, torch.full_like(first, depth))


def simulate_exit(
    confidence: torch.Tensor,
    margin: torch.Tensor,
    delta: torch.Tensor,
    ce: torch.Tensor,
    thresholds: tuple[float, float, float],
    min_depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tau_confidence, tau_margin, tau_delta = thresholds
    depth = ce.shape[1]
    depths = torch.arange(1, depth + 1)
    satisfied = (
        (confidence >= tau_confidence)
        & (margin >= tau_margin)
        & (delta <= tau_delta)
        & (depths >= min_depth)[None, :]
    )
    found = satisfied.any(dim=1)
    first = satisfied.float().argmax(dim=1) + 1
    chosen = torch.where(found, first, torch.full_like(first, depth))
    chosen_ce = ce.gather(1, (chosen - 1)[:, None]).squeeze(1)
    return chosen, chosen_ce


def evaluate_policy(
    confidence: torch.Tensor,
    margin: torch.Tensor,
    delta: torch.Tensor,
    ce: torch.Tensor,
    thresholds: tuple[float, float, float],
    min_depth: int,
    budget: float,
) -> dict[str, float]:
    chosen, chosen_ce = simulate_exit(confidence, margin, delta, ce, thresholds, min_depth)
    reference = ce[:, -1]
    degradation = float((chosen_ce - reference).mean())
    relative = degradation / max(1e-6, float(reference.mean()))
    depth_count = float(chosen.float().mean())
    return {
        "relative_degradation": relative,
        "mean_exit_depth": depth_count,
        "speedup": ce.shape[1] / max(1.0, depth_count),
        "within_budget": bool(relative <= budget),
        "share_full_depth": float((chosen == ce.shape[1]).float().mean()),
    }


def grid_candidates(args: argparse.Namespace) -> list[tuple[float, float, float]]:
    lo, hi, step = (float(value) for value in args.grid_confidence.split(":"))
    confidence = list(np.arange(lo, hi + 1e-9, step))
    margin = [float(value) for value in args.grid_margin.split(",")]
    delta = [float(value) for value in args.grid_delta.split(",")]
    return [
        (round(c, 4), round(m, 4), round(d, 4))
        for c in confidence
        for m in margin
        for d in delta
    ]


def refine(
    features: dict[str, torch.Tensor],
    best: tuple[float, float, float],
    min_depth: int,
    budget: float,
    rounds: int,
) -> tuple[tuple[float, float, float], dict[str, float]]:
    current = best
    current_score = evaluate_policy(
        confidence=features["confidence"],
        margin=features["margin"],
        delta=features["delta"],
        ce=features["ce"],
        thresholds=current,
        min_depth=min_depth,
        budget=budget,
    )
    scales = (1.5, 1.0 / 1.5)
    for _ in range(rounds):
        improved = False
        for axis in range(3):
            for scale in scales:
                candidate = list(current)
                candidate[axis] = round(max(0.0, candidate[axis] * scale), 6)
                if candidate[axis] == current[axis]:
                    continue
                score = evaluate_policy(
                    confidence=features["confidence"],
                    margin=features["margin"],
                    delta=features["delta"],
                    ce=features["ce"],
                    thresholds=tuple(candidate),
                    min_depth=min_depth,
                    budget=budget,
                )
                if score["within_budget"] and score["speedup"] > current_score["speedup"]:
                    current = tuple(candidate)
                    current_score = score
                    improved = True
        if not improved:
            break
    return current, current_score


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    device = (
        torch.device("cuda") if args.device != "cpu" and torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = load_model(args.checkpoint, device, args.precision)
    store = TokenStore(args.dataset_dir)
    features = collect_features(model, store, args, device)
    depth = model.config.num_layers
    min_depth = args.min_depth or min(model.config.early_exit_min_depth, depth)
    half = features["ce"].shape[0] // 2
    search = {key: value[:half] for key, value in features.items()}
    validation = {key: value[half:] for key, value in features.items()}

    def evaluate(table: dict[str, torch.Tensor], thresholds):
        return evaluate_policy(
            confidence=table["confidence"],
            margin=table["margin"],
            delta=table["delta"],
            ce=table["ce"],
            thresholds=thresholds,
            min_depth=min_depth,
            budget=args.budget,
        )

    oracle_search = oracle_depths(search["ce"], args.epsilon, min_depth)
    oracle_validation = oracle_depths(validation["ce"], args.epsilon, min_depth)
    best: tuple[tuple[float, float, float], dict[str, float]] | None = None
    rows = []
    for candidate in grid_candidates(args):
        score = evaluate(search, candidate)
        if not score["within_budget"]:
            continue
        rows.append({"thresholds": candidate, **score})
        if best is None or score["speedup"] > best[1]["speedup"]:
            best = (candidate, score)
    if best is None:
        raise RuntimeError(
            "no threshold combination satisfies the quality budget; "
            "raise --budget or lower --min-depth"
        )
    refined, refined_score = refine(search, best[0], min_depth, args.budget, args.refine_rounds)
    if refined_score["speedup"] > best[1]["speedup"]:
        best = (refined, refined_score)
    heldout = evaluate(validation, best[0])
    if not heldout["within_budget"]:
        raise RuntimeError(f"selected policy fails on the held-out half: {heldout}")
    rows.sort(key=lambda row: -row["speedup"])
    policy = {
        "early_exit_enabled": True,
        "early_exit_min_depth": int(min_depth),
        "early_exit_confidence_threshold": float(best[0][0]),
        "early_exit_margin_threshold": float(best[0][1]),
        "early_exit_state_delta_threshold": float(best[0][2]),
    }
    payload = {
        "policy": policy,
        "metadata": {
            "epsilon": args.epsilon,
            "budget": args.budget,
            "windows": int(features["ce"].shape[0]),
            "search_half": half,
            "oracle_mean_depth_search": float(oracle_search.float().mean()),
            "oracle_mean_depth_heldout": float(oracle_validation.float().mean()),
            "heldout": heldout,
            "pareto": rows[:16],
            "elapsed_seconds": time.monotonic() - started,
            "config": asdict(model.config),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"calibrated thresholds={best[0]} heldout={heldout} "
        f"oracle_depth={payload['metadata']['oracle_mean_depth_heldout']:.2f} "
        f"rows={len(rows)} output={args.output} "
        f"elapsed_s={payload['metadata']['elapsed_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
