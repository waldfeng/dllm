"""Validate the shared-forward joint-training redesign against the current one.

IMPL1 (current): inference-mode 24-layer rollout -> phase-A update on detached
inputs -> per-depth-group forward_until re-forward -> CE backward.

IMPL2 (proposed): ONE grad-enabled 24-layer forward (collect_states=True).
Phase A consumes detached last-position states / deltas / base_logits, so its
backward never touches the backbone graph.  Phase B consumes the ORIGINAL
graph tensors with the readout frozen (requires_grad False) and backwards per
depth group (retain_graph until the last group).  Gradients still reach the
backbone because the frozen readout stays in the hidden->logits path.

Checks per scenario: selected depths, A1/A2, readout params after the A step,
per-group CE, backbone gradient agreement (max abs + relative), phase times,
peak memory.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train as T
from pretrain_data import TokenStore, token_batches

CKPT = "data/checkpoints/joint_scratch_1epoch.pt"


def build_everything(batch_size: int):
    sys.argv = [
        "validate.py", "--train-exit-joint",
        "--dataset-dir", "data/tokens_v2",
        "--tokenizer", "tokenizer/qwen",
        "--sequence-length", "2048",
        "--batch-size", str(batch_size),
        "--steps", "1",
        "--precision", "amp", "--grad-checkpoint", "--ce-chunk-rows", "4096",
        "--lr", "1.5e-4", "--warmup-steps", "600", "--exit-joint-readout-lr", "1e-4",
        "--exit-hard-min-depth", "2", "--exit-min-depth", "3",
        "--save-steps", "0", "--log-steps", "1",
        "--checkpoint", CKPT, "--seed", "42",
    ]
    args = T.parse_args()
    T.seed_everything(args.seed, 0)
    store = TokenStore(args.dataset_dir)
    vocab_size = int(store.manifest["vocab_size"])
    config = T.build_config(args, vocab_size)
    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    saved = payload.get("config") or {}
    mismatched = [k for k in T.ARCHITECTURE_KEYS if saved.get(k) != getattr(config, k)]
    if mismatched:
        raise ValueError(f"checkpoint architecture mismatch: {mismatched}")
    models = []
    for _ in range(2):
        model = T.MemoryTransformerForCausalLM(config)
        if any(n.startswith("exit_readout.") for n in payload.get("model", {})):
            T.ensure_exit_readout(model, rank=int(payload.get("exit_readout_rank", args.exit_readout_rank)))
        model.load_state_dict(payload["model"])
        if not any(n.startswith("exit_readout.") for n in payload.get("model", {})):
            T.ensure_exit_readout(model, rank=args.exit_readout_rank)
        model.to("cuda")
        model.exit_readout.float()
        model.train()
        model.gradient_checkpointing = True
        models.append(model)
    del payload
    torch.cuda.empty_cache()
    fixed_weights = torch.tensor(
        [float(v) for v in args.loss_weight_prior.split(",")], dtype=torch.float32, device="cuda"
    )
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    iterator = token_batches(
        store, sequence_length=args.sequence_length, batch_size=batch_size,
        steps=2, rank=0, world_size=1, start_step=0, seed=args.seed,
    )
    inputs, labels = next(iter(iterator))
    input_tensor = torch.from_numpy(inputs).to("cuda")
    labels_tensor = torch.from_numpy(labels).to("cuda")
    return args, config, models, fixed_weights, autocast, input_tensor, labels_tensor


def readout_params(model):
    return list(model.exit_readout.parameters())


def snapshot_grads(model):
    return {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None and not n.startswith("exit_readout.")
    }


def snapshot_readout(model):
    return {n: p.detach().clone() for n, p in model.exit_readout.named_parameters()}


def chunked_ce(model, flat_hidden, flat_labels, depth_index, ce_chunk_rows):
    head_weight = model.lm_head.weight

    def chunk_ce(h_rows, y_rows, d=depth_index):
        logits = model.exit_readout.logits_at(d, h_rows, head_weight)
        return F.cross_entropy(logits.float(), y_rows, ignore_index=-100, reduction="sum")

    total = flat_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, flat_hidden.size(0), ce_chunk_rows):
        total = total + torch.utils.checkpoint.checkpoint(
            chunk_ce,
            flat_hidden[start:start + ce_chunk_rows],
            flat_labels[start:start + ce_chunk_rows],
            use_reentrant=False,
        )
    return total


def impl1(model, input_tensor, labels_tensor, args, config, fixed_weights, autocast, selection_override=None):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    with autocast:
        rollout = model.joint_rollout(input_tensor)
    states = rollout["states"].clone()
    deltas = rollout["deltas"].clone()
    base_logits = rollout["base_logits"].clone()
    del rollout

    term_rows, _, depths_range = T.assemble_exit_terms(
        model, {"states": states, "deltas": deltas}, labels_tensor[:, -1],
        args, config.num_layers, base_logits=base_logits,
    )
    loss_a1, loss_a2, selected_depths, selection, entropy = T.joint_sample_losses(
        term_rows, fixed_weights, args.exit_temp_end, args.joint_selection_entropy_beta,
        hard_min_depth=args.exit_hard_min_depth,
    )
    loss_a = loss_a1 + loss_a2
    opt_a = torch.optim.AdamW(
        [{"params": readout_params(model), "weight_decay": args.exit_readout_weight_decay}],
        lr=args.exit_joint_readout_lr,
    )
    opt_a.zero_grad()
    loss_a.backward()
    opt_a.step()
    readout_after = snapshot_readout(model)
    torch.cuda.synchronize()
    t_a = time.perf_counter() - t_start

    t_b0 = time.perf_counter()
    for p in readout_params(model):
        p.requires_grad_(False)
    masked_count = labels_tensor.ne(-100).sum().clamp_min(1)
    chosen = selected_depths if selection_override is None else selection_override
    ce_total = 0.0
    group_ce = {}
    for group_depth in sorted(chosen.unique().tolist()):
        indices = (chosen == group_depth).nonzero().flatten()
        with autocast:
            forward = model.forward_until(
                input_tensor[indices], num_cycles=group_depth, collect_states=False)
        flat_hidden, flat_labels = T.joint_token_rows(forward["final_normed"], labels_tensor[indices])
        ce_sum = chunked_ce(model, flat_hidden, flat_labels, group_depth - 1, args.ce_chunk_rows)
        router_term = forward["router_losses"][:group_depth].mean() * indices.numel() / input_tensor.shape[0]
        loss_chunk = ce_sum / masked_count + args.router_weight * router_term
        if not torch.isfinite(loss_chunk):
            raise FloatingPointError("non-finite impl1 phase-B loss")
        loss_chunk.backward()
        ce_sum_value = float(ce_sum.detach())
        ce_total += ce_sum_value
        group_ce[int(group_depth)] = ce_sum_value / float(masked_count)
        del forward, flat_hidden, flat_labels
    for p in readout_params(model):
        p.requires_grad_(True)
    grads = snapshot_grads(model)
    torch.cuda.synchronize()
    t_b = time.perf_counter() - t_b0
    peak = torch.cuda.max_memory_allocated() / 2**30
    return {
        "selection": selected_depths.tolist(), "a1": float(loss_a1.detach()),
        "a2": float(loss_a2.detach()), "ce_mean": ce_total / float(masked_count),
        "group_ce": group_ce, "grads": grads, "readout": readout_after,
        "t_forward": None, "t_a": t_a, "t_b": t_b, "peak_gib": peak,
    }


def impl2(model, input_tensor, labels_tensor, args, config, fixed_weights, autocast, selection_override=None):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    with autocast:
        out = model.forward_until(
            input_tensor, num_cycles=config.num_layers, collect_states=True)
    normed_states = out["normed_states"]
    router_losses = out["router_losses"]

    with torch.no_grad():
        states_last = normed_states[:, :, -1, :]
        deltas = []
        prev = None
        for d in range(config.num_layers):
            normed = normed_states[d]
            if prev is None:
                deltas.append(normed.new_zeros(normed.shape[0]))
            else:
                difference = normed.float() - prev
                denominator = normed.float().norm(dim=-1).clamp_min(1e-6)
                per_position = difference.norm(dim=-1) / denominator
                deltas.append(per_position.mean(-1))
            prev = normed
        deltas = torch.stack(deltas)
        base_logits = F.linear(states_last, model.lm_head.weight)

    term_rows, _, depths_range = T.assemble_exit_terms(
        model, {"states": states_last, "deltas": deltas}, labels_tensor[:, -1],
        args, config.num_layers, base_logits=base_logits,
    )
    loss_a1, loss_a2, selected_depths, selection, entropy = T.joint_sample_losses(
        term_rows, fixed_weights, args.exit_temp_end, args.joint_selection_entropy_beta,
        hard_min_depth=args.exit_hard_min_depth,
    )
    loss_a = loss_a1 + loss_a2
    opt_a = torch.optim.AdamW(
        [{"params": readout_params(model), "weight_decay": args.exit_readout_weight_decay}],
        lr=args.exit_joint_readout_lr,
    )
    opt_a.zero_grad()
    loss_a.backward()
    opt_a.step()
    readout_after = snapshot_readout(model)
    torch.cuda.synchronize()
    t_forward = time.perf_counter() - t_start

    t_b0 = time.perf_counter()
    for p in readout_params(model):
        p.requires_grad_(False)
    masked_count = labels_tensor.ne(-100).sum().clamp_min(1)
    chosen = selected_depths if selection_override is None else selection_override
    groups = sorted(chosen.unique().tolist())
    ce_total = 0.0
    group_ce = {}
    for gi, group_depth in enumerate(groups):
        indices = (chosen == group_depth).nonzero().flatten()
        hidden = normed_states[group_depth - 1][indices]
        flat_hidden, flat_labels = T.joint_token_rows(hidden, labels_tensor[indices])
        ce_sum = chunked_ce(model, flat_hidden, flat_labels, group_depth - 1, args.ce_chunk_rows)
        router_term = router_losses[:group_depth].mean() * indices.numel() / input_tensor.shape[0]
        loss_chunk = ce_sum / masked_count + args.router_weight * router_term
        if not torch.isfinite(loss_chunk):
            raise FloatingPointError("non-finite impl2 phase-B loss")
        loss_chunk.backward(retain_graph=(gi < len(groups) - 1))
        ce_sum_value = float(ce_sum.detach())
        ce_total += ce_sum_value
        group_ce[int(group_depth)] = ce_sum_value / float(masked_count)
    for p in readout_params(model):
        p.requires_grad_(True)
    grads = snapshot_grads(model)
    del normed_states, out, hidden, flat_hidden, flat_labels
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t_b = time.perf_counter() - t_b0
    peak = torch.cuda.max_memory_allocated() / 2**30
    return {
        "selection": selected_depths.tolist(), "a1": float(loss_a1.detach()),
        "a2": float(loss_a2.detach()), "ce_mean": ce_total / float(masked_count),
        "group_ce": group_ce, "grads": grads, "readout": readout_after,
        "t_forward": t_forward, "t_a": t_forward, "t_b": t_b, "peak_gib": peak,
    }


def compare(name, r1, r2, grad_tol):
    print(f"\n=== {name} ===")
    sel_match = r1["selection"] == r2["selection"]
    print(f"selection: impl1={r1['selection']} impl2={r2['selection']} match={sel_match}")
    print(f"A1: impl1={r1['a1']:.6f} impl2={r2['a1']:.6f} diff={abs(r1['a1']-r2['a1']):.2e}")
    print(f"A2: impl1={r1['a2']:.6f} impl2={r2['a2']:.6f} diff={abs(r1['a2']-r2['a2']):.2e}")
    print(f"CE mean: impl1={r1['ce_mean']:.6f} impl2={r2['ce_mean']:.6f} diff={abs(r1['ce_mean']-r2['ce_mean']):.2e}")
    groups = sorted(set(r1["group_ce"]) | set(r2["group_ce"]))
    for g in groups:
        v1, v2 = r1["group_ce"].get(g), r2["group_ce"].get(g)
        print(f"  group d={g}: impl1={v1 if v1 is None else round(v1,6)} "
              f"impl2={v2 if v2 is None else round(v2,6)}")
    readout_max = max(
        (r1["readout"][k] - r2["readout"][k]).abs().max().item()
        for k in r1["readout"]
    )
    print(f"readout params after A step: max abs diff={readout_max:.3e}")
    grad_abs = grad_rel = 0.0
    worst = ("", 0.0)
    missing = 0
    for key, g1 in r1["grads"].items():
        g2 = r2["grads"].get(key)
        if g2 is None:
            missing += 1
            continue
        diff = (g1 - g2).abs().max().item()
        denom = max(g1.abs().max().item(), 1e-12)
        rel = diff / denom
        grad_abs = max(grad_abs, diff)
        grad_rel = max(grad_rel, rel)
        if diff > worst[1]:
            worst = (key, diff)
    print(f"backbone grads: params compared={len(r1['grads'])} missing={missing} "
          f"max_abs={grad_abs:.3e} max_rel={grad_rel:.3e} worst={worst[0]}")
    ok = sel_match and readout_max < 1e-5 and grad_abs < grad_tol
    print(f"VERDICT: {'PASS' if ok else 'CHECK'} (grad_tol={grad_tol:.0e})")
    return ok


def main():
    for batch_size, grad_tol in [(2, 1e-4), (8, 1e-4), (32, 1e-4)]:
        print(f"\n{'#'*70}\n# batch_size={batch_size}\n{'#'*70}")
        args, config, models, fixed_weights, autocast, input_tensor, labels_tensor = \
            build_everything(batch_size)
        model1, model2 = models
        scenarios = [
            ("natural selection", None),
            ("forced all depth-24", torch.full((input_tensor.shape[0],), config.num_layers, device="cuda", dtype=torch.long)),
            ("forced mixed depths", torch.tensor(
                [2, 4, 7, 10, 13, 16, 19, 22, 24] * ((input_tensor.shape[0] // 9) + 1),
                device="cuda", dtype=torch.long)[:input_tensor.shape[0]]),
        ]
        for name, override in scenarios:
            r1 = impl1(model1, input_tensor, labels_tensor, args, config, fixed_weights, autocast, override)
            r2 = impl2(model2, input_tensor, labels_tensor, args, config, fixed_weights, autocast, override)
            compare(name, r1, r2, grad_tol)
            print(f"timing: impl1 A={r1['t_a']:.2f}s B={r1['t_b']:.2f}s peak={r1['peak_gib']:.2f}GiB | "
                  f"impl2 shared_fwd+A={r2['t_forward']:.2f}s B={r2['t_b']:.2f}s peak={r2['peak_gib']:.2f}GiB")
        if batch_size == 8:
            # noise floor: two impl1 runs from identical starting conditions;
            # their gradient spread is the atomics-nondeterminism baseline that
            # impl1-vs-impl2 differences must be judged against
            reference = torch.load(CKPT, map_location="cpu", weights_only=False)
            model1.load_state_dict(reference["model"])
            r1a = impl1(model1, input_tensor, labels_tensor, args, config, fixed_weights, autocast)
            model1.load_state_dict(reference["model"])
            del reference
            r1b = impl1(model1, input_tensor, labels_tensor, args, config, fixed_weights, autocast)
            compare("noise floor (impl1 vs impl1, identical start)", r1a, r1b, grad_tol)
        del models, model1, model2
        torch.cuda.empty_cache()

    print(f"\n{'#'*70}\n# batch_size=160 (production scale, timing/memory focus)\n{'#'*70}")
    try:
        args, config, models, fixed_weights, autocast, input_tensor, labels_tensor = \
            build_everything(160)
        model1, model2 = models
        r1 = impl1(model1, input_tensor, labels_tensor, args, config, fixed_weights, autocast)
        print(f"impl1 (current):  A={r1['t_a']:.2f}s B={r1['t_b']:.2f}s "
              f"total={r1['t_a']+r1['t_b']:.2f}s peak={r1['peak_gib']:.2f}GiB")
        r2 = impl2(model2, input_tensor, labels_tensor, args, config, fixed_weights, autocast)
        print(f"impl2 (shared):   shared_fwd+A={r2['t_forward']:.2f}s B={r2['t_b']:.2f}s "
              f"total={r2['t_forward']+r2['t_b']:.2f}s peak={r2['peak_gib']:.2f}GiB")
        compare("natural selection @B=160", r1, r2, 1e-4)
    except torch.OutOfMemoryError as error:
        print(f"batch=160 OOM: {str(error).splitlines()[0]}")
    del models
    torch.cuda.empty_cache()
    print("\ndone")


if __name__ == "__main__":
    main()
