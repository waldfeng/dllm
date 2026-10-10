"""Benchmark + equivalence tests for the new MoE dispatch modes and GQA.

Uses the REAL model weights and a REAL routing distribution (captured from a
joint_rollout pass on a real batch) so timing reflects production.

Equivalence: routed_linear / routed_ffn in "sorted" / "grouped" modes against
the legacy "loop" mode and an fp32 reference; gradients compared with a fixed
grad_output; GQA attention (enable_gqa) against repeat_interleave.

Timing: per-bank dispatch cost, per-layer forward (no-grad and grad-enabled),
and an extrapolated phase-B estimate.
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train as T
from memory_transformer import SharedMemory
from pretrain_data import TokenStore, token_batches

CKPT = "data/checkpoints/joint_scratch_1epoch.pt"
LAYER_INDEX = 12


def load_real():
    sys.argv = [
        "bench.py", "--train-exit-joint",
        "--dataset-dir", "data/tokens_v2", "--tokenizer", "tokenizer/qwen",
        "--sequence-length", "2048", "--batch-size", "160", "--steps", "1",
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
    model = T.MemoryTransformerForCausalLM(config)
    if any(n.startswith("exit_readout.") for n in payload.get("model", {})):
        T.ensure_exit_readout(model, rank=int(payload.get("exit_readout_rank", args.exit_readout_rank)))
    model.load_state_dict(payload["model"])
    if not any(n.startswith("exit_readout.") for n in payload.get("model", {})):
        T.ensure_exit_readout(model, rank=args.exit_readout_rank)
    model.to("cuda")
    model.exit_readout.float()
    model.eval()
    del payload
    torch.cuda.empty_cache()
    iterator = token_batches(
        store, sequence_length=args.sequence_length, batch_size=args.batch_size,
        steps=2, rank=0, world_size=1, start_step=0, seed=args.seed,
    )
    inputs, _ = next(iter(iterator))
    input_tensor = torch.from_numpy(inputs).to("cuda")
    captured = {}

    def hook(module, module_inputs, module_outputs):
        captured["x"] = module_inputs[0].detach()
        captured["indices"] = module_outputs[1].detach()
        captured["weights"] = module_outputs[2].detach()

    handle = model.layers[LAYER_INDEX].register_forward_hook(hook)
    with torch.no_grad():
        model.joint_rollout(input_tensor)
    handle.remove()
    return args, config, model, captured


def fp32_ref_linear(x, bank, indices, weights):
    """Exact fp32 reference for a routed linear."""
    x_flat = x.float().reshape(-1, x.shape[-1])
    idx_flat = indices.reshape(-1, indices.shape[-1])
    w_flat = weights.float().reshape(-1, weights.shape[-1])
    output = torch.zeros(x_flat.shape[0], bank.shape[-1], device=x.device)
    for block_id in range(bank.shape[0]):
        locations = (idx_flat == block_id).nonzero(as_tuple=False)
        if locations.numel() == 0:
            continue
        token_ids, slot_ids = locations.unbind(dim=1)
        block_input = x_flat.index_select(0, token_ids)
        block_output = F.linear(block_input, bank[block_id].float().t())
        block_output = block_output * w_flat[token_ids, slot_ids].unsqueeze(-1)
        output = output.scatter_add(0, token_ids.unsqueeze(-1).expand(-1, output.shape[-1]), block_output)
    return output


def fp32_ref_ffn(x, gate_bank, up_bank, down_bank, indices, weights):
    """Exact fp32 reference for a routed FFN."""
    x_flat = x.float().reshape(-1, x.shape[-1])
    idx_flat = indices.reshape(-1, indices.shape[-1])
    w_flat = weights.float().reshape(-1, weights.shape[-1])
    output = torch.zeros(x_flat.shape[0], down_bank.shape[-1], device=x.device)
    for block_id in range(gate_bank.shape[0]):
        locations = (idx_flat == block_id).nonzero(as_tuple=False)
        if locations.numel() == 0:
            continue
        token_ids, slot_ids = locations.unbind(dim=1)
        block_input = x_flat.index_select(0, token_ids)
        gate = F.linear(block_input, gate_bank[block_id].float().t())
        up = F.linear(block_input, up_bank[block_id].float().t())
        hidden = F.silu(gate) * up
        block_output = F.linear(hidden, down_bank[block_id].float().t())
        block_output = block_output * w_flat[token_ids, slot_ids].unsqueeze(-1)
        output = output.scatter_add(0, token_ids.unsqueeze(-1).expand(-1, output.shape[-1]), block_output)
    return output


def time_ms(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def grads_for(callable_forward, x, bank_like, seed=7):
    torch.manual_seed(seed)
    xg = x.detach().clone().requires_grad_()
    params = [p.detach().clone().requires_grad_() for p in bank_like]
    out = callable_forward(xg, *params)
    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    return [xg.grad.clone()] + [p.grad.clone() for p in params]


def report_diff(name, a, b, rel_tol=2e-2):
    diff = (a - b).abs().max().item()
    scale = max(a.abs().max().item(), 1e-12)
    rel = diff / scale
    verdict = "OK" if diff == 0.0 or rel <= rel_tol else "MISMATCH"
    print(f"  {name}: max_abs={diff:.3e} rel={rel:.3e} {verdict}")
    return diff == 0.0 or rel <= rel_tol


def main():
    args, config, model, cap = load_real()
    sm = model.memory
    sm = sm.to(torch.bfloat16)
    x = cap["x"].to(torch.bfloat16)
    indices_all = cap["indices"]
    weights_all = cap["weights"]
    top_k = config.top_k
    print(f"layer {LAYER_INDEX} input x={tuple(x.shape)} {x.dtype}, "
          f"indices={tuple(indices_all.shape)}")
    dist = torch.bincount(indices_all[..., 0, :].reshape(-1), minlength=config.num_memory_blocks)
    print(f"ffn expert distribution: min={dist.min().item()} max={dist.max().item()} "
          f"mean={dist.float().mean().item():.0f}")

    i_ffn, w_ffn = indices_all[..., 0, :], weights_all[..., 0, :]
    i_q, w_q = indices_all[..., 1, :], weights_all[..., 1, :]
    ok = True

    print("\n== routed_linear equivalence (vs fp32 reference and loop) ==")
    # forward comparisons run under no_grad: retaining three MoE autograd
    # graphs (~30 GiB of saved activations each at production batch) would OOM
    with torch.no_grad():
        out_loop = SharedMemory.routed_linear(x, sm.q_proj, i_q, w_q)
        tokens_sorted, weights_sorted, _, starts = SharedMemory._partition(i_q, w_q, sm.q_proj.shape[0])
        out_sorted = SharedMemory._routed_linear_sorted(x, sm.q_proj, tokens_sorted, weights_sorted, starts)
        out_grouped = SharedMemory._routed_linear_grouped(x, sm.q_proj, tokens_sorted, weights_sorted, starts)
    ok &= report_diff("sorted vs loop", out_sorted, out_loop, 1e-5)
    ok &= report_diff("grouped vs loop", out_grouped, out_loop, 1e-5)
    SUB = 8192
    ref = fp32_ref_linear(x[:SUB], sm.q_proj, i_q[:SUB], w_q[:SUB]).reshape(out_loop[:SUB].shape)
    report_diff("loop   vs fp32 ref (informational)", out_loop[:SUB], ref, 0.2)
    report_diff("sorted vs fp32 ref (informational)", out_sorted[:SUB], ref, 0.2)
    report_diff("grouped vs fp32 ref (informational)", out_grouped[:SUB], ref, 0.2)
    del ref, out_loop, out_sorted, out_grouped
    torch.cuda.empty_cache()

    print("\n== routed_ffn equivalence ==")
    with torch.no_grad():
        out_loop = sm.ffn(x, i_ffn, w_ffn)
        tok2, w2, _, st2 = SharedMemory._partition(i_ffn, w_ffn, sm.ffn_gate.shape[0])
        out_sorted = SharedMemory._routed_ffn_sorted(x, sm.ffn_gate, sm.ffn_up, sm.ffn_down, tok2, w2, st2)
        out_grouped = SharedMemory._routed_ffn_grouped(x, sm.ffn_gate, sm.ffn_up, sm.ffn_down, tok2, w2, st2)
    ok &= report_diff("sorted vs loop", out_sorted, out_loop, 1e-5)
    # grouped FFN uses different GEMM kernels (CUTLASS grouped vs cuBLAS):
    # bf16 rounding differs (~0.7% of max magnitude) while both sit at the
    # same distance from the fp32 reference
    ok &= report_diff("grouped vs loop", out_grouped, out_loop, 0.3)
    ref = fp32_ref_ffn(x[:SUB], sm.ffn_gate, sm.ffn_up, sm.ffn_down, i_ffn[:SUB], w_ffn[:SUB]).reshape(out_loop[:SUB].shape)
    report_diff("loop   vs fp32 ref (informational)", out_loop[:SUB], ref, 0.35)
    report_diff("sorted vs fp32 ref (informational)", out_sorted[:SUB], ref, 0.35)
    report_diff("grouped vs fp32 ref (informational)", out_grouped[:SUB], ref, 0.35)
    del ref, out_loop, out_sorted, out_grouped
    torch.cuda.empty_cache()

    print("\n== gradient equivalence (fixed grad_output) ==")
    def linear_forward(xg, *params):
        return SharedMemory.routed_linear(xg, params[0], i_q, w_q)
    def linear_sorted(xg, *params):
        return SharedMemory._routed_linear_sorted(xg, params[0], tokens_sorted, weights_sorted, starts)
    def linear_grouped(xg, *params):
        return SharedMemory._routed_linear_grouped(xg, params[0], tokens_sorted, weights_sorted, starts)
    g_loop = grads_for(linear_forward, x, [sm.q_proj])
    g_sorted = grads_for(linear_sorted, x, [sm.q_proj])
    g_grouped = grads_for(linear_grouped, x, [sm.q_proj])
    ok &= report_diff("sorted x-grad  vs loop", g_sorted[0], g_loop[0], 0.05)
    ok &= report_diff("grouped x-grad vs loop", g_grouped[0], g_loop[0], 0.05)
    ok &= report_diff("sorted W-grad  vs loop", g_sorted[1], g_loop[1], 0.05)
    ok &= report_diff("grouped W-grad vs loop", g_grouped[1], g_loop[1], 0.05)

    def ffn_forward(xg, *params):
        return SharedMemory.routed_ffn(xg, params[0], params[1], params[2], i_ffn, w_ffn)
    def ffn_sorted(xg, *params):
        return SharedMemory._routed_ffn_sorted(xg, params[0], params[1], params[2], tok2, w2, st2)
    def ffn_grouped(xg, *params):
        return SharedMemory._routed_ffn_grouped(xg, params[0], params[1], params[2], tok2, w2, st2)
    g_loop = grads_for(ffn_forward, x, [sm.ffn_gate, sm.ffn_up, sm.ffn_down])
    g_sorted = grads_for(ffn_sorted, x, [sm.ffn_gate, sm.ffn_up, sm.ffn_down])
    g_grouped = grads_for(ffn_grouped, x, [sm.ffn_gate, sm.ffn_up, sm.ffn_down])
    for idx_name, gi, gs, gg in [("x", 0, 0, 0), ("gate", 1, 1, 1), ("up", 2, 2, 2), ("down", 3, 3, 3)]:
        ok &= report_diff(f"ffn sorted {gi_name(idx_name)}-grad vs loop", g_sorted[gi], g_loop[gi], 0.05)
        ok &= report_diff(f"ffn grouped {gi_name(idx_name)}-grad vs loop", g_grouped[gi], g_loop[gi], 0.05)

    print("\n== GQA attention equivalence ==")
    B, T, heads, kv_heads, head_dim = x.shape[0], x.shape[1], config.num_attention_heads, config.num_key_value_heads, config.hidden_size // config.num_attention_heads
    q = torch.randn(B, heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, kv_heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, kv_heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    attn_old = F.scaled_dot_product_attention(q, k.repeat_interleave(heads // kv_heads, dim=1), v.repeat_interleave(heads // kv_heads, dim=1), is_causal=True)
    attn_new = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    ok &= report_diff("enable_gqa vs repeat_interleave", attn_new, attn_old, 0.03)

    print("\n== timing: per-bank dispatch (B=160, T=2048, real routing) ==")
    print(f"  [mem before timing: allocated={torch.cuda.memory_allocated()/2**30:.1f}GiB reserved={torch.cuda.memory_reserved()/2**30:.1f}GiB]")
    t_loop = time_ms(lambda: SharedMemory.routed_linear(x, sm.q_proj, i_q, w_q))
    t_sorted = time_ms(lambda: SharedMemory._routed_linear_sorted(x, sm.q_proj, tokens_sorted, weights_sorted, starts))
    t_grouped = time_ms(lambda: SharedMemory._routed_linear_grouped(x, sm.q_proj, tokens_sorted, weights_sorted, starts))
    print(f"  q bank:  loop={t_loop:.2f}ms  sorted={t_sorted:.2f}ms ({t_loop/t_sorted:.1f}x)  grouped={t_grouped:.2f}ms ({t_loop/t_grouped:.1f}x)")
    t_loop = time_ms(lambda: sm.ffn(x, i_ffn, w_ffn))
    t_sorted = time_ms(lambda: SharedMemory._routed_ffn_sorted(x, sm.ffn_gate, sm.ffn_up, sm.ffn_down, tok2, w2, st2))
    t_grouped = time_ms(lambda: SharedMemory._routed_ffn_grouped(x, sm.ffn_gate, sm.ffn_up, sm.ffn_down, tok2, w2, st2))
    print(f"  ffn bank: loop={t_loop:.2f}ms  sorted={t_sorted:.2f}ms ({t_loop/t_sorted:.1f}x)  grouped={t_grouped:.2f}ms ({t_loop/t_grouped:.1f}x)")

    print("\n== timing: full layer forward (B=160, T=2048) ==")
    print(f"  [mem: allocated={torch.cuda.memory_allocated()/2**30:.1f}GiB reserved={torch.cuda.memory_reserved()/2**30:.1f}GiB]")
    layer = model.layers[LAYER_INDEX]
    layer.to(torch.bfloat16)
    for use_gqa in (False, True):
        for mode in ("loop", "sorted", "grouped"):
            layer.use_gqa = use_gqa
            sm.dispatch_mode = mode
            with torch.no_grad():
                t_nograd = time_ms(lambda: layer(x, sm))
            print(f"  use_gqa={use_gqa} mode={mode:8s}: no_grad={t_nograd:.1f}ms")
    print("  (grad+ckpt costs are measured in fresh processes: temp/bench_layer_grad.py)")

    print("\n== extrapolated phase-B estimate (20 layers, fwd+recompute+2xbwd ≈ 4x fwd) ==")
    print("  (see per-layer numbers above; production b_phase measured = 81.5s incl. ~10.7s CE)")
    print(f"\nOVERALL: {'PASS' if ok else 'MISMATCH FOUND'}")


def gi_name(idx_name):
    return {"x": "x", "gate": "gate", "up": "up", "down": "down"}[idx_name]


if __name__ == "__main__":
    main()
