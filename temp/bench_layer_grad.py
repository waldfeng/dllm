"""Time one full layer's grad+checkpoint cost in a FRESH process.

The main bench process accumulates expandable-segment fragmentation across
sections; a fresh CUDA context per dispatch mode gives clean measurements.
Usage: bench_layer_grad.py <dispatch_mode> <use_gqa 0|1>
"""

from __future__ import annotations

import sys
import time

import torch
import torch.utils.checkpoint
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train as T
from pretrain_data import TokenStore, token_batches

CKPT = "data/checkpoints/joint_scratch_1epoch.pt"
LAYER_INDEX = 12


def main():
    mode = sys.argv[1]
    use_gqa = bool(int(sys.argv[2]))
    sys.argv = [
        "bench_layer_grad.py", "--train-exit-joint",
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

    handle = model.layers[LAYER_INDEX].register_forward_hook(hook)
    with torch.no_grad():
        model.joint_rollout(input_tensor)
    handle.remove()
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    from memory_transformer import MemoryTransformerConfig, MemoryReasoner, SharedMemory
    sm = SharedMemory(config).to("cuda").to(torch.bfloat16)
    layer = MemoryReasoner(config).to("cuda").to(torch.bfloat16)
    layer.eval()
    layer.use_gqa = use_gqa
    sm.dispatch_mode = mode
    x = captured["x"].to(torch.bfloat16)

    def grad_call():
        out = torch.utils.checkpoint.checkpoint(layer, x, sm, use_reentrant=False)
        out[0].sum().backward()
        layer.zero_grad(set_to_none=True)
        sm.zero_grad(set_to_none=True)

    for _ in range(3):
        grad_call()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(5):
        grad_call()
    torch.cuda.synchronize()
    grad_ms = (time.perf_counter() - start) / 5 * 1000.0
    peak = torch.cuda.max_memory_allocated() / 2**30

    with torch.no_grad():
        for _ in range(3):
            layer(x, sm)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            layer(x, sm)
        torch.cuda.synchronize()
        no_grad_ms = (time.perf_counter() - start) / 10 * 1000.0

    print(f"mode={mode} use_gqa={use_gqa}: no_grad={no_grad_ms:.1f}ms  "
          f"grad+ckpt={grad_ms:.1f}ms  peak={peak:.2f}GiB")


if __name__ == "__main__":
    main()
