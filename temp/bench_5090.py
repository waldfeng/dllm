"""Benchmark the real 1.09B model on the RTX 5090 for the training launch.

Measures peak CUDA memory and steady-state micro-batch time (fwd+bwd) for
candidate (batch, grad-checkpoint, ce-chunk) configurations at seq 2048 with
AMP (fp32 master + bf16 autocast), including the AdamW step every GA=2 micros.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_transformer import MemoryTransformerConfig, MemoryTransformerForCausalLM

SEQ = 2048
VOCAB = 151665
GA = 2


def build(device: torch.device, grad_ckpt: bool, ce_chunk: int) -> MemoryTransformerForCausalLM:
    config = MemoryTransformerConfig(
        vocab_size=VOCAB,
        hidden_size=1536,
        intermediate_size=4096,
        num_layers=24,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_memory_blocks=24,
        top_k=3,
        max_position_embeddings=32768,
    )
    model = MemoryTransformerForCausalLM(config)
    model.gradient_checkpointing = grad_ckpt
    model.loss_chunk_rows = ce_chunk
    model.to(device=device)
    model.train()
    return model


def run_config(device, batch, grad_ckpt, ce_chunk, micros=6):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build(device, grad_ckpt, ce_chunk)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    ids = torch.randint(0, VOCAB, (batch, SEQ), device=device)
    times = []
    tokens = batch * SEQ
    oom = False
    try:
        for micro in range(micros):
            if micro % GA == 0:
                optimizer.zero_grad(set_to_none=True)
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(ids, labels=ids)
                loss = out["loss"] + 0.01 * out["router_loss"]
            loss.backward()
            if (micro + 1) % GA == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            if micro == 1:
                torch.cuda.reset_peak_memory_stats(device)
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        steady = min(times[2:]) if len(times) > 2 else times[-1]
        tps = tokens / steady
        print(
            f"B={batch:>2} ckpt={int(grad_ckpt)} chunk={ce_chunk or 'off':>5} "
            f"peak={peak:5.1f}GiB micro={steady*1e3:6.0f}ms "
            f"tokens/s={tps:7.0f} loss={loss.item():.3f}",
            flush=True,
        )
    except torch.OutOfMemoryError:
        oom = True
        print(f"B={batch:>2} ckpt={int(grad_ckpt)} chunk={ce_chunk or 'off':>5} OOM", flush=True)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return oom


def main() -> None:
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    total_gib = torch.cuda.get_device_properties(device).total_memory / 2**30
    print(f"device={torch.cuda.get_device_name(device)} total={total_gib:.1f}GiB seq={SEQ} GA={GA}")
    # warmup kernels
    run_config(device, 2, True, 2048, micros=2)
    candidates = [
        (8, True, 2048),
        (10, True, 2048),
        (11, True, 2048),
        (12, True, 2048),
        (10, True, 4096),
    ]
    for batch, ckpt, chunk in candidates:
        if run_config(device, batch, ckpt, chunk):
            continue


if __name__ == "__main__":
    main()
