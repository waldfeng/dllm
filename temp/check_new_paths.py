"""Equivalence checks for the new single-GPU training paths.

1. chunked CE == monolithic CE (same value, same grads)
2. gradient-checkpointed cycle == plain cycle (same grads)
3. AMP (fp32 master + bf16 autocast) runs and produces finite grads
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_transformer import MemoryTransformerConfig, MemoryTransformerForCausalLM


def small_model() -> MemoryTransformerForCausalLM:
    config = MemoryTransformerConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_memory_blocks=4,
        top_k=2,
        max_position_embeddings=512,
    )
    return MemoryTransformerForCausalLM(config)


def ce_reference(model, input_ids, labels):
    out = model(input_ids, labels=labels)
    return out["loss"]


def main() -> None:
    torch.manual_seed(7)
    input_ids = torch.randint(0, 256, (3, 33))
    labels = input_ids.clone()

    # --- 1. chunked CE equivalence ---------------------------------------
    model = small_model()
    model.loss_chunk_rows = 7  # forces 3 chunks over 3*32=96 rows
    loss_chunked = ce_reference(model, input_ids, labels)
    grads_chunked = torch.autograd.grad(loss_chunked, [model.memory.ffn_gate])
    model.zero_grad()
    model.loss_chunk_rows = 0
    loss_plain = ce_reference(model, input_ids, labels)
    grads_plain = torch.autograd.grad(loss_plain, [model.memory.ffn_gate])
    print(f"chunked={loss_chunked.item():.6f} plain={loss_plain.item():.6f}")
    assert abs(loss_chunked.item() - loss_plain.item()) < 1e-5, "CE value mismatch"
    assert torch.allclose(grads_chunked[0], grads_plain[0], atol=1e-6), "CE grads mismatch"
    print("PASS chunked CE == monolithic CE (value and grads)")

    # --- 2. gradient checkpointing equivalence ---------------------------
    model = small_model()
    model.gradient_checkpointing = True
    model.train()
    loss_ckpt = ce_reference(model, input_ids, labels)
    grads_ckpt = torch.autograd.grad(loss_ckpt, [model.memory.q_proj])
    model.zero_grad()
    model.gradient_checkpointing = False
    loss_plain = ce_reference(model, input_ids, labels)
    grads_plain = torch.autograd.grad(loss_plain, [model.memory.q_proj])
    print(f"ckpt={loss_ckpt.item():.6f} plain={loss_plain.item():.6f}")
    assert abs(loss_ckpt.item() - loss_plain.item()) < 1e-6, "ckpt loss mismatch"
    assert torch.allclose(grads_ckpt[0], grads_plain[0], atol=1e-7), "ckpt grads mismatch"
    print("PASS grad-checkpoint == plain backward (value and grads)")

    # --- 3. logits path untouched when chunking disabled -----------------
    model = small_model()
    model.eval()
    with torch.no_grad():
        out = model(input_ids)
    assert out["logits"].shape == (3, 33, 256)
    print("PASS legacy logits path intact")

    # --- 4. AMP on CUDA (if available) ------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = small_model().to(device)
        model.train()
        model.gradient_checkpointing = True
        model.loss_chunk_rows = 16
        ids = torch.randint(0, 256, (2, 64), device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ce_reference(model, ids, ids)
        assert torch.isfinite(loss), "non-finite AMP loss"
        loss.backward()
        assert all(
            torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
        ), "non-finite AMP grads"
        assert model.memory.ffn_gate.dtype == torch.float32, "AMP master weights must stay fp32"
        optimizer.step()
        print(f"PASS AMP step on {torch.cuda.get_device_name(0)} loss={loss.item():.4f}")
    else:
        print("SKIP AMP check (no CUDA)")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
