"""Inspect the relocated training checkpoints (mmap, no full RAM load)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

CKPTS = [
    Path("/home/datasets/dllm/datasets/checkpoints/memory-transformer-24h-router03.pt"),
    Path("/home/datasets/dllm/data/checkpoints/exit-readout-v2.pt"),
]


def main() -> None:
    torch.set_num_threads(4)
    for path in CKPTS:
        print(f"===== {path}")
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        print("keys:", sorted(payload.keys()))
        print("step:", payload.get("step"))
        print("data:", payload.get("data") or payload.get("schedule"))
        config = payload.get("config") or {}
        print("config:", {k: config.get(k) for k in (
            "vocab_size", "hidden_size", "intermediate_size", "num_layers",
            "num_attention_heads", "num_key_value_heads", "num_memory_blocks",
            "top_k", "tie_word_embeddings")})
        model_state = payload.get("model") or {}
        n_tensors = len(model_state)
        n_params = sum(v.numel() for v in model_state.values())
        print(f"model tensors={n_tensors} params={n_params:,}")
        optim = payload.get("optimizer")
        if optim:
            groups = optim.get("param_groups") or []
            sizes = [len(g["params"]) for g in groups]
            n_state = len(optim.get("state") or {})
            print(f"optimizer groups={len(groups)} sizes={sizes} state_entries={n_state}")
            print("optimizer lr:", [g.get("lr") for g in groups][:3])
        sched = payload.get("scheduler") or {}
        if sched:
            print("scheduler last_epoch:", sched.get("last_epoch"))
        readout_keys = [k for k in model_state if k.startswith("exit_readout.")]
        print("exit_readout tensors:", len(readout_keys))
        if payload.get("exit_loss_weights") is not None:
            print("exit_loss_weights:", payload["exit_loss_weights"])
        del payload
        print()


if __name__ == "__main__":
    sys.exit(main())
