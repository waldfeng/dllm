"""Minimal causal-LM training entry point. Use Qwen2.5 tokenizer when available."""

import argparse

import torch
from transformers import AutoTokenizer

from memory_transformer import QWEN25_TOKENIZER, MemoryTransformerConfig, MemoryTransformerForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--recurrent-steps", type=int, default=1)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(QWEN25_TOKENIZER)
    config = MemoryTransformerConfig(vocab_size=len(tokenizer))
    model = MemoryTransformerForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    model.train()
    for step in range(args.steps):
        ids = torch.randint(tokenizer.vocab_size, (args.batch_size, args.seq_len))
        out = model(ids, labels=ids, recurrent_steps=args.recurrent_steps)
        (out["loss"] + 0.01 * out["router_loss"]).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"step={step} loss={out['loss'].item():.4f} router={out['router_loss'].item():.4f}")


if __name__ == "__main__":
    main()
