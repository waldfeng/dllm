"""Greedy decode from a saved Memory Transformer checkpoint.

Reads a prompt, then iteratively appends argmax tokens until 100 characters
of generated text. EOS is ignored so generation does not stop early. After
each completion, reads the next prompt until EOF / empty line.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from memory_transformer import MemoryTransformerConfig, MemoryTransformerForCausalLM
from pretrain_data import load_tokenizer

MAX_CHARS = 100
DEFAULT_CHECKPOINT = Path("datasets/checkpoints/memory-transformer-24h-router03.pt")
DEFAULT_TOKENIZER = "tokenizer/qwen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> MemoryTransformerForCausalLM:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = MemoryTransformerConfig(**payload["config"]) if payload.get("config") else MemoryTransformerConfig()
    if payload.get("config") and payload["config"] != asdict(config):
        raise ValueError("checkpoint config does not match MemoryTransformerConfig")
    model = MemoryTransformerForCausalLM(config)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model


def apply_repetition_penalty(logits: torch.Tensor, token_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0 or token_ids.numel() == 0:
        return logits
    unique = token_ids.unique()
    selected = logits[unique]
    adjusted = torch.where(selected > 0, selected / penalty, selected * penalty)
    logits = logits.clone()
    logits[unique] = adjusted
    return logits


def sample_next(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(logits.argmax(dim=-1).item())
    scores = logits / temperature
    probs = torch.softmax(scores, dim=-1)
    if 0 < top_p < 1:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        denom = sorted_probs.sum()
        if float(denom) == 0:
            return int(logits.argmax(dim=-1).item())
        sorted_probs = sorted_probs / denom
        choice = torch.multinomial(sorted_probs, 1)
        return int(sorted_idx[choice].item())
    return int(torch.multinomial(probs, 1).item())


def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_chars: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    if input_ids.numel() == 0:
        fill = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        if fill is None:
            raise ValueError("empty prompt and tokenizer has no bos/eos token")
        input_ids = torch.tensor([[fill]], device=device)
    generated = ""
    print(prompt, end="", flush=True)
    with torch.no_grad():
        while len(generated) < max_chars:
            logits = model(input_ids)["logits"][0, -1]
            logits = apply_repetition_penalty(logits, input_ids[0], repetition_penalty)
            next_id = sample_next(logits, temperature, top_p)
            piece = tokenizer.decode([next_id], skip_special_tokens=True)
            if not piece:
                remain = max_chars - len(generated)
                piece = " "[:remain]
            remain = max_chars - len(generated)
            if len(piece) > remain:
                piece = piece[:remain]
            generated += piece
            print(piece, end="", flush=True)
            input_ids = torch.cat(
                (input_ids, torch.tensor([[next_id]], device=device)), dim=-1
            )
    print(flush=True)
    return generated


def main() -> None:
    args = parse_args()
    if args.max_chars <= 0:
        raise ValueError("--max-chars must be positive")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.tokenizer, local_files_only=Path(args.tokenizer).exists())
    model = load_model(args.checkpoint, device)
    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_chars=args.max_chars,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    if args.prompt:
        generate_text(prompt=args.prompt, **kwargs)
        return
    while True:
        try:
            prompt = input("prompt: ")
        except EOFError:
            break
        if not prompt.strip():
            break
        generate_text(prompt=prompt, **kwargs)


if __name__ == "__main__":
    main()
