"""Greedy decode from a saved Memory Transformer checkpoint.

Reads a prompt, then iteratively appends argmax tokens until 100 characters
of generated text. EOS is ignored so generation does not stop early. After
each completion, reads the next prompt until EOF / empty line.

Early exit: two-stage checkpoints (ones carrying the trained
``CalibratedExitReadout``) enable the reasoner loop's depth-selection halting
-- the exit depth is the argmax of ``softmax(-l_t / temperature)`` over the
per-depth halting costs, with the trained six-term weights and temperature
restored from the checkpoint.  ``--benchmark`` measures tokens/second with the
early exit on and off and reports the speedup.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch

from memory_transformer import MemoryTransformerConfig, MemoryTransformerForCausalLM, ensure_exit_readout
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
    parser.add_argument(
        "--no-early-exit",
        action="store_true",
        help="disable the early exit even if a trained readout is present",
    )
    parser.add_argument(
        "--benchmark-tokens",
        type=int,
        default=0,
        help="greedy tokens per benchmark run; 0 disables benchmarking",
    )
    parser.add_argument("--benchmark-prompt", default="The meaning of life is")
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> MemoryTransformerForCausalLM:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = MemoryTransformerConfig(**payload["config"]) if payload.get("config") else MemoryTransformerConfig()
    if payload.get("config") and payload["config"] != asdict(config):
        raise ValueError("checkpoint config does not match MemoryTransformerConfig")
    model = MemoryTransformerForCausalLM(config)
    if any(name.startswith("exit_readout.") for name in payload["model"]):
        ensure_exit_readout(
            model, rank=int(payload.get("exit_readout_rank", config.exit_readout_rank))
        )
        model.config.early_exit_enabled = True
        if payload.get("exit_loss_weights") is not None:
            model.config.exit_selection_weights = tuple(
                float(value) for value in payload["exit_loss_weights"]
            )
        if payload.get("exit_temperature") is not None:
            model.config.exit_selection_temperature = float(payload["exit_temperature"])
        model.config.__post_init__()
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
    early_exit: bool = False,
) -> str:
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    if input_ids.numel() == 0:
        fill = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        if fill is None:
            raise ValueError("empty prompt and tokenizer has no bos/eos token")
        input_ids = torch.tensor([[fill]], device=device)
    generated = ""
    exit_depths: list[int] = []
    print(prompt, end="", flush=True)
    with torch.no_grad():
        while len(generated) < max_chars:
            output = model(input_ids, early_exit=early_exit)
            if "exit_depth" in output:
                exit_depths.append(int(output["exit_depth"]))
            logits = output["logits"][0, -1]
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
    if exit_depths:
        depth_count = model.config.num_layers
        average = sum(exit_depths) / len(exit_depths)
        print(
            f"\n[early_exit avg_depth={average:.2f}/{depth_count} "
            f"speedup={depth_count / max(average, 1e-6):.2f}x]",
            end="",
            flush=True,
        )
    print(flush=True)
    return generated


def benchmark(model, tokenizer, device: torch.device, args: argparse.Namespace) -> None:
    if args.benchmark_tokens <= 0:
        return
    prompt = args.benchmark_prompt
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    if input_ids.numel() == 0:
        raise ValueError("benchmark prompt tokenizes to an empty sequence")
    depth_count = model.config.num_layers
    results = {}
    for label, enabled in (("full_depth", False), ("early_exit", True)):
        if enabled and (args.no_early_exit or model.config.early_exit_enabled is False):
            continue
        ids = input_ids.clone()
        depths: list[int] = []
        with torch.no_grad():
            for _ in range(min(2, args.benchmark_tokens)):
                output = model(ids, early_exit=enabled)
                ids = torch.cat((ids, output["logits"][:, -1].argmax(-1, keepdim=True)), dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.monotonic()
        with torch.no_grad():
            ids = input_ids.clone()
            for _ in range(args.benchmark_tokens):
                output = model(ids, early_exit=enabled)
                if "exit_depth" in output:
                    depths.append(int(output["exit_depth"]))
                ids = torch.cat((ids, output["logits"][:, -1].argmax(-1, keepdim=True)), dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        results[label] = {
            "tokens_per_second": args.benchmark_tokens / max(elapsed, 1e-9),
            "avg_exit_depth": (sum(depths) / len(depths)) if depths else float(depth_count),
        }
    print("benchmark", flush=True)
    for label, stats in results.items():
        print(
            f"  {label}: tokens_per_s={stats['tokens_per_second']:.2f} "
            f"avg_depth={stats['avg_exit_depth']:.2f}/{depth_count}",
            flush=True,
        )
    if "full_depth" in results and "early_exit" in results:
        speedup = (
            results["early_exit"]["tokens_per_second"]
            / max(results["full_depth"]["tokens_per_second"], 1e-9)
        )
        print(f"  speedup={speedup:.2f}x", flush=True)


def main() -> None:
    args = parse_args()
    if args.max_chars <= 0:
        raise ValueError("--max-chars must be positive")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.tokenizer, local_files_only=Path(args.tokenizer).exists())
    model = load_model(args.checkpoint, device)
    if args.no_early_exit:
        model.config.early_exit_enabled = False
    benchmark(model, tokenizer, device, args)
    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_chars=args.max_chars,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        early_exit=bool(model.config.early_exit_enabled) and not args.no_early_exit,
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
