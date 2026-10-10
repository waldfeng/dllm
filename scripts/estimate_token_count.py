"""Estimate corpus token count from a small complete-JSONL sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import PreTrainedTokenizerBase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretrain_data import load_tokenizer


def sample_file(
    path: Path,
    tokenizer: PreTrainedTokenizerBase,
    target_bytes: int,
) -> dict[str, int | float | str]:
    raw_bytes = 0
    records: list[str] = []
    with path.open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text") if isinstance(record, dict) else None
            if not isinstance(text, str):
                raise ValueError(f"missing string field 'text' in {path}")
            raw_bytes += len(line)
            records.append(text)
            if raw_bytes >= target_bytes:
                break
    encoded = tokenizer(
        records,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    tokens = sum(len(ids) + 1 for ids in encoded)
    return {
        "file": str(path),
        "records": len(records),
        "raw_bytes": raw_bytes,
        "tokens_with_eos": tokens,
        "tokens_per_byte": tokens / raw_bytes if raw_bytes else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/datasets/dllm/datasets/manifest.json"),
    )
    parser.add_argument("--tokenizer", default="tokenizer/qwen")
    parser.add_argument("--sample-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--sample-files", type=int, default=1)
    parser.add_argument("--target-tokens", type=int, default=20_000_000_000)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if args.sample_bytes <= 0 or args.sample_files <= 0:
        raise ValueError("sample-bytes and sample-files must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(args.tokenizer, local_files_only=Path(args.tokenizer).exists())
    samples = []
    total_bytes = 0
    missing = []
    for item in manifest["files"]:
        path = args.manifest.parent / item["repo_id"].replace("/", "__") / item["path"]
        if not path.exists():
            missing.append(str(path))
            continue
        total_bytes += int(item["size"])
        if len(samples) < args.sample_files:
            samples.append(sample_file(path, tokenizer, args.sample_bytes))
    if not samples:
        raise ValueError("none of the manifest files are present")
    sample_bytes = sum(int(sample["raw_bytes"]) for sample in samples)
    sample_tokens = sum(int(sample["tokens_with_eos"]) for sample in samples)
    ratio = sample_tokens / sample_bytes
    estimated_tokens = int(total_bytes * ratio)
    deficit = max(0, args.target_tokens - estimated_tokens)
    estimate = {
        "method": "complete_jsonl_prefix_sample",
        "tokenizer": getattr(tokenizer, "name_or_path", args.tokenizer),
        "sample_bytes": sample_bytes,
        "sample_tokens_with_eos": sample_tokens,
        "tokens_per_byte": ratio,
        "estimated_total_bytes": total_bytes,
        "estimated_total_tokens": estimated_tokens,
        "target_tokens": args.target_tokens,
        "estimated_deficit_tokens": deficit,
        "estimated_extra_bytes_needed": int(deficit / ratio) if ratio else None,
        "samples": samples,
        "missing_manifest_files": missing,
    }
    if args.write_manifest:
        manifest["token_estimate"] = estimate
        temporary = args.manifest.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(args.manifest)
    print(json.dumps(estimate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
