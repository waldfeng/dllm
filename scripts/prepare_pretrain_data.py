"""Tokenize the downloaded JSONL corpus into mmap-friendly uint32 shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretrain_data import count_tokens, load_tokenizer, prepare_token_shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/tokens"))
    parser.add_argument(
        "--tokenizer",
        default="temp/qwen-tokenizer",
        help="local tokenizer directory or a Hugging Face/ModelScope identifier",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--tokenizer-batch-size", type=int, default=256)
    parser.add_argument("--min-tokens", type=int, default=20_000_000_000)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer, local_files_only=args.local_files_only)
    if args.count_only:
        tokens, documents = count_tokens(args.input_dir, tokenizer, args.max_tokens)
        print(f"documents={documents} tokens={tokens}")
        return

    manifest = prepare_token_shards(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        shard_tokens=args.shard_tokens,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        tokenizer_batch_size=args.tokenizer_batch_size,
    )
    print(
        f"prepared documents={manifest['documents']} "
        f"tokens={manifest['total_tokens']} shards={len(manifest['shards'])}"
    )


if __name__ == "__main__":
    main()
