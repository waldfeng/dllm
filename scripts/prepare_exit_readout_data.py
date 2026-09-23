#!/usr/bin/env python3
"""Extract plain-text records from the WIT parquet shards for readout training.

Reads ONLY the text columns of ``wit_filtered`` parquet shards (the shards hold
no image bytes at all); image archives are never touched.  Each row becomes one
``{"text": ...}`` JSONL record so ``prepare_pretrain_data.py`` can tokenize it
directly.  The source directory is opened strictly read-only and all outputs
are written under this repository's ``data/`` directory.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

TEXT_COLUMNS = (
    "caption",
    "caption_attribution_description",
    "caption_alt_text_description",
    "page_title",
    "section_title",
    "context_page_description",
    "context_section_description",
)
MIN_CHARS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/src/dvlm/data/wit_filtered/data"),
        help="read-only directory holding wit_filtered parquet shards",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/text"),
        help="destination for wit-XXXXX.jsonl.gz records",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=0,
        help="process only the first N parquet shards (0 = all)",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=0,
        help="cap rows per shard (0 = all rows)",
    )
    parser.add_argument(
        "--row-batch-size",
        type=int,
        default=2048,
        help="rows per streaming parquet read batch",
    )
    return parser.parse_args()


def convert_shard(
    source: Path, destination: Path, max_rows: int, row_batch_size: int
) -> tuple[int, int]:
    rows_written = 0
    characters = 0
    parquet = pq.ParquetFile(source)
    with gzip.open(destination, "wt", encoding="utf-8", compresslevel=6) as handle:
        for table in parquet.iter_batches(
            batch_size=max(64, row_batch_size), columns=list(TEXT_COLUMNS)
        ):
            columns = {name: table.column(name).to_pylist() for name in TEXT_COLUMNS}
            for values in zip(*(columns[name] for name in TEXT_COLUMNS)):
                text = "\n".join(
                    str(value).strip() for value in values if value is not None and str(value).strip()
                )
                if len(text) < MIN_CHARS:
                    continue
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                rows_written += 1
                characters += len(text)
                if max_rows and rows_written >= max_rows:
                    return rows_written, characters
    return rows_written, characters


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"input directory {args.input_dir} does not exist")
    shards = sorted(args.input_dir.glob("*.parquet"))
    if args.num_files:
        shards = shards[: args.num_files]
    if not shards:
        raise SystemExit(f"no parquet shards under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_rows, total_chars = 0, 0
    for position, source in enumerate(shards):
        destination = args.output_dir / f"wit-{position:05d}.jsonl.gz"
        rows, characters = convert_shard(
            source, destination, args.max_rows_per_file, args.row_batch_size
        )
        total_rows += rows
        total_chars += characters
        print(
            f"extracted {source.name} -> {destination.name} "
            f"rows={rows} chars={characters} total_rows={total_rows}",
            flush=True,
        )
    print(
        f"done shards={len(shards)} rows={total_rows} "
        f"chars={total_chars} (~{total_chars // 4} tokens est.)",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
