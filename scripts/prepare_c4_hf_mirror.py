"""Fill an existing C4 token directory from the HF mirror stream."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from transformers import AutoTokenizer


def write_metadata(path: Path, metadata: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(path)


def write_progress(path: Path, progress: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(progress, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2_000_000_000)
    parser.add_argument(
        "--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face tokenizer identifier",
    )
    parser.add_argument("--dataset-dir", default="datasets/c4_2b")
    parser.add_argument("--cache-dir", default="datasets/.hf-mirror")
    parser.add_argument("--chunk-tokens", type=int, default=4_000_000)
    parser.add_argument(
        "--skip-shards",
        default="",
        help="comma-separated source shard numbers already materialized from another source",
    )
    args = parser.parse_args()
    os.environ["HF_HOME"] = str(Path(args.cache_dir).resolve())

    root = Path(args.dataset_dir)
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(root.glob("train-*.bin"))
    progress_path = root / "progress.json"
    metadata_path = root / "metadata.json"
    offsets_path = root / "samples.offsets"
    previous_metadata = (
        json.loads(metadata_path.read_text())
        if metadata_path.exists()
        else {}
    )
    if not existing:
        np.asarray([0], dtype=np.uint64).tofile(offsets_path)
    elif not offsets_path.exists():
        raise RuntimeError("cannot resume without samples.offsets; use a new dataset-dir")
    if existing and previous_metadata:
        if previous_metadata.get("tokenizer") != args.tokenizer:
            raise RuntimeError("existing dataset uses a different tokenizer; use a new dataset-dir")
    if existing:
        metadata_path = root / "metadata.json"
        if metadata_path.exists():
            existing_metadata = json.loads(metadata_path.read_text())
            if not any(
                source in existing_metadata.get("source", "")
                for source in ("hf-mirror", "modelscope")
            ):
                raise RuntimeError(
                    "Refusing to append HF mirror data to shards from another source"
                )
        if not progress_path.exists():
            raise RuntimeError(
                "Cannot safely resume without progress.json; use a new dataset-dir"
            )
    def recover_pending() -> None:
        if not progress_path.exists():
            return
        state = json.loads(progress_path.read_text())
        for name in state.get("pending_outputs", []):
            temporary = root / name
            if temporary.exists():
                temporary.replace(temporary.with_suffix(""))

    recover_pending()
    existing = sorted(root.glob("train-*.bin"))
    produced = sum(path.stat().st_size // 4 for path in existing)
    offsets = np.memmap(offsets_path, dtype=np.uint64, mode="r")
    committed_count = int(np.searchsorted(offsets, produced, side="right"))
    if committed_count == 0 or int(offsets[committed_count - 1]) != produced:
        raise RuntimeError("samples.offsets does not match committed token shards")
    if committed_count != len(offsets):
        del offsets
        with offsets_path.open("r+b") as stream:
            stream.truncate(committed_count * np.dtype(np.uint64).itemsize)
    else:
        del offsets
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define both eos_token_id and pad_token_id")
    eos_token_id = tokenizer.eos_token_id
    if produced >= args.tokens:
        metadata = {
            "source": "hf-mirror:allenai/c4",
            "tokenizer": args.tokenizer,
            "tokenizer_vocab_size": len(tokenizer),
            "tokenizer_eos_id": tokenizer.eos_token_id,
            "tokenizer_pad_id": tokenizer.pad_token_id,
            "tokens": produced,
            "sample_count": max(0, len(np.memmap(offsets_path, dtype=np.uint64, mode="r")) - 1),
            "offsets_file": offsets_path.name,
            "dtype": "uint32",
            "block_size": 8,
        }
        metadata_path = root / "metadata.json"
        metadata["tokens_verified"] = True
        write_metadata(metadata_path, metadata)
        print(json.dumps({"tokens": produced, "status": "already_complete"}))
        return
    state = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    start_shard = int(state.get("next_shard", 0))
    shard_index = len(existing)
    buffer: list[int] = []
    sample_count = committed_count - 1
    offset_total = produced
    offsets_file = offsets_path.open("ab")

    def sync_offsets() -> None:
        offsets_file.flush()
        os.fsync(offsets_file.fileno())
    pending_outputs: list[Path] = []
    # The datasets client currently sends tree/list requests to huggingface.co
    # even when HF_ENDPOINT is set. Resolve known C4 shards directly instead.
    source_shards = [f"{i:05d}" for i in range(1024)]
    skip_shards = {
        item.strip().zfill(5)
        for item in args.skip_shards.split(",")
        if item.strip()
    }

    def flush(force: bool = False) -> None:
        nonlocal shard_index
        if not buffer or (len(buffer) < args.chunk_tokens and not force):
            return
        target = root / f"train-{shard_index:05d}.bin"
        temporary = target.with_suffix(target.suffix + ".tmp")
        np.asarray(buffer, dtype=np.uint32).tofile(temporary)
        pending_outputs.append(temporary)
        buffer.clear()
        shard_index += 1

    def commit_outputs() -> None:
        for temporary in pending_outputs:
            temporary.replace(temporary.with_suffix(""))
        pending_outputs.clear()

    consumed_shards = start_shard
    for shard_number, shard in enumerate(source_shards[start_shard:], start=start_shard):
        if shard in skip_shards:
            consumed_shards = shard_number + 1
            continue
        if produced >= args.tokens:
            break
        url = f"https://hf-mirror.com/datasets/allenai/c4/resolve/main/en/c4-train.{shard}-of-01024.json.gz"
        source_path = Path(args.cache_dir) / f"c4-train.{shard}-of-01024.json.gz"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [
                    "curl", "-L", "--fail", "--retry", "8",
                    "--retry-all-errors", "--connect-timeout", "60",
                    "--max-time", "3600", "-o", str(source_path), url,
                ],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"failed to download source shard {shard}") from exc
        with gzip.open(source_path, "rb") as stream:
            for raw_line in stream:
                remaining = args.tokens - produced
                if remaining <= 0:
                    break
                row = json.loads(raw_line)
                ids = tokenizer.encode(
                    row.get("text", ""), add_special_tokens=False
                )[: max(0, remaining - 1)]
                if not ids:
                    continue
                ids.append(eos_token_id)
                buffer.extend(ids)
                sample_count += 1
                offset_total += len(ids)
                np.asarray([offset_total], dtype=np.uint64).tofile(offsets_file)
                produced += len(ids)
                flush()
            flush(force=True)
            sync_offsets()
            write_progress(
                progress_path,
                {
                    "source": "hf-mirror:allenai/c4",
                    "next_shard": shard_number + 1,
                    "tokens": produced,
                    "sample_count": sample_count,
                    "offsets_count": sample_count + 1,
                    "pending_outputs": [item.name for item in pending_outputs],
                },
            )
            commit_outputs()
            source_path.unlink(missing_ok=True)
            write_progress(
                progress_path,
                {
                    "source": "hf-mirror:allenai/c4",
                    "next_shard": shard_number + 1,
                    "tokens": produced,
                },
            )
        consumed_shards = shard_number + 1
    flush(force=True)
    sync_offsets()
    commit_outputs()
    metadata = {
        "source": "modelscope+hf-mirror:allenai/c4",
        "tokenizer": args.tokenizer,
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_eos_id": tokenizer.eos_token_id,
        "tokenizer_pad_id": tokenizer.pad_token_id,
        "tokens": sum(
            path.stat().st_size // np.dtype(np.uint32).itemsize
            for path in root.glob("train-*.bin")
        ),
        "dtype": "uint32",
        "block_size": 8,
    }
    metadata["source_shards"] = consumed_shards
    metadata["tokens_verified"] = True
    metadata["sample_count"] = sample_count
    metadata["offsets_file"] = offsets_path.name
    offsets_file.close()
    write_metadata(root / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
