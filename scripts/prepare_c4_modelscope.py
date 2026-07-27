"""Download and tokenize a bounded C4 slice from ModelScope.

The full C4 corpus is far larger than this workspace. This utility downloads
only enough compressed C4 shards to produce the requested token budget, then
writes compact uint32 token shards under ``datasets/c4_2b``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
from modelscope.hub.file_download import dataset_file_download
from modelscope.hub.api import HubApi
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
    parser.add_argument("--shards", type=int, default=32, help="maximum source shards to consume")
    parser.add_argument(
        "--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face tokenizer identifier",
    )
    parser.add_argument("--dataset-dir", default="datasets/c4_2b")
    parser.add_argument("--modelscope-cache", default="datasets/.modelscope")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-index", type=int, default=-1)
    parser.add_argument(
        "--keep-source-cache",
        action="store_true",
        help="keep downloaded compressed source shards after tokenization",
    )
    args = parser.parse_args()

    # ModelScope remains the primary dataset source. The tokenizer fallback is
    # explicitly pointed at hf-mirror because the default HF endpoint is not
    # reachable in this environment.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    out_dir = Path(args.dataset_dir)
    cache_dir = Path(args.modelscope_cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if list(out_dir.glob("train-*.bin")) and not args.resume:
        raise RuntimeError("dataset directory is not empty; use --resume or a new directory")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define both eos_token_id and pad_token_id")
    eos_token_id = tokenizer.eos_token_id

    progress_path = out_dir / "progress.json"
    offsets_path = out_dir / "samples.offsets"
    existing = sorted(out_dir.glob("train-*.bin")) if args.resume else []
    metadata_path = out_dir / "metadata.json"
    previous_metadata = (
        json.loads(metadata_path.read_text())
        if args.resume and metadata_path.exists()
        else {}
    )
    if not existing:
        np.asarray([0], dtype=np.uint64).tofile(offsets_path)
    elif not offsets_path.exists():
        raise RuntimeError("cannot resume without samples.offsets; use a new dataset-dir")
    if existing and previous_metadata:
        if previous_metadata.get("tokenizer") != args.tokenizer:
            raise RuntimeError("existing dataset uses a different tokenizer; use a new dataset-dir")
    if existing and args.resume and not progress_path.exists() and args.source_index < 0:
        raise RuntimeError(
            "Cannot safely resume without progress.json; provide a new empty "
            "dataset-dir or an explicit --source-index"
        )
    def recover_pending() -> None:
        if not progress_path.exists():
            return
        state = json.loads(progress_path.read_text())
        for name in state.get("pending_outputs", []):
            temporary = out_dir / name
            if temporary.exists():
                temporary.replace(temporary.with_suffix(""))

    recover_pending()
    existing = sorted(out_dir.glob("train-*.bin")) if args.resume else []
    produced = sum(path.stat().st_size // np.dtype(np.uint32).itemsize for path in existing)
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
    shard_index = len(existing)
    token_buffer: list[int] = []
    sample_count = committed_count - 1
    offset_total = produced
    offsets_file = offsets_path.open("ab")

    def sync_offsets() -> None:
        offsets_file.flush()
        os.fsync(offsets_file.fileno())
    output_chunk_tokens = 4_000_000
    pending_outputs: list[Path] = []

    def flush(force: bool = False) -> None:
        nonlocal shard_index
        if len(token_buffer) < output_chunk_tokens and not force:
            return
        if not token_buffer:
            return
        target = out_dir / f"train-{shard_index:05d}.bin"
        temporary = target.with_suffix(target.suffix + ".tmp")
        np.asarray(token_buffer, dtype=np.uint32).tofile(temporary)
        pending_outputs.append(temporary)
        token_buffer.clear()
        shard_index += 1

    def commit_outputs() -> None:
        for temporary in pending_outputs:
            temporary.replace(temporary.with_suffix(""))
        pending_outputs.clear()

    files = HubApi().get_dataset_files("allenai/c4", page_size=2000)
    paths = sorted(
        item["Path"] for item in files
        if item["Type"] == "blob"
        and item["Path"].startswith("en.noblocklist/c4-train.")
        and item["Path"].endswith(".json.gz")
    )
    if not paths:
        raise RuntimeError("ModelScope returned no C4 training shards")
    paths = paths[: args.shards]
    # Existing files correspond to the first source shard consumed by the
    # previous run. Continue with the next manifest entry, not numeric names.
    if args.source_index >= 0:
        start_index = args.source_index
    elif existing:
        state = json.loads(progress_path.read_text())
        start_index = int(state["next_source_index"])
    else:
        start_index = 0
    consumed_shards = start_index
    for source_position, path in enumerate(paths[start_index:], start=start_index):
        local = dataset_file_download(
            "allenai/c4", path, cache_dir=str(cache_dir), revision="master"
        )
        with gzip.open(local, "rt", encoding="utf-8") as stream:
            for line in stream:
                text = json.loads(line).get("text", "")
                if not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                ids.append(eos_token_id)
                remaining = args.tokens - produced
                if remaining <= 0:
                    break
                ids = ids[:remaining]
                if not ids:
                    continue
                produced += len(ids)
                token_buffer.extend(ids)
                sample_count += 1
                offset_total += len(ids)
                np.asarray([offset_total], dtype=np.uint64).tofile(offsets_file)
                flush()
                if produced >= args.tokens:
                    break
        if not args.keep_source_cache:
            Path(local).unlink(missing_ok=True)
        flush(force=True)
        sync_offsets()
        write_progress(
            progress_path,
            {
                "source": "modelscope:allenai/c4",
                "source_path": path,
                "next_source_index": source_position + 1,
                "tokens": produced,
                "sample_count": sample_count,
                "offsets_count": sample_count + 1,
                "pending_outputs": [item.name for item in pending_outputs],
            },
        )
        commit_outputs()
        write_progress(
            progress_path,
            {
                "source": "modelscope:allenai/c4",
                "source_path": path,
                "next_source_index": source_position + 1,
                "tokens": produced,
            },
        )
        consumed_shards = source_position + 1
        if produced >= args.tokens:
            break

    flush(force=True)
    sync_offsets()
    commit_outputs()
    metadata = {
        "source": "modelscope:allenai/c4",
        "source_path": "en.noblocklist/c4-train.*-of-01024.json.gz",
        "tokenizer": args.tokenizer,
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_eos_id": tokenizer.eos_token_id,
        "tokenizer_pad_id": tokenizer.pad_token_id,
        "tokens": sum(
            path.stat().st_size // np.dtype(np.uint32).itemsize
            for path in out_dir.glob("train-*.bin")
        ),
        "dtype": "uint32",
        "block_size": 8,
    }
    metadata["source_shards"] = consumed_shards
    metadata["tokens_verified"] = True
    metadata["sample_count"] = sample_count
    metadata["offsets_file"] = offsets_path.name
    offsets_file.close()
    write_metadata(out_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
