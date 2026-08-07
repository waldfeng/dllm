"""Streaming tokenizer preprocessing and mmap-backed token batches."""

from __future__ import annotations

import gzip
import io
import json
import math
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(name_or_path: str, local_files_only: bool = False) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        name_or_path,
        local_files_only=local_files_only,
        use_fast=True,
    )


def _open_text(path: Path) -> io.TextIOBase:
    if path.name.endswith(".zst"):
        import zstandard as zstd

        raw = path.open("rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(reader, encoding="utf-8")
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def jsonl_paths(input_dir: Path) -> list[Path]:
    paths = list(input_dir.rglob("*.jsonl"))
    paths.extend(input_dir.rglob("*.jsonl.zst"))
    paths.extend(input_dir.rglob("*.jsonl.gz"))
    return sorted(path for path in paths if path.is_file())


def iter_text_records(input_dir: Path) -> Iterator[tuple[Path, str]]:
    for path in jsonl_paths(input_dir):
        with _open_text(path) as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
                text = record.get("text") if isinstance(record, dict) else None
                if not isinstance(text, str):
                    raise ValueError(f"missing string field 'text' in {path}:{line_number}")
                yield path, text


def _batched_records(
    input_dir: Path, batch_size: int
) -> Iterator[list[tuple[Path, str]]]:
    if batch_size <= 0:
        raise ValueError("tokenizer batch size must be positive")
    batch: list[tuple[Path, str]] = []
    for record in iter_text_records(input_dir):
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class TokenShardWriter:
    def __init__(self, output_dir: Path, shard_tokens: int) -> None:
        if shard_tokens <= 0:
            raise ValueError("shard_tokens must be positive")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.buffer = np.empty(shard_tokens, dtype=np.uint32)
        self.filled = 0
        self.shard_index = 0
        self.shards: list[dict[str, int | str]] = []

    def append(self, tokens: np.ndarray) -> None:
        tokens = np.asarray(tokens, dtype=np.uint32)
        offset = 0
        while offset < len(tokens):
            count = min(self.shard_tokens - self.filled, len(tokens) - offset)
            self.buffer[self.filled : self.filled + count] = tokens[offset : offset + count]
            self.filled += count
            offset += count
            if self.filled == self.shard_tokens:
                self.flush()

    def flush(self) -> None:
        if self.filled == 0:
            return
        path = self.output_dir / f"train-{self.shard_index:05d}.bin"
        self.buffer[: self.filled].tofile(path)
        self.shards.append({"path": path.name, "tokens": self.filled})
        self.shard_index += 1
        self.filled = 0

    def close(self) -> None:
        self.flush()


def _write_manifest(output_dir: Path, manifest: dict) -> None:
    temporary = output_dir / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output_dir / "manifest.json")


def count_tokens(
    input_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: Optional[int] = None,
) -> tuple[int, int]:
    """Return exact token and document counts without writing shards."""
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    total_tokens = 0
    documents = 0
    for _, text in iter_text_records(input_dir):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(encoded) + 1
        documents += 1
        if max_tokens is not None and total_tokens >= max_tokens:
            return max_tokens, documents
    return total_tokens, documents


def prepare_token_shards(
    input_dir: Path,
    output_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    shard_tokens: int = 100_000_000,
    min_tokens: int = 20_000_000_000,
    max_tokens: Optional[int] = None,
    tokenizer_batch_size: int = 256,
) -> dict:
    """Tokenize every JSONL document and write uint32 mmap shards.

    An EOS token is appended to every document.  The default minimum catches
    accidental partial downloads before a training run is started.
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    if len(tokenizer) >= 2**32:
        raise ValueError("tokenizer vocabulary does not fit in uint32")

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = TokenShardWriter(output_dir, shard_tokens)
    total_tokens = 0
    documents = 0
    source_documents: dict[str, int] = {}
    source_tokens: dict[str, int] = {}
    stopped_at_limit = False
    current_path: Optional[Path] = None
    current_documents = 0
    current_tokens = 0
    for batch in _batched_records(input_dir, tokenizer_batch_size):
        texts = [text for _, text in batch]
        if callable(tokenizer):
            encoded_batch = tokenizer(
                texts,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
        else:  # Keep the small tokenizer test doubles intentionally simple.
            encoded_batch = [
                tokenizer.encode(text, add_special_tokens=False) for text in texts
            ]
        for (path, _), encoded in zip(batch, encoded_batch):
            if current_path is not None and path != current_path:
                print(
                    f"processed source={current_path.relative_to(input_dir)} "
                    f"documents={current_documents} tokens={current_tokens} "
                    f"total_tokens={total_tokens}",
                    flush=True,
                )
                current_documents = 0
                current_tokens = 0
            current_path = path
            tokens = np.asarray((*encoded, eos_id), dtype=np.uint32)
            if max_tokens is not None and total_tokens + len(tokens) > max_tokens:
                tokens = tokens[: max_tokens - total_tokens]
                stopped_at_limit = True
            if len(tokens) == 0:
                break
            writer.append(tokens)
            total_tokens += len(tokens)
            documents += 1
            current_documents += 1
            current_tokens += len(tokens)
            key = str(path.relative_to(input_dir))
            source_documents[key] = source_documents.get(key, 0) + 1
            source_tokens[key] = source_tokens.get(key, 0) + len(tokens)
            if stopped_at_limit:
                break
        if stopped_at_limit:
            break
    if current_path is not None:
        print(
            f"processed source={current_path.relative_to(input_dir)} "
            f"documents={current_documents} tokens={current_tokens} "
            f"total_tokens={total_tokens}",
            flush=True,
        )
    writer.close()
    manifest = {
        "format": "uint32_token_shards_v1",
        "tokenizer": getattr(tokenizer, "name_or_path", "unknown"),
        "vocab_size": len(tokenizer),
        "eos_token_id": int(eos_id),
        "input_dir": str(input_dir),
        "total_tokens": total_tokens,
        "documents": documents,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "shards": writer.shards,
        "source_documents": source_documents,
        "source_tokens": source_tokens,
    }
    _write_manifest(output_dir, manifest)
    if total_tokens < min_tokens:
        raise ValueError(
            f"only {total_tokens:,} tokens were prepared; expected at least "
            f"{min_tokens:,}. Check the dataset directory or use --min-tokens 0 "
            "for a smoke dataset."
        )
    return manifest


class TokenStore:
    """Read token windows across multiple uint32 mmap shards."""

    def __init__(self, dataset_dir: Path) -> None:
        manifest_path = dataset_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing token manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.shards: list[np.memmap] = []
        self.starts = [0]
        for shard in self.manifest["shards"]:
            path = dataset_dir / shard["path"]
            mapped = np.memmap(path, dtype=np.uint32, mode="r")
            expected = int(shard["tokens"])
            if len(mapped) != expected:
                raise ValueError(f"token count mismatch in {path}")
            self.shards.append(mapped)
            self.starts.append(self.starts[-1] + expected)
        self.total_tokens = self.starts[-1]

    def read(self, start: int, count: int) -> np.ndarray:
        if start < 0 or count < 0 or start + count > self.total_tokens:
            raise IndexError("requested token window is outside the token store")
        result = np.empty(count, dtype=np.int64)
        remaining = count
        destination = 0
        position = start
        shard_index = max(0, int(np.searchsorted(self.starts, position, side="right") - 1))
        while remaining:
            shard_start = self.starts[shard_index]
            local_start = position - shard_start
            available = min(remaining, len(self.shards[shard_index]) - local_start)
            result[destination : destination + available] = self.shards[shard_index][
                local_start : local_start + available
            ]
            position += available
            destination += available
            remaining -= available
            if remaining:
                shard_index += 1
        return result


def _permutation_parameters(sample_count: int, seed: int) -> tuple[int, int]:
    if sample_count <= 1:
        return 1, 0
    rng = np.random.default_rng(seed)
    stride = int(rng.integers(1, sample_count))
    while math.gcd(stride, sample_count) != 1:
        stride = (stride + 1) % sample_count or 1
    offset = int(rng.integers(0, sample_count))
    return stride, offset


def token_batches(
    store: TokenStore,
    sequence_length: int,
    batch_size: int,
    steps: int,
    rank: int = 0,
    world_size: int = 1,
    start_step: int = 0,
    seed: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield equal-sized distributed batches with deterministic epoch shuffling."""
    if sequence_length <= 0 or batch_size <= 0 or steps < 0:
        raise ValueError("sequence_length, batch_size and steps are invalid")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    samples = (store.total_tokens - 1) // sequence_length
    global_batch = batch_size * world_size
    steps_per_epoch = samples // global_batch
    if steps_per_epoch <= 0:
        raise ValueError("token store is too small for one complete distributed batch")
    usable_samples = steps_per_epoch * global_batch
    for local_step in range(steps):
        global_step = start_step + local_step
        epoch, step_in_epoch = divmod(global_step, steps_per_epoch)
        stride, offset = _permutation_parameters(usable_samples, seed + epoch)
        first = step_in_epoch * global_batch + rank * batch_size
        positions = (offset + stride * (first + np.arange(batch_size))) % usable_samples
        inputs = np.empty((batch_size, sequence_length), dtype=np.int64)
        labels = np.empty((batch_size, sequence_length), dtype=np.int64)
        for row, sample in enumerate(positions.tolist()):
            window = store.read(sample * sequence_length, sequence_length + 1)
            inputs[row] = window[:-1]
            labels[row] = window[1:]
        yield inputs, labels
