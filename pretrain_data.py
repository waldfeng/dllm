"""Streaming tokenizer preprocessing and mmap-backed token batches."""

from __future__ import annotations

import gzip
import io
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

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


class _ConcatReader(io.RawIOBase):
    """File-like object that concatenates split archive parts for streaming."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.index = 0
        self.current: Optional[BinaryIO] = None

    def readable(self) -> bool:
        return True

    def _next_part(self) -> bool:
        if self.current is not None:
            self.current.close()
            self.current = None
        while self.index < len(self.paths):
            self.current = self.paths[self.index].open("rb")
            self.index += 1
            return True
        return False

    def read(self, size: int = -1) -> bytes:
        chunks = []
        remaining = size
        while remaining != 0:
            if self.current is None and not self._next_part():
                break
            data = self.current.read(size if size < 0 else remaining)
            if not data:
                self._next_part()
                continue
            chunks.append(data)
            if remaining > 0:
                remaining -= len(data)
                if remaining <= 0:
                    break
        return b"".join(chunks)

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None
        super().close()


def _iter_part_groups(paths: list[Path]) -> list[Path]:
    """Return one representative path per split-archive group (e.g. MAP-CC)."""
    groups: dict[str, Path] = {}
    plain: list[Path] = []
    for path in paths:
        name = path.name
        if ".part" in name and name.rsplit(".part", 1)[-1].isdigit():
            base = name.rsplit(".part", 1)[0]
            existing = groups.get(base)
            if existing is None or path.name < existing.name:
                groups[base] = path
        else:
            plain.append(path)
    return plain + list(groups.values())


def _parquet_texts(path: Path) -> Iterator[str]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    names = parquet_file.schema_arrow.names
    for column in ("text", "content"):
        if column in names:
            break
    else:
        raise ValueError(
            f"missing string field 'text' or 'content' in parquet {path}"
        )
    for row_group in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group, columns=[column])
        for value in table.column(0).to_pylist():
            if isinstance(value, str) and value:
                yield value


def _json_array_zst_texts(path: Path) -> Iterator[str]:
    import zstandard

    raw = zstandard.ZstdDecompressor().stream_reader(path.open("rb"))
    with io.TextIOWrapper(raw, encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError(f"expected a JSON array in {path}")
    for record in records:
        if not isinstance(record, dict):
            continue
        title = record.get("title")
        content = record.get("content")
        if isinstance(content, str) and content:
            if isinstance(title, str) and title:
                yield f"{title}\n{content}"
            else:
                yield content


def _concat_gzip_texts(parts: list[Path]) -> Iterator[str]:
    """Stream JSONL records from a gzip stream split across multiple files."""
    with gzip.open(_ConcatReader(parts), "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                base = parts[0].name.rsplit(".part", 1)[0]
                raise ValueError(
                    f"invalid JSON in {parts[0].parent / base}:{line_number}"
                ) from exc
            text = record.get("text") if isinstance(record, dict) else None
            if not isinstance(text, str):
                base = parts[0].name.rsplit(".part", 1)[0]
                raise ValueError(
                    f"missing string field 'text' in {parts[0].parent / base}:{line_number}"
                )
            yield text


def _mathpile_stackexchange_texts(record: dict) -> Optional[str]:
    question = record.get("question")
    answers = record.get("answers")
    if not isinstance(question, dict):
        return None
    title = question.get("Title")
    body = question.get("Body")
    parts = []
    if isinstance(title, str) and title:
        parts.append(title)
    if isinstance(body, str) and body:
        parts.append(body)
    if isinstance(answers, list):
        for answer in answers:
            if isinstance(answer, dict) and isinstance(answer.get("Body"), str):
                parts.append(answer["Body"])
    if not parts:
        return None
    return "\n\n".join(parts)


def _jsonl_texts(path: Path) -> Iterator[str]:
    with _open_text(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
            text = record.get("text") if isinstance(record, dict) else None
            if not isinstance(text, str) and isinstance(record, dict):
                text = _mathpile_stackexchange_texts(record)
            if not isinstance(text, str):
                raise ValueError(f"missing string field 'text' in {path}:{line_number}")
            yield text


def jsonl_paths(input_dir: Path) -> list[Path]:
    paths = list(input_dir.rglob("*.jsonl"))
    paths.extend(input_dir.rglob("*.jsonl.zst"))
    paths.extend(input_dir.rglob("*.jsonl.gz"))
    paths.extend(input_dir.rglob("*.parquet"))
    paths.extend(input_dir.rglob("*.zst"))
    paths.extend(input_dir.rglob("*.part*"))
    paths = [
        path
        for path in paths
        if path.is_file()
        and (
            ".part" not in path.name
            or path.name.rsplit(".part", 1)[-1].isdigit()
        )
    ]
    return sorted(set(_iter_part_groups(paths)), key=lambda path: str(path))


def iter_text_records(input_dir: Path) -> Iterator[tuple[Path, str]]:
    for path in jsonl_paths(input_dir):
        if path.name.endswith(".parquet"):
            texts = _parquet_texts(path)
        elif path.name.endswith(".json.zst"):
            texts = _json_array_zst_texts(path)
        elif ".part" in path.name and path.name.rsplit(".part", 1)[-1].isdigit():
            base = path.name.rsplit(".part", 1)[0]
            siblings = sorted(
                entry
                for entry in path.parent.iterdir()
                if entry.name.startswith(base + ".part")
                and entry.name.rsplit(".part", 1)[-1].isdigit()
            )
            texts = _concat_gzip_texts(siblings)
        else:
            texts = _jsonl_texts(path)
        for text in texts:
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
        self.prefix = ""
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
        path = self.output_dir / f"{self.prefix}train-{self.shard_index:05d}.bin"
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


def _tokenize_source_file(
    path_str: str,
    input_dir_str: str,
    output_dir_str: str,
    shard_tokens: int,
    tokenizer_name: str,
    tokenizer_batch_size: int,
    max_tokens: Optional[int],
) -> dict:
    """Tokenize one logical source file into its own uint32 shards.

    Runs in a worker process; returns stats and the written shard files in
    order so the caller can concatenate them deterministically.
    """
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)
    path = Path(path_str)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, local_files_only=True, use_fast=True
    )
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer must define eos_token_id")

    stem = path.relative_to(input_dir).as_posix().replace("/", "__")
    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    writer = TokenShardWriter(work_dir, shard_tokens)
    writer.prefix = f"{stem}__"

    total_tokens = 0
    documents = 0
    stopped = False
    if path.name.endswith(".parquet"):
        text_iter = _parquet_texts(path)
    elif path.name.endswith(".json.zst"):
        text_iter = _json_array_zst_texts(path)
    elif ".part" in path.name and path.name.rsplit(".part", 1)[-1].isdigit():
        base = path.name.rsplit(".part", 1)[0]
        siblings = sorted(
            entry
            for entry in path.parent.iterdir()
            if entry.name.startswith(base + ".part")
            and entry.name.rsplit(".part", 1)[-1].isdigit()
        )
        text_iter = _concat_gzip_texts(siblings)
    else:
        text_iter = _jsonl_texts(path)

    batch: list[str] = []
    while not stopped:
        for text in text_iter:
            batch.append(text)
            if len(batch) >= tokenizer_batch_size:
                break
        if not batch:
            break
        encoded_batch = tokenizer(
            batch, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        for encoded in encoded_batch:
            tokens = np.asarray((*encoded, eos_id), dtype=np.uint32)
            if max_tokens is not None and total_tokens + len(tokens) > max_tokens:
                tokens = tokens[: max_tokens - total_tokens]
                stopped = True
            if len(tokens) == 0:
                stopped = True
                break
            writer.append(tokens)
            total_tokens += len(tokens)
            documents += 1
        batch = []
    writer.close()
    return {
        "path": path_str,
        "relative": stem,
        "documents": documents,
        "tokens": total_tokens,
        "shards": writer.shards,
    }


def prepare_token_shards_parallel(
    input_dir: Path,
    output_dir: Path,
    tokenizer_name: str,
    shard_tokens: int = 100_000_000,
    min_tokens: int = 0,
    max_tokens: Optional[int] = None,
    tokenizer_batch_size: int = 256,
    workers: int = 8,
) -> dict:
    """Parallel variant of prepare_token_shards.

    Each logical source file is tokenized in its own process into private
    shard files (written under output_dir/_work with a per-source prefix).
    The partial shards are then concatenated in manifest order into the final
    train-*.bin sequence, so the on-disk format and TokenStore stay identical
    to the sequential implementation.
    """
    paths = jsonl_paths(input_dir)
    if not paths:
        raise ValueError(f"no source files found under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(workers, len(paths)))
    jobs = [
        (
            str(path),
            str(input_dir),
            str(output_dir),
            shard_tokens,
            tokenizer_name,
            tokenizer_batch_size,
            max_tokens,
        )
        for path in paths
    ]
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_tokenize_source_file, *zip(*jobs)):
            results.append(result)
            print(
                f"tokenized source={result['relative']} "
                f"documents={result['documents']} tokens={result['tokens']}",
                flush=True,
            )

    work_dir = output_dir / "_work"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, local_files_only=True, use_fast=True
    )
    eos_id = int(tokenizer.eos_token_id)
    vocab_size = len(tokenizer)

    total_tokens = 0
    documents = 0
    source_documents: dict[str, int] = {}
    source_tokens: dict[str, int] = {}
    writer = TokenShardWriter(output_dir, shard_tokens)
    for result in results:
        key = result["relative"]
        source_documents[key] = result["documents"]
        source_tokens[key] = result["tokens"]
        documents += result["documents"]
        total_tokens += result["tokens"]
        for shard in result["shards"]:
            data = np.fromfile(work_dir / shard["path"], dtype=np.uint32)
            writer.append(data)
            (work_dir / shard["path"]).unlink()
    writer.close()
    work_dir.rmdir()

    manifest = {
        "format": "uint32_token_shards_v1",
        "tokenizer": tokenizer_name,
        "vocab_size": vocab_size,
        "eos_token_id": eos_id,
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
    if min_tokens and total_tokens < min_tokens:
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
