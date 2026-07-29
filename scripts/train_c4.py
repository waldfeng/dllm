"""Small smoke trainer over prepared uint32 C4 block shards."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import contextlib
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from dynamic_router import DynamicRouterLM, GrowthController, train_step


class RollingLog:
    """Keep only the newest fixed number of newline-delimited log records."""

    def __init__(self, path: Path, max_lines: int = 1000) -> None:
        self.path = path
        self.max_lines = max_lines
        self.lines = deque(maxlen=max_lines)
        if path.exists():
            self.lines.extend(path.read_text().splitlines(keepends=True)[-max_lines:])
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, value: str) -> None:
        self.lines.extend(value.splitlines(keepends=True))
        self.path.write_text("".join(self.lines))

    def flush(self) -> None:
        pass


class DualRollingLog:
    """Keep normal output and non-finite diagnostics in separate rolling files."""

    def __init__(self, normal_path: Path, nan_path: Path, max_lines: int = 1000) -> None:
        self.normal = RollingLog(normal_path, max_lines)
        self.nan = RollingLog(nan_path, max_lines)

    def write(self, value: str) -> None:
        self.normal.write(value)

    def write_exception(self, value: str) -> None:
        self.nan.path.parent.mkdir(parents=True, exist_ok=True)
        with self.nan.path.open("a") as stream:
            stream.write("-------------\n")
            stream.write(value)
            if not value.endswith("\n"):
                stream.write("\n")

    def flush(self) -> None:
        self.normal.flush()
        self.nan.flush()


def sample_block_pairs(tokens: np.ndarray, block_size: int, pad_token_id: int):
    """Yield consecutive block pairs from one text sample without crossing it."""
    if len(tokens) < 2:
        return
    block_count = (len(tokens) + block_size - 1) // block_size
    blocks = []
    for index in range(block_count):
        block = tokens[index * block_size : (index + 1) * block_size].tolist()
        blocks.append(block + [pad_token_id] * (block_size - len(block)))
    for index in range(block_count - 1):
        target_length = min(block_size, max(0, len(tokens) - (index + 1) * block_size))
        yield (
            blocks[index],
            blocks[index + 1],
            [position < target_length for position in range(block_size)],
        )


class TokenStore:
    def __init__(self, files: list[Path]) -> None:
        self.files = files
        self.starts = []
        total = 0
        for path in files:
            self.starts.append(total)
            total += path.stat().st_size // np.dtype(np.uint32).itemsize
        self.total = total

    def slice(self, start: int, end: int) -> np.ndarray:
        chunks = []
        for path, file_start in reversed(list(zip(self.files, self.starts))):
            if file_start <= start:
                local_start = start - file_start
                local_end = min(end - file_start, path.stat().st_size // 4)
                chunks.append(np.asarray(np.memmap(path, dtype=np.uint32, mode="r")[local_start:local_end]))
                break
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint32)

    def sample(self, offset: int, length: int) -> np.ndarray:
        end = offset + length
        parts = []
        for path, file_start in zip(self.files, self.starts):
            file_end = file_start + path.stat().st_size // 4
            if end <= file_start or offset >= file_end:
                continue
            local_start = max(offset, file_start) - file_start
            local_end = min(end, file_end) - file_start
            ids = np.memmap(path, dtype=np.uint32, mode="r")
            parts.append(np.asarray(ids[local_start:local_end]))
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.uint32)


def sequential_batches(
    store: TokenStore,
    sample_offsets: np.ndarray,
    batch_size: int,
    block_size: int,
    pad_token_id: int,
    limit: int,
    start_sample: int = 0,
    start_block: int = 0,
):
    sample_count = len(sample_offsets) - 1
    emitted = 0
    for batch_start in range(start_sample, sample_count, batch_size):
        lengths = [
            int(sample_offsets[index + 1] - sample_offsets[index])
            for index in range(batch_start, min(batch_start + batch_size, sample_count))
        ]
        samples = []
        for index, length in enumerate(lengths, batch_start):
            samples.append(store.sample(int(sample_offsets[index]), length))
        pairs = [list(sample_block_pairs(sample, block_size, pad_token_id)) for sample in samples]
        max_pairs = max((len(items) for items in pairs), default=0)
        first_block = start_block if batch_start == start_sample else 0
        for block_index in range(first_block, max_pairs):
            if emitted >= limit:
                return
            x_rows, y_rows, masks, active = [], [], [], []
            for items in pairs:
                if block_index < len(items):
                    x, y, mask = items[block_index]
                    active.append(True)
                else:
                    x = [pad_token_id] * block_size
                    y = [pad_token_id] * block_size
                    mask = [False] * block_size
                    active.append(False)
                x_rows.append(x)
                y_rows.append(y)
                masks.append(mask)
            yield (
                torch.tensor(x_rows, dtype=torch.long),
                torch.tensor(y_rows, dtype=torch.long),
                torch.tensor(masks, dtype=torch.bool),
                active,
                batch_start,
                block_index,
                block_index == max_pairs - 1,
            )
            emitted += 1


def count_epoch_steps(sample_offsets: np.ndarray, batch_size: int, block_size: int) -> int:
    """Count sequential block transitions without loading token shards."""
    sample_count = len(sample_offsets) - 1
    steps = 0
    for batch_start in range(0, sample_count, batch_size):
        batch_end = min(batch_start + batch_size, sample_count)
        pair_counts = [
            max(
                0,
                (int(sample_offsets[index + 1] - sample_offsets[index]) + block_size - 1)
                // block_size
                - 1,
            )
            for index in range(batch_start, batch_end)
        ]
        steps += max(pair_counts, default=0)
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/c4_2b")
    parser.add_argument("--steps", type=int, default=0, help="steps per epoch; 0 means all pairs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--max-modules", type=int, default=1024)
    parser.add_argument("--max-depth", type=int, default=128)
    parser.add_argument("--cooldown", type=int, default=32)
    parser.add_argument("--length-weight", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=151665)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--warmup-steps", type=int, default=0,
        help="0 means 5% of total steps capped at 10000 steps",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument(
        "--log-file", default=None,
        help="write only the newest 1000 output lines to this file",
    )
    parser.add_argument(
        "--nan-log-file", default=None,
        help="write the newest 1000 NaN/Inf diagnostic lines to this file",
    )
    parser.add_argument(
        "--checkpoint-hours", type=float, default=1.0,
        help="overwrite the single checkpoint at this wall-clock interval; 0 disables periodic saves",
    )
    args = parser.parse_args()
    if args.log_file:
        nan_path = (
            Path(args.nan_log_file)
            if args.nan_log_file
            else Path(args.log_file).with_name(
                f"{Path(args.log_file).stem}-nan{Path(args.log_file).suffix}"
            )
        )
        log_stream = DualRollingLog(Path(args.log_file), nan_path)
    else:
        log_stream = None
    if (
        args.epochs <= 0
        or args.steps < 0
        or args.warmup_steps < 0
        or args.max_depth <= 0
        or args.cooldown < 0
        or args.length_weight < 0
        or args.top_k <= 0
    ):
        raise ValueError(
            "epochs, steps, warmup_steps, max_depth, cooldown, and length_weight "
            "must be non-negative/positive"
        )
    if not 0 < args.min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in (0, 1]")
    if args.precision == "bf16" and args.device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is not supported by the selected CUDA device")
    root = Path(args.dataset_dir)
    metadata = json.loads((root / "metadata.json").read_text())
    files = sorted(root.glob("train-*.bin"))
    if not files:
        raise FileNotFoundError("No prepared token shards found; run prepare_c4_modelscope.py")
    actual_tokens = sum(path.stat().st_size // np.dtype(np.uint32).itemsize for path in files)
    if actual_tokens != metadata.get("tokens"):
        raise ValueError(
            f"metadata token count {metadata.get('tokens')} does not match "
            f"shards {actual_tokens}; regenerate metadata before training"
        )
    block_size = int(metadata.get("block_size", 8))
    if block_size != 8:
        raise ValueError("DynamicRouterLM currently requires metadata block_size=8")
    if int(metadata.get("tokenizer_vocab_size", args.vocab_size)) != args.vocab_size:
        raise ValueError("--vocab-size does not match the dataset tokenizer vocabulary")
    max_token_id = max(
        int(np.memmap(path, dtype=np.uint32, mode="r").max()) for path in files
    )
    if max_token_id >= args.vocab_size:
        raise ValueError(
            f"dataset token ID {max_token_id} is outside vocab size {args.vocab_size}"
        )
    offsets_path = root / metadata.get("offsets_file", "samples.offsets")
    if not offsets_path.exists():
        raise ValueError(
            "dataset metadata must contain samples.offsets matching all tokens; "
            "regenerate the dataset with the updated preparation script"
        )
    sample_offsets = np.memmap(offsets_path, dtype=np.uint64, mode="r")
    if len(sample_offsets) < 2 or int(sample_offsets[-1]) != actual_tokens:
        raise ValueError("samples.offsets does not match the token shards")
    pad_token_id = int(metadata.get("tokenizer_pad_id", 151643))
    model = DynamicRouterLM(
        d_model=args.d_model,
        vocab_size=args.vocab_size,
        max_modules=args.max_modules,
        block_size=block_size,
        n_heads=args.n_heads,
        max_depth=args.max_depth,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    controller = GrowthController(cooldown=args.cooldown)
    memory = None
    start_step = 0
    resume_cursor = None
    global_step = 0
    if args.resume_checkpoint:
        # Stage the large model and optimizer checkpoint in host memory first;
        # loading directly into CUDA can cause severe paging on large restores.
        checkpoint = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        if checkpoint.get("metadata", {}).get("tokens") != metadata.get("tokens"):
            raise ValueError("checkpoint and dataset metadata do not match")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        controller.load_state_dict(checkpoint.get("controller", {}))
        if controller.remaining_cooldown:
            if controller.cooldown_source is not None and controller.cooldown_slot is not None:
                model.set_cooldown_connection(
                    controller.cooldown_source, controller.cooldown_slot
                )
        memory = checkpoint.get("memory_state")
        start_step = int(checkpoint.get("step", 0))
        global_step = start_step
        resume_cursor = checkpoint.get("data_cursor")
        saved_config = checkpoint.get("model_config", {})
        for key, value in {
            "d_model": args.d_model,
            "vocab_size": args.vocab_size,
            "block_size": block_size,
            "batch_size": args.batch_size,
            "max_modules": args.max_modules,
            "n_heads": args.n_heads,
            "max_depth": args.max_depth,
            "epochs": args.epochs,
            "warmup_steps": args.warmup_steps,
        }.items():
            if saved_config.get(key, value) != value:
                raise ValueError(f"checkpoint {key} does not match current configuration")
        if memory is not None:
            memory = memory.to(args.device)
    def save_checkpoint(step: int) -> None:
        if not args.checkpoint:
            return
        checkpoint = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "memory_state": memory,
            "metadata": metadata,
            "controller": controller.state_dict(),
            "data_cursor": data_cursor,
            "model_config": {
                "d_model": args.d_model,
                "vocab_size": args.vocab_size,
                "block_size": block_size,
                "batch_size": args.batch_size,
                "max_modules": args.max_modules,
                "n_heads": args.n_heads,
                "max_depth": args.max_depth,
                "epochs": args.epochs,
            "warmup_steps": args.warmup_steps,
            },
        }
        target = Path(args.checkpoint)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(checkpoint, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    if args.checkpoint_hours < 0:
        raise ValueError("checkpoint-hours must be non-negative")
    data_cursor = resume_cursor
    last_checkpoint_time = time.monotonic()
    cursor_pending = resume_cursor is not None and not resume_cursor.get("completed", False)
    cursor_seen = False
    has_cursor = resume_cursor is not None
    if resume_cursor is not None and resume_cursor.get("completed", False):
        print("checkpoint is already complete", file=log_stream)
        return
    pairs_per_epoch = count_epoch_steps(sample_offsets, args.batch_size, 8)
    if pairs_per_epoch <= 0:
        raise ValueError("dataset contains no complete input/target block pairs")
    steps_per_epoch = args.steps or pairs_per_epoch
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = args.warmup_steps or min(10_000, max(100, int(0.05 * total_steps)))
    for epoch in range(args.epochs):
        epoch_limit = steps_per_epoch
        for step, (x, y, target_mask, active, sample_index, block_index, batch_done) in enumerate(
            sequential_batches(
                TokenStore(files), sample_offsets, args.batch_size, 8, pad_token_id,
                epoch_limit,
                int(resume_cursor.get("sample_index", 0)) if resume_cursor and epoch == 0 else 0,
                int(resume_cursor.get("block_index", 0)) if resume_cursor and epoch == 0 else 0,
            ), 1
        ):
            if not has_cursor and step <= start_step and epoch == 0:
                continue
            if cursor_pending:
                if sample_index == int(resume_cursor.get("sample_index", -1)):
                    if block_index < int(resume_cursor.get("block_index", 0)):
                        continue
                    cursor_pending = False
                    cursor_seen = True
                else:
                    continue
            if batch_done:
                next_sample = sample_index + len(active)
                data_cursor = (
                    {"completed": True}
                    if next_sample >= len(sample_offsets) - 1
                    else {"sample_index": next_sample, "block_index": 0}
                )
            else:
                data_cursor = {"sample_index": sample_index, "block_index": block_index + 1}
            active_mask = torch.tensor(active, dtype=torch.bool)
            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / max(1, warmup_steps)
            else:
                progress = (global_step - warmup_steps) / max(
                    1, total_steps - warmup_steps
                )
                lr_scale = args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (
                    1 + np.cos(np.pi * min(1.0, progress))
                )
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_scale
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if args.precision == "bf16" and args.device.startswith("cuda")
                else contextlib.nullcontext()
            )
            with autocast:
                metrics = train_step(
                    model,
                    optimizer,
                    x.to(args.device),
                    y.to(args.device),
                    controller,
                    length_weight=args.length_weight,
                    top_k=args.top_k,
                    memory_state=memory,
                    target_mask=target_mask,
                    sample_mask=active_mask,
                )
            memory = metrics.pop("memory_state")
            if memory is not None:
                memory[~torch.tensor(active, dtype=torch.bool, device=memory.device)] = 0
            if batch_done:
                memory = None
            global_step += 1
            metrics["lr"] = args.lr * lr_scale
            metrics["active_modules"] = model.active_count
            print(epoch, step, metrics, file=log_stream)
            if metrics.get("skipped_update"):
                exception = (
                    f"epoch={epoch}\n"
                    f"step={step}\n"
                    f"active_modules={model.active_count}\n"
                    f"terminations={metrics['terminations']}\n"
                    f"invalid_reasons={metrics['invalid_reasons']}\n"
                    f"diagnostics={metrics.get('diagnostics', {})}\n"
                    "skipped_update=true\n"
                )
                if isinstance(log_stream, DualRollingLog):
                    log_stream.write_exception(exception)
            if (
                args.checkpoint
                and args.checkpoint_hours
                and time.monotonic() - last_checkpoint_time >= args.checkpoint_hours * 3600
            ):
                save_checkpoint(global_step)
                last_checkpoint_time = time.monotonic()
    if resume_cursor is not None and not resume_cursor.get("completed", False) and not cursor_seen:
        raise ValueError("checkpoint data cursor does not exist in the current dataset")
    save_checkpoint(global_step)
    print("dataset", metadata, file=log_stream)


if __name__ == "__main__":
    main()
