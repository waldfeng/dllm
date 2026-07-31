"""Small smoke trainer over prepared uint32 C4 block shards."""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import tempfile
import contextlib
import signal
import sys
import time
from datetime import timedelta
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from dynamic_router import (
    DynamicRouterLM,
    GrowthController,
    MiddleModuleUtilization,
    route_gradient_diagnostics,
    train_step,
)


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
    rank: int = 0,
    world_size: int = 1,
):
    sample_count = len(sample_offsets) - 1
    emitted = 0
    for group_start in range(start_sample, sample_count, batch_size * world_size):
        batch_start = group_start + rank * batch_size
        lengths = [
            int(sample_offsets[index + 1] - sample_offsets[index])
            for index in range(batch_start, min(batch_start + batch_size, sample_count))
        ]
        samples = []
        for index, length in enumerate(lengths, batch_start):
            samples.append(store.sample(int(sample_offsets[index]), length))
        pairs = [list(sample_block_pairs(sample, block_size, pad_token_id)) for sample in samples]
        # All ranks must execute the same number of DDP collectives. Samples
        # are sharded by rank, so compute the group maximum from offsets rather
        # than allowing a shorter rank-local shard to end the iterator early.
        group_max_pairs = 0
        for index in range(group_start, min(group_start + batch_size * world_size, sample_count)):
            length = int(sample_offsets[index + 1] - sample_offsets[index])
            group_max_pairs = max(
                group_max_pairs,
                max(0, (length + block_size - 1) // block_size - 1),
            )
        max_pairs = group_max_pairs
        first_block = (
            start_block
            if start_sample <= batch_start < start_sample + batch_size * world_size
            else 0
        )
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


def global_pair_batches(
    store: TokenStore,
    sample_offsets: np.ndarray,
    batch_size: int,
    block_size: int,
    pad_token_id: int,
    limit: int,
    rank: int = 0,
    world_size: int = 1,
    start_pair_index: int = 0,
):
    """Batch real sample/block pairs globally before splitting them by rank."""
    pairs_per_global_batch = batch_size * world_size
    pending: list[tuple[list[int], list[int], list[bool], int, int]] = []
    emitted = 0
    global_pair_index = 0
    for sample_index in range(len(sample_offsets) - 1):
        length = int(sample_offsets[sample_index + 1] - sample_offsets[sample_index])
        pairs = sample_block_pairs(
            store.sample(int(sample_offsets[sample_index]), length),
            block_size,
            pad_token_id,
        )
        for block_index, (x, y, mask) in enumerate(pairs):
            pending.append((x, y, mask, sample_index, block_index))
            if len(pending) < pairs_per_global_batch:
                continue
            if global_pair_index + pairs_per_global_batch <= start_pair_index:
                global_pair_index += len(pending)
                pending.clear()
                continue
            yield from _split_global_pair_batch(pending, global_pair_index, batch_size, rank, world_size)
            emitted += 1
            global_pair_index += len(pending)
            pending.clear()
            if emitted >= limit:
                return
    if pending and global_pair_index + len(pending) > start_pair_index:
        yield from _split_global_pair_batch(pending, global_pair_index, batch_size, rank, world_size)


def _split_global_pair_batch(
    pending: list[tuple[list[int], list[int], list[bool], int, int]],
    global_pair_index: int,
    batch_size: int,
    rank: int,
    world_size: int,
):
    """Return one rank's equally sized slice of a global pair batch."""
    total = batch_size * world_size
    padded = pending + [
        ([0] * len(pending[0][0]), [0] * len(pending[0][1]), [False] * len(pending[0][2]), -1, -1)
    ] * (total - len(pending))
    local = padded[rank * batch_size : (rank + 1) * batch_size]
    yield (
        torch.tensor([item[0] for item in local], dtype=torch.long),
        torch.tensor([item[1] for item in local], dtype=torch.long),
        torch.tensor([item[2] for item in local], dtype=torch.bool),
        [item[3] >= 0 for item in local],
        local[0][3],
        local[0][4],
        global_pair_index + total,
    )


def count_total_pairs(sample_offsets: np.ndarray, block_size: int) -> int:
    total = 0
    for index in range(len(sample_offsets) - 1):
        length = int(sample_offsets[index + 1] - sample_offsets[index])
        total += max(0, (length + block_size - 1) // block_size - 1)
    return total


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
    parser.add_argument("--dataset-dir", default="datasets/c4_qwen2.5")
    parser.add_argument("--steps", type=int, default=0, help="steps per epoch; 0 means all pairs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--max-modules", type=int, default=1024)
    parser.add_argument("--max-depth", type=int, default=128)
    parser.add_argument("--cooldown", type=int, default=32)
    parser.add_argument("--length-weight", type=float, default=1e-3)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=151665)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--distributed", action="store_true",
        help="use torchrun multi-GPU training; batch-size is per GPU",
    )
    parser.add_argument(
        "--reset-data-cursor", action="store_true",
        help="resume weights/optimizer but start at the beginning of this dataset",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--warmup-steps", type=int, default=0,
        help="0 means five percent of total steps capped at 10000 steps",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--checkpoint",
        default="datasets/checkpoints/lastest_weight.pt",
        help="checkpoint output path; defaults to datasets/checkpoints",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default="datasets/checkpoints/lastest_weight.pt",
        help="checkpoint to resume from; pass an explicit path to override",
    )
    parser.add_argument(
        "--log-file",
        default="datasets/logs/qwen10b-epoch1-b8-c32-top1-batched-w1e-4.log",
        help="write only the newest 1000 output lines to this file",
    )
    parser.add_argument(
        "--nan-log-file",
        default="datasets/logs/qwen10b-epoch1-b8-c32-top1-batched-w1e-4-nan.log",
        help="write the newest 1000 NaN/Inf diagnostic lines to this file",
    )
    parser.add_argument(
        "--checkpoint-hours", type=float, default=1.0,
        help="overwrite the single checkpoint at this wall-clock interval; 0 disables periodic saves",
    )
    parser.add_argument("--ready-file", default="")
    args = parser.parse_args()
    distributed = args.distributed or int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0")) if distributed else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if distributed else 0
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if distributed else 1
    if distributed:
        if world_size < 2:
            raise ValueError("--distributed requires torchrun with at least two processes")
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
        dist.init_process_group(
            backend="nccl", timeout=timedelta(seconds=900), device_id=local_rank
        )
    if args.log_file and rank == 0:
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

    diagnostic_started = time.monotonic()

    def diagnostic(message: str) -> None:
        value = f"[rank={rank} t={time.monotonic() - diagnostic_started:.1f}s] {message}"
        print(value, file=sys.stderr, flush=True)
        if rank == 0 and log_stream is not None:
            print(value, file=log_stream, flush=True)

    # A stalled CUDA/NCCL operation does not produce a Python exception. Keep
    # periodic Python stack snapshots in the watchdog log to identify it.
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
    diagnostic(f"process_group_ready distributed={distributed} device={args.device}")
    if (
        args.epochs <= 0
        or args.steps < 0
        or args.warmup_steps < 0
        or args.max_depth <= 0
        or args.cooldown < 0
        or args.length_weight < 0
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
    diagnostic("dataset_validation_start")
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
    diagnostic(f"dataset_validation_done samples={len(sample_offsets) - 1} batch_size={args.batch_size}")
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
    middle_utilization = MiddleModuleUtilization(window_size=1000)
    previous_route_gradient: dict[int, float] = {}
    previous_route_velocity: dict[int, float] = {}
    memory = None
    start_step = 0
    resume_cursor = None
    global_step = 0
    if args.resume_checkpoint:
        diagnostic(f"checkpoint_load_start path={args.resume_checkpoint}")
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
        middle_utilization.load_state_dict(checkpoint.get("middle_utilization", {}))
        if controller.remaining_cooldown:
            if controller.cooldown_source is not None and controller.cooldown_slot is not None:
                model.set_cooldown_connection(
                    controller.cooldown_source, controller.cooldown_slot
                )
        memory = checkpoint.get("memory_state")
        start_step = 0 if args.reset_data_cursor else int(checkpoint.get("step", 0))
        global_step = start_step
        resume_cursor = None if args.reset_data_cursor else checkpoint.get("data_cursor")
        saved_config = checkpoint.get("model_config", {})
        batch_size_changed = saved_config.get("batch_size", args.batch_size) != args.batch_size
        for key, value in {
            "d_model": args.d_model,
            "vocab_size": args.vocab_size,
            "block_size": block_size,
            "max_modules": args.max_modules,
            "n_heads": args.n_heads,
            "max_depth": args.max_depth,
            "epochs": args.epochs,
        }.items():
            if saved_config.get(key, value) != value:
                raise ValueError(f"checkpoint {key} does not match current configuration")
        if batch_size_changed:
            # Memory state is aligned to the old batch lanes and cannot be reused.
            memory = None
        elif memory is not None:
            memory = memory.to(args.device)
        diagnostic(
            f"checkpoint_load_done step={global_step} cursor={resume_cursor} "
            f"batch_size_changed={batch_size_changed}"
        )
    if distributed:
        # Do not let a rank enter DDP forward while another rank is still
        # restoring the large host-memory checkpoint.
        diagnostic("ddp_barrier_before_init")
        dist.barrier()
        diagnostic("ddp_init_start")
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        diagnostic("ddp_init_done")
        dist.barrier()
        diagnostic("ddp_barrier_after_init")
    base_model = model.module if distributed else model
    def save_checkpoint(step: int) -> None:
        if not args.checkpoint or rank != 0:
            return
        checkpoint = {
            "step": step,
            "model": base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "memory_state": memory,
            "metadata": metadata,
            "controller": controller.state_dict(),
            "middle_utilization": middle_utilization.state_dict(),
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

    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if args.checkpoint_hours < 0:
        raise ValueError("checkpoint-hours must be non-negative")
    legacy_cursor = resume_cursor if resume_cursor and "global_pair_index" not in resume_cursor else None
    if legacy_cursor is not None:
        # Option A: old rank-local cursors cannot be converted safely after a
        # batch-size change, so restart data traversal from pair zero.
        resume_cursor = None
        diagnostic(f"legacy_cursor_reset old_cursor={legacy_cursor}")
    data_cursor = resume_cursor
    last_checkpoint_time = time.monotonic()
    cursor_pending = (
        resume_cursor is not None
        and not resume_cursor.get("completed", False)
        and (not distributed or rank == 0)
    )
    cursor_seen = False
    # A legacy cursor was deliberately reset to pair zero (option A), so do
    # not apply the old global-step skip to the newly scheduled data stream.
    has_cursor = resume_cursor is not None or legacy_cursor is not None
    if resume_cursor is not None and resume_cursor.get("completed", False):
        if rank == 0:
            print("checkpoint is already complete", file=log_stream)
        if distributed:
            dist.destroy_process_group()
        return
    total_pairs = count_total_pairs(sample_offsets, 8)
    pairs_per_epoch = (total_pairs + args.batch_size * world_size - 1) // (args.batch_size * world_size)
    if pairs_per_epoch <= 0:
        raise ValueError("dataset contains no complete input/target block pairs")
    steps_per_epoch = args.steps or pairs_per_epoch
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = args.warmup_steps or min(10_000, max(100, int(0.05 * total_steps)))
    if args.ready_file and rank == 0:
        Path(args.ready_file).write_text(f"{global_step}\n")
    diagnostic(f"training_loop_start global_step={global_step} pairs_per_epoch={pairs_per_epoch}")

    def cuda_memory_metrics() -> dict[str, float]:
        if not args.device.startswith("cuda"):
            return {}
        return {
            "cuda_allocated_mib": torch.cuda.memory_allocated() / 2**20,
            "cuda_reserved_mib": torch.cuda.memory_reserved() / 2**20,
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }

    for epoch in range(args.epochs):
        epoch_limit = steps_per_epoch
        for step, (x, y, target_mask, active, sample_index, block_index, pair_cursor) in enumerate(
            global_pair_batches(
                TokenStore(files), sample_offsets, args.batch_size, 8, pad_token_id,
                epoch_limit,
                rank, world_size,
                int(resume_cursor.get("global_pair_index", 0)) if resume_cursor and epoch == 0 else 0,
            ), 1
        ):
            batch_done = pair_cursor >= total_pairs
            if step == 1:
                diagnostic(
                    f"first_batch_ready sample_index={sample_index} block_index={block_index} "
                    f"batch_done={batch_done} active_samples={sum(active)}"
                )
            if not has_cursor and step <= start_step and epoch == 0:
                continue
            data_cursor = (
                {"completed": True, "global_pair_index": pair_cursor}
                if batch_done
                else {"global_pair_index": pair_cursor}
            )
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
                if step == 1:
                    diagnostic("first_train_step_start")
                metrics = train_step(
                    model,
                    optimizer,
                    x.to(args.device),
                    y.to(args.device),
                    controller if (not distributed or rank == 0) else None,
                    length_weight=args.length_weight,
                    memory_state=memory,
                    target_mask=target_mask,
                    sample_mask=active_mask,
                )
            metrics.update(cuda_memory_metrics())
            if step == 1:
                diagnostic("first_train_step_done")
            if distributed:
                # All ranks must execute the same growth decision. Rank 0
                # owns the controller; broadcast the decision and the full
                # model state only when a new slot is activated.
                growth_signal = torch.tensor(
                    [metrics.get("new_slot", -1) if rank == 0 else -1],
                    device=args.device, dtype=torch.int64,
                )
                dist.broadcast(growth_signal, src=0)
                new_slot = int(growth_signal.item())
                if new_slot >= 0:
                    for parameter in base_model.parameters():
                        dist.broadcast(parameter.data, src=0)
                    for buffer in base_model.buffers():
                        dist.broadcast(buffer, src=0)
                    metrics["new_slot"] = new_slot
            memory = metrics.pop("memory_state")
            middle_route_counts = metrics.pop("middle_route_counts", {})
            # Batch-level status lists are useful for model logic but too large
            # to serialize in every training record.
            metrics.pop("terminations", None)
            metrics.pop("invalid_reasons", None)
            metrics["active_sample_count"] = int(active_mask.sum())
            route_stats = route_gradient_diagnostics(
                metrics.pop("route_records", []),
                metrics.pop("route_lengths", None),
                previous_route_gradient,
                previous_route_velocity,
            )
            if distributed:
                stats = torch.tensor(
                    [
                        route_stats["route_gradient_norm_sum"],
                        route_stats["route_gradient_norm_count"],
                        route_stats["route_gradient_norm_max"],
                        route_stats["route_velocity_sum"],
                        route_stats["route_velocity_count"],
                        route_stats["route_velocity_min"],
                        route_stats["route_velocity_max"],
                        route_stats["route_observed_sources"],
                    ], device=args.device, dtype=torch.float64,
                )
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                extrema = torch.tensor(
                    [route_stats["route_gradient_norm_max"], route_stats["route_velocity_min"], route_stats["route_velocity_max"]],
                    device=args.device, dtype=torch.float64,
                )
                dist.all_reduce(extrema[0:1], op=dist.ReduceOp.MAX)
                dist.all_reduce(extrema[1:2], op=dist.ReduceOp.MIN)
                dist.all_reduce(extrema[2:3], op=dist.ReduceOp.MAX)
                route_stats = {
                    "route_gradient_norm_sum": float(stats[0]),
                    "route_gradient_norm_count": int(stats[1]),
                    "route_gradient_norm_max": float(extrema[0]),
                    "route_velocity_sum": float(stats[3]),
                    "route_velocity_count": int(stats[4]),
                    "route_velocity_min": float(extrema[1]),
                    "route_velocity_max": float(extrema[2]),
                    "route_observed_sources": int(stats[7]),
                }
            gradient_count = max(1, route_stats["route_gradient_norm_count"])
            velocity_count = max(1, route_stats["route_velocity_count"])
            metrics.update(
                {
                    "route_gradient_norm_mean": route_stats["route_gradient_norm_sum"] / gradient_count,
                    "route_gradient_norm_max": route_stats["route_gradient_norm_max"],
                    "route_velocity_min": route_stats["route_velocity_min"],
                    "route_velocity_median": route_stats["route_velocity_sum"] / velocity_count,
                    "route_velocity_max": route_stats["route_velocity_max"],
                    "route_observed_sources": route_stats["route_observed_sources"],
                }
            )
            excluded = torch.zeros_like(base_model.active)
            excluded[base_model.input_id] = True
            excluded[base_model.output_id] = True
            metrics.update(
                middle_utilization.update(
                    middle_route_counts,
                    base_model.active & ~excluded,
                )
            )
            if memory is not None and not metrics.get("skipped_update"):
                memory[~torch.tensor(active, dtype=torch.bool, device=memory.device)] = 0
            if batch_done:
                memory = None
            global_step += 1
            metrics["lr"] = args.lr * lr_scale
            metrics["active_modules"] = base_model.active_count
            if rank == 0:
                print(epoch, step, metrics, file=log_stream)
            if metrics.get("skipped_update"):
                exception = (
                    f"epoch={epoch}\n"
                    f"step={step}\n"
                    f"active_modules={base_model.active_count}\n"
                    f"skipped_update={metrics.get('skipped_update')}\n"
                    f"diagnostics={metrics.get('diagnostics', {})}\n"
                    "skipped_update=true\n"
                )
                if rank == 0 and isinstance(log_stream, DualRollingLog):
                    log_stream.write_exception(exception)
            if (
                args.checkpoint
                and args.checkpoint_hours
                and time.monotonic() - last_checkpoint_time >= args.checkpoint_hours * 3600
            ):
                save_checkpoint(global_step)
                last_checkpoint_time = time.monotonic()
            if stop_requested:
                save_checkpoint(global_step)
                if args.ready_file and rank == 0:
                    Path(args.ready_file).unlink(missing_ok=True)
                if distributed:
                    dist.barrier()
                    dist.destroy_process_group()
                return 0
    if (
        rank == 0
        and resume_cursor is not None
        and not resume_cursor.get("completed", False)
        and not cursor_seen
    ):
        raise ValueError("checkpoint data cursor does not exist in the current dataset")
    save_checkpoint(global_step)
    if rank == 0:
        print("dataset", metadata, file=log_stream)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
