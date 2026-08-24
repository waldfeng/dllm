"""Two-GPU FSDP training entry point for the shared-memory language model."""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from memory_transformer import (
    MemoryReasoner,
    MemoryTransformerConfig,
    MemoryTransformerForCausalLM,
    SharedMemory,
)
from pretrain_data import TokenStore, load_tokenizer, token_batches

try:
    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
except ImportError:  # pragma: no cover - only old torch versions take this path
    FSDP = None
    ModuleWrapPolicy = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/tokens"))
    parser.add_argument("--tokenizer", default="tokenizer/qwen")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--schedule-steps",
        type=int,
        default=0,
        help="total optimizer steps covered by the LR schedule; 0 uses --steps",
    )
    parser.add_argument("--sequence-length", "--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1, help="per-GPU batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=1536)
    parser.add_argument("--intermediate-size", type=int, default=4096)
    parser.add_argument("--num-attention-heads", type=int, default=16)
    parser.add_argument("--num-key-value-heads", type=int, default=8)
    parser.add_argument("--num-memory-blocks", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--router-weight", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--no-fsdp", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("datasets/checkpoints/last.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument(
        "--save-interval-hours",
        type=float,
        default=0.0,
        help="overwrite the checkpoint after this wall-clock interval; 0 disables it",
    )
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=0.0,
        help="stop training after this many wall-clock hours (0 disables it)",
    )
    parser.add_argument(
        "--no-final-save",
        action="store_true",
        help="skip the final checkpoint save when training ends",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def setup_distributed(device_choice: str) -> tuple[int, int, int, torch.device, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = device_choice != "cpu" and torch.cuda.is_available()
    if device_choice == "cuda" and not use_cuda:
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if world_size > 1:
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if use_cuda:
        if world_size > 1:
            torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank if world_size > 1 else 0)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device, world_size > 1


def seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def build_config(args: argparse.Namespace, vocab_size: int) -> MemoryTransformerConfig:
    return MemoryTransformerConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        num_memory_blocks=args.num_memory_blocks,
        top_k=args.top_k,
    )


def wrap_model(
    model: MemoryTransformerForCausalLM,
    args: argparse.Namespace,
    device: torch.device,
    distributed: bool,
    local_rank: int,
) -> tuple[torch.nn.Module, bool]:
    use_fsdp = distributed and not args.no_fsdp
    if use_fsdp and FSDP is None:
        raise RuntimeError("this torch build does not provide FSDP")
    if args.precision == "bf16" and device.type == "cuda":
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mixed_precision = None
    if use_fsdp:
        policy = ModuleWrapPolicy({SharedMemory, MemoryReasoner})
        wrapped = FSDP(
            model,
            auto_wrap_policy=policy,
            mixed_precision=mixed_precision,
            device_id=device,
            use_orig_params=True,
        )
        return wrapped, True

    dtype = torch.bfloat16 if args.precision == "bf16" and device.type == "cuda" else None
    model.to(device=device, dtype=dtype)
    if distributed:
        return DDP(model, device_ids=[local_rank] if device.type == "cuda" else None), False
    return model, False


def unwrap_model(model: torch.nn.Module) -> MemoryTransformerForCausalLM:
    if isinstance(model, DDP) or (FSDP is not None and isinstance(model, FSDP)):
        return model.module  # type: ignore[return-value]
    return model  # type: ignore[return-value]


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def reduce_mean(value: Tensor, distributed: bool) -> Tensor:
    if not distributed:
        return value.detach()
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return reduced


def scheduler_for(
    optimizer: torch.optim.Optimizer,
    steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if warmup_steps < 0 or not 0 <= min_lr_ratio <= 1:
        raise ValueError("invalid scheduler settings")

    def scale(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, (step + 1) / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def build_resume_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    current_lr: float,
    start_step: int,
    schedule_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a continuation scheduler from the current checkpoint LR."""
    return build_continuation_scheduler(
        optimizer,
        peak_lr=args.lr,
        start_lr=current_lr,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        schedule_start_step=start_step,
        schedule_steps=schedule_steps,
    )


def build_continuation_scheduler(
    optimizer: torch.optim.Optimizer,
    peak_lr: float,
    start_lr: float,
    warmup_steps: int,
    min_lr_ratio: float,
    schedule_start_step: int,
    schedule_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a scheduler whose phase is defined by absolute optimizer steps."""
    if peak_lr <= 0 or start_lr < 0:
        raise ValueError("learning rates must be non-negative and peak lr must be positive")
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be positive for resume continuation")
    if schedule_start_step < 0 or schedule_steps <= schedule_start_step:
        raise ValueError("schedule must extend beyond its starting step")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    start_ratio = max(1e-8, start_lr / peak_lr)

    def scale(step: int) -> float:
        relative = max(0, step - schedule_start_step)
        if relative < warmup_steps:
            progress = relative / warmup_steps
            return start_ratio + progress * (1.0 - start_ratio)
        decay_steps = schedule_steps - schedule_start_step - warmup_steps
        if decay_steps <= 0:
            return 1.0
        progress = (relative - warmup_steps) / decay_steps
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
        return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    for pg in optimizer.param_groups:
        pg["lr"] = peak_lr
        pg["initial_lr"] = peak_lr
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, scale, last_epoch=schedule_start_step - 1
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    config: MemoryTransformerConfig,
    rank: int,
    distributed: bool,
    use_fsdp: bool,
    schedule_metadata: dict[str, object] | None = None,
    data_metadata: dict[str, object] | None = None,
) -> None:
    if use_fsdp:
        state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, state_config, optim_config
        ):
            model_state = model.state_dict()
            optimizer_state = FSDP.optim_state_dict(model, optimizer)
    else:
        base = unwrap_model(model)
        model_state = base.state_dict()
        optimizer_state = optimizer.state_dict()
    local_rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    if distributed:
        rng_states: list[dict | None] | None = [None] * dist.get_world_size()
        dist.gather_object(local_rng, rng_states if rank == 0 else None, dst=0)
    else:
        rng_states = [local_rng]
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": step,
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": scheduler.state_dict(),
            "config": asdict(config),
            "rng_states": rng_states,
            "schedule": schedule_metadata,
            "data": data_metadata,
        }
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if distributed:
        dist.barrier()


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    distributed: bool,
    use_fsdp: bool,
    device: torch.device,
    expected_config: MemoryTransformerConfig,
) -> tuple[
    int,
    dict[str, object] | None,
    dict[str, object] | None,
    dict[str, object],
]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("config") != asdict(expected_config):
        raise ValueError("checkpoint model config does not match the requested config")
    if use_fsdp:
        state_config = FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
        optim_config = FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False)
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, state_config, optim_config
        ):
            model.load_state_dict(payload["model"])
            optimizer_state = FSDP.optim_state_dict_to_load(
                model, optimizer, payload["optimizer"]
            )
            optimizer.load_state_dict(optimizer_state)
    else:
        unwrap_model(model).load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    rng_states = payload.get("rng_states") or []
    if rng_states:
        state = rng_states[dist.get_rank() if distributed else 0]
        if state is not None:
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"])
            if state["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(state["cuda"])
    if distributed:
        dist.barrier()
    return (
        int(payload["step"]),
        payload.get("schedule"),
        payload.get("data"),
        payload["scheduler"],
    )


def build_data_metadata(
    store: TokenStore, args: argparse.Namespace, world_size: int
) -> dict[str, object]:
    return {
        "total_tokens": store.total_tokens,
        "vocab_size": int(store.manifest["vocab_size"]),
        "eos_token_id": int(store.manifest["eos_token_id"]),
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "world_size": world_size,
        "seed": args.seed,
    }


def batch_iterator(
    store: TokenStore,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    start_step: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    return token_batches(
        store,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        steps=args.steps * args.gradient_accumulation,
        rank=rank,
        world_size=world_size,
        start_step=start_step * args.gradient_accumulation,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("steps and gradient-accumulation must be positive")
    if args.log_steps <= 0:
        raise ValueError("log-steps must be positive")
    if (
        args.save_steps < 0
        or args.save_interval_hours < 0
        or args.max_runtime_hours < 0
        or args.schedule_steps < 0
    ):
        raise ValueError("save, runtime, and schedule limits must be non-negative")
    schedule_steps = args.schedule_steps or args.steps
    if schedule_steps <= 0:
        raise ValueError("schedule-steps must be positive")
    rank, world_size, local_rank, device, distributed = setup_distributed(args.device)
    seed_everything(args.seed, rank)
    store = TokenStore(args.dataset_dir)
    vocab_size = int(store.manifest["vocab_size"])
    tokenizer = load_tokenizer(args.tokenizer)
    if len(tokenizer) != vocab_size:
        raise ValueError(
            f"tokenizer length {len(tokenizer)} does not match token manifest vocab "
            f"size {vocab_size}"
        )
    config = build_config(args, vocab_size)
    model = MemoryTransformerForCausalLM(config)
    model, use_fsdp = wrap_model(model, args, device, distributed, local_rank)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = scheduler_for(
        optimizer, args.steps, args.warmup_steps, args.min_lr_ratio
    )
    start_step = 0
    continuation_lr: float | None = None
    schedule_metadata: dict[str, object] | None = None
    data_metadata = build_data_metadata(store, args, world_size)
    if args.resume is not None:
        start_step, _saved_schedule, saved_data, _saved_scheduler_state = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            distributed,
            use_fsdp,
            device,
            config,
        )
        if saved_data is not None and saved_data != data_metadata:
            raise ValueError("checkpoint data metadata does not match the requested data stream")
        if saved_data is None and rank == 0:
            print(
                "resume_warning checkpoint has no data metadata; "
                "data continuity relies on unchanged loader arguments",
                flush=True,
            )
        if start_step >= args.steps:
            raise ValueError("checkpoint step must be less than requested --steps")
        if args.warmup_steps > 0 and start_step > 0:
            # A resume keeps the optimizer state but starts the next phase from
            # the schedule requested by the current command line. This allows
            # a shorter cosine phase after a long-running schedule has plateaued.
            continuation_lr = float(optimizer.param_groups[0]["lr"])
            schedule_metadata = {
                "mode": "continuation",
                "start_step": start_step,
                "start_lr": continuation_lr,
                "peak_lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "min_lr_ratio": args.min_lr_ratio,
                "schedule_steps": schedule_steps,
            }
            scheduler = build_resume_scheduler(
                optimizer, args, continuation_lr, start_step, schedule_steps
            )
    iterator = batch_iterator(store, args, rank, world_size, start_step)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    run_started = time.monotonic()
    checkpoint_interval = args.save_interval_hours * 3600.0
    next_wall_checkpoint = (
        run_started + checkpoint_interval if checkpoint_interval > 0 else None
    )
    checkpoint_signal = (
        torch.zeros(1, dtype=torch.int32, device=device)
        if distributed and checkpoint_interval > 0
        else None
    )
    stop_signal = (
        torch.zeros(1, dtype=torch.int32, device=device)
        if distributed and args.max_runtime_hours > 0
        else None
    )
    tokens_per_step = (
        args.batch_size * world_size * args.sequence_length * args.gradient_accumulation
    )
    last_log_time = run_started
    last_log_step = start_step
    if rank == 0:
        extra = ""
        if continuation_lr is not None:
            extra = (
                f" resume_lr={continuation_lr:.3e}"
            )
        print(
            f"start step={start_step} target_steps={args.steps} "
            f"params={config.parameter_count():,} vocab={vocab_size} "
            f"global_tokens_per_step={tokens_per_step:,} "
            f"lr={args.lr:.3e} warmup_steps={args.warmup_steps} "
            f"min_lr_ratio={args.min_lr_ratio:.3f} precision={args.precision} "
            f"fsdp={use_fsdp} checkpoint={args.checkpoint}"
            f" schedule_steps={schedule_steps}"
            f" max_runtime_h={args.max_runtime_hours:.1f}"
            f"{extra}",
            flush=True,
        )
    if rank == 0 and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    early_exit = False
    for step in range(start_step, args.steps):
        loss_value = torch.zeros((), device=device)
        router_value = torch.zeros((), device=device)
        for micro_step in range(args.gradient_accumulation):
            inputs, labels = next(iterator)
            input_tensor = torch.from_numpy(inputs).to(device=device, non_blocking=True)
            label_tensor = torch.from_numpy(labels).to(device=device, non_blocking=True)
            if micro_step + 1 == args.gradient_accumulation or not hasattr(
                model, "no_sync"
            ):
                sync_context = contextlib.nullcontext()
            else:
                sync_context = model.no_sync()
            with sync_context:
                with autocast_context(device, args.precision):
                    output = model(input_tensor, labels=input_tensor)
                    loss = output["loss"] + args.router_weight * output["router_loss"]
                    scaled_loss = loss / args.gradient_accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                scaled_loss.backward()
            loss_value += output["loss"].detach()
            router_value += output["router_loss"].detach()
        if use_fsdp:
            grad_norm = model.clip_grad_norm_(args.max_grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed_step = step + 1
        if completed_step % args.log_steps == 0 or completed_step == 1:
            metrics = torch.stack(
                [
                    loss_value / args.gradient_accumulation,
                    router_value / args.gradient_accumulation,
                    torch.as_tensor(grad_norm, device=device),
                ]
            )
            metrics = reduce_mean(metrics, distributed)
            if rank == 0:
                now = time.monotonic()
                window_seconds = max(now - last_log_time, 1e-6)
                window_steps = completed_step - last_log_step
                tokens_per_second = window_steps * tokens_per_step / window_seconds
                if device.type == "cuda":
                    allocated_gib = torch.cuda.memory_allocated(device) / 2**30
                    reserved_gib = torch.cuda.memory_reserved(device) / 2**30
                    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
                    memory = (
                        f" alloc_gib={allocated_gib:.2f}"
                        f" reserved_gib={reserved_gib:.2f} peak_gib={peak_gib:.2f}"
                    )
                    torch.cuda.reset_peak_memory_stats(device)
                else:
                    memory = ""
                print(
                    f"step={completed_step} loss={metrics[0].item():.5f} "
                    f"router={metrics[1].item():.5f} grad={metrics[2].item():.5f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} "
                    f"elapsed_h={(now - run_started) / 3600.0:.3f} "
                    f"tokens_per_s={tokens_per_second:.1f}{memory}",
                    flush=True,
                )
                last_log_time = now
                last_log_step = completed_step
        step_checkpoint_due = args.save_steps > 0 and completed_step % args.save_steps == 0
        wall_checkpoint_due = (
            rank == 0
            and next_wall_checkpoint is not None
            and time.monotonic() >= next_wall_checkpoint
        )
        max_runtime_due = False
        if args.max_runtime_hours > 0:
            elapsed_h = (time.monotonic() - run_started) / 3600.0
            if rank == 0 and elapsed_h >= args.max_runtime_hours:
                max_runtime_due = True
            if stop_signal is not None:
                stop_signal.fill_(int(max_runtime_due))
                dist.broadcast(stop_signal, src=0)
                max_runtime_due = bool(stop_signal.item())
        should_save = step_checkpoint_due or wall_checkpoint_due or max_runtime_due
        if checkpoint_signal is not None:
            checkpoint_signal.fill_(int(should_save))
            dist.broadcast(checkpoint_signal, src=0)
            should_save = bool(checkpoint_signal.item())
        if should_save:
            save_checkpoint(
                args.checkpoint,
                model,
                optimizer,
                scheduler,
                completed_step,
                config,
                rank,
                distributed,
                use_fsdp,
                schedule_metadata,
                data_metadata,
            )
            if rank == 0:
                print(
                    f"checkpoint_saved step={completed_step} path={args.checkpoint} "
                    f"elapsed_h={(time.monotonic() - run_started) / 3600.0:.3f}",
                    flush=True,
                )
                if next_wall_checkpoint is not None:
                    next_wall_checkpoint = time.monotonic() + checkpoint_interval
        if max_runtime_due:
            early_exit = True
            break
    if not early_exit and not args.no_final_save:
        save_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            scheduler,
            args.steps,
            config,
            rank,
            distributed,
            use_fsdp,
            schedule_metadata,
            data_metadata,
        )
        if rank == 0:
            print(
                f"checkpoint_saved final_step={args.steps} path={args.checkpoint} "
                f"elapsed_h={(time.monotonic() - run_started) / 3600.0:.3f}",
                flush=True,
            )
    elif early_exit and rank == 0:
        print(
            f"max_runtime_stopped step={completed_step} "
            f"elapsed_h={(time.monotonic() - run_started) / 3600.0:.3f}",
            flush=True,
        )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
