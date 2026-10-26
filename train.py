"""Two-GPU FSDP training entry point for the shared-memory language model."""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.utils.checkpoint
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from memory_transformer import (
    MemoryReasoner,
    MemoryTransformerConfig,
    MemoryTransformerForCausalLM,
    SharedMemory,
    ensure_exit_readout,
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
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/tokens_v2")
    )
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
    parser.add_argument("--max-position-embeddings", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--router-weight", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16", "fp32", "amp"), default="amp")
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="recompute each reasoner cycle's activations in backward "
        "(trades ~30%% extra compute for a much larger usable batch size)",
    )
    parser.add_argument(
        "--ce-chunk-rows",
        type=int,
        default=0,
        help="when positive, compute the label CE over vocabulary-sized row "
        "chunks with per-chunk activation checkpointing instead of one "
        "monolithic [batch*seq, vocab] tensor (0 keeps the legacy path)",
    )
    parser.add_argument(
        "--dispatch-mode",
        choices=["loop", "sorted", "grouped"],
        default="loop",
        help="routed-bank dispatch: loop = legacy per-expert nonzero() scan, "
        "sorted = one-pass argsort partition with per-expert GEMMs and "
        "in-place accumulation, grouped = fused torch._grouped_mm GEMMs",
    )
    parser.add_argument(
        "--use-gqa",
        action="store_true",
        help="use grouped-query attention (enable_gqa) instead of "
        "repeat_interleave head expansion",
    )
    parser.add_argument("--no-fsdp", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/last.pt"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-repartition", action="store_true",
                        help="resume with a different global batch, preserving sample cursor and LR schedule")
    parser.add_argument(
        "--init-weights-from",
        type=Path,
        help="warm start full-parameter training: load only the model weights "
        "from the given checkpoint and start a fresh optimizer and schedule on "
        "the requested data stream (mutually exclusive with --resume)",
    )
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument(
        "--save-interval-hours",
        type=float,
        default=1.0,
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
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--train-exit-readout",
        action="store_true",
        help="continued-training mode: freeze the backbone and fit only the "
        "per-depth calibrated exit readout (r=16 LoRA + bias + temperature)",
    )
    parser.add_argument("--exit-readout-rank", type=int, default=16)
    parser.add_argument("--exit-readout-lr", type=float, default=1e-3)
    parser.add_argument("--exit-readout-weight-decay", type=float, default=1e-4)
    parser.add_argument("--exit-shortlist", type=int, default=64)
    parser.add_argument("--exit-min-depth", type=int, default=3)
    parser.add_argument("--exit-hard-min-depth", type=int, default=2,
                        help="hard lower bound for joint-training depth candidates; exit-min-depth is the soft guard threshold")
    parser.add_argument("--exit-bphase-mem-target", type=float, default=0.92,
                        help="phase-B sub-chunks are sized so the estimated allocated peak stays near this fraction of total memory")
    parser.add_argument("--exit-bphase-min-chunk", type=int, default=8,
                        help="smallest phase-B sub-chunk before a micro-batch is skipped as OOM")
    parser.add_argument("--profile-timing", action="store_true",
                        help="log a per-phase wall-clock breakdown each step (adds host syncs)")
    parser.add_argument("--exit-tau-confidence", type=float, default=0.9)
    parser.add_argument("--exit-tau-margin", type=float, default=0.3)
    parser.add_argument("--exit-tau-delta", type=float, default=0.05)
    parser.add_argument(
        "--exit-delta-weight",
        type=float,
        default=1.1,
        help="lossB coefficient for the selected depth's state-change term "
        "(delta / --exit-tau-delta). Calibrate for a target share s of lossB as "
        "s/(1-s) * (lossB_ce + router_weight * router_loss) / mean(delta/tau) "
        "from past joint logs; 1.1 ~= 20%% share at lossB_ce 8.5, d=0.99",
    )
    parser.add_argument(
        "--exit-selection-delta-weight",
        type=float,
        default=0.05,
        help="kappa: weight of kappa*delta/tau_s added to the five-term cost for "
        "the depth-selection argmin only (patent rule d_i=argmin(l+kappa*delta/"
        "tau_s)); loss_A keeps the five-term cost and tau_s is --exit-tau-delta",
    )
    parser.add_argument(
        "--loss-weight-prior",
        default="0.65,0.06,0.06,0.17,0.06",
        help="fixed weights over the five cost terms [CE, confidence, margin, "
        "speed, guard] (C_1..C_5), summing to one; the state-change term is not "
        "part of the weighted cost -- it enters the joint depth argmin through "
        "--exit-selection-delta-weight and lossB through --exit-delta-weight. "
        "The joint trainer rejects the legacy six-value input (with a delta "
        "slot); the readout trainer still maps it onto its six-term loss",
    )
    parser.add_argument(
        "--pi1-anneal-start",
        type=float,
        default=0.65,
        help="prior mass of the CE term at training start",
    )
    parser.add_argument(
        "--pi1-anneal-end",
        type=float,
        default=0.0,
        help="prior mass of the CE term at training end; 0 keeps the training "
        "depth-selection distribution identical to the inference-time one",
    )
    parser.add_argument("--exit-temp-start", type=float, default=2.0)
    parser.add_argument("--exit-temp-end", type=float, default=0.25)
    parser.add_argument("--joint-selection-entropy-beta", type=float, default=0.05)
    parser.add_argument(
        "--memory-target",
        type=float,
        default=0.9,
        help="readout mode: fraction of total GPU memory the auto batch probe "
        "should fill (batch_size <= 0 enables the probe)",
    )
    parser.add_argument(
        "--probe-batch",
        type=int,
        default=16,
        help="readout mode: seed batch size of the memory probe",
    )
    parser.add_argument(
        "--train-exit-joint",
        action="store_true",
        help="joint two-phase mode: full-parameter backbone training on the "
        "argmax-selected depth CE while the halting losses fit only the exit "
        "readout LoRA and the six loss weights (replicated tensors, no FSDP)",
    )
    parser.add_argument(
        "--exit-joint-readout-lr",
        type=float,
        default=1.25e-5,
        help="joint mode: readout/loss-weight lr; applied once per micro-batch",
    )
    return parser.parse_args()


def setup_distributed(device_choice: str) -> tuple[int, int, int, torch.device, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only policy: training requires CUDA")
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank if world_size > 1 else 0)
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
        max_position_embeddings=args.max_position_embeddings,
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
    if args.precision in ("bf16", "amp") and device.type == "cuda":
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

    # "amp": keep fp32 master parameters and rely on the surrounding autocast
    # for bf16 compute (single-GPU mixed precision without FSDP).  "bf16":
    # cast the whole module to bf16 (legacy pure-bf16 behavior).
    if args.precision == "amp" and device.type == "cuda":
        model.to(device=device)
    else:
        dtype = (
            torch.bfloat16
            if args.precision == "bf16" and device.type == "cuda"
            else None
        )
        model.to(device=device, dtype=dtype)
    if distributed:
        return DDP(model, device_ids=[local_rank] if device.type == "cuda" else None), False
    return model, False


def unwrap_model(model: torch.nn.Module) -> MemoryTransformerForCausalLM:
    if isinstance(model, DDP) or (FSDP is not None and isinstance(model, FSDP)):
        return model.module  # type: ignore[return-value]
    return model  # type: ignore[return-value]


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision in ("bf16", "amp"):
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
        rng_rank = dist.get_rank() if distributed else 0
        state = rng_states[rng_rank] if rng_rank < len(rng_states) else None
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


def data_streams_match(
    saved: dict[str, object] | None, requested: dict[str, object]
) -> bool:
    """Compare two data-stream metadata dicts up to rank re-partitioning.

    The sample sequence an optimizer step consumes depends only on the token
    store, sequence length, seed, and the *global* batch
    (batch_size * gradient_accumulation * world_size); how that global batch
    is split across ranks or micro-batches does not change the stream.  This
    lets a checkpoint trained on N GPUs resume on M GPUs with an equivalent
    split.
    """
    if saved is None:
        return True
    fixed_keys = (
        "total_tokens",
        "vocab_size",
        "eos_token_id",
        "sequence_length",
        "seed",
    )
    for key in fixed_keys:
        if saved.get(key) != requested.get(key):
            return False
    saved_global = (
        saved["batch_size"] * saved["gradient_accumulation"] * saved["world_size"]
    )
    requested_global = (
        requested["batch_size"]
        * requested["gradient_accumulation"]
        * requested["world_size"]
    )
    return saved_global == requested_global


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
        "samples_consumed": getattr(args, "samples_consumed", 0),
        "permutation_batch": getattr(args, "permutation_batch", args.batch_size * world_size),
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
        start_sample=getattr(args, "samples_consumed", None),
        permutation_batch=getattr(args, "permutation_batch", None),
    )


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


ARCHITECTURE_KEYS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "num_memory_blocks",
    "top_k",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "tie_word_embeddings",
    "dropout",
)


def _anneal(start: float, end: float, progress: float) -> float:
    return start + (end - start) * min(1.0, max(0.0, progress))


def assemble_exit_terms(
    model: MemoryTransformerForCausalLM,
    rollout: dict[str, Tensor],
    target_tokens: Tensor,
    args: argparse.Namespace,
    num_depths: int,
    base_logits: Optional[Tensor] = None,
    include_delta: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build the per-depth halting terms ``[6 or 5, D, B]`` and selection logits.

    Linear costs with no extra nonlinearity: bounded CE mapping
    ``CE/(ln V + CE)`` in [0, 1), confidence and margin deficits in [0, 1],
    relative state change over ``--exit-tau-delta``, depth fraction and the
    clamped guard ``max(0, (d_min - d) / D)``.  No cost is hard-clipped apart
    from the guard's zero floor. ``rollout["states"]`` holds
    the per-depth last-position normed states; when ``base_logits`` is given
    (the cached last-position ``W h`` products from :meth:`joint_rollout`) the
    readout logits are reconstructed from it plus the batched LoRA correction,
    skipping a redundant vocabulary matmul.  ``include_delta=False`` drops the
    state-change row (five terms) -- lossB trains it through
    :meth:`forward_until` instead, so the phase-A cost matrix omits it.
    """
    device = rollout["states"].device
    depths_range = torch.arange(num_depths, device=device, dtype=torch.float32)
    depth_fraction = (depths_range + 1.0) / num_depths
    guard = ((args.exit_min_depth - (depths_range + 1.0)) / num_depths).clamp_min(0.0)
    speed = depth_fraction
    deltas = rollout["deltas"].float() if include_delta else None
    states = rollout["states"].float()
    target = target_tokens
    log_vocab = float(np.log(model.lm_head.weight.shape[0]))
    if args.exit_tau_delta <= 0:
        raise ValueError("exit_tau_delta must be positive")

    compute_dtype = torch.bfloat16 if states.is_cuda else states.dtype
    readout = model.exit_readout
    if readout is None:
        raise ValueError("exit readout module missing")
    if base_logits is not None:
        corrections = readout.corrections_at(
            rollout["states"].to(compute_dtype)
        ).float()
        logits = (base_logits.to(compute_dtype).float() + corrections) * readout.temperature.float().unsqueeze(-1).unsqueeze(-1)
    else:
        head_weight = model.lm_head.weight.to(compute_dtype)
        logits = torch.stack(
            [
                readout.logits_at(depth_index, states[depth_index].to(compute_dtype), head_weight)
                for depth_index in range(num_depths)
            ]
        ).float()
    log_normalizer = torch.logsumexp(logits, dim=-1)  # [D, B]
    top2 = torch.topk(logits, 2, dim=-1).values
    confidence = (top2[..., 0] - log_normalizer).exp()
    margin = confidence - (top2[..., 1] - log_normalizer).exp()
    target_logit = logits.gather(-1, target[None, :, None].expand(num_depths, -1, 1)).squeeze(-1)
    cross_entropy = (log_normalizer - target_logit) / log_vocab
    term_rows = [
        cross_entropy / (1.0 + cross_entropy),
        1.0 - confidence,
        1.0 - margin,
    ]
    if include_delta:
        term_rows.append(deltas / args.exit_tau_delta)
    term_rows.append(speed[:, None].expand(num_depths, cross_entropy.shape[-1]))
    term_rows.append(guard[:, None].expand(num_depths, cross_entropy.shape[-1]))
    per_depth_terms = torch.stack(term_rows, dim=0)  # [5 or 6, D, B]
    selection_logits = -(per_depth_terms.mean(dim=-1).sum(dim=0))  # [D]
    return per_depth_terms, selection_logits, depths_range


def exit_readout_losses(
    model: MemoryTransformerForCausalLM,
    rollout: dict[str, Tensor],
    target_tokens: Tensor,
    weights: Tensor,
    prior: Tensor,
    args: argparse.Namespace,
    progress: float,
    num_depths: int,
) -> tuple[Tensor, dict[str, float]]:
    """Six weighted costs plus contribution-prior alignment.

    The depth distribution is used for diagnostics, not entropy regularization.
    CE is the normalized cross entropy CE/ln(vocabulary size); inference omits
    this label term.  The guard term is excluded from the alignment and the
    remaining target weights are renormalized.
    """
    term_rows, _, depths_range = assemble_exit_terms(
        model, rollout, target_tokens, args, num_depths
    )
    term_means = term_rows.mean(dim=(1, 2))  # [6]
    weighted_terms = weights * term_means  # [6]
    align_prior = prior[:-1] / prior[:-1].sum().clamp_min(1e-12)
    alignment = torch.sigmoid(
        torch.nn.functional.cross_entropy(
            torch.log(weighted_terms[:-1].clamp_min(1e-12)).unsqueeze(0), align_prior.unsqueeze(0)
        )
    )
    selection_logits = -(
        (weights[:, None, None] * term_rows).sum(dim=0).mean(dim=1)
    )  # [D]
    temperature = _anneal(args.exit_temp_start, args.exit_temp_end, progress)
    selection = torch.softmax(selection_logits / temperature, dim=0)
    total = weighted_terms.sum() + alignment
    stats = {
        name: value.detach().item()
        for name, value in zip(
            (
                "ce",
                "confidence",
                "margin",
                "delta",
                "speed",
                "guard",
                "weight_alignment",
            ),
            [*weighted_terms.detach(), alignment.detach()],
        )
    }
    stats["total"] = total.detach().item()
    stats["selection_argmin"] = int(selection_logits.argmin().item()) + 1
    stats["expected_depth"] = float((selection * (depths_range + 1.0)).sum().item())
    stats["temperature"] = temperature
    return total, stats


def _distributed_average(value: float, distributed: bool, device: torch.device) -> float:
    if not distributed:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor)
    return float(tensor.item()) / max(1, torch.distributed.get_world_size())


def _probe_memory_step(
    model: MemoryTransformerForCausalLM,
    store: TokenStore,
    args: argparse.Namespace,
    config,
    weights: Tensor,
    prior: Tensor,
    device: torch.device,
    autocast: contextlib.AbstractContextManager,
    batch_size: int,
) -> float:
    """Run one full readout step and return its peak CUDA allocation (-1 on OOM)."""
    iterator = token_batches(
        store,
        sequence_length=args.sequence_length,
        batch_size=batch_size,
        steps=1,
        seed=args.seed,
    )
    inputs, labels = next(iterator)
    input_tensor = torch.from_numpy(inputs).to(device)
    target_tensor = torch.from_numpy(labels).to(device)[:, -1]
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.no_grad():
            with autocast:
                rollout = model.exit_rollout(input_tensor)
        total, _ = exit_readout_losses(
            model, rollout, target_tensor, weights, prior.to(device), args, 0.0,
            config.num_layers,
        )
        total.backward()
        model.zero_grad(set_to_none=True)
        del rollout, total
        return float(torch.cuda.max_memory_allocated(device))
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return -1.0


def _probe_target_batch(
    model: MemoryTransformerForCausalLM,
    store: TokenStore,
    args: argparse.Namespace,
    config,
    weights: Tensor,
    prior: Tensor,
    device: torch.device,
    autocast: contextlib.AbstractContextManager,
) -> int:
    """Pick the per-rank batch that fills --memory-target of total GPU memory."""
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    budget = args.memory_target * total_bytes
    seed_batch = max(1, args.probe_batch)
    batch = seed_batch
    first = _probe_memory_step(
        model, store, args, config, weights, prior, device, autocast, batch
    )
    while first < 0 and batch > 1:
        batch //= 2
        first = _probe_memory_step(
            model, store, args, config, weights, prior, device, autocast, batch
        )
    if first < 0:
        raise RuntimeError("memory probe OOMed even at batch size 1")
    points = [(batch, first)]
    for factor in (3.0, 2.0):
        larger = _probe_memory_step(
            model, store, args, config, weights, prior, device, autocast,
            int(batch * factor),
        )
        if larger >= 0:
            points.append((int(batch * factor), larger))
            break
    if len(points) == 1:
        return batch
    (b0, p0), (b1, p1) = points[0], points[1]
    slope = (p1 - p0) / (b1 - b0)
    fixed = p0 - slope * b0
    target = max(points[0][0], int((budget - fixed) / max(1.0, slope)))
    print(
        f"exit_readout_probe seed={points[0][0]} peak={points[0][1] / 2**30:.1f}GiB "
        f"scaled={points[1][0]} peak={points[1][1] / 2**30:.1f}GiB "
        f"fixed={fixed / 2**30:.1f}GiB slope={slope / 2**20:.1f}MiB/row "
        f"target={target}",
        flush=True,
    )
    while target > points[0][0]:
        if _probe_memory_step(
            model, store, args, config, weights, prior, device, autocast, target
        ) >= 0:
            break
        target = max(points[0][0], int(target * 0.85))
    torch.cuda.empty_cache()
    return target


def train_exit_readout(args: argparse.Namespace) -> None:
    rank, world_size, local_rank, device, distributed = setup_distributed(args.device)
    if args.resume is None:
        raise ValueError("--resume must point at the base pretrained checkpoint")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    prior = torch.tensor(
        [float(value) for value in args.loss_weight_prior.split(",")], dtype=torch.float32
    )
    if prior.numel() == 5:
        # Five-value prior maps to [CE, confidence, margin, speed, guard];
        # this loss keeps a delta slot with zero weight (the state-change
        # term belongs to lossB in joint training).
        prior = torch.cat([prior[:3], prior.new_zeros(1), prior[3:]])
    if prior.numel() != 6 or abs(float(prior.sum()) - 1.0) > 1e-4:
        raise ValueError("--loss-weight-prior must hold five or six values summing to one")
    seed_everything(args.seed, rank)
    store = TokenStore(args.dataset_dir)
    vocab_size = int(store.manifest["vocab_size"])
    if Path(args.tokenizer).exists():
        tokenizer = load_tokenizer(args.tokenizer)
        if len(tokenizer) != vocab_size:
            raise ValueError(
                f"tokenizer length {len(tokenizer)} does not match token manifest vocab "
                f"size {vocab_size}"
            )
    elif rank == 0:
        print(
            f"exit_readout_warning tokenizer path {args.tokenizer} missing; "
            "skipping tokenizer check (readout training never tokenizes text)",
            flush=True,
        )
    config = build_config(args, vocab_size)
    payload = torch.load(args.resume, map_location="cpu", weights_only=False)
    saved = payload.get("config") or {}
    mismatched = [key for key in ARCHITECTURE_KEYS if saved.get(key) != getattr(config, key)]
    if mismatched:
        raise ValueError(
            f"base checkpoint architecture does not match the requested config: {mismatched}"
        )
    model = MemoryTransformerForCausalLM(config)
    has_readout = any(
        name.startswith("exit_readout.") for name in payload["model"]
    )
    if has_readout:
        ensure_exit_readout(model, rank=int(payload.get("exit_readout_rank", args.exit_readout_rank)))
    model.load_state_dict(payload["model"])
    if not has_readout:
        ensure_exit_readout(model, rank=args.exit_readout_rank)
    dtype = torch.bfloat16 if args.precision == "bf16" and device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype)
    model.exit_readout.float()
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("exit_readout."))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    start_step = int(payload.get("step", 0)) if has_readout else 0
    remaining_steps = args.steps - start_step
    if remaining_steps <= 0:
        raise ValueError(f"--steps {args.steps} already reached by resumed step {start_step}")
    weights = prior.to(device)
    optimizer_parameters = trainable
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.exit_readout_lr,
        weight_decay=args.exit_readout_weight_decay,
    )
    if has_readout and payload.get("optimizer"):
        optimizer.load_state_dict(readout_optimizer_without_weight_group(payload["optimizer"]))
    warmup_steps = min(20, max(1, remaining_steps // 10))
    scheduler = scheduler_for(optimizer, remaining_steps, warmup_steps, 0.1)
    prior_start = prior.clone()
    prior_end = prior.clone()
    prior_end[0] = args.pi1_anneal_end
    prior_end[1:] *= (1.0 - args.pi1_anneal_end) / max(1e-6, float(prior_end[1:].sum()))
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda" and args.precision in ("bf16", "amp")
        else contextlib.nullcontext()
    )
    if args.batch_size <= 0:
        if device.type != "cuda":
            raise ValueError("--batch-size 0 (memory probe) requires a CUDA device")
        args.batch_size = _probe_target_batch(
            model, store, args, config, weights, prior_start, device, autocast
        )
        torch.cuda.reset_peak_memory_stats(device)
    sync_parameters = trainable
    run_started = time.monotonic()
    last_wall_save = run_started

    iterator = token_batches(
        store,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        steps=remaining_steps,
        rank=rank,
        world_size=world_size,
        start_step=start_step,
        seed=args.seed,
    )
    for local_step, (inputs, labels) in enumerate(iterator):
        global_step = start_step + local_step
        progress = global_step / max(1, args.steps - 1)
        input_tensor = torch.from_numpy(inputs).to(device)
        target_tensor = torch.from_numpy(labels).to(device)[:, -1]
        with torch.no_grad():
            with autocast:
                rollout = model.exit_rollout(input_tensor)
        total, stats = exit_readout_losses(
            model,
            rollout,
            target_tensor,
            weights,
            prior_start.lerp(prior_end, min(1.0, progress)).to(device),
            args,
            progress,
            config.num_layers,
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite exit-readout loss at step {global_step}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        del rollout, total
        if distributed:
            for parameter in sync_parameters:
                if parameter.grad is not None:
                    torch.distributed.all_reduce(parameter.grad)
                    parameter.grad.div_(world_size)
        torch.nn.utils.clip_grad_norm_(sync_parameters, 1.0)
        optimizer.step()
        scheduler.step()
        stats = {
            key: _distributed_average(value, distributed, device)
            for key, value in stats.items()
        }
        if rank == 0 and ((local_step + 1) % args.log_steps == 0 or local_step == 0):
            weights_display = weights.detach().softmax(dim=0)
            memory_note = ""
            if device.type == "cuda":
                peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
                utilization = 100.0 * peak_gib / (
                    torch.cuda.get_device_properties(device).total_memory / 2**30
                )
                memory_note = f"peak={peak_gib:.1f}GiB({utilization:.0f}%) "
            print(
                f"exit_readout step={global_step + 1} total={stats['total']:.5f} "
                f"ce={stats['ce']:.5f} conf={stats['confidence']:.5f} "
                f"margin={stats['margin']:.5f} delta={stats['delta']:.5f} "
                f"speed={stats['speed']:.5f} guard={stats['guard']:.5f} "
                f"align={stats['weight_alignment']:.5f} "
                f"argmin={int(stats['selection_argmin'])} E[t]={stats['expected_depth']:.2f} "
                f"T={stats['temperature']:.3f} "
                f"pi={','.join(f'{value:.3f}' for value in weights_display.tolist())} "
                f"batch={args.batch_size} {memory_note}"
                f"lr={scheduler.get_last_lr()[0]:.3e} "
                f"elapsed_h={(time.monotonic() - run_started) / 3600.0:.3f}",
                flush=True,
            )
        if args.save_steps > 0 and (global_step + 1) % args.save_steps == 0 and rank == 0:
            atomic_torch_save(
                {
                    "step": global_step + 1,
                    "model": {key: value.cpu() for key, value in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(config),
                    "exit_readout_rank": model.exit_readout.rank,
                    "exit_loss_weights": weights.detach().cpu(),
                    "exit_temperature": stats["temperature"],
                },
                args.checkpoint,
            )
            print(
                f"exit_readout_saved step={global_step + 1} path={args.checkpoint} "
                f"elapsed_h={(time.monotonic() - run_started) / 3600.0:.3f}",
                flush=True,
            )
        # Wall-clock checkpointing: hourly overwrite so a crash between step
        # saves loses at most one hour of progress.
        if (
            rank == 0
            and args.save_interval_hours > 0
            and time.monotonic() - last_wall_save >= args.save_interval_hours * 3600.0
        ):
            last_wall_save = time.monotonic()
            atomic_torch_save(
                {
                    "step": global_step + 1,
                    "model": {key: value.cpu() for key, value in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(config),
                    "exit_readout_rank": model.exit_readout.rank,
                    "exit_loss_weights": weights.detach().cpu(),
                    "exit_temperature": stats["temperature"],
                },
                args.checkpoint,
            )
            print(
                f"exit_readout_saved step={global_step + 1} path={args.checkpoint} "
                f"elapsed_h={(time.monotonic() - run_started) / 3600.0:.3f} "
                f"trigger=wall",
                flush=True,
            )
    if rank == 0:
        atomic_torch_save(
            {
                "step": args.steps,
                "model": {key: value.cpu() for key, value in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "config": asdict(config),
                "exit_readout_rank": model.exit_readout.rank,
                "exit_loss_weights": weights.detach().cpu(),
                "exit_temperature": stats["temperature"],
            },
            args.checkpoint,
        )
        print(
            f"exit_readout_saved step={args.steps} path={args.checkpoint} "
            f"elapsed_h={(time.monotonic() - run_started) / 3600.0:.3f}",
            flush=True,
        )
    if distributed:
        torch.distributed.destroy_process_group()


def _joint_select_depth(
    model: MemoryTransformerForCausalLM,
    rollout: dict[str, Tensor],
    target_tokens: Tensor,
    weights: Tensor,
    args: argparse.Namespace,
    temperature: float,
    num_depths: int,
) -> tuple[int, Tensor, Tensor, Tensor]:
    """Phase-A selection: #10 softmax over the six weighted terms -> argmax.

    Uses the freshly updated halting parameters (call after ``opt_a.step()``)
    plus the cached last-position base logits from :meth:`joint_rollout`.
    Returns the 1-based exit depth, the selection distribution ``rho``, the
    per-depth terms ``[6, D, B]``, and the raw selection logits ``[D]``.
    The hard floor restricts candidates; the guard penalizes shallow depths softly.
    """
    term_rows, selection_logits, depths_range = assemble_exit_terms(
        model, rollout, target_tokens, args, num_depths,
        base_logits=rollout["base_logits"],
    )
    weighted = (weights[:, None, None] * term_rows).sum(dim=0).mean(dim=1)  # [D]
    logits = -weighted / temperature
    logits[:args.exit_hard_min_depth - 1] = float("-inf")
    selection = torch.softmax(logits, dim=0)
    depth = int(selection.argmax().item()) + 1
    return depth, selection, term_rows.detach(), weighted.detach()


def joint_token_rows(hidden, next_token_labels):
    """Pair every state with iterator labels, which are already shifted once."""
    if hidden.shape[:2] != next_token_labels.shape:
        raise ValueError("hidden states and next-token labels must share batch/sequence dimensions")
    return hidden.reshape(-1, hidden.size(-1)), next_token_labels.reshape(-1)


def joint_sample_losses(terms, weights, temperature, beta, hard_min_depth=1,
                        delta_costs=None, kappa=0.0):
    """A1 selected weighted cost; A2 per-sample self-label CE + entropy.

    Candidates start at hard_min_depth; the guard independently penalizes
    depths below its soft threshold.  The contribution alignment term
    is dropped: the fixed weights already set each cost's share of the total.

    ``delta_costs`` is the per-depth state-change row ``delta / tau_s`` with
    shape ``[D, N]``; ``kappa`` scales it into the SELECTION cost only, per
    the patent rule ``d_i = argmin(ℓ + κ·δ/τ_s)``.  loss_A stays on the
    five-term cost: A1, the selection distribution and the entropy never see
    the state-change term (it carries no readout gradient and trains the
    backbone through lossB instead).
    """
    if temperature <= 0 or beta < 0:
        raise ValueError("temperature must be positive and beta non-negative")
    weighted = weights[:, None, None] * terms
    costs = weighted.sum(0)
    selection_costs = costs
    if delta_costs is not None and kappa != 0.0:
        if delta_costs.shape != costs.shape:
            raise ValueError("delta_costs must share the [depth, sample] cost shape")
        selection_costs = costs + kappa * delta_costs
    if not 1 <= hard_min_depth <= costs.shape[0]:
        raise ValueError("hard_min_depth must be within the available depths")
    floor = hard_min_depth - 1
    chosen = selection_costs[floor:].detach().argmin(0) + floor
    sample = torch.arange(costs.shape[1], device=costs.device)
    a1 = costs[chosen, sample].mean()
    candidate = costs[floor:]
    log_p = (-candidate / temperature).log_softmax(0)
    p = log_p.exp()
    ce = -log_p[chosen - floor, sample].mean()
    entropy = -(p * log_p).sum(0).mean() / np.log(candidate.shape[0]) if candidate.shape[0] > 1 else ce * 0
    a2 = ce + beta * entropy
    return a1, a2, chosen + 1, p, entropy


def _joint_prior_and_kappa(args):
    """Five cost weights [CE, confidence, margin, speed, guard] plus kappa.

    The legacy six-value prior (with a delta slot) is retired: the
    state-change term now enters the joint depth argmin through the
    standalone ``--exit-selection-delta-weight`` and lossB through
    --exit-delta-weight, so the five weights must sum to one.
    """
    prior = torch.tensor(
        [float(value) for value in args.loss_weight_prior.split(",")], dtype=torch.float32
    )
    if prior.numel() != 5 or abs(float(prior.sum()) - 1.0) > 1e-4:
        raise ValueError(
            "--loss-weight-prior must hold exactly five values summing to one: "
            "[CE, confidence, margin, speed, guard]; the legacy delta slot moved "
            "to --exit-selection-delta-weight"
        )
    kappa = float(args.exit_selection_delta_weight)
    if kappa < 0:
        raise ValueError("--exit-selection-delta-weight must be non-negative")
    return prior, kappa


def readout_optimizer_without_weight_group(state):
    """Retain readout moments from the former readout + u optimizer."""
    if len(state["param_groups"]) == 1:
        return state
    group = state["param_groups"][0]
    return {"param_groups": [group], "state": {
        key: value for key, value in state["state"].items() if key in group["params"]}}


def train_joint_exit(args: argparse.Namespace) -> None:
    """Cache all depths without backbone grads; select separately per sample.

    A1 trains selected readouts; A2 trains all candidate readouts using self-label
    CE and entropy. Their sum makes one A update. B groups samples by the original
    choices and trains backbone CE + router loss through fixed readouts.
    """
    rank, world_size, local_rank, device, distributed = setup_distributed(args.device)
    if args.gradient_accumulation != 1:
        raise ValueError("joint training currently requires --gradient-accumulation 1")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    prior, kappa = _joint_prior_and_kappa(args)
    seed_everything(args.seed, rank)
    store = TokenStore(args.dataset_dir)
    vocab_size = int(store.manifest["vocab_size"])
    if Path(args.tokenizer).exists():
        tokenizer = load_tokenizer(args.tokenizer)
        if len(tokenizer) != vocab_size:
            raise ValueError(
                f"tokenizer length {len(tokenizer)} does not match token manifest vocab "
                f"size {vocab_size}"
            )
    elif rank == 0:
        print(
            "joint_exit_warning tokenizer path missing; skipping tokenizer check "
            "(joint training only consumes the token store)",
            flush=True,
        )
    config = build_config(args, vocab_size)
    source_path = args.resume if args.resume is not None else args.init_weights_from
    payload = (torch.load(source_path, map_location="cpu", weights_only=False)
               if source_path is not None else {})
    saved = payload.get("config") or {}
    mismatched = [key for key in ARCHITECTURE_KEYS if saved.get(key) != getattr(config, key)] if payload else []
    if mismatched:
        raise ValueError(
            f"checkpoint architecture does not match the requested config: {mismatched}"
        )
    model = MemoryTransformerForCausalLM(config)
    has_readout = any(name.startswith("exit_readout.") for name in payload.get("model", {}))
    if has_readout:
        ensure_exit_readout(model, rank=int(payload.get("exit_readout_rank", args.exit_readout_rank)))
    if payload:
        model.load_state_dict(payload["model"])
    if not has_readout:
        ensure_exit_readout(model, rank=args.exit_readout_rank)
    model.memory.dispatch_mode = args.dispatch_mode
    for layer in model.layers:
        layer.use_gqa = bool(args.use_gqa)
    resume_step = int(payload.get("step", 0)) if args.resume is not None else 0
    resume_optimizers = payload if args.resume is not None else {}
    del payload
    dtype = torch.bfloat16 if args.precision == "bf16" and device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype)
    model.exit_readout.float()
    model.train()
    if distributed:
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=0)
    start_step = resume_step
    if rank == 0:
        print(f"joint_start source={source_path or 'random'} step={start_step} seed={args.seed} "
              f"kappa={kappa} init=linear:xavier,gate:kaiming_relu_approx,router:xavier", flush=True)
    remaining_steps = args.steps - start_step
    if remaining_steps <= 0:
        raise ValueError(f"--steps {args.steps} already reached by resumed step {start_step}")

    # Parameter split: group A = exit readout; six cost weights are fixed.
    # group B = everything else (the full backbone including lm_head).
    readout_parameters = [p for p in model.exit_readout.parameters()]
    backbone_parameters = [
        p for n, p in model.named_parameters() if not n.startswith("exit_readout.")
    ]
    fixed_weights = prior.to(device)
    saved_kappa = resume_optimizers.get("exit_selection_delta_weight")
    if saved_kappa is not None and abs(float(saved_kappa) - kappa) > 1e-9:
        print(
            f"joint_kappa checkpoint={float(saved_kappa):.6g} cli={kappa:.6g}; "
            "keeping the CLI value",
            flush=True,
        )
    optimizer_a = torch.optim.AdamW(
        [
            {"params": readout_parameters, "weight_decay": args.exit_readout_weight_decay},
        ],
        lr=args.exit_joint_readout_lr,
    )
    optimizer_b = torch.optim.AdamW(
        backbone_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    # phase A updates once per micro-batch, so its schedule spans optimizer
    # steps times gradient_accumulation
    micro_total = remaining_steps * args.gradient_accumulation
    scheduler_a = scheduler_for(
        optimizer_a, micro_total, min(args.warmup_steps * args.gradient_accumulation, micro_total // 10 or 1), args.min_lr_ratio
    )
    scheduler_b = scheduler_for(
        optimizer_b, remaining_steps, args.warmup_steps, args.min_lr_ratio
    )
    if args.resume is not None:
        ckpt = resume_optimizers
        if ckpt.get("optimizer_readout"):
            optimizer_a.load_state_dict(readout_optimizer_without_weight_group(ckpt["optimizer_readout"]))
        if ckpt.get("optimizer"):
            optimizer_b.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler_readout"):
            saved_scheduler = dict(ckpt["scheduler_readout"])
            for key in ("base_lrs", "_last_lr", "lr_lambdas"):
                if key in saved_scheduler:
                    saved_scheduler[key] = saved_scheduler[key][:1]
            scheduler_a.load_state_dict(saved_scheduler)
        if ckpt.get("scheduler"):
            scheduler_b.load_state_dict(ckpt["scheduler"])
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda" and args.precision in ("bf16", "amp")
        else contextlib.nullcontext()
    )
    temperature = args.exit_temp_end  # fixed; training selection == inference halting
    model.gradient_checkpointing = bool(args.grad_checkpoint)
    run_started = time.monotonic()
    last_wall_save = run_started

    iterator = token_batches(
        store,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        # one micro-batch per yielded item: each runs its own phase-A halting
        # update, and gradient_accumulation of them form one optimizer step
        steps=remaining_steps * args.gradient_accumulation,
        rank=rank,
        world_size=world_size,
        start_step=start_step * args.gradient_accumulation,
        seed=args.seed,
    )
    micro_step_in_step = 0
    accumulated_stats: dict[str, float] = {}
    accumulated_depths: list[int] = []
    skipped_steps = 0
    grad_norm_value = 0.0
    oom_skips = 0
    oom_window_micros = 0
    oom_window_skips = 0
    window_oom_skips = 0
    bphase_scale = 1.0
    timing = bool(getattr(args, "profile_timing", False))
    step_timings: dict[str, float] = {}

    def _phase(key, t0):
        if timing:
            torch.cuda.synchronize()
            step_timings[key] = step_timings.get(key, 0.0) + time.perf_counter() - t0

    optimizer_b.zero_grad(set_to_none=True)
    for local_step, (inputs, labels) in enumerate(iterator):
        t_loop = time.perf_counter()
        global_step = start_step + local_step // args.gradient_accumulation
        input_tensor = torch.from_numpy(inputs).to(device, non_blocking=True)
        labels_tensor = torch.from_numpy(labels).to(device, non_blocking=True)
        target_last = labels_tensor[:, -1]
        _phase("h2d", t_loop)

        # Phase A: halting losses (readout + loss weights only).
        # On OOM the micro-batch is skipped, but the collective skeleton
        # (phase-A all-reduce + the gate's backbone all-reduce) must still
        # run with identical call counts on every rank or NCCL deadlocks:
        # a lone `continue` is what killed the B=20 run.
        rollout = None
        t_rollout = time.perf_counter()
        try:
            with autocast:
                rollout = model.joint_rollout(input_tensor, compute_deltas=True)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            oom_skips += 1
            window_oom_skips += 1
        phase_a_ok = rollout is not None
        _phase("rollout", t_rollout)
        if phase_a_ok:
            t_aterms = time.perf_counter()
            states = rollout["states"].clone()
            base_logits = rollout["base_logits"].clone()
            deltas = rollout["deltas"].clone()
            router_all = rollout["router_losses"].clone()
            del rollout
            phase_a_rollout = {"states": states}
            term_rows, _, depths_range = assemble_exit_terms(
                model, phase_a_rollout, target_last, args, config.num_layers,
                base_logits=base_logits, include_delta=False,
            )
        if phase_a_ok:
            weights = fixed_weights
            term_means = term_rows.mean(dim=(1, 2))
            weighted_terms = weights * term_means
            loss_a1, loss_a2, selected_depths, selection, entropy = joint_sample_losses(
                term_rows, weights, args.exit_temp_end,
                args.joint_selection_entropy_beta,
                hard_min_depth=args.exit_hard_min_depth,
                delta_costs=deltas / args.exit_tau_delta,
                kappa=kappa,
            )
            loss_a = loss_a1 + loss_a2
            _phase("a_terms", t_aterms)
            if local_step % args.log_steps == 0:
                print(f"joint_A rank={rank} A1={float(loss_a1.detach()):.6f} "
                      f"A2={float(loss_a2.detach()):.6f} entropy={float(entropy.detach()):.6f}", flush=True)
            if not torch.isfinite(loss_a):
                diagnostics = {
                    "weighted_terms": weighted_terms.detach().tolist(),
                    "weights": weights.detach().tolist(),
                    "term_means": term_means.detach().tolist(),
                    "temperature_max": float(
                        model.exit_readout.temperature.detach().abs().max()
                    ),
                    "corrections_absmax": float(
                        term_rows.detach().abs().max()
                    ),
                }
                raise FloatingPointError(
                    f"non-finite phase-A loss at step {global_step}: {diagnostics}"
                )
        optimizer_a.zero_grad(set_to_none=True)
        t_aupdate = time.perf_counter()
        if phase_a_ok:
            loss_a.backward()
        # this all-reduce is unconditional: ranks that OOMed participate with
        # zero grads so every rank issues the same collective sequence
        if distributed:
            phase_a_group = readout_parameters
            for parameter in phase_a_group:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
            for parameter in phase_a_group:
                torch.distributed.all_reduce(parameter.grad)
                parameter.grad.div_(world_size)
        torch.nn.utils.clip_grad_norm_(
            readout_parameters, args.max_grad_norm
        )
        phase_a_grads_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in readout_parameters
        )
        if phase_a_grads_finite and phase_a_ok:
            optimizer_a.step()
            scheduler_a.step()
        else:
            skipped_steps += 1
        optimizer_a.zero_grad(set_to_none=True)
        _phase("a_update", t_aupdate)

        # Keep the original per-sample choices through both A and B updates.
        t_diag = time.perf_counter()
        optimizer_b.zero_grad(set_to_none=True)
        if phase_a_ok:
            with torch.no_grad():
                fresh = fixed_weights
                depth = int(selected_depths.max())
            accumulated_depths.extend(selected_depths.tolist())
            if local_step % args.log_steps == 0:
                with torch.no_grad():
                    # per-depth loop keeps the fp32 diagnostic transient at
                    # [B, V] (~97MB) instead of a monolithic [D, B, V] fp32
                    # tensor (~7GB at batch 160) that fragments the allocator
                    corrections_all = model.exit_readout.corrections_at(states)
                    raw_ce = []
                    for depth_index in range(config.num_layers):
                        logits_d = (base_logits[depth_index].float() + corrections_all[depth_index].float()) * float(model.exit_readout.temperature[depth_index])
                        lse = torch.logsumexp(logits_d, dim=-1)
                        target_logit = logits_d.gather(-1, target_last[:, None]).squeeze(-1)
                        raw_ce.append((lse - target_logit).mean().item())
                contributions = (fresh[:, None, None] * term_rows).mean(dim=-1).detach().tolist()
                print(f"joint_depth rank={rank} micro={local_step+1} selected={depth} "
                      f"ce_by_depth={raw_ce} weighted_terms={contributions} weights={fresh.tolist()} "
                      f"kappa={kappa}", flush=True)
                diagnostic = {"rank": rank, "micro": local_step + 1, "selected": selected_depths.tolist(),
                              "loss_a1": float(loss_a1.detach()), "loss_a2": float(loss_a2.detach()),
                              "selection_entropy": float(entropy.detach()),
                              "training_min_depth": args.exit_min_depth,
                              "hard_min_depth": args.exit_hard_min_depth,
                              "ce_by_depth": raw_ce, "weighted_terms": contributions,
                              "weights": fresh.tolist(), "selection_kappa": kappa}
        _phase("diag", t_diag)

        # Phase B: backbone CE at the selected depth (readout detached).
        # Each depth group is processed in sub-chunks sized from a per-depth
        # memory model, and every sub-chunk backward()s immediately so the
        # activation graph never spans more than one sub-chunk.  This keeps
        # the allocated peak near --exit-bphase-mem-target of total memory
        # regardless of how deep the batch selects, without OOM-skips.
        skip_phase_b = not phase_a_ok
        t_bphase = time.perf_counter()
        for parameter in readout_parameters:
            parameter.requires_grad_(False)
        total_memory = torch.cuda.get_device_properties(device).total_memory

        def depth_chunk_samples(group_depth, remaining, scale):
            # Linear per-sample cost model anchored on measured peaks (batch
            # 160, 95.6 GiB cards): depth~14 = 0.49%/sample, depth 20 = 0.527%,
            # depth 24 = 0.90%, base (params+optimizer) ~= 17%.  Per-sample
            # memory is nearly FLAT below depth 20 (checkpoint inputs +
            # graph dominate), so the extrapolation floor is 0.50% -- the
            # earlier 0.002% floor let shallow-depth batches run as ONE
            # 160-sample chunk and blew the peak to 95.7% (OOM-STORM).
            # `scale` backs off after OOM retries and recovers toward 1.0 on
            # successful micro-batches.
            per_sample = max(0.0938 * group_depth - 1.349, 0.50) / 88.0
            current = torch.cuda.memory_allocated(device) / total_memory
            budget = max(args.exit_bphase_mem_target - current - 0.01, 0.02) * scale
            return int(min(remaining, max(args.exit_bphase_min_chunk, budget / per_sample)))

        if phase_a_ok:
            masked_count = labels_tensor.ne(-100).sum().clamp_min(1)
            # fp32 softmax/CE: shallow-depth logits are large-magnitude and
            # bf16 log-softmax over 151k classes loses precision.  The CE is
            # row-chunked with per-chunk checkpointing so the fp32 copy and
            # its backward grads never span the whole sub-chunk: a monolithic
            # [B*(T-1), vocab] fp32 tensor is ~9.3 GiB (plus ~15 GiB of bf16
            # intermediates) and OOMs both ranks when the selected depth
            # runs deep.
            ce_chunk_size = args.ce_chunk_rows if args.ce_chunk_rows > 0 else 8192
            ce_total = 0.0
            router_total = 0.0
            delta_total = 0.0
            # An OOM inside a sub-chunk may have already deposited partial
            # gradients, so the whole micro-batch restarts from scratch with
            # smaller chunks; gradients from the aborted attempt are dropped.
            bphase_attempts = 0
            phase_b_failed = False
            while True:
                bphase_attempts += 1
                if bphase_attempts > 3:
                    phase_b_failed = True
                    break
                forward = flat_hidden = flat_labels = ce_sum = loss_chunk = None
                try:
                    ce_total = 0.0
                    router_total = 0.0
                    delta_total = 0.0
                    optimizer_b.zero_grad(set_to_none=True)
                    for group_depth in selected_depths.unique().tolist():
                        indices = (selected_depths == group_depth).nonzero().flatten()
                        pos = 0
                        while pos < indices.numel():
                            chunk = depth_chunk_samples(group_depth, indices.numel() - pos, bphase_scale)
                            sub = indices[pos:pos + chunk]
                            with autocast:
                                forward = model.forward_until(
                                    input_tensor[sub], num_cycles=group_depth, collect_states=False,
                                    collect_delta=True)
                            # iter_training_batches already supplies window[1:]
                            # labels.  Shifting again would train a
                            # two-position-ahead target.
                            flat_hidden, flat_labels = joint_token_rows(
                                forward["final_normed"], labels_tensor[sub])

                            def chunk_ce(h_rows, y_rows, selected=group_depth):
                                logits = model.exit_readout.logits_at(selected - 1, h_rows, model.lm_head.weight)
                                return torch.nn.functional.cross_entropy(logits.float(), y_rows, ignore_index=-100, reduction="sum")

                            ce_sum = flat_hidden.new_zeros((), dtype=torch.float32)
                            for start in range(0, flat_hidden.size(0), ce_chunk_size):
                                ce_sum = ce_sum + torch.utils.checkpoint.checkpoint(
                                    chunk_ce, flat_hidden[start:start+ce_chunk_size], flat_labels[start:start+ce_chunk_size], use_reentrant=False)
                            router_term = forward["router_losses"][:group_depth].mean() * sub.numel() / input_tensor.shape[0]
                            delta_term = forward["final_delta"].float().sum() / input_tensor.shape[0]
                            loss_chunk = (ce_sum / masked_count + args.router_weight * router_term + args.exit_delta_weight * delta_term / args.exit_tau_delta) / args.gradient_accumulation
                            if not torch.isfinite(loss_chunk):
                                raise FloatingPointError(f"non-finite phase-B loss at step {global_step}")
                            loss_chunk.backward()
                            ce_total += float(ce_sum.detach())
                            router_total += float(router_term.detach())
                            delta_total += float(delta_term.detach())
                            del forward, flat_hidden, flat_labels, ce_sum, loss_chunk
                            pos += chunk
                    break
                except torch.OutOfMemoryError:
                    forward = flat_hidden = flat_labels = ce_sum = loss_chunk = None
                    optimizer_b.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    bphase_scale = max(0.25, bphase_scale * 0.5)
            bphase_scale = min(1.0, bphase_scale * 2.0) if not phase_b_failed else bphase_scale
            if not phase_b_failed:
                ce_mean = ce_total / float(masked_count)
                router_sum = router_total
                delta_mean = delta_total
                loss_b = ce_mean + args.router_weight * router_sum + args.exit_delta_weight * delta_mean / args.exit_tau_delta
                if not np.isfinite(loss_b):
                    raise FloatingPointError(f"non-finite phase-B loss at step {global_step}")
                if local_step % args.log_steps == 0:
                    router_grad = [None if layer.router.weight.grad is None else
                                   float(layer.router.weight.grad.float().norm()) for layer in model.layers]
                    print(f"joint_grad rank={rank} micro={local_step+1} routers={router_grad} "
                          f"ce={ce_mean:.5f} delta={delta_mean:.5f} "
                          f"peak_gib={torch.cuda.max_memory_allocated(device)/2**30:.2f}", flush=True)
                    import json
                    diagnostic.update(router_grad=router_grad, sequence_ce=ce_mean,
                                      sequence_delta=delta_mean,
                                      peak_gib=torch.cuda.max_memory_allocated(device)/2**30)
                    diagnostic_path = args.checkpoint.with_suffix(f".rank{rank}.diagnostics.jsonl")
                    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
                    with diagnostic_path.open("a") as diagnostic_file:
                        diagnostic_file.write(json.dumps(diagnostic) + "\n")
            else:
                # persistent OOM even at the minimum sub-chunk; drop this
                # micro-batch's CE.  No `continue` here: the gate below (and
                # its backbone all-reduce) must run on every rank.
                torch.cuda.empty_cache()
                oom_skips += 1
                window_oom_skips += 1
                optimizer_b.zero_grad(set_to_none=True)
                skip_phase_b = True
        _phase("b_phase", t_bphase)
        for parameter in readout_parameters:
            parameter.requires_grad_(True)
        t_bupdate = time.perf_counter()
        micro_step_in_step += 1
        if not skip_phase_b:
            with torch.no_grad():
                accumulated_stats["ce_at_depth"] = (
                    accumulated_stats.get("ce_at_depth", 0.0) + ce_mean
                )
                accumulated_stats["delta_term"] = (
                    accumulated_stats.get("delta_term", 0.0) + delta_mean
                )
                accumulated_stats["loss_a"] = (
                    accumulated_stats.get("loss_a", 0.0) + float(loss_a.detach())
                )
                accumulated_stats["expected_depth"] = (
                    accumulated_stats.get("expected_depth", 0.0)
                    + float((selection * (depths_range[-selection.shape[0]:] + 1.0)[:, None]).sum(0).mean())
                )

        if micro_step_in_step == args.gradient_accumulation:
            if distributed:
                # the selected depth differs per rank, so cycles beyond a rank's
                # depth produce no gradient; materialize zero grads for the
                # missing ones to keep the all-reduce set identical everywhere
                for parameter in backbone_parameters:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                for parameter in backbone_parameters:
                    torch.distributed.all_reduce(parameter.grad)
                    parameter.grad.div_(world_size)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                backbone_parameters, args.max_grad_norm
            )
            grad_norm_value = float(grad_norm)
            if not np.isfinite(grad_norm_value):
                # shallow-depth CE gradients can spike early in training;
                # drop the step instead of corrupting the backbone with NaNs
                optimizer_b.zero_grad(set_to_none=True)
                skipped_steps += 1
                grad_norm_value = float("nan")
            else:
                optimizer_b.step()
                scheduler_b.step()
        optimizer_b.zero_grad(set_to_none=True)
        _phase("b_update", t_bupdate)
        completed_step = global_step + 1
        # circuit breaker: silent OOM-skip storms look healthy to any process
        # check (no traceback, log keeps rolling) while the run does no work;
        # if >20% of the last window's micro-batches were skipped, abort with
        # a distinctive exit code so the watchdog wakes the operator instead
        # of burning GPU hours on a stalled run
        oom_window_micros += args.gradient_accumulation
        oom_window_skips += window_oom_skips
        skip_rate = oom_window_skips / max(1, oom_window_micros)
        if skip_rate > 0.2:
            if rank == 0:
                print(
                    f"OOM-STORM skip_rate={skip_rate:.0%} "
                    f"window_skips={oom_window_skips} "
                    f"window_micros={oom_window_micros} "
                    f"batch={args.batch_size} step={completed_step} "
                    f"oom_total={oom_skips}",
                    flush=True,
                )
            sys.exit(42)
        oom_window_micros = 0
        oom_window_skips = 0
        window_oom_skips = 0
        divisor = max(1, micro_step_in_step)
        means = {key: value / divisor for key, value in accumulated_stats.items()}
        mean_depth = (
            sum(accumulated_depths) / len(accumulated_depths)
            if accumulated_depths
            else 0.0
        )
        if rank == 0 and completed_step % args.log_steps == 0:
            peak_note = ""
            if device.type == "cuda":
                peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
                peak_note = f" peak={peak_gib:.1f}GiB"
                torch.cuda.reset_peak_memory_stats(device)
            weights_display = fixed_weights.cpu()
            mem_note = ""
            if device.type == "cuda":
                used_gib = torch.cuda.memory_allocated(device) / 2**30
                total_gib = (
                    torch.cuda.get_device_properties(device).total_memory / 2**30
                )
                mem_note = (
                    f" mem={used_gib:.1f}/{total_gib:.0f}GiB"
                    f"({100.0 * used_gib / total_gib:.0f}%)"
                )
            print(
                f"joint step={completed_step} lossB_ce={means.get('ce_at_depth', 0.0):.5f} "
                f"lossB_delta={means.get('delta_term', 0.0):.5f} "
                f"lossA={means.get('loss_a', 0.0):.5f} depth={mean_depth:.2f} "
                f"E[t]={means.get('expected_depth', 0.0):.2f} "
                f"w={','.join(f'{value:.3f}' for value in weights_display.tolist())} "
                f"grad={grad_norm_value:.3f} skipped={skipped_steps} "
                f"oom={oom_skips} "
                f"lrB={scheduler_b.get_last_lr()[0]:.3e} "
                f"lrA={scheduler_a.get_last_lr()[0]:.3e}{peak_note}{mem_note} "
                f"elapsed_h={(time.monotonic() - run_started) / 3600.0:.3f}",
                flush=True,
            )
        if timing and rank == 0:
            _phase("log", t_loop)
            wall = step_timings.pop("log", 0.0)
            breakdown = " ".join(f"{k}={v:.2f}" for k, v in step_timings.items())
            print(
                f"joint_timing step={completed_step} {breakdown} "
                f"sum={sum(step_timings.values()):.2f} wall={wall:.2f} "
                f"other={max(0.0, wall - sum(step_timings.values())):.2f}",
                flush=True,
            )
            step_timings = {}
        micro_step_in_step = 0
        accumulated_stats = {}
        accumulated_depths = []
        if args.save_steps > 0 and completed_step % args.save_steps == 0:
            if rank == 0:
                atomic_torch_save(
                    {
                        "step": completed_step,
                        "model": {
                            key: value.cpu()
                            for key, value in model.state_dict().items()
                        },
                        "optimizer": optimizer_b.state_dict(),
                        "optimizer_readout": optimizer_a.state_dict(),
                        "exit_readout_rank": model.exit_readout.rank,
                        "scheduler": scheduler_b.state_dict(),
                        "scheduler_readout": scheduler_a.state_dict(),
                        "config": asdict(config),
                        "exit_cost_mapping": "linear_v1",
                        "joint_objective": "sample_a1_a2_v4",
                        "joint_selection_entropy_beta": args.joint_selection_entropy_beta,
                        "exit_hard_min_depth": args.exit_hard_min_depth,
                        "exit_guard_min_depth": args.exit_min_depth,
                        "exit_loss_weights": fixed_weights.detach().cpu(),
                        "exit_temperature": args.exit_temp_end,
                        "exit_selection_delta_weight": kappa,
                    },
                    args.checkpoint,
                )
                print(
                    f"joint_saved step={completed_step} path={args.checkpoint} "
                    f"elapsed_h={(time.monotonic() - run_started) / 3600.0:.3f}",
                    flush=True,
                )
            if distributed:
                dist.barrier()
        # Wall-clock checkpointing: the legacy --train-exit path honors
        # --save-interval-hours but this joint loop only saved on step
        # multiples.  A crash between step saves (e.g. OOM-STORM at step 72
        # with --save-steps 250) then lost every step since step 0.
        if (
            rank == 0
            and args.save_interval_hours > 0
            and time.monotonic() - last_wall_save >= args.save_interval_hours * 3600.0
        ):
            last_wall_save = time.monotonic()
            atomic_torch_save(
                {
                    "step": completed_step,
                    "model": {
                        key: value.cpu()
                        for key, value in model.state_dict().items()
                    },
                    "optimizer": optimizer_b.state_dict(),
                    "optimizer_readout": optimizer_a.state_dict(),
                    "exit_readout_rank": model.exit_readout.rank,
                    "scheduler": scheduler_b.state_dict(),
                    "scheduler_readout": scheduler_a.state_dict(),
                    "config": asdict(config),
                    "exit_cost_mapping": "linear_v1",
                    "joint_objective": "sample_a1_a2_v4",
                    "joint_selection_entropy_beta": args.joint_selection_entropy_beta,
                    "exit_hard_min_depth": args.exit_hard_min_depth,
                    "exit_guard_min_depth": args.exit_min_depth,
                    "exit_loss_weights": fixed_weights.detach().cpu(),
                    "exit_temperature": args.exit_temp_end,
                    "exit_selection_delta_weight": kappa,
                },
                args.checkpoint,
            )
            print(
                f"joint_saved step={completed_step} path={args.checkpoint} "
                f"elapsed_h={(time.monotonic() - run_started) / 3600.0:.3f} "
                f"trigger=wall",
                flush=True,
            )
        if distributed:
            dist.barrier()
        if args.max_runtime_hours > 0 and (
            (time.monotonic() - run_started) / 3600.0 >= args.max_runtime_hours
        ):
            if rank == 0:
                print(
                    f"max_runtime_stopped step={completed_step} "
                    f"elapsed_h={(time.monotonic() - run_started) / 3600.0:.3f}",
                    flush=True,
                )
            break
    if rank == 0 and micro_step_in_step == 0 and not args.no_final_save:
        atomic_torch_save(
            {
                "step": completed_step,
                "model": {key: value.cpu() for key, value in model.state_dict().items()},
                "optimizer": optimizer_b.state_dict(),
                "optimizer_readout": optimizer_a.state_dict(),
                "exit_readout_rank": model.exit_readout.rank,
                "scheduler": scheduler_b.state_dict(),
                "scheduler_readout": scheduler_a.state_dict(),
                "config": asdict(config),
                "exit_cost_mapping": "linear_v1",
                "joint_objective": "sample_a1_a2_v4",
                "joint_selection_entropy_beta": args.joint_selection_entropy_beta,
                "exit_hard_min_depth": args.exit_hard_min_depth,
                "exit_guard_min_depth": args.exit_min_depth,
                "exit_loss_weights": fixed_weights.detach().cpu(),
                "exit_temperature": args.exit_temp_end,
                "exit_selection_delta_weight": kappa,
            },
            args.checkpoint,
        )
        print(
            f"joint_saved step={completed_step} path={args.checkpoint} "
            f"elapsed_h={(time.monotonic() - run_started) / 3600.0:.3f}",
            flush=True,
        )
    if distributed:
        torch.distributed.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.train_exit_readout:
        train_exit_readout(args)
        return
    if args.train_exit_joint:
        train_joint_exit(args)
        return
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
    model.gradient_checkpointing = bool(args.grad_checkpoint)
    model.loss_chunk_rows = max(0, int(args.ce_chunk_rows))
    if args.init_weights_from is not None:
        if args.resume is not None:
            raise ValueError("--resume and --init-weights-from are mutually exclusive")
        payload = torch.load(args.init_weights_from, map_location="cpu", weights_only=False)
        saved = payload.get("config") or {}
        mismatched = [key for key in ARCHITECTURE_KEYS if saved.get(key) != getattr(config, key)]
        if mismatched:
            raise ValueError(
                f"init weights checkpoint architecture does not match the "
                f"requested config: {mismatched}"
            )
        model.load_state_dict(payload["model"])
        del payload
        if rank == 0:
            print(
                f"init_weights loaded from {args.init_weights_from} "
                "(model weights only; optimizer and schedule restart at step 0)",
                flush=True,
            )
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
    args.samples_consumed = 0
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
        if saved_data is not None and args.resume_repartition:
            comparable = dict(data_metadata)
            for key in ("batch_size", "gradient_accumulation", "world_size"):
                comparable[key] = saved_data[key]
            if not data_streams_match(saved_data, comparable):
                raise ValueError("repartition requires the same dataset, sequence length and seed")
            args.samples_consumed = int(saved_data.get("samples_consumed", start_step * saved_data["batch_size"] * saved_data["gradient_accumulation"] * saved_data["world_size"]))
            args.permutation_batch = int(saved_data.get("permutation_batch", saved_data["batch_size"] * saved_data["world_size"]))
            data_metadata = build_data_metadata(store, args, world_size)
            if rank == 0:
                print(f"resume_data samples_consumed={args.samples_consumed} permutation_batch={args.permutation_batch}", flush=True)
        elif saved_data is not None:
            if not data_streams_match(saved_data, data_metadata):
                raise ValueError(
                    "checkpoint data metadata does not match the requested data stream: "
                    f"saved={saved_data} requested={data_metadata}"
                )
            args.samples_consumed = int(saved_data.get("samples_consumed", start_step * saved_data["batch_size"] * saved_data["gradient_accumulation"] * saved_data["world_size"]))
            args.permutation_batch = int(saved_data.get("permutation_batch", saved_data["batch_size"] * saved_data["world_size"]))
            if rank == 0:
                print(f"resume_data samples_consumed={args.samples_consumed} permutation_batch={args.permutation_batch}", flush=True)
        if saved_data is None and rank == 0:
            print(
                "resume_warning checkpoint has no data metadata; "
                "data continuity relies on unchanged loader arguments",
                flush=True,
            )
        if start_step >= args.steps:
            raise ValueError("checkpoint step must be less than requested --steps")
        if args.resume_repartition and _saved_schedule:
            schedule_metadata = _saved_schedule
            scheduler = build_continuation_scheduler(
                optimizer, peak_lr=float(_saved_schedule["peak_lr"]),
                start_lr=float(_saved_schedule["start_lr"]),
                warmup_steps=int(_saved_schedule["warmup_steps"]),
                min_lr_ratio=float(_saved_schedule["min_lr_ratio"]),
                schedule_start_step=int(_saved_schedule["start_step"]),
                schedule_steps=int(_saved_schedule["schedule_steps"]),
            )
            scheduler.load_state_dict(_saved_scheduler_state)
            for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
                group["lr"] = lr
        elif args.warmup_steps > 0 and start_step > 0:
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
        log_due = (step + 1) % args.log_steps == 0 or step + 1 == 1
        health = None
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
                    want_health = log_due and micro_step + 1 == args.gradient_accumulation
                    output = model(
                        input_tensor,
                        labels=input_tensor,
                        return_router_health=want_health,
                    )
                    if want_health:
                        health = output.get("router_health")
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
        data_metadata["samples_consumed"] = args.samples_consumed + (completed_step - start_step) * args.batch_size * world_size * args.gradient_accumulation
        if log_due:
            if device.type == "cuda":
                total_gib = torch.cuda.get_device_properties(device).total_memory / 2**30
                print(f"gpu_memory rank={rank} allocated_gib={torch.cuda.memory_allocated(device) / 2**30:.2f} "
                      f"reserved_gib={torch.cuda.memory_reserved(device) / 2**30:.2f} "
                      f"peak_gib={torch.cuda.max_memory_allocated(device) / 2**30:.2f} total_gib={total_gib:.2f}", flush=True)
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
                health_field = ""
                if health is not None:
                    layer_losses = health["layer_losses"]
                    block_load = health["block_load"]
                    dead = int((block_load < 1e-4).sum())
                    entropy = -(
                        block_load * (block_load + 1e-12).log()
                    ).sum() / math.log(block_load.numel())
                    health_field = (
                        f" rl_min={min(layer_losses):.3f} rl_max={max(layer_losses):.3f}"
                        f" rb_dead={dead}/{block_load.numel()}"
                        f" rb_max={block_load.max().item():.4f}"
                        f" rb_min={block_load.min().item():.4f}"
                        f" rb_ent={entropy.item():.3f}"
                    )
                exit_enabled = bool(getattr(config, "early_exit_enabled", False))
                readout = (
                    "present" if getattr(model, "exit_readout", None) is not None else "absent"
                )
                exit_min_depth = min(
                    getattr(config, "early_exit_min_depth", 0), config.num_layers
                )
                exit_field = (
                    f" exit={'on' if exit_enabled else 'off'}"
                    f" ro={readout} md={exit_min_depth}"
                )
                print(
                    f"step={completed_step} loss={metrics[0].item():.5f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e}"
                    f"{health_field}{exit_field}{memory} tokens_per_s={tokens_per_second:.1f}",
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
