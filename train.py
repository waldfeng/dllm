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
    parser.add_argument("--max-position-embeddings", type=int, default=32768)
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
    parser.add_argument("--exit-min-depth", type=int, default=8)
    parser.add_argument("--exit-tau-confidence", type=float, default=0.9)
    parser.add_argument("--exit-tau-margin", type=float, default=0.3)
    parser.add_argument("--exit-tau-delta", type=float, default=0.05)
    parser.add_argument(
        "--loss-weight-prior",
        default="0.65,0.05,0.05,0.05,0.15,0.05",
        help="fixed target distribution over the six loss terms "
        "[CE, confidence, margin, delta, speed, guard]",
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
        default=0.20,
        help="prior mass of the CE term at training end (inference-side signals)",
    )
    parser.add_argument("--exit-temp-start", type=float, default=2.0)
    parser.add_argument("--exit-temp-end", type=float, default=0.25)
    parser.add_argument("--exit-sparsity-weight", type=float, default=1.0)
    parser.add_argument(
        "--freeze-loss-weights",
        action="store_true",
        help="pin the six loss weights to the prior instead of learning them",
    )
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
    """Assemble the eight readout-training losses (all minimization-direction).

    Six per-depth normalized terms (sigmoid CE, confidence deficit, probability
    margin deficit, state-delta deficit, speed t/24, min-depth guard), weighted
    by the learnable distribution ``weights``; term seven aligns
    ``softmax(weighted terms)`` with the (annealed) prior through a sigmoid;
    term eight sharpens the depth-selection distribution and normalizes its
    entropy by ``log(depths)``.
    """
    device = rollout["states"].device
    depths_range = torch.arange(num_depths, device=device, dtype=torch.float32)
    depth_fraction = (depths_range + 1.0) / num_depths
    guard = (args.exit_min_depth - (depths_range + 1.0)).clamp_min(0.0) / num_depths
    speed = depth_fraction
    deltas = rollout["deltas"].float()
    states = rollout["states"].float()
    teacher_logits = rollout["teacher_logits"].float()
    target = target_tokens
    shortlist_size = max(1, min(args.exit_shortlist, teacher_logits.shape[-1] - 1))
    top_indices = teacher_logits.topk(shortlist_size, dim=-1).indices
    in_top = (top_indices == target[:, None]).any(-1)
    shortlist = torch.cat((top_indices, target[:, None]), dim=-1)
    keep_last = ~in_top

    head_weight = model.lm_head.weight.to(states.dtype)
    per_depth_terms = [[] for _ in range(6)]
    for depth_index in range(num_depths):
        logits = model.exit_readout.logits_at(depth_index, states[depth_index], head_weight)
        log_normalizer = torch.logsumexp(logits, dim=-1)
        top2 = torch.topk(logits, 2, dim=-1).values
        confidence = (top2[:, 0] - log_normalizer).exp()
        margin = confidence - (top2[:, 1] - log_normalizer).exp()
        target_logit = logits.gather(1, target[:, None]).squeeze(1)
        short_logits = logits.gather(1, shortlist)
        last_column = torch.where(
            keep_last[:, None],
            short_logits[:, -1:],
            torch.full_like(short_logits[:, -1:], float("-inf")),
        )
        short_logits = torch.cat((short_logits[:, :-1], last_column), dim=-1)
        cross_entropy = -(
            target_logit - torch.logsumexp(short_logits, dim=-1)
        )
        per_depth_terms[0].append(torch.sigmoid(cross_entropy))
        per_depth_terms[1].append((args.exit_tau_confidence - confidence).clamp_min(0.0))
        per_depth_terms[2].append((args.exit_tau_margin - margin).clamp_min(0.0))
        per_depth_terms[3].append((deltas[depth_index] - args.exit_tau_delta).clamp_min(0.0))
        per_depth_terms[4].append(speed[depth_index].expand(confidence.shape))
        per_depth_terms[5].append(guard[depth_index].expand(confidence.shape))
    term_rows = torch.stack(
        [torch.stack(column, dim=0) for column in per_depth_terms], dim=0
    )  # [6, D, B]
    term_means = term_rows.mean(dim=(1, 2))  # [6]
    weighted_terms = weights * term_means  # [6]
    alignment = torch.sigmoid(
        torch.nn.functional.cross_entropy(
            torch.log(weighted_terms.clamp_min(1e-12)).unsqueeze(0), prior.unsqueeze(0)
        )
    )
    selection_logits = -(
        (weights[:, None, None] * term_rows).sum(dim=0).mean(dim=1)
    )  # [D]
    temperature = _anneal(args.exit_temp_start, args.exit_temp_end, progress)
    selection = torch.softmax(selection_logits / temperature, dim=0)
    sparsity = -(selection * selection.clamp_min(1e-12).log()).sum() / max(
        1e-6, float(np.log(num_depths))
    )
    total = weighted_terms.sum() + alignment + args.exit_sparsity_weight * sparsity
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
                "sparsity",
            ),
            [*weighted_terms.detach(), alignment.detach(), sparsity.detach()],
        )
    }
    stats["total"] = total.detach().item()
    stats["selection_argmin"] = int(selection_logits.argmin().item()) + 1
    stats["expected_depth"] = float((selection * (depths_range + 1.0)).sum().item())
    stats["temperature"] = temperature
    return total, stats


def train_exit_readout(args: argparse.Namespace) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("exit-readout training runs on a single device only")
    if args.resume is None:
        raise ValueError("--resume must point at the base pretrained checkpoint")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    prior = torch.tensor(
        [float(value) for value in args.loss_weight_prior.split(",")], dtype=torch.float32
    )
    if prior.numel() != 6 or abs(float(prior.sum()) - 1.0) > 1e-4:
        raise ValueError("--loss-weight-prior must hold six values summing to one")
    device = (
        torch.device("cuda") if args.device != "cpu" and torch.cuda.is_available()
        else torch.device("cpu")
    )
    seed_everything(args.seed, 0)
    store = TokenStore(args.dataset_dir)
    vocab_size = int(store.manifest["vocab_size"])
    if Path(args.tokenizer).exists():
        tokenizer = load_tokenizer(args.tokenizer)
        if len(tokenizer) != vocab_size:
            raise ValueError(
                f"tokenizer length {len(tokenizer)} does not match token manifest vocab "
                f"size {vocab_size}"
            )
    else:
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
    if args.freeze_loss_weights:
        weights = prior.to(device)
        optimizer_parameters = trainable
    else:
        weights = torch.nn.Parameter(torch.log(prior.to(device) / prior.to(device)[-1:]).float())
        optimizer_parameters = [
            {"params": trainable},
            {"params": [weights], "weight_decay": 0.0},
        ]
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.exit_readout_lr,
        weight_decay=args.exit_readout_weight_decay,
    )
    warmup_steps = min(20, max(1, args.steps // 10))
    scheduler = scheduler_for(optimizer, args.steps, warmup_steps, 0.1)
    prior_start = prior.clone()
    prior_end = prior.clone()
    prior_end[0] = args.pi1_anneal_end
    prior_end[1:] *= (1.0 - args.pi1_anneal_end) / max(1e-6, float(prior_end[1:].sum()))
    iterator = token_batches(
        store,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        steps=args.steps,
        rank=0,
        world_size=1,
        seed=args.seed,
    )
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and args.precision == "bf16"
        else contextlib.nullcontext()
    )
    run_started = time.monotonic()
    for step, (inputs, labels) in enumerate(iterator):
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
            prior_start.lerp(prior_end, min(1.0, step / max(1, args.steps - 1))).to(device),
            args,
            step / max(1, args.steps - 1),
            config.num_layers,
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite exit-readout loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable + ([] if args.freeze_loss_weights else [weights]), 1.0)
        optimizer.step()
        scheduler.step()
        if (step + 1) % args.log_steps == 0 or step == 0:
            weights_display = weights.detach().softmax(dim=0)
            print(
                f"exit_readout step={step + 1} total={stats['total']:.5f} "
                f"ce={stats['ce']:.5f} conf={stats['confidence']:.5f} "
                f"margin={stats['margin']:.5f} delta={stats['delta']:.5f} "
                f"speed={stats['speed']:.5f} guard={stats['guard']:.5f} "
                f"align={stats['weight_alignment']:.5f} sparse={stats['sparsity']:.5f} "
                f"argmin={stats['selection_argmin']} E[t]={stats['expected_depth']:.2f} "
                f"T={stats['temperature']:.3f} "
                f"pi={','.join(f'{value:.3f}' for value in weights_display.tolist())} "
                f"lr={scheduler.get_last_lr()[0]:.3e} "
                f"elapsed_h={(time.monotonic() - run_started) / 3600.0:.3f}",
                flush=True,
            )
    atomic_torch_save(
        {
            "step": args.steps,
            "model": {key: value.cpu() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "exit_readout_rank": model.exit_readout.rank,
        },
        args.checkpoint,
    )
    print(
        f"exit_readout_saved step={args.steps} path={args.checkpoint} "
        f"elapsed_h={(time.monotonic() - run_started) / 3600.0:.3f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.train_exit_readout:
        train_exit_readout(args)
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
