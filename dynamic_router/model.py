"""Block-level language model with a fixed-capacity, fully dynamic router."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class RouteRecord:
    source_id: int
    logits: Tensor
    sample_index: int = 0
    depth: int = 0
    path_length: int = 0
    gradient_norm: float | None = None
    gradient: Tensor | None = None


class MemoryModule(nn.Module):
    """Process one block with recurrent K/V generation and residual attention."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.gru_norm = nn.LayerNorm(d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use Xavier input weights and orthogonal recurrent GRU gates."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                if module._qkv_same_embed_dim:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                else:
                    nn.init.xavier_uniform_(module.q_proj_weight)
                    nn.init.xavier_uniform_(module.k_proj_weight)
                    nn.init.xavier_uniform_(module.v_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)
            elif isinstance(module, nn.GRU):
                for name, parameter in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "weight_hh" in name:
                        for gate in parameter.chunk(3, dim=0):
                            nn.init.orthogonal_(gate)
                    else:
                        nn.init.zeros_(parameter)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _forward_impl(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        gru_input = self.gru_norm(x)
        attention_input = self.attn_norm(x)
        query = self.q_proj(attention_input)
        # GRU backward can overflow under BF16 autocast on long dynamic paths.
        # Keep recurrent state and its backward path in FP32 while leaving the
        # attention and feed-forward branches eligible for autocasting.
        with torch.autocast(device_type=x.device.type, enabled=False):
            gru_output, next_hidden = self.gru(
                gru_input.float(), hidden.float()
            )
        keys = self.k_proj(gru_output)
        values = self.v_proj(gru_output)
        attended, _ = self.attn(query, keys, values, need_weights=False)
        attention_hidden = x + attended
        output = attention_hidden + self.ffn(self.ffn_norm(attention_hidden))
        return output, next_hidden

    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or hidden.ndim != 3:
            raise ValueError("module input and hidden state must be rank-3 tensors")
        if hidden.shape[0] != 1 or hidden.shape[1] != x.shape[0]:
            raise ValueError("hidden state must have shape [1, batch, d_model]")
        if self.training and torch.is_grad_enabled():
            return checkpoint(self._forward_impl, x, hidden, use_reentrant=False)
        return self._forward_impl(x, hidden)


class DynamicRouterLM(nn.Module):
    """A block language model whose active modules form a dynamic complete graph.

    There are no stored edges. At every route decision, the current block query
    scores every active module embedding. The visited mask provides a per-sample
    acyclic execution path, while ``active`` controls capacity growth.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        max_modules: int = 1024,
        block_size: int = 8,
        n_heads: int = 4,
        max_depth: int = 8,
    ) -> None:
        super().__init__()
        if min(d_model, vocab_size, max_modules, block_size, n_heads, max_depth) <= 0:
            raise ValueError("model dimensions must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if max_modules < 3:
            raise ValueError("max_modules must include input, middle, and output slots")
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.max_modules = max_modules
        self.max_depth = max_depth
        self.use_batched_group_execution = True
        self.input_id, self.output_id = 0, 2
        self.pool = nn.ModuleList(
            MemoryModule(d_model, n_heads) for _ in range(max_modules)
        )
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(block_size, d_model))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.router = nn.Parameter(torch.randn(max_modules, d_model) * 0.02)
        self.head = nn.Linear(d_model, vocab_size)
        self.register_buffer("active", torch.zeros(max_modules, dtype=torch.bool))
        self.register_buffer("exploration_bias", torch.zeros(max_modules))
        self.register_buffer("bias_age", torch.zeros(max_modules, dtype=torch.long))
        self.cooldown_source: int | None = None
        self.cooldown_slot: int | None = None
        with torch.no_grad():
            self.active[[self.input_id, 1, self.output_id]] = True
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def active_count(self) -> int:
        return int(self.active.sum().item())

    def activate_next(self, initial_bias: float = 2.0) -> int | None:
        free = torch.where(~self.active)[0]
        if not len(free):
            return None
        slot = int(free[0])
        with torch.no_grad():
            self.pool[slot].reset_parameters()
            self.active[slot] = True
            self.router[slot].zero_()
            self.exploration_bias[slot] = initial_bias
            self.bias_age[slot] = 0
        return slot

    def set_cooldown_connection(self, source_id: int, slot: int) -> None:
        self.cooldown_source = source_id
        self.cooldown_slot = slot

    def clear_cooldown_connection(self) -> None:
        self.cooldown_source = None
        self.cooldown_slot = None

    def decay_exploration(self, amount: float = 0.99) -> None:
        with torch.no_grad():
            self.exploration_bias.mul_(amount)
            self.exploration_bias.masked_fill_(self.exploration_bias.abs() < 1e-3, 0)
            self.bias_age[self.exploration_bias != 0] += 1

    def _prepare_input(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            if x.shape[1] != self.block_size or x.dtype not in (
                torch.int64, torch.int32, torch.int16, torch.uint8
            ):
                raise ValueError(f"expected integer token IDs [batch, {self.block_size}]")
            if torch.any((x < 0) | (x >= self.vocab_size)):
                raise ValueError("token ID is outside the configured vocabulary")
            return self.embedding(x.long())
        if x.ndim == 3 and tuple(x.shape[1:]) == (self.block_size, self.d_model):
            return x
        raise ValueError(
            f"expected token IDs [batch, {self.block_size}] or hidden states "
            f"[batch, {self.block_size}, {self.d_model}]"
        )

    def _route_logits(self, q: Tensor, source_id: int, visited: Tensor) -> Tensor:
        query = q.mean(dim=1)
        logits = query @ self.router.T + self.exploration_bias
        forbidden = ~self.active.clone()
        forbidden[self.input_id] = True
        forbidden |= visited
        forbidden[self.output_id] = False
        return logits.masked_fill(forbidden.unsqueeze(0), torch.finfo(logits.dtype).min)

    def _route_logits_batch(self, q: Tensor, visited: Tensor) -> Tensor:
        """Compute route logits for all live branches at one depth."""
        logits = q.mean(dim=1) @ self.router.T + self.exploration_bias
        forbidden = (~self.active).unsqueeze(0).expand(q.shape[0], -1).clone()
        forbidden[:, self.input_id] = True
        forbidden |= visited
        forbidden[:, self.output_id] = False
        return logits.masked_fill(forbidden, torch.finfo(logits.dtype).min)

    def _execute_target_groups_batch(
        self,
        target: int,
        groups: list[list[tuple[Tensor, Tensor]]],
    ) -> list[tuple[Tensor, Tensor]]:
        """Batch independent visited-history groups for one target module."""
        q_batch = torch.cat(
            [torch.stack([item[0] for item in group]).mean(dim=0) for group in groups],
            dim=0,
        )
        hidden_batch = torch.cat(
            [
                torch.stack(
                    [
                        item[1].unsqueeze(0) if item[1].ndim == 2 else item[1]
                        for item in group
                    ]
                ).mean(dim=0)
                for group in groups
            ],
            dim=1,
        )
        output_batch, next_hidden_batch = self.pool[target](q_batch, hidden_batch)
        return [
            (output_batch[index : index + 1], next_hidden_batch[:, index : index + 1])
            for index in range(len(groups))
        ]

    def _execute_target_groups_loop(
        self,
        target: int,
        groups: list[list[tuple[Tensor, Tensor]]],
    ) -> list[tuple[Tensor, Tensor]]:
        """Reference implementation for grouped module execution."""
        results = []
        for group in groups:
            q = torch.stack([item[0] for item in group]).mean(dim=0)
            hidden = torch.stack([
                item[1].unsqueeze(0) if item[1].ndim == 2 else item[1]
                for item in group
            ]).mean(dim=0)
            results.append(self.pool[target](q, hidden))
        return results

    def forward(
        self,
        x: Tensor,
        tau: float = 1.0,
        memory_state: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        if tau <= 0:
            raise ValueError("tau must be positive")
        x = self._prepare_input(x)
        x = x + self.position_embedding.unsqueeze(0)
        batch_size = x.shape[0]
        expected_shape = (batch_size, self.max_modules, self.d_model)
        if memory_state is None:
            memory_state = x.new_zeros(expected_shape)
        elif tuple(memory_state.shape) != expected_shape:
            raise ValueError(f"memory_state must have shape {expected_shape}")

        outputs = torch.zeros(
            (batch_size, self.block_size, self.vocab_size), device=x.device, dtype=x.dtype
        )
        memories = [memory_state[:, module_id, :].clone() for module_id in range(self.max_modules)]
        records: list[RouteRecord] = []
        route_targets: list[tuple[int, list[int], int, int]] = []
        branch_paths: list[tuple[int, tuple[int, ...]]] = []
        route_lengths: list[int] = []
        valid_samples = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        expected_lengths = x.new_full((batch_size,), 2.0)
        terminations = ["output"] * batch_size
        invalid_reasons: list[str | None] = [None] * batch_size
        forward_diagnostics = {
            "max_query_abs": 0.0,
            "max_route_logits_abs": 0.0,
            "max_candidate_output_abs": 0.0,
            "max_memory_abs": 0.0,
            "max_route_depth": 0,
            "max_frontier_count": 0,
            "total_frontier_states": 0,
            "nonfinite_query": False,
            "nonfinite_route_logits": False,
            "nonfinite_candidate_output": False,
            "nonfinite_memory": False,
        }

        def observe_tensor(name: str, value: Tensor) -> None:
            finite = torch.isfinite(value)
            all_finite = bool(finite.all())
            forward_diagnostics[f"nonfinite_{name}"] |= not all_finite
            if all_finite:
                forward_diagnostics[f"max_{name}_abs"] = max(
                    forward_diagnostics[f"max_{name}_abs"],
                    float(value.detach().abs().amax()),
                )

        # Input is the only externally-entered module. All samples are processed together.
        q, next_hidden = self.pool[self.input_id](x, memories[self.input_id].unsqueeze(0))
        memories[self.input_id] = next_hidden.squeeze(0)
        live_ids = torch.arange(batch_size, device=x.device)
        frontier_q = q
        frontier_hidden = memories[self.input_id]
        frontier_modules = torch.full((batch_size,), self.input_id, device=x.device, dtype=torch.long)
        frontier_visited = torch.zeros((batch_size, self.max_modules), device=x.device, dtype=torch.bool)
        frontier_visited[:, self.input_id] = True
        frontier_lengths = torch.zeros(batch_size, device=x.device, dtype=torch.long)
        frontier_paths: list[tuple[int, ...]] = [(self.input_id,) for _ in range(batch_size)]
        output_q_parts: list[Tensor] = []
        output_ids: list[Tensor] = []
        for depth in range(self.max_depth):
            forward_diagnostics["max_route_depth"] = max(forward_diagnostics["max_route_depth"], depth)
            if frontier_q.shape[0] == 0:
                break
            forward_diagnostics["max_frontier_count"] = max(
                forward_diagnostics["max_frontier_count"], frontier_q.shape[0]
            )
            forward_diagnostics["total_frontier_states"] += frontier_q.shape[0]
            observe_tensor("query", frontier_q)
            logits_batch = self._route_logits_batch(frontier_q, frontier_visited)
            if not torch.isfinite(logits_batch).all():
                invalid = ~torch.isfinite(logits_batch).all(dim=1)
                for sample_id in live_ids[invalid].tolist():
                    invalid_reasons[sample_id] = "nonfinite_logits"
                    terminations[sample_id] = "nonfinite_logits"
                break
            router_probs = F.softmax(logits_batch.float() / tau, dim=-1)
            targets = router_probs.argmax(dim=-1)
            for row, target in enumerate(targets.tolist()):
                source_id = int(frontier_modules[row])
                forced = self.cooldown_slot if (
                    source_id == self.cooldown_source
                    and self.cooldown_slot is not None
                    and not bool(frontier_visited[row, self.cooldown_slot])
                ) else None
                if forced is not None:
                    targets[row] = forced
            selected_probs = router_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            routing_weights = selected_probs
            next_groups: dict[int, list[int]] = defaultdict(list)
            depth_records: list[RouteRecord] = []
            for row, target in enumerate(targets.tolist()):
                source_id = int(frontier_modules[row])
                next_groups[target].append(row)
                record_logits = logits_batch[row : row + 1]
                record = RouteRecord(source_id, record_logits, int(live_ids[row]), depth, int(frontier_lengths[row]))
                depth_records.append(record)
                records.append(record)
                route_targets.append((source_id, [target], int(live_ids[row]), depth))
                observe_tensor("route_logits", record_logits)
                path = (*frontier_paths[row], target)
                length = int(frontier_lengths[row]) + 1
                if target == self.output_id:
                    output_q_parts.append(frontier_q[row : row + 1] * routing_weights[row].view(1, 1, 1))
                    output_ids.append(live_ids[row : row + 1])
                    sample_id = int(live_ids[row])
                    valid_samples[sample_id] = True
                    route_lengths.append(length)
                    branch_paths.append((sample_id, path))
                else:
                    frontier_paths[row] = path
            if logits_batch.requires_grad:
                def capture_gradient(gradient: Tensor, depth_records: list[RouteRecord] = depth_records) -> Tensor:
                    for row, record in enumerate(depth_records):
                        row_gradient = gradient[row].detach()
                        record.gradient = row_gradient
                        record.gradient_norm = float(row_gradient.float().norm().cpu())
                    return gradient
                logits_batch.register_hook(capture_gradient)
            next_q_parts: list[Tensor] = []
            next_h_parts: list[Tensor] = []
            next_modules: list[int] = []
            next_visited_parts: list[Tensor] = []
            next_lengths: list[int] = []
            next_ids: list[int] = []
            for target, rows in next_groups.items():
                rows_tensor = torch.tensor(rows, device=x.device, dtype=torch.long)
                non_output = target != self.output_id
                if not non_output:
                    continue
                target_q, target_h = self.pool[target](
                    torch.cat([frontier_q[row : row + 1] * routing_weights[row].view(1, 1, 1) for row in rows], dim=0),
                    frontier_hidden[rows_tensor].unsqueeze(0),
                )
                observe_tensor("candidate_output", target_q)
                observe_tensor("memory", target_h)
                memories[target] = memories[target].index_copy(0, live_ids[rows_tensor], target_h.squeeze(0))
                for index, row in enumerate(rows):
                    next_q_parts.append(target_q[index : index + 1])
                    next_h_parts.append(target_h[:, index, :])
                    next_modules.append(target)
                    visited = frontier_visited[row].clone()
                    visited[target] = True
                    next_visited_parts.append(visited)
                    next_lengths.append(int(frontier_lengths[row]) + 1)
                    next_ids.append(int(live_ids[row]))
            frontier_q = torch.cat(next_q_parts, dim=0) if next_q_parts else frontier_q[:0]
            frontier_hidden = torch.cat(next_h_parts, dim=0) if next_h_parts else frontier_hidden[:0]
            frontier_modules = torch.tensor(next_modules, device=x.device, dtype=torch.long)
            frontier_visited = torch.stack(next_visited_parts) if next_visited_parts else frontier_visited[:0]
            frontier_lengths = torch.tensor(next_lengths, device=x.device, dtype=torch.long)
            live_ids = torch.tensor(next_ids, device=x.device, dtype=torch.long)
            frontier_paths = [frontier_paths[row] for target, rows in next_groups.items() if target != self.output_id for row in rows]

        if output_q_parts:
            output_ids_tensor = torch.cat(output_ids)
            output_input = torch.cat(output_q_parts, dim=0)
            output_hidden = memories[self.output_id][output_ids_tensor].unsqueeze(0)
            output_q, output_hidden_next = self.pool[self.output_id](output_input, output_hidden)
            memories[self.output_id] = memories[self.output_id].index_copy(
                0, output_ids_tensor, output_hidden_next.squeeze(0)
            )
            outputs.index_copy_(0, output_ids_tensor, self.head(output_q).to(outputs.dtype))
            observe_tensor("candidate_output", output_q)
            observe_tensor("memory", output_hidden_next)
        route_length_by_sample = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        for sample_id, length in zip([item[0] for item in branch_paths], route_lengths):
            route_length_by_sample[sample_id] = length
        route_lengths = route_length_by_sample.tolist()
        for sample_id in range(batch_size):
            if not valid_samples[sample_id]:
                invalid_reasons[sample_id] = invalid_reasons[sample_id] or "no_valid_path"
        next_memories = torch.stack(memories, dim=1)
        observe_tensor("memory", next_memories)

        return outputs, {
            "route_records": records,
            "route_targets": route_targets,
            "branch_paths": branch_paths,
            "expected_length_per_sample": expected_lengths,
            "expected_length": expected_lengths.mean(),
            "route_lengths": torch.tensor(route_lengths, device=x.device),
            "valid_sample_mask": valid_samples,
            "terminations": terminations,
            "invalid_reasons": invalid_reasons,
            "memory_state": next_memories,
            "forward_diagnostics": forward_diagnostics,
        }


def language_model_loss(
    logits: Tensor,
    targets: Tensor,
    expected_length: Tensor,
    route_lengths: Tensor,
    length_weight: float = 1e-3,
    target_mask: Tensor | None = None,
    sample_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    if logits.shape[:2] != targets.shape:
        raise ValueError("targets must have shape [batch, block_size]")
    if route_lengths.shape != (targets.shape[0],):
        raise ValueError("route_lengths must have shape [batch]")
    if target_mask is None:
        lm = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    else:
        if target_mask.shape != targets.shape or target_mask.dtype != torch.bool:
            raise ValueError("target_mask must be bool with shape [batch, block_size]")
        target_mask = target_mask.to(logits.device)
        flat_mask = target_mask.flatten()
        if not bool(flat_mask.any()):
            raise ValueError("target_mask must contain at least one valid target")
        lm = F.cross_entropy(logits.flatten(0, 1)[flat_mask], targets.flatten()[flat_mask])
    if sample_mask is not None:
        if sample_mask.shape != (targets.shape[0],) or sample_mask.dtype != torch.bool:
            raise ValueError("sample_mask must be bool with shape [batch]")
        sample_mask = sample_mask.to(expected_length.device)
        if not bool(sample_mask.any()):
            raise ValueError("sample_mask must contain at least one active sample")
        if expected_length.ndim == 1:
            expected_length = expected_length[sample_mask].mean()
        route_lengths = route_lengths[sample_mask]
    elif expected_length.ndim == 1:
        expected_length = expected_length.mean()
    excess_lengths = (route_lengths.float() - expected_length).clamp_min(0.0)
    length_loss = length_weight * excess_lengths.mean()
    total = lm + length_loss
    return total, {
        "loss": total.detach(),
        "lm_loss": lm.detach(),
        "length_loss": length_loss.detach(),
        "expected_length": expected_length.detach(),
    }


def _reset_module(module: nn.Module) -> None:
    reset = getattr(module, "reset_parameters", None)
    if reset is not None:
        reset()
