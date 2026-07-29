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
        top_k: int = 3,
    ) -> tuple[Tensor, dict[str, Any]]:
        if tau <= 0:
            raise ValueError("tau must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        x = self._prepare_input(x)
        x = x + self.position_embedding.unsqueeze(0)
        batch_size = x.shape[0]
        expected_shape = (batch_size, self.max_modules, self.d_model)
        if memory_state is None:
            memory_state = x.new_zeros(expected_shape)
        elif tuple(memory_state.shape) != expected_shape:
            raise ValueError(f"memory_state must have shape {expected_shape}")

        outputs: list[Tensor] = []
        next_memories: list[Tensor] = []
        records: list[RouteRecord] = []
        route_targets: list[tuple[int, list[int], int, int]] = []
        branch_paths: list[tuple[int, tuple[int, ...]]] = []
        route_lengths: list[int] = []
        valid_samples: list[bool] = []
        expected_lengths: list[Tensor] = []
        terminations: list[str] = []
        invalid_reasons: list[str | None] = []
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

        for sample_index, sample in enumerate(x.unbind(0)):
            q = sample.unsqueeze(0)
            hidden_states = [memory_state[sample_index, i : i + 1].clone() for i in range(self.max_modules)]
            expected_length = q.new_zeros(1)
            output_inputs: list[Tensor] = []
            route_length = 0
            termination = "output"
            invalid_reason = None
            completed_lengths: list[int] = []

            # Input is the only externally-entered module.
            q, next_hidden = self.pool[self.input_id](q, hidden_states[self.input_id].unsqueeze(0))
            hidden_states[self.input_id] = next_hidden.squeeze(0)
            frontier_q = q
            frontier_hidden = hidden_states[self.input_id]
            frontier_modules = torch.tensor([self.input_id], device=x.device, dtype=torch.long)
            frontier_visited = torch.zeros(
                (1, self.max_modules), device=x.device, dtype=torch.bool
            )
            frontier_visited[0, self.input_id] = True
            frontier_lengths = torch.zeros(1, device=x.device, dtype=torch.long)
            frontier_paths: list[tuple[int, ...]] = [(self.input_id,)]
            for depth in range(self.max_depth):
                forward_diagnostics["max_route_depth"] = max(
                    forward_diagnostics["max_route_depth"], depth
                )
                if frontier_q.shape[0] == 0:
                    break
                forward_diagnostics["max_frontier_count"] = max(
                    forward_diagnostics["max_frontier_count"], frontier_q.shape[0]
                )
                forward_diagnostics["total_frontier_states"] += frontier_q.shape[0]
                observe_tensor("query", frontier_q)
                logits_batch = self._route_logits_batch(frontier_q, frontier_visited)
                if not torch.isfinite(logits_batch).all():
                    invalid_reason = "nonfinite_logits"
                    termination = invalid_reason
                    break
                next_groups: dict[int, list[tuple[int, Tensor, Tensor, tuple[int, ...], int]]] = defaultdict(list)
                next_q_parts: list[Tensor] = []
                next_h_parts: list[Tensor] = []
                next_modules: list[int] = []
                next_visited_parts: list[Tensor] = []
                next_lengths: list[int] = []
                for branch_index in range(frontier_q.shape[0]):
                    logits = logits_batch[branch_index : branch_index + 1]
                    source_id = int(frontier_modules[branch_index])
                    path_length = int(frontier_lengths[branch_index])
                    record = RouteRecord(source_id, logits, sample_index, depth, path_length)
                    if logits.requires_grad:
                        def capture_gradient(gradient: Tensor, record: RouteRecord = record) -> Tensor:
                            record.gradient_norm = float(gradient.detach().float().norm().cpu())
                            return gradient
                        logits.register_hook(capture_gradient)
                    records.append(record)
                    observe_tensor("route_logits", logits)
                    legal = torch.where(logits[0] > torch.finfo(logits.dtype).min / 2)[0].tolist()
                    if not legal:
                        invalid_reason = "no_legal_candidate"
                        termination = invalid_reason
                        continue
                    ranked = torch.topk(logits[0], k=min(top_k, len(legal))).indices.tolist()
                    ranked = [target for target in ranked if target in legal]
                    forced = self.cooldown_slot if source_id == self.cooldown_source else None
                    if forced is not None and forced in legal:
                        ranked = [forced] + [target for target in ranked if target != forced]
                    targets = ranked[:top_k]
                    route_targets.append((source_id, targets, sample_index, depth))
                    target_tensor = torch.tensor(targets, device=x.device, dtype=torch.long)
                    topk_probs = (logits[:, target_tensor].float() / tau).softmax(dim=-1)
                    for target_index, target in enumerate(targets):
                        gate = 1.0 + topk_probs[:, target_index] - topk_probs[:, target_index].detach()
                        gated_q = frontier_q[branch_index : branch_index + 1] * gate.view(1, 1, 1)
                        if target == self.output_id:
                            output_inputs.append(gated_q)
                            completed_length = path_length + 1
                            completed_lengths.append(completed_length)
                            route_length = max(route_length, completed_length)
                            branch_paths.append((sample_index, (*frontier_paths[branch_index], target)))
                        else:
                            next_visited = frontier_visited[branch_index].clone()
                            next_visited[target] = True
                            next_groups[target].append(
                                (branch_index, gated_q, frontier_hidden[branch_index : branch_index + 1],
                                 (*frontier_paths[branch_index], target), path_length + 1)
                            )
                if not next_groups:
                    break
                for target, group in next_groups.items():
                    q_mean = torch.cat([item[1] for item in group], dim=0).mean(dim=0, keepdim=True)
                    h_mean = torch.cat([item[2] for item in group], dim=0).mean(
                        dim=0, keepdim=True
                    ).unsqueeze(0)
                    target_q, target_h = self.pool[target](q_mean, h_mean)
                    observe_tensor("candidate_output", target_q)
                    observe_tensor("memory", target_h)
                    for _, _, _, path, length in group:
                        next_q_parts.append(target_q)
                        next_h_parts.append(target_h.squeeze(0))
                        next_modules.append(target)
                        next_visited = torch.zeros(self.max_modules, device=x.device, dtype=torch.bool)
                        next_visited[list(path)] = True
                        next_visited_parts.append(next_visited)
                        next_lengths.append(length)
                        frontier_paths_next = path
                frontier_q = torch.cat(next_q_parts, dim=0)
                frontier_hidden = torch.cat(next_h_parts, dim=0)
                frontier_modules = torch.tensor(next_modules, device=x.device, dtype=torch.long)
                frontier_visited = torch.stack(next_visited_parts)
                frontier_lengths = torch.tensor(next_lengths, device=x.device, dtype=torch.long)
                frontier_paths = [item[3] for group in next_groups.values() for item in group]
            if output_inputs:
                output_input = torch.stack(output_inputs).mean(dim=0)
                output_hidden = hidden_states[self.output_id]
                output_q, output_hidden_next = self.pool[self.output_id](
                    output_input, output_hidden.unsqueeze(0)
                )
                hidden_states[self.output_id] = output_hidden_next.squeeze(0)
                outputs.append(self.head(output_q))
                valid_samples.append(True)
            else:
                outputs.append(torch.zeros_like(self.head(q)))
                valid_samples.append(False)
                if invalid_reason is None:
                    invalid_reason = "no_valid_path"
            if completed_lengths:
                expected_length = q.new_tensor(completed_lengths, dtype=torch.float32).mean()
            next_memories.append(torch.cat(hidden_states, dim=0))
            observe_tensor("memory", next_memories[-1])
            route_lengths.append(route_length)
            expected_lengths.append(expected_length.squeeze(0))
            terminations.append(termination)
            invalid_reasons.append(invalid_reason)

        return torch.cat(outputs, dim=0), {
            "route_records": records,
            "route_targets": route_targets,
            "branch_paths": branch_paths,
            "expected_length_per_sample": torch.stack(expected_lengths),
            "expected_length": torch.stack(expected_lengths).mean(),
            "route_lengths": torch.tensor(route_lengths, device=x.device),
            "valid_sample_mask": torch.tensor(valid_samples, dtype=torch.bool, device=x.device),
            "terminations": terminations,
            "invalid_reasons": invalid_reasons,
            "memory_state": torch.stack(next_memories),
            "forward_diagnostics": forward_diagnostics,
        }


def language_model_loss(
    logits: Tensor,
    targets: Tensor,
    expected_length: Tensor,
    length_weight: float = 1e-3,
    target_mask: Tensor | None = None,
    sample_mask: Tensor | None = None,
    length_ratio: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    if logits.shape[:2] != targets.shape:
        raise ValueError("targets must have shape [batch, block_size]")
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
    elif expected_length.ndim == 1:
        expected_length = expected_length.mean()
    raw_length_loss = length_weight * expected_length
    if length_ratio is not None:
        if not 0 <= length_ratio < 1:
            raise ValueError("length_ratio must be in [0, 1)")
        length_loss = lm.detach() * length_ratio / max(1e-12, 1.0 - length_ratio)
    else:
        length_loss = raw_length_loss
    total = lm + length_loss
    return total, {
        "loss": total.detach(),
        "lm_loss": lm.detach(),
        "length_loss": length_loss.detach(),
        "raw_length_loss": raw_length_loss.detach(),
        "length_ratio": (length_loss / total).detach(),
        "expected_length": expected_length.detach(),
    }


def _reset_module(module: nn.Module) -> None:
    reset = getattr(module, "reset_parameters", None)
    if reset is not None:
        reset()
