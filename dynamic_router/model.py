"""Block-level language model with a fixed-capacity, fully dynamic router."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class RouteRecord:
    source_id: int
    logits: Tensor


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

    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or hidden.ndim != 3:
            raise ValueError("module input and hidden state must be rank-3 tensors")
        if hidden.shape[0] != 1 or hidden.shape[1] != x.shape[0]:
            raise ValueError("hidden state must have shape [1, batch, d_model]")
        gru_input = self.gru_norm(x)
        attention_input = self.attn_norm(x)
        query = self.q_proj(attention_input)
        gru_output, next_hidden = self.gru(gru_input, hidden)
        keys = self.k_proj(gru_output)
        values = self.v_proj(gru_output)
        attended, _ = self.attn(query, keys, values, need_weights=False)
        attention_hidden = x + attended
        output = attention_hidden + self.ffn(self.ffn_norm(attention_hidden))
        return output, next_hidden


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
        forbidden[source_id] = True
        forbidden |= visited
        forbidden[self.output_id] = False
        return logits.masked_fill(forbidden.unsqueeze(0), torch.finfo(logits.dtype).min)

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

        outputs: list[Tensor] = []
        next_memories: list[Tensor] = []
        records: list[RouteRecord] = []
        route_lengths: list[int] = []
        expected_lengths: list[Tensor] = []
        terminations: list[str] = []
        invalid_reasons: list[str | None] = []

        for sample_index, sample in enumerate(x.unbind(0)):
            q = sample.unsqueeze(0)
            hidden_states = [memory_state[sample_index, i : i + 1].clone() for i in range(self.max_modules)]
            current = self.input_id
            visited = torch.zeros(self.max_modules, dtype=torch.bool, device=x.device)
            survival = q.new_ones(1)
            expected_length = q.new_zeros(1)
            route_length = 0
            termination = "output"
            invalid_reason = None
            output_processed = False

            # Input is the only externally-entered module.
            q, next_hidden = self.pool[current](q, hidden_states[current].unsqueeze(0))
            hidden_states[current] = next_hidden.squeeze(0)
            visited[current] = True
            for _ in range(self.max_depth):
                logits = self._route_logits(q, current, visited)
                if logits.requires_grad:
                    logits.retain_grad()
                records.append(RouteRecord(current, logits))
                if not torch.isfinite(logits).all():
                    termination = "nonfinite_logits"
                    invalid_reason = termination
                    break
                route_logits = logits.float()
                probs = (route_logits / tau).softmax(dim=-1)
                expected_length = expected_length + survival
                survival = survival * (1 - probs[:, self.output_id])

                legal = torch.where(
                    logits[0] > torch.finfo(logits.dtype).min / 2
                )[0].tolist()
                if not legal:
                    termination = "no_legal_candidate"
                    invalid_reason = termination
                    break
                if self.training:
                    route = F.gumbel_softmax(
                        route_logits, tau=tau, hard=True, dim=-1
                    )
                    choice = legal[int(route[0, legal].argmax().item())]
                else:
                    route = F.one_hot(
                        logits.argmax(dim=-1), num_classes=self.max_modules
                    ).to(logits.dtype)
                    choice = int(logits[0, legal].argmax().item())
                    choice = legal[choice]
                if bool(visited[choice]) or choice not in legal:
                    termination = "cycle"
                    invalid_reason = termination
                    break

                # Candidate computation supplies the straight-through gradient.
                # Only the selected candidate's memory is committed.
                candidate_outputs: dict[int, Tensor] = {}
                candidate_memories: dict[int, Tensor] = {}
                candidate_targets = legal if self.training else [choice]
                for target in candidate_targets:
                    candidate_outputs[target], candidate_memories[target] = self.pool[target](
                        q, hidden_states[target].clone().unsqueeze(0)
                    )
                if not self.training:
                    hidden_states[choice] = candidate_memories[choice].squeeze(0)
                    q = candidate_outputs[choice]
                    current = choice
                    route_length += 1
                    visited[current] = True
                    if current == self.output_id:
                        output_processed = True
                        break
                    continue
                soft_q = sum(
                    candidate_outputs[target] * route[:, target].view(1, 1)
                    for target in legal
                )
                if self.training:
                    hard_q = sum(
                        candidate_outputs[target]
                        * route[:, target].detach().view(1, 1)
                        for target in legal
                    )
                    q = hard_q + (soft_q - soft_q.detach())
                hidden_states[choice] = candidate_memories[choice].squeeze(0)
                current = choice
                route_length += 1
                visited[current] = True
                if current == self.output_id:
                    output_processed = True
                    break
            else:
                termination = "max_depth"
                current = self.output_id

            if not output_processed and invalid_reason is None:
                if termination == "output":
                    termination = "no_legal_candidate"
                    invalid_reason = termination
            if not output_processed and invalid_reason is None:
                q, next_hidden = self.pool[self.output_id](
                    q, hidden_states[self.output_id].unsqueeze(0)
                )
                hidden_states[self.output_id] = next_hidden.squeeze(0)
            outputs.append(self.head(q))
            next_memories.append(torch.cat(hidden_states, dim=0))
            route_lengths.append(route_length)
            expected_lengths.append(expected_length.squeeze(0))
            terminations.append(termination)
            invalid_reasons.append(invalid_reason)

        return torch.cat(outputs, dim=0), {
            "route_records": records,
            "expected_length_per_sample": torch.stack(expected_lengths),
            "expected_length": torch.stack(expected_lengths).mean(),
            "route_lengths": torch.tensor(route_lengths, device=x.device),
            "terminations": terminations,
            "invalid_reasons": invalid_reasons,
            "memory_state": torch.stack(next_memories),
        }


def language_model_loss(
    logits: Tensor,
    targets: Tensor,
    expected_length: Tensor,
    length_weight: float = 1e-3,
    target_mask: Tensor | None = None,
    sample_mask: Tensor | None = None,
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
    total = lm + length_weight * expected_length
    return total, {
        "loss": total.detach(),
        "lm_loss": lm.detach(),
        "length_loss": (length_weight * expected_length).detach(),
        "expected_length": expected_length.detach(),
    }


def _reset_module(module: nn.Module) -> None:
    reset = getattr(module, "reset_parameters", None)
    if reset is not None:
        reset()
