"""Shared-memory recurrent Transformer language model.

The model has one globally shared memory with 24 addressable blocks.  Each of
the 24 reasoner cycles owns a router and two RMSNorms, but all cycles access the
same FFN/Q/KV/O banks.  Routing is top-3 per head and K/V deliberately share
the same route head and weights.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


QWEN25_TOKENIZER = "Qwen/Qwen2.5-0.5B"


@dataclass
class MemoryTransformerConfig:
    vocab_size: int = 151665
    hidden_size: int = 1536
    intermediate_size: int = 4096
    num_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_memory_blocks: int = 24
    top_k: int = 3
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = False
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_memory_blocks": self.num_memory_blocks,
            "max_position_embeddings": self.max_position_embeddings,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"configuration values must be positive: {invalid}")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_layers != self.num_memory_blocks:
            raise ValueError(
                "num_layers must equal num_memory_blocks: each memory block has "
                "one corresponding recurrent reasoner cycle"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if not 1 <= self.top_k <= self.num_memory_blocks:
            raise ValueError("top_k must be in [1, num_memory_blocks]")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("normalization epsilon and RoPE theta must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def parameter_count(self) -> int:
        embedding = self.vocab_size * self.hidden_size * (1 if self.tie_word_embeddings else 2)
        ffn = self.num_memory_blocks * 3 * self.hidden_size * self.intermediate_size
        attention = self.num_memory_blocks * (
            2 * self.hidden_size * self.hidden_size
            + 2 * self.hidden_size * self.kv_size
        )
        per_cycle = 2 * self.hidden_size + 4 * self.num_memory_blocks * self.hidden_size
        final_norm = self.hidden_size
        return embedding + ffn + attention + self.num_layers * per_cycle + final_norm


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, config: MemoryTransformerConfig) -> None:
        super().__init__()
        inv = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim)
        )
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        positions = torch.arange(q.shape[-2], device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        cos = freqs.cos()[None, None, :, :].to(q.dtype)
        sin = freqs.sin()[None, None, :, :].to(q.dtype)

        def rotate(x: Tensor) -> Tensor:
            even, odd = x[..., ::2], x[..., 1::2]
            return torch.stack(
                (even * cos - odd * sin, even * sin + odd * cos), dim=-1
            ).flatten(-2)

        return rotate(q), rotate(k)


class SharedMemory(nn.Module):
    """The globally shared FFN/Q/KV/O memory banks."""

    def __init__(self, config: MemoryTransformerConfig) -> None:
        super().__init__()
        n, h, m, kv = (
            config.num_memory_blocks,
            config.hidden_size,
            config.intermediate_size,
            config.kv_size,
        )
        # Banks use [input, output] per block, matching F.linear after transpose.
        self.ffn_gate = nn.Parameter(torch.empty(n, h, m))
        self.ffn_up = nn.Parameter(torch.empty(n, h, m))
        self.ffn_down = nn.Parameter(torch.empty(n, m, h))
        self.q_proj = nn.Parameter(torch.empty(n, h, h))
        self.k_proj = nn.Parameter(torch.empty(n, h, kv))
        self.v_proj = nn.Parameter(torch.empty(n, h, kv))
        self.o_proj = nn.Parameter(torch.empty(n, h, h))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize each addressable matrix separately. Initializing the whole
        # 3-D tensor makes PyTorch treat the block dimension as a kernel axis.
        for bank in (
            self.ffn_gate,
            self.ffn_up,
            self.ffn_down,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
        ):
            for block in bank:
                nn.init.xavier_uniform_(block)

    @staticmethod
    def _locations(indices: Tensor, block_id: int) -> Tensor:
        # Flatten all leading token dimensions. The second location column is
        # the top-k slot and lets repeated block IDs accumulate correctly.
        return (indices.reshape(-1, indices.shape[-1]) == block_id).nonzero(
            as_tuple=False
        )

    @staticmethod
    def routed_linear(
        x: Tensor, bank: Tensor, indices: Tensor, weights: Tensor
    ) -> Tensor:
        """Apply selected matrices without materializing [tokens, top_k, I, O]."""
        x_flat = x.reshape(-1, x.shape[-1])
        weight_flat = weights.reshape(-1, weights.shape[-1])
        output = x_flat.new_zeros((x_flat.shape[0], bank.shape[-1]))
        for block_id in range(bank.shape[0]):
            locations = SharedMemory._locations(indices, block_id)
            if locations.numel() == 0:
                continue
            token_ids, slot_ids = locations.unbind(dim=1)
            block_input = x_flat.index_select(0, token_ids)
            block_output = F.linear(block_input, bank[block_id].transpose(0, 1))
            block_output = block_output * weight_flat[token_ids, slot_ids].to(
                block_output.dtype
            ).unsqueeze(-1)
            scatter_ids = token_ids.unsqueeze(-1).expand(-1, output.shape[-1])
            output = output.scatter_add(0, scatter_ids, block_output)
        return output.reshape(*x.shape[:-1], bank.shape[-1])

    @staticmethod
    def routed_ffn(
        x: Tensor,
        gate_bank: Tensor,
        up_bank: Tensor,
        down_bank: Tensor,
        indices: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """Dispatch complete FFN experts so gate/up/down stay block-aligned."""
        x_flat = x.reshape(-1, x.shape[-1])
        weight_flat = weights.reshape(-1, weights.shape[-1])
        output = x_flat.new_zeros((x_flat.shape[0], down_bank.shape[-1]))
        for block_id in range(gate_bank.shape[0]):
            locations = SharedMemory._locations(indices, block_id)
            if locations.numel() == 0:
                continue
            token_ids, slot_ids = locations.unbind(dim=1)
            block_input = x_flat.index_select(0, token_ids)
            gate = F.linear(block_input, gate_bank[block_id].transpose(0, 1))
            up = F.linear(block_input, up_bank[block_id].transpose(0, 1))
            hidden = F.silu(gate) * up
            block_output = F.linear(hidden, down_bank[block_id].transpose(0, 1))
            block_output = block_output * weight_flat[token_ids, slot_ids].to(
                block_output.dtype
            ).unsqueeze(-1)
            scatter_ids = token_ids.unsqueeze(-1).expand(-1, output.shape[-1])
            output = output.scatter_add(0, scatter_ids, block_output)
        return output.reshape(*x.shape[:-1], down_bank.shape[-1])

    def ffn(self, x: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
        return self.routed_ffn(
            x,
            self.ffn_gate,
            self.ffn_up,
            self.ffn_down,
            indices,
            weights,
        )

    def forward(
        self, operation: str, x: Tensor, indices: Tensor, weights: Tensor
    ) -> Tensor:
        """Run a bank operation through the module boundary used by FSDP."""
        if operation == "ffn":
            return self.ffn(x, indices, weights)
        bank = getattr(self, f"{operation}_proj", None)
        if bank is None or operation not in {"q", "k", "v", "o"}:
            raise ValueError(f"unknown shared-memory operation: {operation}")
        return self.routed_linear(x, bank, indices, weights)


class MemoryReasoner(nn.Module):
    """One recurrent reasoner cycle; memory is passed explicitly for FSDP safety."""

    def __init__(self, config: MemoryTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.router = nn.Linear(
            config.hidden_size, 4 * config.num_memory_blocks, bias=False
        )
        self.rotary = RotaryEmbedding(config)
        self.dropout = nn.Dropout(config.dropout)

    def route(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.router(x).view(
            *x.shape[:2], 4, self.config.num_memory_blocks
        )
        probabilities = F.softmax(logits.float(), dim=-1)
        values, indices = logits.topk(self.config.top_k, dim=-1)
        weights = F.softmax(values.float(), dim=-1).to(x.dtype)
        return indices, weights, probabilities

    def forward(
        self,
        x: Tensor,
        memory: SharedMemory,
        attention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        normalized = self.norm1(x)
        indices, weights, probabilities = self.route(normalized)
        q = memory("q", normalized, indices[..., 1, :], weights[..., 1, :])
        k = memory("k", normalized, indices[..., 2, :], weights[..., 2, :])
        v = memory("v", normalized, indices[..., 2, :], weights[..., 2, :])
        q = q.view(x.shape[0], x.shape[1], self.config.num_attention_heads, -1)
        k = k.view(x.shape[0], x.shape[1], self.config.num_key_value_heads, -1)
        v = v.view(x.shape[0], x.shape[1], self.config.num_key_value_heads, -1)
        q, k = self.rotary(q.transpose(1, 2), k.transpose(1, 2))
        v = v.transpose(1, 2)
        k = k.repeat_interleave(
            self.config.num_attention_heads // self.config.num_key_value_heads, dim=1
        )
        v = v.repeat_interleave(
            self.config.num_attention_heads // self.config.num_key_value_heads, dim=1
        )
        causal = torch.tril(
            torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool)
        )
        allowed = causal[None, None, :, :]
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, :].bool()
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape_as(x)
        attn = memory("o", attn, indices[..., 3, :], weights[..., 3, :])
        x = x + self.dropout(attn)
        x = x + self.dropout(
            memory("ffn", self.norm2(x), indices[..., 0, :], weights[..., 0, :])
        )
        return x, indices, weights, probabilities


class MemoryTransformerForCausalLM(nn.Module):
    def __init__(self, config: Optional[MemoryTransformerConfig] = None) -> None:
        super().__init__()
        self.config = config or MemoryTransformerConfig()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.memory = SharedMemory(self.config)
        self.layers = nn.ModuleList(
            MemoryReasoner(self.config) for _ in range(self.config.num_layers)
        )
        self.norm = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.apply(self._init)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear) and module.weight is not None:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    @staticmethod
    def load_balance_loss(
        indices: Tensor,
        probabilities: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Switch-style importance/load loss over actual memory block IDs."""
        blocks = probabilities.shape[-1]
        probability_rows = probabilities.reshape(-1, probabilities.shape[-2], blocks)
        index_rows = indices.reshape(-1, indices.shape[-2], indices.shape[-1])
        if attention_mask is not None:
            valid = attention_mask.reshape(-1).bool()
            probability_rows = probability_rows[valid]
            index_rows = index_rows[valid]
        if probability_rows.shape[0] == 0:
            return probabilities.sum() * 0.0
        importance = probability_rows.float().mean(dim=0)
        assignment = F.one_hot(index_rows, num_classes=blocks).float()
        load = assignment.mean(dim=(0, 2))
        return blocks * (importance * load).sum(dim=-1).mean()

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        return_router_stats: bool = False,
    ) -> dict[str, object]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds "
                f"max_position_embeddings={self.config.max_position_embeddings}"
            )
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            attention_mask = attention_mask.bool()
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")

        x = self.embed_tokens(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        route_losses = []
        route_indices = []
        route_weights = []
        route_probabilities = []
        for layer in self.layers:
            x, indices, weights, probabilities = layer(
                x, self.memory, attention_mask=attention_mask
            )
            if attention_mask is not None:
                x = x * attention_mask.unsqueeze(-1).to(x.dtype)
            route_losses.append(
                self.load_balance_loss(indices, probabilities, attention_mask)
            )
            if return_router_stats:
                route_indices.append(indices)
                route_weights.append(weights)
                route_probabilities.append(probabilities)
        logits = self.lm_head(self.norm(x))
        result: dict[str, object] = {
            "logits": logits,
            "router_loss": torch.stack(route_losses).mean(),
        }
        if labels is not None:
            shifted_logits = logits[:, :-1].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            if attention_mask is not None:
                shifted_labels = shifted_labels.masked_fill(
                    ~attention_mask[:, 1:], -100
                )
            result["loss"] = F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.size(-1)),
                shifted_labels.reshape(-1),
                ignore_index=-100,
            )
        if return_router_stats:
            result["route_indices"] = route_indices
            result["route_weights"] = route_weights
            result["route_probabilities"] = route_probabilities
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            logits = self(input_ids)["logits"][:, -1]
            next_token = logits.argmax(-1, keepdim=True)
            if eos_token_id is not None:
                next_token = torch.where(
                    finished[:, None],
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )
                finished |= next_token[:, 0].eq(eos_token_id)
            input_ids = torch.cat((input_ids, next_token), dim=-1)
            if eos_token_id is not None and bool(finished.all()):
                break
        return input_ids
