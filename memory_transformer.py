"""A configurable research implementation of the requested memory Transformer.

The paper's confirmed idea is a globally shared FFN memory.  This implementation
extends that idea by putting Q, K/V and O projections in the same address space,
with four independent top-k routers (FFN, Q, KV and O).
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


QWEN25_TOKENIZER = "Qwen/Qwen2.5-0.5B"


@dataclass
class MemoryTransformerConfig:
    vocab_size: int = 151665  # len(AutoTokenizer.from_pretrained(QWEN25_TOKENIZER))
    hidden_size: int = 1536
    intermediate_size: int = 4096
    num_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_memory_blocks: int = 24
    top_k: int = 3
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    recurrent_steps: int = 1
    tie_word_embeddings: bool = False
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if not 1 <= self.top_k <= self.num_memory_blocks:
            raise ValueError("top_k must be in [1, num_memory_blocks]")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def parameter_count(self) -> int:
        # Includes untied input/output embeddings and the four memory banks.
        embedding = 2 * self.vocab_size * self.hidden_size
        ffn = self.num_memory_blocks * 3 * self.hidden_size * self.intermediate_size
        attention = self.num_memory_blocks * (
            2 * self.hidden_size * self.hidden_size
            + 2 * self.hidden_size * self.kv_size
        )
        reasoners = self.num_layers * (2 * self.hidden_size)  # norms and router
        return embedding + ffn + attention + reasoners


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, config: MemoryTransformerConfig) -> None:
        super().__init__()
        inv = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        positions = torch.arange(q.shape[-2], device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        cos = freqs.cos()[None, None, :, :].to(q.dtype)
        sin = freqs.sin()[None, None, :, :].to(q.dtype)

        def rotate(x: Tensor) -> Tensor:
            even, odd = x[..., ::2], x[..., 1::2]
            return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

        return rotate(q), rotate(k)


class SharedMemory(nn.Module):
    """Global knowledge memory; each bank has 24 addressable blocks."""

    def __init__(self, config: MemoryTransformerConfig) -> None:
        super().__init__()
        n, h, m, kv = (config.num_memory_blocks, config.hidden_size, config.intermediate_size, config.kv_size)
        self.ffn_gate = nn.Parameter(torch.empty(n, h, m))
        self.ffn_up = nn.Parameter(torch.empty(n, h, m))
        self.ffn_down = nn.Parameter(torch.empty(n, m, h))
        self.q_proj = nn.Parameter(torch.empty(n, h, h))
        self.k_proj = nn.Parameter(torch.empty(n, h, kv))
        self.v_proj = nn.Parameter(torch.empty(n, h, kv))
        self.o_proj = nn.Parameter(torch.empty(n, h, h))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Attention and output projections are linear operators and use Xavier.
        for p in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.xavier_uniform_(p)
        # The gated FFN projections feed the nonlinear SwiGLU activation.
        for p in (self.ffn_gate, self.ffn_up, self.ffn_down):
            nn.init.kaiming_uniform_(p, a=0.0, mode="fan_in", nonlinearity="relu")

    def routed_linear(self, x: Tensor, bank: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
        # x: [B,T,I], bank: [N,I,O], indices/weights: [B,T,K]
        selected = bank[indices]  # [B,T,K,I,O]
        result = torch.einsum("bti,btkio->btko", x, selected)
        return (result * weights.unsqueeze(-1)).sum(dim=-2)

    def ffn(self, x: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
        gate = self.routed_linear(x, self.ffn_gate, indices, weights)
        up = self.routed_linear(x, self.ffn_up, indices, weights)
        return self.routed_linear(F.silu(gate) * up, self.ffn_down, indices, weights)


class MemoryReasoner(nn.Module):
    def __init__(self, config: MemoryTransformerConfig, memory: SharedMemory) -> None:
        super().__init__()
        self.config = config
        self.memory = memory
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.router = nn.Linear(config.hidden_size, 4 * config.num_memory_blocks, bias=False)
        self.rotary = RotaryEmbedding(config)
        self.dropout = nn.Dropout(config.dropout)

    def route(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.router(x).view(*x.shape[:2], 4, self.config.num_memory_blocks)
        values, indices = logits.topk(self.config.top_k, dim=-1)
        weights = F.softmax(values.float(), dim=-1).to(x.dtype)
        return indices, weights

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        indices, weights = self.route(self.norm1(x))
        q = self.memory.routed_linear(self.norm1(x), self.memory.q_proj, indices[..., 1, :], weights[..., 1, :])
        k = self.memory.routed_linear(self.norm1(x), self.memory.k_proj, indices[..., 2, :], weights[..., 2, :])
        v = self.memory.routed_linear(self.norm1(x), self.memory.v_proj, indices[..., 2, :], weights[..., 2, :])
        q = q.view(x.shape[0], x.shape[1], self.config.num_attention_heads, -1).transpose(1, 2)
        k = k.view(x.shape[0], x.shape[1], self.config.num_key_value_heads, -1).transpose(1, 2)
        v = v.view(x.shape[0], x.shape[1], self.config.num_key_value_heads, -1).transpose(1, 2)
        q, k = self.rotary(q, k)
        k = k.repeat_interleave(self.config.num_attention_heads // self.config.num_key_value_heads, dim=1)
        v = v.repeat_interleave(self.config.num_attention_heads // self.config.num_key_value_heads, dim=1)
        causal = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool), diagonal=1)
        if attention_mask is not None:
            causal = causal | ~attention_mask[:, None, None, :].bool()
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=~causal, dropout_p=self.dropout.p if self.training else 0.0)
        attn = attn.transpose(1, 2).reshape_as(x)
        attn = self.memory.routed_linear(attn, self.memory.o_proj, indices[..., 3, :], weights[..., 3, :])
        x = x + self.dropout(attn)
        x = x + self.dropout(self.memory.ffn(self.norm2(x), indices[..., 0, :], weights[..., 0, :]))
        return x, weights


class MemoryTransformerForCausalLM(nn.Module):
    def __init__(self, config: Optional[MemoryTransformerConfig] = None) -> None:
        super().__init__()
        self.config = config or MemoryTransformerConfig()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.memory = SharedMemory(self.config)
        self.layers = nn.ModuleList(MemoryReasoner(self.config, self.memory) for _ in range(self.config.num_layers))
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

    def forward(self, input_ids: Tensor, labels: Optional[Tensor] = None, recurrent_steps: Optional[int] = None) -> dict[str, Tensor]:
        x = self.embed_tokens(input_ids)
        route_losses = []
        steps = recurrent_steps or self.config.recurrent_steps
        for _ in range(steps):
            for layer in self.layers:
                x, weights = layer(x)
                route_losses.append(self.load_balance_loss(weights))
        logits = self.lm_head(self.norm(x))
        result = {"logits": logits, "router_loss": torch.stack(route_losses).mean()}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
        return result

    @staticmethod
    def load_balance_loss(weights: Tensor) -> Tensor:
        # Switch-style importance loss: each route head should use all blocks.
        importance = weights.mean(dim=(0, 1))
        return importance.var(unbiased=False) * importance.numel()

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int, recurrent_steps: Optional[int] = None) -> Tensor:
        for _ in range(max_new_tokens):
            logits = self(input_ids, recurrent_steps=recurrent_steps)["logits"][:, -1]
            input_ids = torch.cat((input_ids, logits.argmax(-1, keepdim=True)), dim=-1)
        return input_ids
