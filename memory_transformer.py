"""Shared-memory recurrent Transformer language model.

The model has one globally shared memory with 24 addressable blocks.  Each of
the 24 reasoner cycles owns a router and two RMSNorms, but all cycles access the
same FFN/Q/KV/O banks.  Routing is top-3 per head and K/V deliberately share
the same route head and weights.

Early exit (batch-level): after each reasoner cycle the model reads the last
token position and evaluates the weighted halting cost ``l_t`` built from the
top-1 confidence ``p_t``, the top1-minus-top2 probability margin, the
normalized state change ``||norm(x_t) - norm(x_{t-1})|| / ||norm(x_t)||``, the
speed ``t/num_depths``, and the min-depth guard.  The exit depth is the argmax
of the depth-selection distribution ``softmax(-l_t / temperature)`` (the
smallest cost); online, the loop halts at the first depth whose cost rises
above the running minimum.  The optional ``CalibratedExitReadout`` (rank-r LoRA
plus per-depth vocabulary bias and temperature on the frozen lm_head) supplies
the depth-calibrated logits used by both the halting rule and the final
readout.  Design references: CALM (arXiv 2207.07061), FPRM (arXiv 2606.18206),
post-hoc readout diagnostics (arXiv 2607.20519), Huginn deep supervision
(arXiv 2502.05171), RecurTrace (arXiv 2609.03379).
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.utils.checkpoint
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
    early_exit_enabled: bool = False
    early_exit_min_depth: int = 8
    early_exit_confidence_threshold: float = 0.9
    early_exit_margin_threshold: float = 0.3
    early_exit_state_delta_threshold: float = 0.05
    exit_selection_weights: tuple = (
        0.0,
        0.14285714285714285,
        0.14285714285714285,
        0.14285714285714285,
        0.42857142857142855,
        0.14285714285714285,
    )
    exit_selection_temperature: float = 0.25
    exit_readout_rank: int = 16

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
        if not 1 <= self.early_exit_min_depth:
            # Values above num_layers are clamped at use time so the 24-cycle
            # production default (8) also validates for smaller test models.
            raise ValueError("early_exit_min_depth must be at least 1")
        if not 0 < self.early_exit_confidence_threshold <= 1:
            raise ValueError("early_exit_confidence_threshold must be in (0, 1]")
        if self.early_exit_margin_threshold < 0:
            raise ValueError("early_exit_margin_threshold must be non-negative")
        if self.early_exit_state_delta_threshold <= 0:
            raise ValueError("early_exit_state_delta_threshold must be positive")
        if len(self.exit_selection_weights) != 6:
            raise ValueError("exit_selection_weights must hold six values")
        if any(weight < 0 for weight in self.exit_selection_weights):
            raise ValueError("exit_selection_weights must be non-negative")
        if abs(sum(self.exit_selection_weights) - 1.0) > 1e-3:
            raise ValueError("exit_selection_weights must sum to one")
        if self.exit_selection_temperature <= 0:
            raise ValueError("exit_selection_temperature must be positive")
        if self.exit_readout_rank < 1:
            raise ValueError("exit_readout_rank must be positive")

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
        # Expert dispatch implementation: "loop" (per-expert nonzero scan,
        # legacy), "sorted" (one-pass argsort+bincount partition, per-expert
        # GEMMs, single accumulation) or "grouped" (sorted partition plus
        # torch._grouped_mm fused group GEMMs).  "loop" keeps the reference
        # behavior; the alternatives remove the 24-per-bank host syncs.
        self.dispatch_mode = "loop"
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize each addressable matrix separately. Initializing the whole
        # 3-D tensor makes PyTorch treat the block dimension as a kernel axis.
        for bank in (
            self.ffn_up,
            self.ffn_down,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
        ):
            for block in bank:
                nn.init.xavier_uniform_(block)
        # SiLU's input branch uses the common ReLU-gain Kaiming approximation.
        # Banks are [input, output]; transpose for PyTorch's fan-in convention.
        with torch.no_grad():
            for block in self.ffn_gate:
                nn.init.kaiming_uniform_(block.transpose(0, 1), nonlinearity="relu")

    @staticmethod
    def _locations(indices: Tensor, block_id: int) -> Tensor:
        # Flatten all leading token dimensions. The second location column is
        # the top-k slot and lets repeated block IDs accumulate correctly.
        return (indices.reshape(-1, indices.shape[-1]) == block_id).nonzero(
            as_tuple=False
        )

    @staticmethod
    def _partition(indices: Tensor, weights: Tensor, num_blocks: int):
        """One-pass expert partition; ONE host sync replaces 24 nonzero syncs.

        ``indices``/``weights`` have shape ``[..., top_k]``.  Returns
        ``(tokens_sorted, weights_sorted, counts_cpu, starts)`` where
        ``tokens_sorted`` are token ids grouped by expert (stable sort keeps
        the original order inside each group) and ``starts[i]`` is group i's
        start row in the sorted arrays.
        """
        top_k = indices.shape[-1]
        idx_flat = indices.reshape(-1)
        w_flat = weights.reshape(-1)
        order = torch.argsort(idx_flat, stable=True)
        counts = torch.bincount(idx_flat, minlength=num_blocks)
        counts_cpu = counts.tolist()
        starts = [0]
        for count in counts_cpu:
            starts.append(starts[-1] + count)
        tokens_sorted = order.div(top_k, rounding_mode="floor")
        weights_sorted = w_flat[order]
        return tokens_sorted, weights_sorted, counts_cpu, starts

    @staticmethod
    def _routed_linear_sorted(
        x: Tensor,
        bank: Tensor,
        tokens_sorted: Tensor,
        weights_sorted: Tensor,
        starts: list[int],
    ) -> Tensor:
        """Sorted dispatch: partition once (one host sync), per-expert gather,
        per-expert GEMMs, IN-PLACE accumulation.

        Versus the legacy loop: 24 nonzero host syncs become one argsort +
        bincount + tolist; the chained out-of-place scatter_adds (which keep
        every intermediate output alive in the autograd graph, ~1 GiB each for
        the FFN bank) become in-place index_add_ on a plain buffer.
        """
        x_flat = x.reshape(-1, x.shape[-1])
        output = x_flat.new_zeros((x_flat.shape[0], bank.shape[-1]))
        for block_id in range(bank.shape[0]):
            s, e = starts[block_id], starts[block_id + 1]
            if e <= s:
                continue
            token_ids = tokens_sorted[s:e]
            block_input = x_flat.index_select(0, token_ids)
            block_output = F.linear(block_input, bank[block_id].transpose(0, 1))
            block_output = block_output * weights_sorted[s:e].to(
                block_output.dtype
            ).unsqueeze(-1)
            # index_add_ enforces same dtype even under autocast (scatter_add
            # does not); cast the source to the accumulator dtype.
            output.index_add_(0, token_ids, block_output.to(output.dtype))
        return output.reshape(*x.shape[:-1], bank.shape[-1])

    @staticmethod
    def _grouped_mm_chunked_rows(
        grouped_rows: Tensor, weight_stack: Tensor, starts: list[int], chunk_rows: int = 262144
    ) -> Tensor:
        """torch._grouped_mm over rows ALREADY in sorted routed-row order.

        Slices ``grouped_rows[lo:hi]`` per chunk (no gather): for chains where
        the input is a previous grouped GEMM's output (e.g. the FFN down
        projection consuming ``silu(gate)*up``), the rows are already in
        routed-row order and re-gathering with token ids would index the
        wrong rows entirely.
        """
        total = grouped_rows.shape[0]
        num_groups = weight_stack.shape[0]
        outputs: list[Tensor] = []
        lo = 0
        g = 0
        while lo < total:
            hi = min(lo + chunk_rows, total)
            sub = grouped_rows[lo:hi]
            while g < num_groups and starts[g + 1] <= lo:
                g += 1
            g_end = g
            while g_end < num_groups and starts[g_end] < hi:
                g_end += 1
            offs = torch.tensor(
                [min(starts[k + 1], hi) - lo for k in range(g, g_end)],
                dtype=torch.int32,
                device=sub.device,
            )
            outputs.append(torch._grouped_mm(sub, weight_stack[g:g_end], offs=offs))
            lo = hi
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _grouped_mm_chunked(
        x_flat: Tensor, tokens_sorted: Tensor, weight_stack: Tensor,
        starts: list[int], chunk_rows: int = 262144,
    ) -> Tensor:
        """torch._grouped_mm over sorted routed rows, chunked to cap memory.

        The fused group GEMM materializes [routed_rows, out] outputs; at
        production batch (983k routed rows) a single call would hold ~8 GiB
        per bank tensor.  Chunking the sorted rows splits each expert's GEMM
        across chunks (relative offsets), keeping transients ~2 GiB.  Rows are
        gathered per chunk so no full [983k, hidden] copy is retained.
        """
        total = tokens_sorted.shape[0]
        num_groups = weight_stack.shape[0]
        outputs: list[Tensor] = []
        lo = 0
        g = 0
        while lo < total:
            hi = min(lo + chunk_rows, total)
            sub = x_flat.index_select(0, tokens_sorted[lo:hi])
            while g < num_groups and starts[g + 1] <= lo:
                g += 1
            g_end = g
            while g_end < num_groups and starts[g_end] < hi:
                g_end += 1
            offs = torch.tensor(
                [min(starts[k + 1], hi) - lo for k in range(g, g_end)],
                dtype=torch.int32,
                device=sub.device,
            )
            outputs.append(torch._grouped_mm(sub, weight_stack[g:g_end], offs=offs))
            lo = hi
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _routed_linear_grouped(
        x: Tensor,
        bank: Tensor,
        tokens_sorted: Tensor,
        weights_sorted: Tensor,
        starts: list[int],
    ) -> Tensor:
        """Sorted dispatch with fused group GEMMs (torch._grouped_mm)."""
        x_flat = x.reshape(-1, x.shape[-1])
        grouped_output = SharedMemory._grouped_mm_chunked(
            x_flat, tokens_sorted, bank.to(torch.bfloat16), starts
        )
        grouped_output = grouped_output * weights_sorted.to(
            grouped_output.dtype
        ).unsqueeze(-1)
        output = x_flat.new_zeros(
            (x_flat.shape[0], grouped_output.shape[-1]), dtype=grouped_output.dtype
        )
        output = output.index_add(0, tokens_sorted, grouped_output)
        return output.to(x.dtype).reshape(*x.shape[:-1], -1)

    @staticmethod
    def _routed_ffn_sorted(
        x: Tensor,
        gate_bank: Tensor,
        up_bank: Tensor,
        down_bank: Tensor,
        tokens_sorted: Tensor,
        weights_sorted: Tensor,
        starts: list[int],
    ) -> Tensor:
        """Sorted FFN dispatch: partition once, per-expert gather and
        gate/up/down GEMMs, IN-PLACE accumulation."""
        x_flat = x.reshape(-1, x.shape[-1])
        hidden_size = down_bank.shape[-1]
        output = x_flat.new_zeros((x_flat.shape[0], hidden_size))
        for block_id in range(gate_bank.shape[0]):
            s, e = starts[block_id], starts[block_id + 1]
            if e <= s:
                continue
            token_ids = tokens_sorted[s:e]
            block_input = x_flat.index_select(0, token_ids)
            gate = F.linear(block_input, gate_bank[block_id].transpose(0, 1))
            up = F.linear(block_input, up_bank[block_id].transpose(0, 1))
            hidden = F.silu(gate) * up
            block_output = F.linear(hidden, down_bank[block_id].transpose(0, 1))
            block_output = block_output * weights_sorted[s:e].to(
                block_output.dtype
            ).unsqueeze(-1)
            output.index_add_(0, token_ids, block_output.to(output.dtype))
        return output.reshape(*x.shape[:-1], hidden_size)

    @staticmethod
    def _routed_ffn_grouped(
        x: Tensor,
        gate_bank: Tensor,
        up_bank: Tensor,
        down_bank: Tensor,
        tokens_sorted: Tensor,
        weights_sorted: Tensor,
        starts: list[int],
    ) -> Tensor:
        """Sorted FFN dispatch with fused group GEMMs (torch._grouped_mm)."""
        x_flat = x.reshape(-1, x.shape[-1])
        gate = SharedMemory._grouped_mm_chunked(x_flat, tokens_sorted, gate_bank.to(torch.bfloat16), starts)
        up = SharedMemory._grouped_mm_chunked(x_flat, tokens_sorted, up_bank.to(torch.bfloat16), starts)
        hidden = F.silu(gate) * up
        del gate, up
        # hidden is already in routed-row order: slice it, don't re-gather
        block_output = SharedMemory._grouped_mm_chunked_rows(hidden, down_bank.to(torch.bfloat16), starts)
        del hidden
        block_output = block_output * weights_sorted.to(
            block_output.dtype
        ).unsqueeze(-1)
        output = x_flat.new_zeros(
            (x_flat.shape[0], block_output.shape[-1]), dtype=block_output.dtype
        )
        output = output.index_add(0, tokens_sorted, block_output)
        return output.to(x.dtype).reshape(*x.shape[:-1], -1)

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
        mode = getattr(self, "dispatch_mode", "loop")
        if mode != "loop":
            tokens_sorted, weights_sorted, _, starts = self._partition(
                indices, weights, self.ffn_gate.shape[0]
            )
            if mode == "sorted":
                return self._routed_ffn_sorted(
                    x, self.ffn_gate, self.ffn_up, self.ffn_down,
                    tokens_sorted, weights_sorted, starts,
                )
            if mode == "grouped":
                return self._routed_ffn_grouped(
                    x, self.ffn_gate, self.ffn_up, self.ffn_down,
                    tokens_sorted, weights_sorted, starts,
                )
            raise ValueError(f"unknown dispatch mode: {mode}")
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
        mode = getattr(self, "dispatch_mode", "loop")
        if mode != "loop":
            tokens_sorted, weights_sorted, _, starts = self._partition(
                indices, weights, bank.shape[0]
            )
            if mode == "sorted":
                return self._routed_linear_sorted(
                    x, bank, tokens_sorted, weights_sorted, starts
                )
            if mode == "grouped":
                return self._routed_linear_grouped(
                    x, bank, tokens_sorted, weights_sorted, starts
                )
            raise ValueError(f"unknown dispatch mode: {mode}")
        return self.routed_linear(x, bank, indices, weights)


class CalibratedExitReadout(nn.Module):
    """Per-depth calibrated readout on top of the frozen ``lm_head``.

    Depth ``t`` computes ``logits = temperature[t] * (W h + B_t (A_t h) + c_t)``
    with ``W`` the frozen language-model head.  ``B_t`` and ``c_t`` start at
    zero and the temperature starts at one, so an unfitted readout is exactly
    the identity mapping of the shared head (bit-level equal logits).  Only
    these low-rank corrections, the bias, and the temperature are trainable;
    the backbone and the shared head stay frozen.
    """

    def __init__(
        self, hidden_size: int, vocab_size: int, num_depths: int, rank: int
    ) -> None:
        super().__init__()
        if rank < 1 or num_depths < 1:
            raise ValueError("rank and num_depths must be positive")
        self.rank = rank
        self.num_depths = num_depths
        self.project = nn.Parameter(torch.zeros(num_depths, rank, hidden_size))
        self.vocabulary = nn.Parameter(torch.zeros(num_depths, vocab_size, rank))
        self.bias = nn.Parameter(torch.zeros(num_depths, vocab_size))
        self.temperature = nn.Parameter(torch.ones(num_depths))
        for depth in range(num_depths):
            nn.init.xavier_uniform_(self.project[depth])

    def reset_identity(self) -> None:
        """Restore the exact identity initialization (B = c = 0, T = 1)."""
        nn.init.zeros_(self.vocabulary)
        nn.init.zeros_(self.bias)
        nn.init.ones_(self.temperature)
        for depth in range(self.num_depths):
            nn.init.xavier_uniform_(self.project[depth])

    def logits_at(self, depth_index: int, states: Tensor, head_weight: Tensor) -> Tensor:
        """Calibrated logits for normed states ``[..., hidden]`` at one depth."""
        if not 0 <= depth_index < self.num_depths:
            raise ValueError(f"depth index {depth_index} outside [0, {self.num_depths})")
        base = F.linear(states, head_weight)
        dtype = states.dtype
        projected = states @ self.project[depth_index].to(dtype).transpose(0, 1)
        correction = projected @ self.vocabulary[depth_index].to(dtype).transpose(0, 1)
        correction = correction + self.bias[depth_index].to(dtype)
        return (base + correction) * self.temperature[depth_index].to(dtype)

    def corrections_at(
        self, states: Tensor, depth_index: Optional[Tensor] = None
    ) -> Tensor:
        """Batched LoRA correction for every depth at once (no head matmul).

        ``states`` has shape ``[D, ..., hidden]`` (one row set per depth).  The
        correction ``B_t (A_t h) + c_t`` for each depth ``t`` is computed with
        two bmm ops instead of a Python loop.  When ``depth_index`` is given
        (shape ``[...]`` matching the leading depth batch dims), the correction
        for that single depth is selected per row instead.
        """
        if states.shape[0] != self.num_depths:
            raise ValueError(
                f"states leading dim {states.shape[0]} != num_depths {self.num_depths}"
            )
        dtype = states.dtype
        lead = states.shape[1:-1]
        flat = states.reshape(self.num_depths, -1, states.shape[-1])
        projected = torch.bmm(
            flat, self.project.to(dtype).transpose(1, 2)
        )  # [D, N, r]
        correction = torch.bmm(
            projected, self.vocabulary.to(dtype).transpose(1, 2)
        )  # [D, N, V]
        correction = correction + self.bias.to(dtype).unsqueeze(1)
        if depth_index is None:
            return correction.reshape(self.num_depths, *lead, -1)
        gathered = correction[
            torch.arange(self.num_depths, device=states.device), depth_index.reshape(-1)
        ]
        return gathered.reshape(*depth_index.shape, -1)

    def extra_repr(self) -> str:
        return f"depths={self.num_depths} rank={self.rank}"


def ensure_exit_readout(
    model: "MemoryTransformerForCausalLM", rank: Optional[int] = None
) -> "CalibratedExitReadout":
    """Attach (idempotently) the calibrated exit readout to a model."""
    if model.exit_readout is None:
        model.exit_readout = CalibratedExitReadout(
            hidden_size=model.config.hidden_size,
            vocab_size=model.config.vocab_size,
            num_depths=model.config.num_layers,
            rank=rank if rank is not None else model.config.exit_readout_rank,
        )
    elif rank is not None and rank != model.exit_readout.rank:
        raise ValueError("exit readout rank mismatch with the existing module")
    return model.exit_readout


def selection_exit_depth(
    losses: Sequence[float], min_depth: int, num_depths: int
) -> tuple[int, bool]:
    """Batch-level halting rule over per-depth selection costs.

    ``losses[index]`` is the cost ``l_t`` of depth ``t = index + 1``.  The exit
    depth is the argmax of the depth-selection distribution
    ``softmax(-l_t / temperature)``, i.e. the smallest cost; online, the loop
    halts at the first depth whose cost rises above the running minimum.
    Depths below ``min_depth`` are never exit candidates.  Returns the 1-based
    exit depth and whether halting happened before the full depth.
    """
    best: Optional[tuple[float, int]] = None
    for index, loss in enumerate(losses):
        if index + 1 < min_depth:
            continue
        if best is None or loss < best[0]:
            best = (loss, index)
        else:
            return best[1] + 1, True
    return num_depths, False


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
        # GQA attention: True passes k/v to SDPA with enable_gqa=True instead
        # of materializing repeat_interleave'd copies ([B, heads, T, head_dim]
        # each); outputs agree to kernel-level precision.
        self.use_gqa = False

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
        dropout_p = self.dropout.p if self.training else 0.0
        if getattr(self, "use_gqa", False):
            if attention_mask is not None:
                allowed = torch.tril(
                    torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool)
                ) & attention_mask[:, None, None, :].bool()
                attn = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=allowed, dropout_p=dropout_p, enable_gqa=True
                )
            else:
                attn = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=dropout_p, enable_gqa=True
                )
        else:
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
                    dropout_p=dropout_p,
                )
            else:
                attn = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=True,
                    dropout_p=dropout_p,
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
        self.exit_readout: Optional[CalibratedExitReadout] = None
        # Training-time memory optimizations toggled by the trainer:
        # gradient_checkpointing recomputes each cycle in backward;
        # loss_chunk_rows > 0 computes the label CE over checkpointed
        # vocabulary-sized row chunks instead of one monolithic tensor.
        self.gradient_checkpointing = False
        self.loss_chunk_rows = 0
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

    def _exit_readout_logits(self, depth_index: int, states: Tensor) -> Tensor:
        """Last-position logits for the halting rule at one depth."""
        if self.exit_readout is not None:
            return self.exit_readout.logits_at(depth_index, states, self.lm_head.weight)
        return F.linear(states, self.lm_head.weight)

    def _exit_selection_cost(
        self,
        confidence: Tensor,
        margin: Tensor,
        delta: Optional[Tensor],
        depth_index: int,
    ) -> Tensor:
        """Batch-mean halting cost ``l_t`` at one depth (lower is better).

        Weighted sum of the inference-available terms of the readout-training
        objective, matching the training cost definition: confidence deficit
        ``1 - q1``, probability-margin deficit ``1 - (q1 - q2)``, relative
        state change over the delta threshold, speed ``t/num_depths``, and the
        clamped min-depth guard ``max(0, (d_min - d) / D)``.  The shortlist
        cross-entropy term has no ground truth at inference and its annealed
        prior mass is zero, so it is omitted.
        """
        weights = self.config.exit_selection_weights
        td = self.config.early_exit_state_delta_threshold
        if td <= 0:
            raise ValueError("exit state delta threshold must be positive")
        cost = weights[1] * (1.0 - confidence).mean()
        cost = cost + weights[2] * (1.0 - margin).mean()
        if delta is not None and weights[3] > 0:
            cost = cost + weights[3] * (delta.float() / td).mean()
        num_depths = self.config.num_layers
        cost = cost + weights[4] * confidence.new_tensor((depth_index + 1) / num_depths)
        min_depth = min(self.config.early_exit_min_depth, num_depths)
        guard = confidence.new_tensor(
            max(0.0, (min_depth - (depth_index + 1)) / num_depths)
        )
        cost = cost + weights[5] * guard
        return cost

    def apply_exit_selection(
        self,
        weights: Sequence[float],
        temperature: float,
        enable: bool = True,
    ) -> None:
        """Attach trained depth-selection weights and temperature to the config."""
        self.config.exit_selection_weights = tuple(float(value) for value in weights)
        self.config.exit_selection_temperature = float(temperature)
        if enable:
            self.config.early_exit_enabled = True
        self.config.__post_init__()

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

    def _per_sample_exit_depths(
        self,
        confidence: list[Tensor],
        margin: list[Tensor],
        deltas: list[Optional[Tensor]],
    ) -> Tensor:
        """Per-sample halting depths (1-based) via the batch rule applied per row."""
        weights = self.config.exit_selection_weights
        num_depths = len(confidence)
        min_depth = min(self.config.early_exit_min_depth, num_depths)
        device = confidence[0].device
        rows = confidence[0].shape[0]
        best = torch.full((rows,), float("inf"), device=device)
        best_depth = torch.zeros(rows, dtype=torch.long, device=device)
        exited = torch.full((rows,), float(num_depths), device=device)
        done = torch.zeros(rows, dtype=torch.bool, device=device)
        for depth_index in range(num_depths):
            if depth_index + 1 < min_depth:
                continue
            cost = weights[1] * (
                self.config.early_exit_confidence_threshold - confidence[depth_index]
            ).clamp_min(0.0)
            cost = cost + weights[2] * (
                self.config.early_exit_margin_threshold - margin[depth_index]
            ).clamp_min(0.0)
            delta = deltas[depth_index]
            if delta is not None and weights[3] > 0:
                cost = cost + weights[3] * (
                    delta.to(cost.dtype) - self.config.early_exit_state_delta_threshold
                ).clamp_min(0.0)
            cost = cost + weights[4] * ((depth_index + 1) / num_depths)
            cost = cost + weights[5] * (max(0, min_depth - (depth_index + 1)) / num_depths)
            improve = cost < best
            best = torch.where(improve, cost, best)
            best_depth = torch.where(improve, depth_index, best_depth)
            halt = (~improve) & (~done)
            exited = torch.where(halt, best_depth.to(exited.dtype) + 1.0, exited)
            done = done | halt
        return exited

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        return_router_stats: bool = False,
        return_router_health: bool = False,
        early_exit: Optional[bool] = None,
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

        enabled = (
            self.config.early_exit_enabled if early_exit is None else bool(early_exit)
        )
        if enabled and self.training:
            # Halting is an inference-time mechanism; training always runs the
            # full depth so gradients and router statistics stay unchanged.
            enabled = False
        if enabled and input_ids.shape[0] > 1:
            # Independent sample execution: stopped samples perform no further cycles.
            outputs = [self.forward(
                input_ids[i:i+1],
                labels=None if labels is None else labels[i:i+1],
                attention_mask=None if attention_mask is None else attention_mask[i:i+1],
                return_router_stats=return_router_stats,
                return_router_health=return_router_health, early_exit=True,
            ) for i in range(input_ids.shape[0])]
            result = {
                "logits": torch.cat([o["logits"] for o in outputs]),
                "router_loss": torch.stack([o["router_loss"] for o in outputs]).mean(),
                "exit_depth": [o["exit_depth"] for o in outputs],
                "exit_halted": [o["exit_halted"] for o in outputs],
                "sample_outputs": outputs,
            }
            if labels is not None:
                targets = labels[:, 1:]
                if attention_mask is not None:
                    targets = targets.masked_fill(~attention_mask[:, 1:], -100)
                result["loss"] = F.cross_entropy(result["logits"][:, :-1].reshape(-1, self.config.vocab_size), targets.reshape(-1), ignore_index=-100)
            return result

        x = self.embed_tokens(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        route_losses = []
        route_indices = []
        route_weights = []
        route_probabilities = []
        health_load: Optional[Tensor] = None
        health_importance: Optional[Tensor] = None
        exit_depth = self.config.num_layers
        exit_confidence: list[Tensor] = []
        exit_margin: list[Tensor] = []
        exit_delta: list[Optional[Tensor]] = []
        exit_normed: Optional[Tensor] = None
        best_cost: Optional[Tensor] = None
        best_depth: Optional[int] = None
        best_normed: Optional[Tensor] = None
        prev_normed: Optional[Tensor] = None
        halted = False
        for depth_index, layer in enumerate(self.layers):
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                x, indices, weights, probabilities = torch.utils.checkpoint.checkpoint(
                    layer, x, self.memory, attention_mask, use_reentrant=False
                )
            else:
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
            if return_router_health:
                blocks_n = probabilities.shape[-1]
                probability_rows = probabilities.reshape(
                    -1, probabilities.shape[-2], blocks_n
                )
                index_rows = indices.reshape(-1, indices.shape[-2], indices.shape[-1])
                if attention_mask is not None:
                    valid = attention_mask.reshape(-1).bool()
                    probability_rows = probability_rows[valid]
                    index_rows = index_rows[valid]
                if probability_rows.shape[0] > 0:
                    importance = probability_rows.float().mean(dim=0)
                    assignment = F.one_hot(index_rows, num_classes=blocks_n).float()
                    load = assignment.mean(dim=(0, 2))
                    if health_load is None:
                        health_load = torch.zeros(blocks_n, device=x.device)
                        health_importance = torch.zeros_like(health_load)
                    health_load = health_load + load.sum(dim=0).detach()
                    health_importance = health_importance + importance.sum(dim=0).detach()
            if not enabled:
                continue
            normed = self.norm(x)
            if prev_normed is None:
                delta = None
            else:
                difference = normed.float() - prev_normed
                denominator = normed.float().norm(dim=-1).clamp_min(1e-6)
                per_position = difference.norm(dim=-1) / denominator
                if attention_mask is not None:
                    mask = attention_mask.to(per_position.dtype)
                    delta = (per_position * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
                else:
                    delta = per_position.mean(-1)
            last_logits = self._exit_readout_logits(depth_index, normed[:, -1, :])
            last_probabilities = torch.softmax(last_logits.float(), dim=-1)
            top2 = torch.topk(last_probabilities, 2, dim=-1).values
            confidence = top2[:, 0]
            margin = top2[:, 0] - top2[:, 1]
            exit_confidence.append(confidence.detach())
            exit_margin.append(margin.detach())
            exit_delta.append(None if delta is None else delta.detach())
            prev_normed = normed.detach()
            if depth_index + 1 < min(
                self.config.early_exit_min_depth, self.config.num_layers
            ):
                continue
            cost = self._exit_selection_cost(confidence, margin, delta, depth_index)
            if best_cost is None or bool(cost < best_cost):
                best_cost = cost
                best_depth = depth_index
                best_normed = normed
                continue
            exit_depth = best_depth + 1
            exit_normed = best_normed
            halted = True
            break
        # Chunked training CE: skip materializing the [batch, seq, vocab]
        # logits entirely and let the labels branch below compute the loss
        # row-chunk by row-chunk (training mode never early-exits, so the
        # exit-aware logits selection above is irrelevant here).
        chunked_ce = (
            labels is not None
            and self.loss_chunk_rows > 0
            and self.training
            and torch.is_grad_enabled()
        )
        if chunked_ce:
            logits = None
        elif enabled and self.exit_readout is not None:
            normed_final = exit_normed if halted else self.norm(x)
            logits = self.exit_readout.logits_at(
                exit_depth - 1, normed_final, self.lm_head.weight
            )
        elif halted:
            logits = self.lm_head(exit_normed)
        else:
            logits = self.lm_head(self.norm(x))
        result: dict[str, object] = {
            "logits": logits,
            "router_loss": torch.stack(route_losses).mean(),
        }
        if return_router_health:
            depth_count = max(len(route_losses), 1)
            result["router_health"] = {
                "layer_losses": [float(value.detach()) for value in route_losses],
                "block_load": (
                    (health_load / depth_count).cpu()
                    if health_load is not None
                    else torch.zeros(self.config.num_memory_blocks)
                ),
                "block_importance": (
                    (health_importance / depth_count).cpu()
                    if health_importance is not None
                    else torch.zeros(self.config.num_memory_blocks)
                ),
            }
        if enabled:
            result["exit_depth"] = exit_depth
            result["exit_halted"] = halted
            result["exit_confidence"] = [tensor.tolist() for tensor in exit_confidence]
            result["exit_margin"] = [tensor.tolist() for tensor in exit_margin]
            result["exit_delta"] = [
                None if tensor is None else tensor.tolist() for tensor in exit_delta
            ]
        if labels is not None:
            if chunked_ce:
                result["loss"] = self._chunked_cross_entropy(
                    x, labels, attention_mask
                )
            else:
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

    def _chunked_cross_entropy(
        self, hidden_state: Tensor, labels: Tensor, attention_mask: Optional[Tensor]
    ) -> Tensor:
        """Checkpointed chunk-wise CE that never materializes full logits.

        Computes the same masked-mean cross entropy as the monolithic path,
        but the ``[rows, vocab]`` logits and their backward intermediates only
        ever span ``loss_chunk_rows`` rows at a time, each wrapped in an
        activation checkpoint so the saved state per chunk is just the hidden
        rows.  The CE itself runs in fp32: a bf16 log-softmax over a ~150k
        vocabulary loses precision.
        """
        normed = self.norm(hidden_state)
        flat_hidden = normed[:, :-1].reshape(-1, normed.size(-1))
        flat_labels = labels[:, 1:].reshape(-1)
        if attention_mask is not None:
            flat_labels = flat_labels.masked_fill(
                ~attention_mask[:, 1:].reshape(-1), -100
            )
        masked_count = flat_labels.ne(-100).sum().clamp_min(1)
        ce_sum = flat_hidden.new_zeros((), dtype=torch.float32)

        def chunk_ce(rows: Tensor, targets: Tensor) -> Tensor:
            chunk_logits = self.lm_head(rows)
            return F.cross_entropy(
                chunk_logits.float(), targets, ignore_index=-100, reduction="sum"
            )

        for start in range(0, flat_hidden.size(0), self.loss_chunk_rows):
            ce_sum = ce_sum + torch.utils.checkpoint.checkpoint(
                chunk_ce,
                flat_hidden[start : start + self.loss_chunk_rows],
                flat_labels[start : start + self.loss_chunk_rows],
                use_reentrant=False,
            )
        return ce_sum / masked_count

    @torch.no_grad()
    def exit_depth_stats(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Per-sample halting depths (1-based) from one full-depth no-grad probe.

        Diagnostic helper for the trainer's log line: runs every reasoner
        cycle without grad, records per-depth confidence/margin/delta, and
        applies the selection-cost halting rule per row.  Never materializes
        full-vocabulary logits, so the transient peak stays well below the
        training step's.
        """
        x = self.embed_tokens(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        confidence: list[Tensor] = []
        margin: list[Tensor] = []
        deltas: list[Optional[Tensor]] = []
        prev_normed: Optional[Tensor] = None
        for depth_index, layer in enumerate(self.layers):
            x, _indices, _weights, _probabilities = layer(
                x, self.memory, attention_mask=attention_mask
            )
            if attention_mask is not None:
                x = x * attention_mask.unsqueeze(-1).to(x.dtype)
            normed = self.norm(x)
            if prev_normed is None:
                delta = None
            else:
                difference = normed.float() - prev_normed
                denominator = normed.float().norm(dim=-1).clamp_min(1e-6)
                per_position = difference.norm(dim=-1) / denominator
                if attention_mask is not None:
                    mask = attention_mask.to(per_position.dtype)
                    delta = (per_position * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
                else:
                    delta = per_position.mean(-1)
            last_logits = self._exit_readout_logits(depth_index, normed[:, -1, :])
            last_probabilities = torch.softmax(last_logits.float(), dim=-1)
            top2 = torch.topk(last_probabilities, 2, dim=-1).values
            confidence.append(top2[:, 0])
            margin.append(top2[:, 0] - top2[:, 1])
            deltas.append(delta)
            prev_normed = normed
        return self._per_sample_exit_depths(confidence, margin, deltas)

    @torch.no_grad()
    def exit_rollout(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> dict[str, Tensor]:
        """Full-depth backbone pass collecting per-depth readout inputs.

        Runs every reasoner cycle without grad and records, per depth: the
        final-normed last-position state (the calibrated readout input), the
        mask-normalized state delta against the previous depth (zero at depth
        one), and the plain lm_head teacher logits at full depth.  Used by the
        exit-readout trainer and by threshold calibration.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            attention_mask = attention_mask.bool()
        x = self.embed_tokens(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        states: list[Tensor] = []
        deltas: list[Tensor] = []
        prev_normed: Optional[Tensor] = None
        for layer in self.layers:
            x, _indices, _weights, _probabilities = layer(
                x, self.memory, attention_mask=attention_mask
            )
            if attention_mask is not None:
                x = x * attention_mask.unsqueeze(-1).to(x.dtype)
            normed = self.norm(x)
            states.append(normed[:, -1, :])
            if prev_normed is None:
                deltas.append(x.new_zeros(x.shape[0]))
            else:
                difference = normed.float() - prev_normed
                denominator = normed.float().norm(dim=-1).clamp_min(1e-6)
                per_position = difference.norm(dim=-1) / denominator
                if attention_mask is not None:
                    mask = attention_mask.to(per_position.dtype)
                    deltas.append(
                        (per_position * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
                    )
                else:
                    deltas.append(per_position.mean(-1))
            prev_normed = normed.detach()
        return {
            "states": torch.stack(states),
            "deltas": torch.stack(deltas),
            "teacher_logits": F.linear(states[-1], self.lm_head.weight),
        }

    @torch.inference_mode()
    def joint_rollout(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> dict[str, Tensor]:
        """Full-depth no-grad pass collecting phase-A inputs for joint training.

        Like :meth:`exit_rollout` but additionally caches the last-position
        ``W h`` products (the ``lm_head`` base logits at each depth) so the
        phase-A halting losses never re-run the vocabulary matmul, and the
        per-cycle router losses needed by the phase-B CE objective.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            attention_mask = attention_mask.bool()
        x = self.embed_tokens(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        states: list[Tensor] = []
        deltas: list[Tensor] = []
        router_losses: list[Tensor] = []
        prev_normed: Optional[Tensor] = None
        for layer in self.layers:
            x, indices, _weights, probabilities = layer(
                x, self.memory, attention_mask=attention_mask
            )
            if attention_mask is not None:
                x = x * attention_mask.unsqueeze(-1).to(x.dtype)
            router_losses.append(
                self.load_balance_loss(indices, probabilities, attention_mask)
            )
            normed = self.norm(x)
            states.append(normed[:, -1, :])
            if prev_normed is None:
                deltas.append(x.new_zeros(x.shape[0]))
            else:
                difference = normed.float() - prev_normed
                denominator = normed.float().norm(dim=-1).clamp_min(1e-6)
                per_position = difference.norm(dim=-1) / denominator
                if attention_mask is not None:
                    mask = attention_mask.to(per_position.dtype)
                    deltas.append(
                        (per_position * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
                    )
                else:
                    deltas.append(per_position.mean(-1))
            prev_normed = normed.detach()
        stacked = torch.stack(states)
        return {
            "states": stacked,
            "deltas": torch.stack(deltas),
            "base_logits": F.linear(stacked, self.lm_head.weight),
            "router_losses": torch.stack(router_losses),
        }

    def forward_until(
        self,
        input_ids: Tensor,
        num_cycles: int,
        attention_mask: Optional[Tensor] = None,
        collect_states: bool = False,
    ) -> dict[str, Tensor]:
        """Gradient pass through the first ``num_cycles`` reasoner cycles.

        Phase-B half of joint training: runs only the deployed prefix of the
        recurrent loop so the autograd graph (and the backward cost) scales
        with the selected exit depth instead of the full 24 cycles.  Returns
        the final-normed full-sequence state and, optionally, the per-cycle
        router losses and the intermediate normed states.
        """
        if not 1 <= num_cycles <= self.config.num_layers:
            raise ValueError(
                f"num_cycles must be in [1, {self.config.num_layers}]"
            )
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            attention_mask = attention_mask.bool()
        x = self.embed_tokens(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        router_losses: list[Tensor] = []
        normed_states: list[Tensor] = []
        for layer in self.layers[:num_cycles]:
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                x, indices, _weights, probabilities = torch.utils.checkpoint.checkpoint(
                    layer, x, self.memory, attention_mask, use_reentrant=False
                )
            else:
                x, indices, _weights, probabilities = layer(
                    x, self.memory, attention_mask=attention_mask
                )
            if attention_mask is not None:
                x = x * attention_mask.unsqueeze(-1).to(x.dtype)
            router_losses.append(
                self.load_balance_loss(indices, probabilities, attention_mask)
            )
            if collect_states:
                normed_states.append(self.norm(x))
        result: dict[str, Tensor] = {
            "final_normed": self.norm(x),
            "router_losses": torch.stack(router_losses),
        }
        if collect_states:
            result["normed_states"] = torch.stack(normed_states)
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        early_exit: Optional[bool] = None,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, float]]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        exit_depths: list[int] = []
        for _ in range(max_new_tokens):
            output = self(input_ids, early_exit=early_exit)
            if "exit_depth" in output:
                depths = output["exit_depth"]
                exit_depths.extend(depths if isinstance(depths, list) else [int(depths)])
            logits = output["logits"][:, -1]
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
        if return_stats:
            steps = len(exit_depths)
            if steps:
                average_depth = sum(exit_depths) / steps
            else:
                average_depth = float(self.config.num_layers)
            stats = {
                "steps": float(steps),
                "avg_exit_depth": average_depth,
            }
            return input_ids, stats
        return input_ids
