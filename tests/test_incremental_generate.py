"""Incremental (causal-recursion) generation must match the legacy recompute.

The GPU-only policy applies: conftest.py pins the default device to CUDA.

Equivalence contract: each decode step computes only the newest position,
which routes on its own hidden state at every cycle; earlier positions reuse
their cached per-cycle K/V.  Greedy tokens, per-step exit depths, and logits
must therefore match a whole-prefix recomputation (float tolerance for SDPA
kernel reduction-order differences between [1, 1] and [1, T] queries).
"""

import pytest
import torch
from torch import Tensor

from memory_transformer import (
    MemoryTransformerForCausalLM,
    _IncrementalRecursionState,
    ensure_exit_readout,
)
from test_early_exit import always_exit_model, tiny_model


def greedy_incremental(model, ids, steps, enabled):
    """Drive the incremental path manually, recording per-step outputs."""
    states = [
        _IncrementalRecursionState(model, ids[i : i + 1], enabled)
        for i in range(ids.shape[0])
    ]
    outputs = [(state.logits, state.exit_depth) for state in states]
    rows = [outputs]
    produced: list[Tensor] = []
    for _ in range(steps):
        produced.append(
            torch.cat([row.argmax(-1, keepdim=True) for row, _depth in outputs], dim=0)
        )
        if len(rows) == steps:
            break
        tokens = produced[-1]
        outputs = [
            state.step(tokens[i : i + 1]) for i, state in enumerate(states)
        ]
        rows.append(outputs)
    sequence = torch.cat([ids] + produced, dim=1)
    return rows, sequence


def greedy_legacy(model, ids, steps, enabled):
    """Whole-prefix recomputation every step (the pre-incremental behaviour)."""
    rows = []
    current = ids
    with torch.no_grad():
        for _ in range(steps):
            output = model(current, early_exit=enabled)
            logits = output["logits"][:, -1]
            depths = output.get("exit_depth", model.config.num_layers)
            if not isinstance(depths, list):
                depths = [int(depths)] * ids.shape[0]
            rows.append([
                (logits[i : i + 1], int(depths[i])) for i in range(ids.shape[0])
            ])
            current = torch.cat((current, logits.argmax(-1, keepdim=True)), dim=-1)
    return rows, current


def assert_steps_match(new_rows, old_rows, atol):
    assert len(new_rows) == len(old_rows)
    for step_new, step_old in zip(new_rows, old_rows):
        assert len(step_new) == len(step_old)
        for (new_logits, new_depth), (old_logits, old_depth) in zip(step_new, step_old):
            assert new_depth == old_depth
            torch.testing.assert_close(new_logits, old_logits, rtol=0, atol=atol)


@pytest.mark.parametrize("enabled", [True, False])
def test_incremental_matches_legacy_greedy(enabled):
    torch.manual_seed(0)
    model = tiny_model(early_exit_enabled=enabled).eval()
    ensure_exit_readout(model)
    ids = torch.arange(10).view(2, 5) % 32
    new_rows, new_sequence = greedy_incremental(model, ids, 5, enabled)
    old_rows, old_sequence = greedy_legacy(model, ids, 5, enabled)
    assert_steps_match(new_rows, old_rows, atol=1e-5)
    assert torch.equal(new_sequence, old_sequence)
    assert new_sequence.shape == (2, 10)


def test_incremental_matches_legacy_with_gqa():
    torch.manual_seed(1)
    model = tiny_model(early_exit_enabled=True).eval()
    ensure_exit_readout(model)
    for layer in model.layers:
        layer.use_gqa = True
    ids = torch.arange(8).view(2, 4) % 32
    new_rows, new_sequence = greedy_incremental(model, ids, 4, True)
    old_rows, old_sequence = greedy_legacy(model, ids, 4, True)
    assert_steps_match(new_rows, old_rows, atol=1e-5)
    assert torch.equal(new_sequence, old_sequence)


def test_always_exit_drives_cache_extension():
    """Shallow exits exercise the backfill and deeper-cycle extension paths."""
    torch.manual_seed(2)
    model = always_exit_model(early_exit_min_depth=1).eval()
    ids = torch.arange(6).view(1, 6) % 32
    state = _IncrementalRecursionState(model, ids, True)
    depths = [state.exit_depth]
    for _ in range(6):
        _logits, depth = state.step(
            torch.tensor([[int(state.logits.argmax(-1))]])
        )
        depths.append(depth)
    old_rows, old_sequence = greedy_legacy(model, ids, 7, True)
    assert depths == [row[0][1] for row in old_rows]
    # the cache must have grown beyond the prefill halt depth and stay coherent
    assert 1 <= state.depth <= model.config.num_layers
    assert all(
        key.shape[2] == ids.shape[1] + 6 for key in state.keys
    )


def test_generate_incremental_equals_legacy():
    torch.manual_seed(3)
    model = tiny_model(early_exit_enabled=True).eval()
    ensure_exit_readout(model)
    ids = torch.arange(12).view(3, 4) % 32
    with torch.no_grad():
        fast = model.generate(ids, 6, early_exit=True, incremental=True)
        slow = model.generate(ids, 6, early_exit=True, incremental=False)
    assert torch.equal(fast, slow)
    with torch.no_grad():
        fast, fast_stats = model.generate(
            ids, 6, early_exit=True, incremental=True, return_stats=True
        )
        slow, slow_stats = model.generate(
            ids, 6, early_exit=True, incremental=False, return_stats=True
        )
    assert fast_stats == slow_stats


def test_generate_full_depth_incremental_equals_legacy():
    torch.manual_seed(4)
    model = tiny_model().eval()
    ids = torch.arange(9).view(3, 3) % 32
    with torch.no_grad():
        fast = model.generate(ids, 5, early_exit=False, incremental=True)
        slow = model.generate(ids, 5, early_exit=False, incremental=False)
    assert torch.equal(fast, slow)


def test_generate_eos_and_zero_steps():
    torch.manual_seed(5)
    model = tiny_model(early_exit_enabled=True).eval()
    ids = torch.arange(4).view(1, 4) % 32
    with torch.no_grad():
        generated = model.generate(
            ids, 4, eos_token_id=int(model.generate(
                ids, 1, early_exit=True
            )[0, -1].item()), early_exit=True
        )
    assert generated.shape[1] <= ids.shape[1] + 4
    same = model.generate(ids, 0, early_exit=True, incremental=True)
    assert torch.equal(same, ids)


def test_incremental_rejects_training_mode():
    model = tiny_model(early_exit_enabled=True).train()
    ids = torch.arange(4).view(1, 4) % 32
    with pytest.raises(ValueError, match="eval mode"):
        model.generate(ids, 1, early_exit=True, incremental=True)
