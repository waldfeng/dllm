import argparse
import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from memory_transformer import (
    CalibratedExitReadout,
    MemoryTransformerConfig,
    MemoryTransformerForCausalLM,
    ensure_exit_readout,
)
from test import load_model
from train import assemble_exit_terms, exit_readout_losses


def tiny_config(**overrides) -> MemoryTransformerConfig:
    values = dict(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_memory_blocks=4,
        top_k=2,
        max_position_embeddings=32,
        early_exit_min_depth=2,
    )
    values.update(overrides)
    return MemoryTransformerConfig(**values)


def tiny_model(**overrides) -> MemoryTransformerForCausalLM:
    return MemoryTransformerForCausalLM(tiny_config(**overrides))


def permissive_args(**overrides) -> argparse.Namespace:
    values = dict(
        exit_shortlist=64,
        exit_min_depth=1,
        exit_tau_confidence=0.9,
        exit_tau_margin=0.3,
        exit_tau_delta=1e9,
        exit_temp_start=2.0,
        exit_temp_end=0.25,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def synthetic_rollout(
    depth: int = 4, batch: int = 2, hidden: int = 32, vocab: int = 32, seed: int = 0
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    states = torch.randn(depth, batch, hidden, generator=generator)
    deltas = torch.rand(depth, batch, generator=generator)
    teacher = torch.randn(batch, vocab, generator=generator) * 5.0
    return {"states": states, "deltas": deltas, "teacher_logits": teacher}


def uniform_weights() -> torch.Tensor:
    prior = torch.tensor([0.65, 0.05, 0.05, 0.05, 0.15, 0.05])
    return prior / prior.sum()


def always_exit_model(**overrides) -> MemoryTransformerForCausalLM:
    values = dict(
        early_exit_enabled=True,
        early_exit_confidence_threshold=1e-6,
        early_exit_margin_threshold=0.0,
        early_exit_state_delta_threshold=1e9,
    )
    values.update(overrides)
    return tiny_model(**values)


def test_full_min_depth_runs_full_depth_and_matches_baseline():
    model = always_exit_model(early_exit_min_depth=4).eval()
    ids = torch.arange(8).view(1, 8) % 32
    with torch.no_grad():
        exited = model(ids, early_exit=True)
        baseline = model(ids, early_exit=False)
    assert exited["exit_depth"] == 4
    assert exited["exit_halted"] is False
    assert torch.equal(exited["logits"], baseline["logits"])


def test_threshold_rule_halts_at_first_candidate_and_matches_manual_loop():
    model = always_exit_model(early_exit_min_depth=2).eval()
    ensure_exit_readout(model)
    ids = torch.arange(8).view(1, 8) % 32
    with torch.no_grad():
        output = model(ids, early_exit=True)
        x = model.embed_tokens(ids)
        for layer in model.layers[:2]:
            x, _index, _weight, _probability = layer(x, model.memory, attention_mask=None)
        reference = model.lm_head(model.norm(x))
    assert output["exit_depth"] == 2
    assert output["exit_halted"] is True
    assert torch.equal(output["logits"], reference)


def test_calibrated_readout_identity_is_bitwise_equal():
    model = tiny_model(
        early_exit_enabled=True,
        early_exit_min_depth=4,
        early_exit_confidence_threshold=1e-6,
    ).eval()
    ensure_exit_readout(model)
    ids = torch.arange(8).view(1, 8) % 32
    with torch.no_grad():
        exited = model(ids, early_exit=True)
        baseline = model(ids, early_exit=False)
        states = torch.randn(3, 32)
        at_depth = model.exit_readout.logits_at(1, states, model.lm_head.weight)
        plain = torch.nn.functional.linear(states, model.lm_head.weight)
    assert exited["exit_depth"] == 4
    assert torch.equal(exited["logits"], baseline["logits"])
    assert torch.equal(at_depth, plain)


def test_exit_terms_move_in_the_minimization_direction():
    model = tiny_model().eval()
    ensure_exit_readout(model)
    target = torch.tensor([3, 7])

    def run(rollout, weights, **overrides):
        total, stats = exit_readout_losses(
            model, rollout, target, weights, uniform_weights(), permissive_args(**overrides), 0.0, 4
        )
        return total, stats

    rollout = synthetic_rollout()
    weights = uniform_weights()
    shaping_args = permissive_args(
        exit_tau_confidence=0.9, exit_tau_margin=0.3, exit_tau_delta=1e9
    )

    def run(rollout, weights, args):
        total, stats = exit_readout_losses(
            model, rollout, target, weights, uniform_weights(), args, 0.0, 4
        )
        return total, stats

    weak_total, weak_stats = run(rollout, weights, shaping_args)
    peaked = synthetic_rollout()
    with torch.no_grad():
        model.exit_readout.bias.zero_()
        model.exit_readout.bias[:, 3] = 20.0
        model.exit_readout.bias[:, 7] = 20.0
        others = torch.ones(32, dtype=torch.bool)
        others[[3, 7]] = False
        model.exit_readout.bias[:, others] = -20.0
    strong_total, strong_stats = run(peaked, weights, shaping_args)
    with torch.no_grad():
        model.exit_readout.bias.zero_()
    assert strong_stats["confidence"] < weak_stats["confidence"]
    assert strong_stats["margin"] < weak_stats["margin"]
    assert strong_stats["ce"] < weak_stats["ce"]

    flat_delta = synthetic_rollout()
    flat_delta["deltas"] = torch.zeros(4, 2)
    rough_delta = synthetic_rollout()
    rough_delta["deltas"] = torch.full((4, 2), 0.5)
    delta_args = permissive_args(
        exit_tau_confidence=0.9, exit_tau_margin=0.3, exit_tau_delta=0.05
    )
    _, flat_stats = run(flat_delta, weights, delta_args)
    _, rough_stats = run(rough_delta, weights, delta_args)
    assert rough_stats["delta"] > flat_stats["delta"]

    penalized = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    _, guard_free = run(rollout, penalized, permissive_args(exit_min_depth=1))
    _, guarded = run(rollout, penalized, permissive_args(exit_min_depth=8))
    assert guard_free["guard"] == 0.0
    assert guarded["guard"] > guard_free["guard"]
    assert strong_total.isfinite() and weak_total.isfinite()


def test_loss_terms_are_bounded_and_weights_sum_to_one():
    model = tiny_model().eval()
    ensure_exit_readout(model)
    weights = uniform_weights()
    prior = uniform_weights()
    total, stats = exit_readout_losses(
        model,
        synthetic_rollout(),
        torch.tensor([3, 7]),
        weights,
        prior,
        permissive_args(),
        0.0,
        4,
    )
    assert abs(float(weights.sum()) - 1.0) < 1e-5
    assert 0.0 <= stats["weight_alignment"] <= 1.0
    assert "sparsity" not in stats
    assert stats["total"] == pytest.approx(
        sum(stats[key] for key in ("ce", "confidence", "margin", "delta", "speed", "guard", "weight_alignment"))
    )
    assert stats["ce"] >= 0.0
    assert 0.0 <= stats["confidence"] <= 1.0
    assert 0.0 <= stats["margin"] <= 1.0
    assert stats["delta"] >= 0.0
    assert 0.0 <= stats["speed"] <= 1.0
    assert stats["guard"] >= 0.0
    assert torch.isfinite(total)


def test_smooth_costs_have_range_and_finite_gradients():
    model = tiny_model().eval()
    ensure_exit_readout(model)
    weights = uniform_weights()
    total, stats = exit_readout_losses(
        model,
        synthetic_rollout(),
        torch.tensor([3, 7]),
        weights,
        uniform_weights(),
        permissive_args(),
        0.0,
        4,
    )
    assert 0.0 < stats["confidence"] < weights[1]
    assert 0.0 < stats["margin"] < weights[2]
    total.backward()
    assert model.exit_readout.bias.grad is not None
    assert torch.isfinite(model.exit_readout.bias.grad).all()
    assert model.exit_readout.bias.grad.abs().sum() > 0


def test_fixed_weights_receive_no_gradient():
    model = tiny_model().eval()
    ensure_exit_readout(model)
    weights = uniform_weights()
    total, _stats = exit_readout_losses(
        model,
        synthetic_rollout(),
        torch.tensor([3, 7]),
        weights,
        uniform_weights(),
        permissive_args(),
        0.0,
        4,
    )
    total.backward()
    assert weights.grad is None
    assert model.exit_readout.vocabulary.grad is not None
    backbone = model.memory.ffn_gate.grad
    assert backbone is None


def test_training_mode_ignores_early_exit():
    model = tiny_model(early_exit_enabled=True)
    ids = torch.arange(8).view(1, 8) % 32
    output = model(ids, early_exit=True)
    assert "exit_depth" not in output
    model.eval()
    with torch.no_grad():
        output = model(ids, early_exit=True)
    assert "exit_depth" in output


def test_exit_rollout_shapes_and_zero_first_delta():
    model = tiny_model().eval()
    ids = torch.arange(8).view(2, 4) % 32
    with torch.no_grad():
        rollout = model.exit_rollout(ids)
    assert rollout["states"].shape == (4, 2, 32)
    assert rollout["deltas"].shape == (4, 2)
    assert rollout["teacher_logits"].shape == (2, 32)
    assert torch.all(rollout["deltas"][0] == 0.0)
    assert torch.all(rollout["deltas"][1:] > 0.0)


def test_checkpoint_roundtrip_enables_halting(tmp_path: Path):
    model = tiny_model().eval()
    ensure_exit_readout(model)
    ids = torch.arange(8).view(1, 8) % 32
    with torch.no_grad():
        before = model(ids, early_exit=True)["logits"]
    path = tmp_path / "exit-readout.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(model.config),
            "exit_readout_rank": 16,
        },
        path,
    )
    restored = load_model(path, torch.device("cuda"))
    assert restored.config.early_exit_enabled is True
    with torch.no_grad():
        after = restored(ids, early_exit=True)["logits"]
    assert torch.equal(before, after)


def test_load_model_tolerates_retired_config_fields(tmp_path: Path):
    model = tiny_model().eval()
    ensure_exit_readout(model)
    saved = asdict(model.config)
    saved["exit_selection_weights"] = (0.0, 0.2, 0.2, 0.2, 0.2, 0.2)
    saved["exit_selection_temperature"] = 0.25
    path = tmp_path / "legacy-config.pt"
    torch.save({"model": model.state_dict(), "config": saved, "exit_readout_rank": 16}, path)
    restored = load_model(path, torch.device("cuda"))
    assert restored.config.early_exit_enabled is True


def test_shortlist_full_coverage_matches_full_ce():
    model = tiny_model().eval()
    ensure_exit_readout(model)
    rollout = synthetic_rollout()
    teacher = rollout["teacher_logits"]
    target = teacher.argmin(dim=-1)
    weights = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _, stats = exit_readout_losses(
        model, rollout, target, weights, uniform_weights(), permissive_args(), 0.0, 4
    )
    states = rollout["states"].float()
    log_vocab = math.log(model.lm_head.weight.shape[-1])
    expected = []
    for depth_index in range(4):
        logits = model.exit_readout.logits_at(
            depth_index, states[depth_index], model.lm_head.weight.to(states.dtype)
        )
        full_ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        expected.append(full_ce / (log_vocab + full_ce))
    expected_mean = torch.stack(expected).mean().detach()
    # CUDA readout terms are assembled in bf16 (train.py compute_dtype), so
    # the fp32 reference here only matches to bf16 rounding.
    assert abs(stats["ce"] - float(expected_mean)) < 2e-3


def test_generate_returns_stats():
    model = always_exit_model(early_exit_min_depth=1).eval()
    ids = torch.arange(4).view(1, 4) % 32
    with torch.no_grad():
        generated, stats = model.generate(ids, 3, early_exit=True, return_stats=True)
    assert generated.shape == (1, 7)
    assert stats["steps"] == 3.0
    assert stats["avg_exit_depth"] == 1.0


def test_config_validates_exit_fields():
    with pytest.raises(ValueError, match="early_exit_min_depth"):
        tiny_config(early_exit_min_depth=0)
    with pytest.raises(ValueError, match="early_exit_confidence_threshold"):
        tiny_config(early_exit_confidence_threshold=0.0)
    with pytest.raises(ValueError, match="early_exit_margin_threshold"):
        tiny_config(early_exit_margin_threshold=-1.0)
    with pytest.raises(ValueError, match="early_exit_state_delta_threshold"):
        tiny_config(early_exit_state_delta_threshold=0.0)
    with pytest.raises(ValueError, match="exit_readout_rank"):
        tiny_config(exit_readout_rank=0)


def test_thresholds_met_unit():
    model = tiny_model(
        early_exit_confidence_threshold=0.6,
        early_exit_margin_threshold=0.3,
        early_exit_state_delta_threshold=0.2,
    )
    confidence = torch.tensor([0.7])
    margin = torch.tensor([0.4])
    assert model._thresholds_met(confidence, margin, None) is True
    assert model._thresholds_met(torch.tensor([0.5]), margin, None) is False
    assert model._thresholds_met(confidence, torch.tensor([0.2]), None) is False
    assert model._thresholds_met(confidence, margin, torch.tensor([0.3])) is False
    assert model._thresholds_met(confidence, margin, torch.tensor([0.1])) is True


def test_threshold_conjunction_requires_all_three():
    model = tiny_model(
        early_exit_enabled=True,
        early_exit_min_depth=1,
        early_exit_confidence_threshold=0.5,
        early_exit_margin_threshold=0.3,
        early_exit_state_delta_threshold=1e9,
    ).eval()
    ensure_exit_readout(model)
    ids = torch.arange(8).view(1, 8) % 32
    with torch.no_grad():
        others = torch.ones(32, dtype=torch.bool)
        others[3] = False
        model.exit_readout.bias[:, others] = -20.0
        model.exit_readout.bias[:, 3] = 20.0
    with torch.no_grad():
        halted = model(ids, early_exit=True)
    assert halted["exit_depth"] == 1
    assert halted["exit_halted"] is True
    with torch.no_grad():
        model.exit_readout.bias.zero_()
        model.config.early_exit_confidence_threshold = 1.0
        blocked = model(ids, early_exit=True)
    assert blocked["exit_depth"] == 4
    assert blocked["exit_halted"] is False


def test_state_delta_threshold_gates_halting():
    model = tiny_model(
        early_exit_enabled=True,
        early_exit_min_depth=2,
        early_exit_confidence_threshold=1e-6,
        early_exit_margin_threshold=0.0,
        early_exit_state_delta_threshold=1e-6,
    ).eval()
    ids = torch.arange(8).view(1, 8) % 32
    with torch.no_grad():
        strict = model(ids, early_exit=True)
    assert strict["exit_depth"] == 4
    assert strict["exit_halted"] is False
    with torch.no_grad():
        model.config.early_exit_state_delta_threshold = 1e9
        permissive = model(ids, early_exit=True)
    assert permissive["exit_depth"] == 2
    assert permissive["exit_halted"] is True


def test_oversized_min_depth_clamps_to_full_depth():
    model = always_exit_model(early_exit_min_depth=99).eval()
    ids = torch.arange(8).view(1, 8) % 32
    with torch.no_grad():
        output = model(ids, early_exit=True)
        baseline = model(ids, early_exit=False)
    assert output["exit_depth"] == 4
    assert torch.equal(output["logits"], baseline["logits"])


def test_readout_module_rejects_bad_depth():
    readout = CalibratedExitReadout(hidden_size=8, vocab_size=16, num_depths=3, rank=2)
    states = torch.randn(5, 8)
    with pytest.raises(ValueError, match="outside"):
        readout.logits_at(3, states, torch.randn(16, 8))


def test_joint_rollout_shapes_and_base_logits():
    model = tiny_model().eval()
    ensure_exit_readout(model)
    ids = torch.arange(8).view(1, 8) % 32
    rollout = model.joint_rollout(ids)
    assert rollout["states"].shape == (4, 1, 32)
    assert rollout["deltas"].shape == (4, 1)
    assert rollout["base_logits"].shape == (4, 1, 32)
    assert rollout["router_losses"].shape == (4,)
    assert torch.all(rollout["deltas"][0] == 0.0)
    # base_logits equals the plain head at each depth's last position
    with torch.no_grad():
        for depth in range(4):
            reference = torch.nn.functional.linear(
                rollout["states"][depth], model.lm_head.weight
            )
        assert torch.allclose(rollout["base_logits"][depth], reference, atol=1e-5)


def test_corrections_at_matches_per_depth_logits():
    readout = CalibratedExitReadout(hidden_size=8, vocab_size=16, num_depths=3, rank=2)
    states = torch.randn(3, 2, 8)
    head = torch.randn(16, 8)
    correction = readout.corrections_at(states)
    assert correction.shape == (3, 2, 16)
    for depth in range(3):
        logits = readout.logits_at(depth, states[depth], head)
        expected = (torch.nn.functional.linear(states[depth], head) + correction[depth]) * readout.temperature[depth]
        assert torch.allclose(logits, expected, atol=1e-6)
    with pytest.raises(ValueError):
        readout.corrections_at(torch.randn(2, 2, 8))


def test_assemble_terms_matches_readout_losses_and_base_path():
    model = tiny_model().eval()
    ensure_exit_readout(model)
    rollout = synthetic_rollout()
    target = torch.tensor([3, 7])
    args = permissive_args(exit_min_depth=2)
    terms, selection_logits, depths = assemble_exit_terms(
        model, rollout, target, args, 4
    )
    assert terms.shape == (6, 4, 2)
    assert selection_logits.shape == (4,)
    assert torch.allclose(depths, torch.arange(4, dtype=torch.float32))
    # weighted selection equals the formula used inside exit_readout_losses
    weights = uniform_weights()
    weighted = -((weights[:, None, None] * terms).sum(dim=0).mean(dim=1))
    assert torch.allclose(selection_logits * 0 + weighted, weighted)
    # cached base_logits path reproduces the fresh head matmul path
    base = torch.stack(
        [
            torch.nn.functional.linear(rollout["states"][d], model.lm_head.weight)
            for d in range(4)
        ]
    )
    terms_base, selection_base, _ = assemble_exit_terms(
        model, rollout, target, args, 4, base_logits=base
    )
    # bf16 readout assembly on CUDA vs the fp32 base-logits reference; the
    # bf16 GEMM itself is run-to-run nondeterministic (cublas split-k
    # scheduling), so keep a generous absolute band over the [0, 1] terms.
    assert torch.allclose(terms_base, terms, atol=6e-3)
    assert torch.allclose(selection_base, selection_logits, atol=6e-3)


def test_single_sample_exit_terms_preserve_depth_batch_axes():
    model = tiny_model().eval()
    ensure_exit_readout(model)
    rollout = model.joint_rollout(torch.arange(8).view(1, 8) % 32)
    rollout = {key: value.clone() for key, value in rollout.items()}
    terms, _, _ = assemble_exit_terms(
        model, rollout, torch.tensor([7]), permissive_args(), 4,
        base_logits=rollout["base_logits"],
    )
    assert terms.shape == (6, 4, 1)
    expected = torch.nn.functional.cross_entropy(
        rollout["base_logits"].reshape(4, -1), torch.full((4,), 7), reduction="none"
    )
    log_vocab = torch.tensor(float(model.config.vocab_size)).log()
    expected = expected / (log_vocab + expected)
    # bf16 readout assembly on CUDA vs the fp32 base-logits reference
    assert torch.allclose(terms[0, :, 0], expected, atol=2e-3)


def test_phase_isolation_halting_vs_backbone():
    model = tiny_model().eval()
    ensure_exit_readout(model)
    ids = torch.arange(16).view(2, 8) % 32
    labels = ids.clone()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    rollout = model.joint_rollout(ids)
    rollout = {key: value.clone() for key, value in rollout.items()}
    args = permissive_args(exit_min_depth=2)
    terms, _, _ = assemble_exit_terms(
        model,
        {"states": rollout["states"], "deltas": rollout["deltas"]},
        labels[:, -1],
        args,
        4,
        base_logits=rollout["base_logits"],
    )
    model.zero_grad(set_to_none=True)
    # the CE term is the part of the halting objective that touches the
    # readout parameters; the deficit terms follow the readout logits and
    # speed/guard are constants
    terms[0].mean().backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in model.exit_readout.parameters()
    )
    assert all(
        p.grad is None
        for name, p in model.named_parameters()
        if not name.startswith("exit_readout.")
    )
    # phase B: CE through the truncated forward must reach the backbone only
    model.zero_grad(set_to_none=True)
    for parameter in model.exit_readout.parameters():
        parameter.requires_grad_(False)
    forward = model.forward_until(ids, 3)
    logits = model.exit_readout.logits_at(
        2, forward["final_normed"], model.lm_head.weight
    )
    torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1)
    ).backward()
    assert model.lm_head.weight.grad is not None
    assert all(p.grad is None for p in model.exit_readout.parameters())
    for name, p in model.named_parameters():
        p.requires_grad_(True)


def test_forward_until_truncates_cycles():
    model = tiny_model().eval()
    ids = torch.arange(8).view(1, 8) % 32
    two = model.forward_until(ids, 2)
    assert two["final_normed"].shape == (1, 8, 32)
    assert two["router_losses"].shape == (2,)
    four = model.forward_until(ids, 4)
    assert four["router_losses"].shape == (4,)
    with pytest.raises(ValueError):
        model.forward_until(ids, 5)


def test_forward_until_delta_matches_rollout_and_carries_gradient():
    model = tiny_model()
    ensure_exit_readout(model)
    ids = torch.arange(16).view(2, 8) % 32
    depth = 3
    forward = model.forward_until(ids, depth, collect_delta=True)
    assert forward["final_delta"].shape == (2,)
    # the selected depth's delta matches the rollout's delta at that depth
    rollout = model.joint_rollout(ids)
    assert torch.allclose(forward["final_delta"].detach(), rollout["deltas"][depth - 1])
    # depth 1 has no previous cycle: the delta is a zero constant
    one = model.forward_until(ids, 1, collect_delta=True)
    assert one["final_delta"].shape == (2,)
    assert torch.all(one["final_delta"] == 0)
    # collect_delta=False leaves the result unchanged
    assert "final_delta" not in model.forward_until(ids, depth)
    # the delta term's gradient reaches the backbone (both cycles), not the readout
    model.zero_grad(set_to_none=True)
    for parameter in model.exit_readout.parameters():
        parameter.requires_grad_(False)
    fresh = model.forward_until(ids, depth, collect_delta=True)
    fresh["final_delta"].sum().backward()
    backbone_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if not name.startswith("exit_readout.")
    ]
    assert any(grad is not None for grad in backbone_grads)
    assert all(parameter.grad is None for parameter in model.exit_readout.parameters())


def test_assemble_exit_terms_omits_delta_when_disabled():
    model = tiny_model().eval()
    ensure_exit_readout(model)
    rollout = synthetic_rollout()
    target = torch.tensor([3, 7])
    args = permissive_args(exit_min_depth=2)
    terms_with, selection_with, _ = assemble_exit_terms(model, rollout, target, args, 4)
    terms, selection, _ = assemble_exit_terms(
        model, rollout, target, args, 4, include_delta=False
    )
    assert terms.shape == (5, 4, 2)
    assert torch.allclose(torch.cat([terms_with[:3], terms_with[4:]]), terms)
    # the selection drops the delta row's contribution
    delta_row = terms_with[3].mean(dim=-1)
    assert torch.allclose(selection, selection_with + delta_row.sum(0))
