import argparse
import math

import pytest
import torch

from memory_transformer import ensure_exit_readout
from train import (
    _joint_prior_and_kappa,
    assemble_exit_terms,
    joint_sample_losses,
    joint_token_rows,
)
from test_early_exit import permissive_args, tiny_model


def test_sample_choices_and_gradient_scopes():
    terms = torch.full((6, 4, 2), 0.7)
    terms[:, 1, 0] = 0.2
    terms[:, 3, 1] = 0.1
    terms.requires_grad_()
    weights = torch.tensor([0.65, 0.05, 0.05, 0.05, 0.15, 0.05])
    a1, a2, depths, p, _ = joint_sample_losses(terms, weights, 0.25, 0.01)
    assert depths.tolist() == [2, 4]
    assert p.argmax(0).tolist() == [1, 3]
    g1 = torch.autograd.grad(a1, terms, retain_graph=True)[0]
    assert g1[:, 0].abs().sum() == 0
    assert g1[:, 2].abs().sum() == 0
    assert g1[:, 1, 1].abs().sum() == 0
    assert g1[:, 3, 0].abs().sum() == 0
    g2 = torch.autograd.grad(a2, terms)[0]
    assert (g2.abs().sum(0) > 0).all()


def test_uniform_costs_give_uniform_selection_and_max_entropy():
    terms = torch.full((6, 4, 2), 0.7, requires_grad=True)
    _, a2, depths, p, entropy = joint_sample_losses(terms, torch.ones(6)/6, 0.25, 0.01)
    assert torch.allclose(p, torch.full_like(p, 0.25))
    assert torch.allclose(entropy, torch.tensor(1.0))
    assert torch.allclose(a2, torch.log(torch.tensor(4.0)) + 0.01)


def test_selection_loss_direction_and_entropy_sharpening():
    weights = torch.ones(6)/6
    terms = torch.tensor([0.7, 0.2, 0.6])[None, :, None].expand(6, -1, -1).clone().requires_grad_()
    _, ce, depths, p, _ = joint_sample_losses(terms, weights, 0.25, 0)
    grad = torch.autograd.grad(ce, terms)[0].sum(0).flatten()
    assert depths.tolist() == [2]
    assert grad[1] > 0  # Gradient descent lowers the selected cost.
    assert grad[0] < 0 and grad[2] < 0  # Self-label CE raises alternatives.
    lower = terms.detach().clone(); lower[:, 1] -= 0.05
    assert joint_sample_losses(lower, weights, 0.25, 0)[1] < ce
    flat = torch.full_like(terms, 0.5)
    h_flat = joint_sample_losses(flat, weights, 0.25, 1)[-1]
    h_sharp = joint_sample_losses(lower, weights, 0.25, 1)[-1]
    assert h_sharp < h_flat


def test_selected_cost_is_plain_weighted_mean():
    terms = torch.full((6, 3, 2), 0.5)
    terms[:, 1, 0] = 0.1
    weights = torch.tensor([0.65, 0.05, 0.05, 0.05, 0.15, 0.05])
    a1, _, depths, _, _ = joint_sample_losses(terms, weights, 0.25, 0.01)
    expected_sample0 = (terms[:, 1, 0] * weights).sum()
    expected_sample1 = (terms[:, 0, 1] * weights).sum()
    assert depths.tolist() == [2, 1]
    assert torch.allclose(a1, (expected_sample0 + expected_sample1) / 2)


def test_batched_inference_matches_independent_samples():
    model = tiny_model(early_exit_min_depth=2).eval()
    ids = torch.arange(16).view(2, 8) % 32
    with torch.no_grad():
        out = model(ids, early_exit=True)
        singles = [model(row[None], early_exit=True) for row in ids]
    assert out["exit_depth"] == [o["exit_depth"] for o in singles]
    assert torch.allclose(out["logits"], torch.cat([o["logits"] for o in singles]))


def test_hard_floor_excludes_first_depth_and_preserves_soft_guard():
    weights = torch.tensor([0.65, 0.05, 0.05, 0.05, 0.15, 0.05])
    terms = torch.full((6, 24, 2), 0.7)
    terms[:, 0] = 0.0  # Would win without the hard constraint.
    terms[:, 1] = 0.1
    terms[5] = ((3 - torch.arange(1, 25)) / 24).clamp_min(0)[:, None]
    terms.requires_grad_()
    a1, a2, depths, p, entropy = joint_sample_losses(
        terms, weights, 0.25, 0.05, hard_min_depth=2)
    assert depths.tolist() == [2, 2]
    assert p.shape == (23, 2)
    assert torch.allclose(a1, (weights[:, None] * terms[:, 1]).sum(0).mean())
    grad = torch.autograd.grad(a1 + a2, terms)[0]
    assert grad[:, 0].count_nonzero() == 0
    assert grad[5, 1].abs().sum() > 0
    assert torch.isfinite(entropy)


def test_hard_floor_uniform_entropy_and_expected_depth():
    terms = torch.full((6, 24, 2), 0.5)
    _, a2, _, p, entropy = joint_sample_losses(
        terms, torch.ones(6)/6, 0.25, 0.05, hard_min_depth=2)
    assert torch.allclose(entropy, torch.tensor(1.0))
    assert torch.allclose(a2, torch.log(torch.tensor(23.0)) + 0.05)
    expected = (p * torch.arange(2, 25)[:, None]).sum(0)
    assert torch.allclose(expected, torch.full((2,), 13.0))


def test_joint_sequence_ce_uses_already_shifted_targets_and_last_position():
    # Iterator contract: inputs=[A,B,C,D], labels=[B,C,D,E].
    # Each position strongly predicts its next token, including the last one.
    logits = torch.full((2, 4, 6), -8.0)
    labels = torch.tensor([[1, 2, 3, 4], [2, 3, -100, 5]])
    for batch in range(2):
        for position in range(4):
            if labels[batch, position] >= 0:
                logits[batch, position, labels[batch, position]] = 8.0
    logits.requires_grad_()
    rows, targets = joint_token_rows(logits, labels)
    loss = torch.nn.functional.cross_entropy(rows, targets, ignore_index=-100,
                                             reduction="sum") / labels.ne(-100).sum()
    assert loss < 0.00001
    wrong_loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 6), labels[:, 1:].reshape(-1), ignore_index=-100)
    assert wrong_loss > 10
    loss.backward()
    assert logits.grad[:, -1].abs().sum() > 0
    assert logits.grad[1, 2].count_nonzero() == 0


def test_selection_delta_term_matches_six_term_reference():
    depth, batch = 4, 3
    torch.manual_seed(7)
    terms6 = torch.rand(6, depth, batch) * 0.5  # [CE, conf, margin, delta, speed, guard]
    weights6 = torch.tensor([0.65, 0.05, 0.05, 0.05, 0.15, 0.05])
    costs_old = (weights6[:, None, None] * terms6).sum(0)
    floor = 1  # hard_min_depth 2
    chosen_old0 = costs_old[floor:].argmin(0) + floor
    terms5 = torch.cat([terms6[:3], terms6[4:]])
    weights5 = torch.cat([weights6[:3], weights6[4:]])
    a1, _, chosen_new, _, _ = joint_sample_losses(
        terms5, weights5, 0.25, 0.05, hard_min_depth=2,
        delta_costs=terms6[3], kappa=float(weights6[3]),
    )
    assert torch.equal(chosen_new, chosen_old0 + 1)  # the function returns 1-based depths
    selection_costs = (weights5[:, None, None] * terms5).sum(0) + float(weights6[3]) * terms6[3]
    assert torch.allclose(selection_costs, costs_old, atol=1e-6)
    # A1 is the five-term cost at the chosen depth, not the selection cost
    a1_reference = (weights5[:, None, None] * terms5).sum(0)[chosen_old0, torch.arange(batch)].mean()
    assert torch.allclose(a1, a1_reference, atol=1e-6)


def test_delta_costs_do_not_enter_loss_a():
    depth, batch = 4, 2
    terms = torch.zeros(5, depth, batch)
    terms[0] = torch.tensor([0.1, 0.2, 0.3, 0.4])[:, None].expand(depth, batch)
    weights = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    delta_costs = torch.tensor([10.0, 5.0, 0.3, 0.9])[:, None].expand(depth, batch).clone()
    a1, a2, chosen, p, entropy = joint_sample_losses(
        terms, weights, 0.25, 0.05, hard_min_depth=1,
        delta_costs=delta_costs, kappa=1.0,
    )
    # the state-change term moves the argmin to the 0-based depth 2 (returned 3)
    assert torch.equal(chosen, torch.full((batch,), 3))
    # A1 and the selection distribution are computed from the five-term cost
    assert torch.allclose(a1, torch.tensor(0.3), atol=1e-6)
    log_p_ref = (-torch.tensor([0.1, 0.2, 0.3, 0.4]) / 0.25).log_softmax(0)
    p_ref = log_p_ref.exp()
    assert torch.allclose(p, p_ref[:, None].expand(depth, batch), atol=1e-6)
    ce_ref = -log_p_ref[2]
    entropy_ref = -(p_ref * log_p_ref).sum() / math.log(depth)
    assert torch.allclose(a2, ce_ref + 0.05 * entropy_ref, atol=1e-6)


def test_delta_term_steers_selection_deeper():
    model = tiny_model().eval()
    ensure_exit_readout(model)
    ids = torch.arange(16).view(2, 8) % 32
    rollout = model.joint_rollout(ids)
    rollout = {key: value.clone() for key, value in rollout.items()}
    target = torch.tensor([7, 11])
    args = permissive_args(exit_min_depth=2)
    terms, _, _ = assemble_exit_terms(
        model, {"states": rollout["states"], "deltas": rollout["deltas"]},
        target, args, 4, base_logits=rollout["base_logits"], include_delta=False,
    )
    weights = torch.tensor([0.65, 0.06, 0.06, 0.17, 0.06])
    _, _, base_chosen, _, _ = joint_sample_losses(terms, weights, 0.25, 0.05, hard_min_depth=2)
    # punish the three shallower candidates so the argmin must land deepest
    delta_costs = torch.zeros(4, 2)
    delta_costs[:3] = 1e3
    _, _, steered_chosen, _, _ = joint_sample_losses(
        terms, weights, 0.25, 0.05, hard_min_depth=2,
        delta_costs=delta_costs, kappa=1.0,
    )
    assert torch.equal(steered_chosen, torch.full((2,), 4))


def test_joint_prior_validation():
    ok = argparse.Namespace(
        loss_weight_prior="0.65,0.06,0.06,0.17,0.06",
        exit_selection_delta_weight=0.05,
    )
    prior, kappa = _joint_prior_and_kappa(ok)
    assert torch.allclose(prior, torch.tensor([0.65, 0.06, 0.06, 0.17, 0.06]))
    assert kappa == 0.05
    with pytest.raises(ValueError, match="five values"):
        _joint_prior_and_kappa(argparse.Namespace(
            loss_weight_prior="0.0,0.15,0.15,0.10,0.45,0.15",  # legacy six-value input
            exit_selection_delta_weight=0.05,
        ))
    with pytest.raises(ValueError, match="five values"):
        _joint_prior_and_kappa(argparse.Namespace(
            loss_weight_prior="0.5,0.2,0.2,0.2,0.2",  # sums to 1.3
            exit_selection_delta_weight=0.05,
        ))
    with pytest.raises(ValueError, match="non-negative"):
        _joint_prior_and_kappa(argparse.Namespace(
            loss_weight_prior="0.65,0.06,0.06,0.17,0.06",
            exit_selection_delta_weight=-0.1,
        ))
