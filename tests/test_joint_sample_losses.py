import torch
from train import joint_sample_losses, joint_token_rows
from test_early_exit import tiny_model


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
