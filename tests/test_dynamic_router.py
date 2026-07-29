import torch
from torch import nn
import numpy as np
from types import SimpleNamespace
from pathlib import Path

from dynamic_router import DynamicRouterLM, GrowthController, language_model_loss, train_step
from scripts.train_c4 import TokenStore, sample_block_pairs, sequential_batches


def test_module_uses_gru_qkv_and_bidirectional_attention():
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    module = model.pool[0]
    assert isinstance(module.gru, nn.GRU)
    assert module.q_proj.in_features == 16
    assert module.k_proj.in_features == 16
    assert module.v_proj.in_features == 16
    x = torch.randn(2, 8, 16)
    hidden = torch.zeros(1, 2, 16)
    output, next_hidden = module(x, hidden)
    assert output.shape == x.shape
    assert next_hidden.shape == hidden.shape


def test_gru_recurrent_branch_stays_float32_under_autocast():
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    dtypes = []

    def capture(_, inputs, output):
        dtypes.extend([output[0].dtype, output[1].dtype])

    handle = model.pool[0].gru.register_forward_hook(capture)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        model.pool[0](torch.randn(1, 8, 16), torch.zeros(1, 1, 16))
    handle.remove()
    assert dtypes == [torch.float32, torch.float32]


def test_module_initialization_uses_xavier_and_orthogonal_gru_families():
    torch.manual_seed(7)
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    module = model.pool[0]
    assert torch.allclose(module.q_proj.bias, torch.zeros_like(module.q_proj.bias))
    assert torch.allclose(module.attn_norm.weight, torch.ones_like(module.attn_norm.weight))
    assert torch.allclose(module.ffn_norm.bias, torch.zeros_like(module.ffn_norm.bias))
    for name, parameter in module.gru.named_parameters():
        if "bias" in name:
            assert torch.allclose(parameter, torch.zeros_like(parameter))
        else:
            assert torch.isfinite(parameter).all()




def test_large_default_training_configuration_is_half_budget():
    model = DynamicRouterLM(d_model=256, vocab_size=32, max_modules=6, n_heads=8)
    assert sum(p.numel() for p in model.pool[0].parameters()) == 1_382_400


def test_block_forward_and_fixed_memory_shapes():
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, block_size=8, n_heads=4)
    x = torch.randn(3, 8, 16)
    logits, aux = model(x)
    assert logits.shape == (3, 8, 32)
    assert aux["route_lengths"].shape == (3,)
    assert aux["memory_state"].shape == (3, 6, 16)
    assert aux["valid_sample_mask"].tolist() == [True, True, True]


def test_training_route_uses_unique_top_three_targets():
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    _, aux = model(torch.randint(0, 32, (1, 8)))
    for _, targets, _, _ in aux["route_targets"]:
        assert len(targets) <= 3
        assert len(targets) == len(set(targets))


def test_each_branch_path_has_unique_modules_and_bounded_length():
    model = DynamicRouterLM(
        d_model=16, vocab_size=32, max_modules=6, n_heads=4, max_depth=32
    )
    _, aux = model(torch.randint(0, 32, (1, 8)))
    assert aux["route_lengths"].item() <= model.active_count
    assert aux["branch_paths"]
    for sample_index, path in aux["branch_paths"]:
        assert sample_index == 0
        assert len(path) == len(set(path))
        assert path[0] == model.input_id
        assert path[-1] == model.output_id


def test_cooldown_source_includes_new_slot_in_top_three():
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    slot = model.activate_next()
    model.set_cooldown_connection(0, slot)
    _, aux = model(torch.randint(0, 32, (1, 8)))
    source_targets = [targets for source, targets, _, _ in aux["route_targets"] if source == 0]
    assert source_targets
    assert slot in source_targets[0]


def test_memory_state_is_per_sample_and_persistent():
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    x = torch.randn(2, 8, 16)
    _, first = model(x)
    assert first["memory_state"].shape == (2, 6, 16)
    _, second = model(x, memory_state=first["memory_state"].detach())
    assert not torch.equal(first["memory_state"], second["memory_state"])


def test_growth_only_activates_preallocated_slot():
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    before = model.active_count
    slot = model.activate_next()
    assert slot == 3
    assert model.active_count == before + 1
    assert model.pool[slot] in model.pool


def test_training_smoke():
    torch.manual_seed(0)
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    controller = GrowthController(gradient_window=48, cooldown=1)
    x = torch.randn(2, 8, 16)
    y = torch.randint(0, 32, (2, 8))
    metrics = train_step(model, optimizer, x, y, controller)
    assert "lm_loss" in metrics
    assert metrics["new_slot"] >= -1
    assert metrics["memory_state"].shape == (2, 6, 16)


def test_token_ids_use_model_embedding():
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    x = torch.randint(0, 32, (2, 8))
    logits, _ = model(x)
    assert logits.shape == (2, 8, 32)
    assert model.position_embedding.shape == (8, 16)


def test_eval_forward_works_without_gradients():
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    model.eval()
    with torch.no_grad():
        logits, aux = model(torch.randint(0, 32, (2, 8)))
    assert logits.shape == (2, 8, 32)
    assert len(aux["terminations"]) == 2


def test_eval_forward_is_deterministic():
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4).eval()
    x = torch.randint(0, 32, (1, 8))
    with torch.no_grad():
        first, _ = model(x)
        second, _ = model(x)
    assert torch.equal(first, second)


def test_language_loss_reaches_route_logits():
    torch.manual_seed(1)
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    x = torch.randint(0, 32, (1, 8))
    y = torch.randint(0, 32, (1, 8))
    logits, aux = model(x)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
    loss.backward()
    assert any(record.gradient_norm is not None for record in aux["route_records"])
    assert model.router.grad is not None


def test_route_gradients_are_masked_outside_top_three():
    torch.manual_seed(2)
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    x = torch.randint(0, 32, (1, 8))
    logits, aux = model(x)
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), x.flatten()).backward()
    routed = [
        (record, targets)
        for record, target_info in zip(aux["route_records"], aux["route_targets"])
        for source, targets, sample_index, depth in [target_info]
        if record.source_id == source
        and record.sample_index == sample_index
        and record.depth == depth
    ]
    assert routed
    for record, targets in routed:
        assert record.gradient_norm is not None


def test_target_visited_group_batch_matches_loop_forward_and_backward():
    torch.manual_seed(3)
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    target = 1
    q_values = [torch.randn(1, 8, 16, requires_grad=True) for _ in range(2)]
    h_values = [torch.randn(1, 1, 16, requires_grad=True) for _ in range(2)]
    inputs = list(zip(q_values, h_values))

    loop_q = torch.stack([q for q, _ in inputs]).mean(dim=0)
    loop_hidden = torch.stack([hidden for _, hidden in inputs]).mean(dim=0)
    loop_output, loop_next_hidden = model.pool[target](loop_q, loop_hidden)
    loop_loss = loop_output.square().mean() + loop_next_hidden.square().mean()
    loop_loss.backward()
    loop_grads = [parameter.grad.detach().clone() for parameter in model.pool[target].parameters()]

    model.zero_grad(set_to_none=True)
    batch_inputs = [
        (q.detach().clone().requires_grad_(), hidden.detach().clone().requires_grad_())
        for q, hidden in inputs
    ]
    batch_results = model._execute_target_groups_batch(target, [batch_inputs])
    batch_output, batch_hidden = batch_results[0]
    batch_loss = batch_output.square().mean() + batch_hidden.square().mean()
    batch_loss.backward()
    batch_grads = [parameter.grad.detach().clone() for parameter in model.pool[target].parameters()]

    assert torch.allclose(batch_output.detach(), loop_output.detach())
    assert torch.allclose(batch_hidden.detach(), loop_next_hidden.detach())
    assert len(loop_grads) == len(batch_grads)
    assert all(torch.allclose(left, right) for left, right in zip(loop_grads, batch_grads))


def test_full_forward_batch_execution_matches_loop_baseline():
    torch.manual_seed(4)
    batch_model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    loop_model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    loop_model.load_state_dict(batch_model.state_dict())
    loop_model.use_batched_group_execution = False
    x = torch.randint(0, 32, (2, 8))
    batch_logits, batch_aux = batch_model(x)
    loop_logits, loop_aux = loop_model(x)
    assert torch.allclose(batch_logits, loop_logits)
    assert batch_aux["route_targets"] == loop_aux["route_targets"]
    assert batch_aux["branch_paths"] == loop_aux["branch_paths"]

    batch_loss = batch_logits.square().mean()
    loop_loss = loop_logits.square().mean()
    batch_loss.backward()
    loop_loss.backward()
    batch_route_grads = [record.gradient_norm for record in batch_aux["route_records"]]
    loop_route_grads = [record.gradient_norm for record in loop_aux["route_records"]]
    assert len(batch_route_grads) == len(loop_route_grads)
    assert all(
        left is not None and right is not None and abs(left - right) < 1e-5
        for left, right in zip(batch_route_grads, loop_route_grads)
    )


def test_train_step_returns_detached_memory():
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    result = train_step(model, optimizer, x, y)
    assert result["memory_state"].shape == (2, 6, 16)
    assert not result["memory_state"].requires_grad


def test_language_loss_reports_weighted_length_loss():
    logits = torch.randn(1, 8, 32)
    targets = torch.randint(0, 32, (1, 8))
    expected = torch.tensor(3.0)
    total, metrics = language_model_loss(
        logits, targets, expected, length_weight=0.25
    )
    assert torch.isclose(metrics["length_loss"], torch.tensor(0.75))
    assert torch.isclose(metrics["expected_length"], expected)
    assert torch.isclose(total, metrics["lm_loss"] + metrics["length_loss"])


def test_sample_pairs_are_next_blocks_with_padding_mask():
    pairs = list(sample_block_pairs(np.arange(20, dtype=np.uint32), 8, 50256))
    assert len(pairs) == 2
    assert pairs[0][0] == list(range(8))
    assert pairs[0][1] == list(range(8, 16))
    assert pairs[1][0] == list(range(8, 16))
    assert pairs[1][1] == list(range(16, 20)) + [50256] * 4
    assert pairs[1][2] == [True, True, True, True, False, False, False, False]


def test_masked_loss_ignores_padding_targets():
    logits = torch.zeros(1, 8, 4)
    targets = torch.tensor([[0, 1, 2, 3, 3, 3, 3, 3]])
    mask = torch.tensor([[True, True, True, True, False, False, False, False]])
    _, metrics = language_model_loss(logits, targets, torch.tensor(0.0), target_mask=mask)
    expected = torch.nn.functional.cross_entropy(logits[:, :4].reshape(-1, 4), targets[:, :4].reshape(-1))
    assert torch.isclose(metrics["lm_loss"], expected)


def test_sample_mask_ignores_finished_lanes_for_length_loss():
    logits = torch.zeros(2, 8, 4)
    targets = torch.zeros(2, 8, dtype=torch.long)
    sample_mask = torch.tensor([True, False])
    _, metrics = language_model_loss(
        logits,
        targets,
        torch.tensor(2.0),
        sample_mask=sample_mask,
    )
    assert torch.isclose(metrics["expected_length"], torch.tensor(2.0))


def test_sequential_batches_use_offsets_and_finish_lanes(tmp_path: Path):
    tokens = np.arange(20, dtype=np.uint32)
    tokens.tofile(tmp_path / "train-00000.bin")
    offsets = np.asarray([0, 20], dtype=np.uint64)
    offsets.tofile(tmp_path / "samples.offsets")
    batches = list(
        sequential_batches(
            TokenStore([tmp_path / "train-00000.bin"]),
            np.memmap(tmp_path / "samples.offsets", dtype=np.uint64, mode="r"),
            batch_size=1,
            block_size=8,
            pad_token_id=31,
            limit=10,
        )
    )
    assert len(batches) == 2
    assert batches[0][0].tolist() == [list(range(8))]
    assert batches[0][1].tolist() == [list(range(8, 16))]
    assert batches[1][0].tolist() == [list(range(8, 16))]
    assert batches[1][1].tolist() == [list(range(16, 20)) + [31] * 4]


def test_invalid_model_dimensions_are_rejected():
    for kwargs in (
        {"block_size": 0},
        {"max_depth": 0},
        {"n_heads": 0},
    ):
        try:
            DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid configuration accepted: {kwargs}")


def test_route_has_safe_termination():
    model = DynamicRouterLM(
        d_model=16, vocab_size=32, max_modules=6, max_depth=4, n_heads=4
    )
    x = torch.randint(0, 32, (2, 8))
    _, aux = model(x)
    assert all(length <= 4 for length in aux["route_lengths"].tolist())
    assert all(reason in {"output", "cycle", "max_depth", "dead_end"} for reason in aux["terminations"])


def test_nonfinite_route_logits_are_reported_separately():
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    with torch.no_grad():
        model.router.fill_(float("nan"))
    _, aux = model(torch.randint(0, 32, (2, 8)))
    assert aux["invalid_reasons"] == ["nonfinite_logits", "nonfinite_logits"]
    assert aux["terminations"] == ["nonfinite_logits", "nonfinite_logits"]


def test_nonfinite_route_skips_update_and_clears_memory():
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    with torch.no_grad():
        model.router.fill_(float("nan"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    memory = torch.full((2, 6, 16), float("nan"))
    result = train_step(
        model,
        optimizer,
        torch.randint(0, 32, (2, 8)),
        torch.randint(0, 32, (2, 8)),
        memory_state=memory,
    )
    assert result["skipped_update"] == 1
    assert all(reason == "nonfinite_logits" for reason in result["invalid_reasons"])
    assert torch.isfinite(result["memory_state"]).all()


def test_inactive_modules_do_not_receive_gradients_or_optimizer_updates():
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    inactive_before = model.pool[4].ffn[0].weight.detach().clone()
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    train_step(model, optimizer, x, y)
    assert model.pool[4].ffn[0].weight.grad is None
    assert torch.equal(model.pool[4].ffn[0].weight, inactive_before)


def test_growth_controller_state_round_trip():
    controller = GrowthController(gradient_window=48, cooldown=3)
    controller.remaining_cooldown = 2
    controller.previous_gradient_norm_ema[1] = 1.0
    controller.previous_velocity_ema[1] = -0.25
    controller.velocity_history[1].append(-0.5)
    restored = GrowthController(gradient_window=48, cooldown=3)
    restored.load_state_dict(controller.state_dict())
    assert restored.remaining_cooldown == 2
    assert restored.previous_gradient_norm_ema[1] == 1.0
    assert restored.previous_velocity_ema[1] == -0.25
    assert list(restored.velocity_history[1]) == [-0.5]


def test_cooldown_reports_velocity_without_affecting_growth_state():
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    controller = GrowthController(gradient_window=2, cooldown=2)
    controller.remaining_cooldown = 2
    first_logits = torch.ones(1, 6, requires_grad=True)
    first_logits.grad = torch.ones_like(first_logits)
    controller.observe_and_maybe_grow(
        model, [SimpleNamespace(source_id=0, logits=first_logits)]
    )
    assert controller.last_diagnostics["diagnostic_only"] is True
    assert controller.last_diagnostics["velocity_min"] is None
    assert controller.previous_gradient_norm_ema == {}
    second_logits = torch.full((1, 6), 0.5, requires_grad=True)
    second_logits.grad = torch.ones_like(second_logits) * 0.5
    controller.observe_and_maybe_grow(
        model, [SimpleNamespace(source_id=0, logits=second_logits)]
    )
    assert torch.isclose(
        torch.tensor(controller.last_diagnostics["velocity_min"]),
        torch.tensor(-0.25 * (6**0.5)),
    )
    assert controller.previous_gradient_norm_ema == {}
    assert controller.velocity_history == {}


def test_velocity_ema_is_used_for_growth_signal():
    controller = GrowthController(gradient_window=2)
    assert controller.previous_velocity_ema == {}


def test_growth_signal_uses_route_length_normalization_and_relative_velocity():
    controller = GrowthController(gradient_window=1)
    controller.previous_gradient_norm_ema[0] = 10.0
    logits = torch.zeros(1, 6, requires_grad=True)
    logits.grad = torch.ones_like(logits) * 4.0
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    controller.observe_and_maybe_grow(
        model,
        [SimpleNamespace(source_id=0, logits=logits, sample_index=0)],
        torch.tensor([2]),
    )
    # The normalized norm is 4 * sqrt(6) / 2, so velocity is its absolute
    # difference from the previous EMA.
    assert torch.isclose(
        torch.tensor(controller.last_diagnostics["velocity_min"]),
        torch.tensor(4 * (6**0.5) / 2 - 10),
    )
    controller.previous_gradient_norm_ema[0] = 1.0
    controller.previous_velocity_ema[0] = -1.0
    controller.velocity_history[0].extend([-1.0, -1.0])
    logits = torch.zeros(1, 6, requires_grad=True)
    logits.grad = torch.ones_like(logits) * 2.0
    model = DynamicRouterLM(d_model=16, vocab_size=32, max_modules=6, n_heads=4)
    slot = controller.observe_and_maybe_grow(
        model, [SimpleNamespace(source_id=0, logits=logits)]
    )
    assert slot == 3
    assert controller.previous_gradient_norm_ema == {}
    assert controller.previous_velocity_ema == {}
