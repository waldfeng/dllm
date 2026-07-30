"""Training utilities and second-order proxy growth control."""

from __future__ import annotations

from collections import defaultdict, deque

import torch
from torch import Tensor

from .model import DynamicRouterLM, language_model_loss


def _finite_gradients(model: DynamicRouterLM) -> bool:
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def route_gradient_diagnostics(
    records: list,
    route_lengths: Tensor | None,
    previous_ema: dict[int, float],
    previous_velocity_ema: dict[int, float],
    beta: float = 0.9375,
) -> dict[str, float | int]:
    """Return compact route-gradient and velocity statistics.

    The returned values are aggregate scalars, so logging remains independent
    of batch size. Gradient and velocity both use fixed-window EMAs of 16.
    """
    source_values: dict[int, list[float]] = defaultdict(list)
    all_norms: list[float] = []
    for record in records:
        norm = getattr(record, "gradient_norm", None)
        if norm is None:
            continue
        sample_index = getattr(record, "sample_index", 0)
        path_length = getattr(record, "path_length", None)
        if path_length is None and route_lengths is not None and sample_index < len(route_lengths):
            path_length = max(1.0, float(route_lengths[sample_index]))
        normalized = float(norm) / max(1.0, float(path_length or 1.0))
        source_values[record.source_id].append(normalized)
        all_norms.append(normalized)

    velocities: list[float] = []
    for source, values in source_values.items():
        current = sum(values) / len(values)
        previous = previous_ema.get(source)
        ema = current if previous is None else beta * previous + (1.0 - beta) * current
        if previous is not None:
            velocity = ema - previous
            old_velocity = previous_velocity_ema.get(source)
            velocity_ema = velocity if old_velocity is None else beta * old_velocity + (1.0 - beta) * velocity
            previous_velocity_ema[source] = velocity_ema
            velocities.append(velocity_ema)
        previous_ema[source] = ema

    return {
        "route_gradient_norm_sum": sum(all_norms),
        "route_gradient_norm_count": len(all_norms),
        "route_gradient_norm_max": max(all_norms, default=0.0),
        "route_velocity_sum": sum(velocities),
        "route_velocity_count": len(velocities),
        "route_velocity_min": min(velocities, default=0.0),
        "route_velocity_max": max(velocities, default=0.0),
        "route_observed_sources": len(source_values),
    }


def _gradient_diagnostics(model: DynamicRouterLM) -> dict:
    bad = []
    largest = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        finite = torch.isfinite(gradient)
        if not bool(finite.all()):
            bad.append(
                {
                    "name": name,
                    "shape": list(gradient.shape),
                    "dtype": str(gradient.dtype),
                    "nan_count": int(torch.isnan(gradient).sum()),
                    "inf_count": int(torch.isinf(gradient).sum()),
                    "finite_max_abs": (
                        float(gradient[finite].abs().max()) if bool(finite.any()) else None
                    ),
                }
            )
        elif gradient.numel():
            largest.append((float(gradient.abs().max()), name))
    largest.sort(reverse=True)
    return {
        "bad_gradients": bad,
        "largest_finite_gradients": [
            {"name": name, "max_abs": value} for value, name in largest[:5]
        ],
    }


def _invalid_result(
    aux: dict,
    memory_state: Tensor | None,
    reason: str | None = None,
    diagnostics: dict | None = None,
) -> dict[str, float | int | Tensor | list[str]]:
    invalid_reasons = aux["invalid_reasons"]
    if reason is not None:
        invalid_reasons = [reason for _ in invalid_reasons]
    safe_memory = (
        torch.zeros_like(aux["memory_state"])
        if memory_state is None
        else memory_state.detach().clone()
    )
    return {
        "loss": 0.0,
        "lm_loss": 0.0,
        "length_loss": 0.0,
        "expected_length": float(aux["expected_length"].detach()),
        "mean_route_length": float(aux["route_lengths"].float().mean()),
        "new_slot": -1,
        "terminations": aux["terminations"],
        "invalid_reasons": invalid_reasons,
        "skipped_update": 1,
        "memory_state": safe_memory.detach(),
        "diagnostics": diagnostics or aux.get("forward_diagnostics", {}),
        "middle_route_counts": {},
    }


class MiddleModuleUtilization:
    """Track middle-module target usage over a sliding step window."""

    def __init__(self, window_size: int = 1000) -> None:
        self.window_size = window_size
        self.window: deque[dict[int, int]] = deque()
        self.counts: dict[int, int] = defaultdict(int)

    def state_dict(self) -> dict:
        return {
            "window_size": self.window_size,
            "window": list(self.window),
            "counts": dict(self.counts),
        }

    def load_state_dict(self, state: dict) -> None:
        self.window_size = int(state.get("window_size", self.window_size))
        self.window = deque(
            ({int(key): int(value) for key, value in item.items()} for item in state.get("window", [])),
        )
        self.counts = defaultdict(int)
        self.counts.update({int(key): int(value) for key, value in state.get("counts", {}).items()})

    def update(self, counts: dict[int, int], active_middle: Tensor) -> dict[str, float | int]:
        current = {int(key): int(value) for key, value in counts.items() if value}
        self.window.append(current)
        for key, value in current.items():
            self.counts[key] += value
        while len(self.window) > self.window_size:
            expired = self.window.popleft()
            for key, value in expired.items():
                self.counts[key] -= value
                if self.counts[key] <= 0:
                    self.counts.pop(key, None)
        module_ids = [int(key) for key in torch.where(active_middle)[0].tolist()]
        values = torch.tensor(
            [float(self.counts.get(key, 0)) for key in module_ids], dtype=torch.float64
        )
        total = float(values.sum())
        used = int((values > 0).sum())
        module_count = len(module_ids)
        if module_count == 0 or total == 0:
            rate = cv = max_share = 0.0
        else:
            mean = float(values.mean())
            std = float(values.std(unbiased=False))
            rate = used / module_count
            cv = std / mean if mean else 0.0
            max_share = float(values.max()) / total
        return {
            "middle_window_size": len(self.window),
            "middle_utilization_rate": rate,
            "middle_utilization_cv": cv,
            "middle_max_share": max_share,
        }


class GrowthController:
    """Activate capacity when route-gradient velocity stays below a threshold.

    The finite difference of consecutive route-logit gradients is used as a
    practical proxy for a second-order signal. It avoids building a full Hessian
    through a sampled dynamic execution path.
    """

    def __init__(
        self,
        gradient_window: int = 48,
        cooldown: int = 100,
    ) -> None:
        if gradient_window <= 0 or cooldown < 0:
            raise ValueError("invalid growth controller configuration")
        self.gradient_window = gradient_window
        self.cooldown = cooldown
        self.remaining_cooldown = 0
        self.cooldown_source: int | None = None
        self.cooldown_slot: int | None = None
        self.previous_gradient_norm_ema: dict[int, float] = {}
        self.diagnostic_previous_gradient_norm_ema: dict[int, float] = {}
        self.previous_velocity_ema: dict[int, float] = {}
        self.diagnostic_previous_velocity_ema: dict[int, float] = {}
        self.last_diagnostics: dict[str, float | int | None] = {}

    def state_dict(self) -> dict:
        return {
            "remaining_cooldown": self.remaining_cooldown,
            "cooldown_source": self.cooldown_source,
            "cooldown_slot": self.cooldown_slot,
            "previous_gradient_norm_ema": dict(self.previous_gradient_norm_ema),
            "diagnostic_previous_gradient_norm_ema": dict(self.diagnostic_previous_gradient_norm_ema),
            "gradient_window": 16,
            "previous_velocity_ema": dict(self.previous_velocity_ema),
            "diagnostic_previous_velocity_ema": dict(self.diagnostic_previous_velocity_ema),
        }

    def load_state_dict(self, state: dict) -> None:
        self.remaining_cooldown = int(state.get("remaining_cooldown", 0))
        self.cooldown_source = state.get("cooldown_source")
        self.cooldown_slot = state.get("cooldown_slot")
        self.previous_gradient_norm_ema = {
            int(key): float(value)
            for key, value in state.get("previous_gradient_norm_ema", {}).items()
        }
        self.diagnostic_previous_gradient_norm_ema = {
            int(key): float(value)
            for key, value in state.get("diagnostic_previous_gradient_norm_ema", {}).items()
        }
        self.previous_velocity_ema = {
            int(key): float(value)
            for key, value in state.get("previous_velocity_ema", {}).items()
        }
        self.diagnostic_previous_velocity_ema = {
            int(key): float(value)
            for key, value in state.get("diagnostic_previous_velocity_ema", {}).items()
        }

    def observe_and_maybe_grow(
        self, model: DynamicRouterLM, records: list, route_lengths: Tensor | None = None
    ) -> int | None:

        def normalized_values(record) -> float | None:
            gradient_norm = getattr(record, "gradient_norm", None)
            if gradient_norm is None and record.logits.grad is not None:
                gradient_norm = float(record.logits.grad.detach().float().norm().cpu())
            if gradient_norm is None:
                return None
            sample_index = getattr(record, "sample_index", 0)
            path_length = getattr(record, "path_length", None)
            if (
                path_length is None
                and route_lengths is not None
                and sample_index < len(route_lengths)
            ):
                path_length = max(1.0, float(route_lengths[sample_index]))
            if path_length is None:
                path_length = 1.0
            return gradient_norm / max(1.0, float(path_length))

        if self.remaining_cooldown:
            if self.cooldown_source is not None and self.cooldown_slot is not None:
                model.set_cooldown_connection(self.cooldown_source, self.cooldown_slot)
            current: dict[int, list[float]] = defaultdict(list)
            for record in records:
                values = normalized_values(record)
                if values is not None:
                    current[record.source_id].append(values)
            velocities: list[float] = []
            for source, values in current.items():
                gradient_norm = sum(values) / len(values)
                previous = self.diagnostic_previous_gradient_norm_ema.get(source)
                gradient_norm_ema = gradient_norm if previous is None else 0.9375 * previous + 0.0625 * gradient_norm
                if previous is not None:
                    velocity = gradient_norm_ema - previous
                    old_velocity = self.diagnostic_previous_velocity_ema.get(source)
                    velocity_ema = velocity if old_velocity is None else 0.9375 * old_velocity + 0.0625 * velocity
                    self.diagnostic_previous_velocity_ema[source] = velocity_ema
                    velocities.append(velocity_ema)
                self.diagnostic_previous_gradient_norm_ema[source] = gradient_norm_ema
            sorted_velocities = sorted(velocities)
            self.last_diagnostics = {
                "observed_sources": len(current),
                "velocity_min": None,
                "velocity_median": (
                    sorted_velocities[len(sorted_velocities) // 2]
                    if sorted_velocities
                    else None
                ),
                "velocity_max": max(velocities) if velocities else None,
                "cooldown": self.remaining_cooldown,
                "growth_triggered": False,
                "diagnostic_only": True,
            }
            if velocities:
                self.last_diagnostics["velocity_min"] = min(velocities)
            self.remaining_cooldown -= 1
            if self.remaining_cooldown == 0:
                model.clear_cooldown_connection()
                self.cooldown_source = None
                self.cooldown_slot = None
            return None

        current: dict[int, list[float]] = defaultdict(list)
        for record in records:
            values = normalized_values(record)
            if values is not None:
                current[record.source_id].append(values)

        growth_triggered = False
        triggered_source: int | None = None
        velocities: list[float] = []
        for source, values in current.items():
            gradient_norm = sum(values) / len(values)
            previous = self.previous_gradient_norm_ema.get(source)
            if previous is not None:
                gradient_norm_ema = 0.9375 * previous + 0.0625 * gradient_norm
                velocity = gradient_norm_ema - previous
                old_velocity = self.previous_velocity_ema.get(source)
                velocity_ema = velocity if old_velocity is None else 0.9375 * old_velocity + 0.0625 * velocity
                self.previous_velocity_ema[source] = velocity_ema
                velocities.append(velocity_ema)
                if abs(velocity_ema) < 1e-8:
                    growth_triggered = True
                    triggered_source = source
                self.previous_gradient_norm_ema[source] = gradient_norm_ema
            else:
                self.previous_gradient_norm_ema[source] = gradient_norm

        sorted_velocities = sorted(velocities)
        median_velocity = (
            sorted_velocities[len(sorted_velocities) // 2]
            if sorted_velocities
            else None
        )
        self.last_diagnostics = {
            "observed_sources": len(current),
            "velocity_min": min(velocities) if velocities else None,
            "velocity_median": median_velocity,
            "velocity_max": max(velocities) if velocities else None,
            "cooldown": 0,
            "growth_triggered": growth_triggered,
            "diagnostic_only": False,
        }

        if not growth_triggered:
            return None
        slot = model.activate_next()
        if slot is not None:
            model.set_cooldown_connection(triggered_source, slot)
            self.cooldown_source = triggered_source
            self.cooldown_slot = slot
            self.remaining_cooldown = self.cooldown
            self.previous_gradient_norm_ema.clear()
            self.previous_velocity_ema.clear()
        self.diagnostic_previous_gradient_norm_ema.clear()
        self.diagnostic_previous_velocity_ema.clear()
        return slot


def _clear_optimizer_state(optimizer: torch.optim.Optimizer, parameter: Tensor) -> None:
    optimizer.state.pop(parameter, None)


def _protect_inactive_optimizer_state(
    optimizer: torch.optim.Optimizer, model: DynamicRouterLM
) -> None:
    inactive = ~model.active
    inactive_parameters = {
        parameter
        for index, module in enumerate(model.pool)
        if not bool(model.active[index])
        for parameter in module.parameters()
    }
    for parameter, state in list(optimizer.state.items()):
        if parameter is model.router:
            for value in state.values():
                if torch.is_tensor(value) and value.ndim and value.shape[0] == model.max_modules:
                    value[inactive] = 0
        elif parameter in inactive_parameters:
            _clear_optimizer_state(optimizer, parameter)


def train_step(
    model: DynamicRouterLM,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    targets: Tensor,
    controller: GrowthController | None = None,
    tau: float = 1.0,
    length_weight: float = 1e-3,
    memory_state: Tensor | None = None,
    grad_clip: float | None = 1.0,
    target_mask: Tensor | None = None,
    sample_mask: Tensor | None = None,
) -> dict[str, float | int | Tensor | list[str]]:
    base_model = model.module if hasattr(model, "module") else model
    optimizer.zero_grad(set_to_none=True)
    if memory_state is not None:
        memory_state = memory_state.detach().clone()
        finite_memory = torch.isfinite(memory_state).all(dim=(1, 2))
        memory_state[~finite_memory] = 0
    logits, aux = model(x, tau=tau, memory_state=memory_state)
    if not torch.isfinite(aux["memory_state"]).all():
        return _invalid_result(aux, memory_state, "nonfinite_memory")
    valid_sample_mask = aux.get("valid_sample_mask")
    if valid_sample_mask is None:
        reason = next(
            (item for item in aux["invalid_reasons"] if item is not None),
            "no_valid_path",
        )
        return _invalid_result(aux, memory_state, reason)
    effective_sample_mask = valid_sample_mask
    if sample_mask is not None:
        effective_sample_mask = sample_mask.to(valid_sample_mask.device) & valid_sample_mask
    if not bool(effective_sample_mask.any()):
        reason = next(
            (item for item in aux["invalid_reasons"] if item is not None),
            "no_valid_active_sample",
        )
        result = _invalid_result(aux, memory_state, reason)
        result["middle_route_counts"] = {}
        return result
    middle_route_counts: dict[int, int] = defaultdict(int)
    for _, targets_for_route, sample_index, _ in aux["route_targets"]:
        if not bool(effective_sample_mask[sample_index]):
            continue
        for target in targets_for_route:
            if target != base_model.input_id and target != base_model.output_id and bool(base_model.active[target]):
                middle_route_counts[target] += 1
    loss, metrics = language_model_loss(
        logits,
        targets,
        aux.get("expected_length_per_sample", aux["expected_length"]),
        aux["route_lengths"],
        length_weight,
        target_mask,
        effective_sample_mask,
    )
    if not torch.isfinite(loss):
        return _invalid_result(aux, memory_state, "nonfinite_loss")
    loss.backward()

    if not _finite_gradients(model):
        return _invalid_result(
            aux, memory_state, "nonfinite_gradient", _gradient_diagnostics(model)
        )

    router_snapshot = base_model.router.detach().clone()
    new_slot = (
        controller.observe_and_maybe_grow(
            base_model, aux["route_records"], aux["route_lengths"]
        )
        if controller
        else None
    )

    if base_model.router.grad is not None:
        base_model.router.grad[~base_model.active] = 0
    if grad_clip is not None:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(total_norm):
            return _invalid_result(
                aux, memory_state, "nonfinite_gradient", _gradient_diagnostics(model)
            )
    parameter_snapshot = {
        parameter: parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.grad is not None
    }
    optimizer.step()

    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        with torch.no_grad():
            for parameter, value in parameter_snapshot.items():
                if not torch.isfinite(parameter).all():
                    parameter.copy_(value)
        optimizer.state.clear()
        return _invalid_result(aux, memory_state, "nonfinite_parameter")

    with torch.no_grad():
        # Preserve inactive route rows against AdamW decay. A newly activated
        # row is intentionally not restored and starts from its zero seed.
        inactive_rows = ~base_model.active
        base_model.router[inactive_rows].copy_(router_snapshot[inactive_rows])
    _protect_inactive_optimizer_state(optimizer, base_model)
    if new_slot is not None:
        # The newly activated row must not inherit a preallocated slot's state.
        state = optimizer.state.get(base_model.router)
        if state is not None:
            for value in state.values():
                if torch.is_tensor(value) and value.ndim and value.shape[0] == base_model.max_modules:
                    value[new_slot] = 0

    base_model.decay_exploration()
    result: dict[str, float | int | Tensor | list[str]] = {
        key: float(value) for key, value in metrics.items()
    }
    result.update(
        {
            "expected_length": float(aux["expected_length"].detach()),
            "mean_route_length": float(aux["route_lengths"].float().mean()),
            "max_frontier_count": aux["forward_diagnostics"]["max_frontier_count"],
            "total_frontier_states": aux["forward_diagnostics"]["total_frontier_states"],
            "new_slot": new_slot if new_slot is not None else -1,
            "terminations": aux["terminations"],
            "invalid_reasons": aux["invalid_reasons"],
            "skipped_update": 0,
            "growth_diagnostics": controller.last_diagnostics if controller else {},
            "memory_state": aux["memory_state"].detach(),
            "middle_route_counts": dict(middle_route_counts),
            "route_records": aux["route_records"],
            "route_lengths": aux["route_lengths"],
        }
    )
    return result
