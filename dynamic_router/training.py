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


def _invalid_result(
    aux: dict, memory_state: Tensor | None, reason: str | None = None
) -> dict[str, float | int | Tensor | list[str]]:
    invalid_reasons = aux["invalid_reasons"]
    if reason is not None:
        invalid_reasons = [reason for _ in invalid_reasons]
    safe_memory = (
        torch.zeros_like(aux["memory_state"])
        if memory_state is None
        else torch.zeros_like(memory_state)
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
        cooldown: int = 200,
    ) -> None:
        if gradient_window <= 0 or cooldown < 0:
            raise ValueError("invalid growth controller configuration")
        self.gradient_window = gradient_window
        self.cooldown = cooldown
        self.gradient_beta = 1.0 - 1.0 / gradient_window
        self.remaining_cooldown = 0
        self.previous_gradient_norm_ema: dict[int, float] = {}
        self.previous_velocity_ema: dict[int, float] = {}
        self.diagnostic_previous_gradient_norm_ema: dict[int, float] = {}
        self.diagnostic_previous_velocity_ema: dict[int, float] = {}
        self.velocity_history: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=3)
        )
        self.last_diagnostics: dict[str, float | int | None] = {}

    def state_dict(self) -> dict:
        return {
            "remaining_cooldown": self.remaining_cooldown,
            "previous_gradient_norm_ema": dict(self.previous_gradient_norm_ema),
            "previous_velocity_ema": dict(self.previous_velocity_ema),
            "diagnostic_previous_gradient_norm_ema": dict(
                self.diagnostic_previous_gradient_norm_ema
            ),
            "diagnostic_previous_velocity_ema": dict(
                self.diagnostic_previous_velocity_ema
            ),
            "gradient_window": self.gradient_window,
            "velocity_history": {key: list(value) for key, value in self.velocity_history.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.remaining_cooldown = int(state.get("remaining_cooldown", 0))
        self.previous_gradient_norm_ema = {
            int(key): float(value)
            for key, value in state.get("previous_gradient_norm_ema", {}).items()
        }
        self.previous_velocity_ema = {
            int(key): float(value)
            for key, value in state.get("previous_velocity_ema", {}).items()
        }
        self.diagnostic_previous_gradient_norm_ema = {
            int(key): float(value)
            for key, value in state.get(
                "diagnostic_previous_gradient_norm_ema", {}
            ).items()
        }
        self.diagnostic_previous_velocity_ema = {
            int(key): float(value)
            for key, value in state.get(
                "diagnostic_previous_velocity_ema", {}
            ).items()
        }
        self.velocity_history = defaultdict(lambda: deque(maxlen=3))
        for key, values in state.get("velocity_history", {}).items():
            self.velocity_history[int(key)].extend(float(value) for value in values)
        self.gradient_window = max(1, int(state.get("gradient_window", self.gradient_window)))
        self.gradient_beta = 1.0 - 1.0 / self.gradient_window

    def observe_and_maybe_grow(
        self, model: DynamicRouterLM, records: list
    ) -> int | None:
        if self.remaining_cooldown:
            current: dict[int, list[Tensor]] = defaultdict(list)
            for record in records:
                if record.logits.grad is not None:
                    current[record.source_id].append(record.logits.grad.detach().flatten())
            velocities: list[float] = []
            for source, values in current.items():
                gradient_norm = sum(value.norm().item() for value in values) / len(values)
                previous = self.diagnostic_previous_gradient_norm_ema.get(source)
                gradient_norm_ema = gradient_norm
                if previous is not None:
                    gradient_norm_ema = (
                        self.gradient_beta * previous
                        + (1.0 - self.gradient_beta) * gradient_norm
                    )
                    raw_velocity = gradient_norm_ema - previous
                    previous_velocity = self.diagnostic_previous_velocity_ema.get(source)
                    velocity = raw_velocity if previous_velocity is None else (
                        self.gradient_beta * previous_velocity
                        + (1.0 - self.gradient_beta) * raw_velocity
                    )
                    velocities.append(velocity)
                    self.diagnostic_previous_velocity_ema[source] = velocity
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
                "gradient_window": self.gradient_window,
                "gradient_beta": self.gradient_beta,
                "growth_triggered": False,
                "diagnostic_only": True,
            }
            if velocities:
                self.last_diagnostics["velocity_min"] = min(velocities)
            self.remaining_cooldown -= 1
            return None

        current: dict[int, list[Tensor]] = defaultdict(list)
        for record in records:
            if record.logits.grad is not None:
                current[record.source_id].append(record.logits.grad.detach().flatten())

        growth_triggered = False
        velocities: list[float] = []
        for source, values in current.items():
            gradient_norm = sum(value.norm().item() for value in values) / len(values)
            previous = self.previous_gradient_norm_ema.get(source)
            if previous is not None:
                gradient_norm_ema = (
                    self.gradient_beta * previous
                    + (1.0 - self.gradient_beta) * gradient_norm
                )
                velocity = gradient_norm_ema - previous
                previous_velocity = self.previous_velocity_ema.get(source)
                velocity_ema = velocity if previous_velocity is None else (
                    self.gradient_beta * previous_velocity
                    + (1.0 - self.gradient_beta) * velocity
                )
                self.previous_velocity_ema[source] = velocity_ema
                self.velocity_history[source].append(velocity_ema)
                velocities.append(velocity_ema)
                if (
                    len(self.velocity_history[source]) == 3
                    and self.velocity_history[source][-3] < 0
                    and self.velocity_history[source][-2] < 0
                    and self.velocity_history[source][-1] >= 0
                ):
                    growth_triggered = True
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
            "gradient_window": self.gradient_window,
            "gradient_beta": self.gradient_beta,
            "growth_triggered": growth_triggered,
            "diagnostic_only": False,
        }

        if not growth_triggered:
            return None
        slot = model.activate_next()
        if slot is not None:
            self.remaining_cooldown = self.cooldown
            self.velocity_history.clear()
            self.previous_gradient_norm_ema.clear()
            self.previous_velocity_ema.clear()
            self.diagnostic_previous_gradient_norm_ema.clear()
            self.diagnostic_previous_velocity_ema.clear()
        return slot


def _snapshot_inactive(model: DynamicRouterLM) -> dict[Tensor, tuple[Tensor, int]]:
    return {
        parameter: (parameter.detach().clone(), index)
        for index, module in enumerate(model.pool)
        if not bool(model.active[index])
        for parameter in module.parameters()
    }


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
    optimizer.zero_grad(set_to_none=True)
    if memory_state is not None:
        memory_state = memory_state.detach().clone()
        finite_memory = torch.isfinite(memory_state).all(dim=(1, 2))
        memory_state[~finite_memory] = 0
    logits, aux = model(x, tau=tau, memory_state=memory_state)
    if not torch.isfinite(aux["memory_state"]).all():
        return _invalid_result(aux, memory_state, "nonfinite_memory")
    invalid_reasons = [reason for reason in aux["invalid_reasons"] if reason is not None]
    if invalid_reasons:
        return _invalid_result(aux, memory_state)
    loss, metrics = language_model_loss(
        logits,
        targets,
        aux.get("expected_length_per_sample", aux["expected_length"]),
        length_weight,
        target_mask,
        sample_mask,
    )
    if not torch.isfinite(loss):
        return _invalid_result(aux, memory_state, "nonfinite_loss")
    loss.backward()

    if not _finite_gradients(model):
        return _invalid_result(aux, memory_state, "nonfinite_gradient")

    inactive_parameters = _snapshot_inactive(model)
    router_snapshot = model.router.detach().clone()
    new_slot = (
        controller.observe_and_maybe_grow(model, aux["route_records"])
        if controller
        else None
    )

    for parameter in inactive_parameters:
        parameter.grad = None
    if model.router.grad is not None:
        model.router.grad[~model.active] = 0
    if grad_clip is not None:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(total_norm):
            return _invalid_result(aux, memory_state, "nonfinite_gradient")
    parameter_snapshot = {
        parameter: parameter.detach().clone()
        for parameter in model.parameters()
    }
    optimizer.step()

    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        with torch.no_grad():
            for parameter, value in parameter_snapshot.items():
                parameter.copy_(value)
        optimizer.state.clear()
        return _invalid_result(aux, memory_state, "nonfinite_parameter")

    with torch.no_grad():
        for parameter, (value, module_index) in inactive_parameters.items():
            if not bool(model.active[module_index]):
                parameter.copy_(value)
        # Preserve inactive route rows against AdamW decay. A newly activated
        # row is intentionally not restored and starts from its zero seed.
        inactive_rows = ~model.active
        model.router[inactive_rows].copy_(router_snapshot[inactive_rows])
    _protect_inactive_optimizer_state(optimizer, model)
    if new_slot is not None:
        # The newly activated row must not inherit a preallocated slot's state.
        state = optimizer.state.get(model.router)
        if state is not None:
            for value in state.values():
                if torch.is_tensor(value) and value.ndim and value.shape[0] == model.max_modules:
                    value[new_slot] = 0

    model.decay_exploration()
    result: dict[str, float | int | Tensor | list[str]] = {
        key: float(value) for key, value in metrics.items()
    }
    result.update(
        {
            "expected_length": float(aux["expected_length"].detach()),
            "mean_route_length": float(aux["route_lengths"].float().mean()),
            "new_slot": new_slot if new_slot is not None else -1,
            "terminations": aux["terminations"],
            "invalid_reasons": aux["invalid_reasons"],
            "skipped_update": 0,
            "growth_diagnostics": controller.last_diagnostics if controller else {},
            "memory_state": aux["memory_state"].detach(),
        }
    )
    return result
