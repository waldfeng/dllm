from types import SimpleNamespace

import pytest
import torch

from train import build_resume_scheduler


def test_resume_scheduler_uses_cli_horizon_and_peak_lr():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    args = SimpleNamespace(lr=2e-4, warmup_steps=2, min_lr_ratio=0.1)

    scheduler = build_resume_scheduler(
        optimizer,
        args,
        current_lr=1e-4,
        start_step=10,
        schedule_steps=14,
    )

    assert scheduler.last_epoch == 10
    assert scheduler.get_last_lr()[0] == pytest.approx(1e-4)

    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(1.5e-4)

    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(2e-4)

    optimizer.step()
    scheduler.step()
    optimizer.step()
    scheduler.step()
    assert scheduler.last_epoch == 14
    assert scheduler.get_last_lr()[0] == pytest.approx(2e-5)
