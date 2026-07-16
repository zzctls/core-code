from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_constant_learning_rate_scheduler_never_changes_optimizer_lr():
    from utils.warmup_lr import ConstantLearningRate

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = ConstantLearningRate(optimizer, lr=1e-4)

    for _ in range(5):
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_constant_learning_rate_resume_keeps_configured_lr():
    from utils.warmup_lr import ConstantLearningRate

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = ConstantLearningRate(optimizer, lr=1e-4)

    scheduler.load_state_dict({"lr": 1e-3, "last_epoch": 20})
    scheduler.set_max_steps(999)

    assert scheduler.lr == pytest.approx(1e-4)
    assert scheduler.last_epoch == 20
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_warmup_constant_learning_rate_warms_up_then_stays_constant():
    from utils.warmup_lr import WarmupConstantLearningRate

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = WarmupConstantLearningRate(
        optimizer,
        lr=1e-4,
        start_lr=1e-6,
        warmup_steps=4,
    )

    observed_lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(5):
        scheduler.step()
        observed_lrs.append(optimizer.param_groups[0]["lr"])

    assert observed_lrs == pytest.approx(
        [1e-6, 2.575e-5, 5.05e-5, 7.525e-5, 1e-4, 1e-4]
    )


def test_warmup_constant_resume_keeps_configured_target_lr():
    from utils.warmup_lr import WarmupConstantLearningRate

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = WarmupConstantLearningRate(
        optimizer,
        lr=1e-4,
        start_lr=1e-6,
        warmup_steps=10,
    )

    scheduler.load_state_dict({"lr": 1e-3, "warmup_steps": 100, "last_epoch": 20})

    assert scheduler.lr == pytest.approx(1e-4)
    assert scheduler.start_lr == pytest.approx(1e-6)
    assert scheduler.warmup_steps == 10
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_diffusion_training_uses_warmup_then_constant_learning_rate_scheduler():
    source = (PROJECT_ROOT / "train_diffusion_network_2.py").read_text(
        encoding="utf-8"
    )

    assert "scheduler = WarmupConstantLearningRate(" in source
    assert 'lr=train_hypers["learning_rate"]' in source
    assert 'start_lr=train_hypers["warmup_start_lr"]' in source
    assert 'warmup_steps=train_hypers["warmup_epochs"] * len(train_dataset_loader)' in source
    assert "scheduler = WarmupCosineLR(" not in source
