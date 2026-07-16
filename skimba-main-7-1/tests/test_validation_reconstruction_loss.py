import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
METRICS_MODULE = ROOT / "utils" / "validation_metrics.py"
TRAIN_SOURCE = ROOT / "train_diffusion_network_2.py"
PLOT_SOURCE = ROOT / "plot_training_curves.py"


def load_validation_metrics():
    assert METRICS_MODULE.is_file(), "validation metric helpers are missing"
    spec = importlib.util.spec_from_file_location(
        "validation_metrics",
        METRICS_MODULE,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_invalid_voxels_do_not_contribute_to_sampled_semantic_ce():
    metrics = load_validation_metrics()
    target = torch.tensor([[[[0, 1]]]], dtype=torch.long)
    invalid_voxels = np.array([[[1, 0]]], dtype=np.uint8)
    logits = torch.tensor(
        [[[[[-10.0, -10.0]]], [[[10.0, 10.0]]]]],
        dtype=torch.float32,
    )

    masked_target = metrics.mask_invalid_voxels(
        target,
        invalid_voxels,
        ignore_label=255,
    )

    unmasked_loss = F.cross_entropy(logits, target, ignore_index=255)
    masked_loss = F.cross_entropy(logits, masked_target, ignore_index=255)
    assert target.tolist() == [[[[0, 1]]]]
    assert masked_target.tolist() == [[[[255, 1]]]]
    assert masked_loss.item() < 1e-6
    assert unmasked_loss.item() > 9.0


def test_training_masks_invalid_voxels_before_reconstruction_loss():
    source = TRAIN_SOURCE.read_text(encoding="utf-8")

    assert "from utils.validation_metrics import mask_invalid_voxels" in source
    mask_position = source.index("reconstruction_target = mask_invalid_voxels(")
    loss_position = source.index("reconstruction_loss = loss_func(")
    assert mask_position < loss_position
    assert "recon_voxel.detach(), reconstruction_target" in source


def test_plot_calls_metric_sampled_semantic_cross_entropy():
    source = PLOT_SOURCE.read_text(encoding="utf-8")

    assert 'label="Validation sampled semantic CE"' in source
    assert 'ax.set_title("Validation Sampled Semantic Cross-Entropy")' in source
    assert "reconstruction loss (no KL)" not in source
