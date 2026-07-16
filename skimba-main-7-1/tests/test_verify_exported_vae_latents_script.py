import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit" / "verify_exported_vae_latents.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("verify_exported_vae_latents", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tensor_comparison_reports_exactness_and_error_counts():
    module = load_script_module()
    reference = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    candidate = torch.tensor([0.0, 1.5, 2.0], dtype=torch.float32)

    result = module.compare_tensors(reference, candidate)

    assert result == {
        "element_count": 3,
        "bit_exact": False,
        "differing_elements": 1,
        "max_abs_error": 0.5,
        "mean_abs_error": pytest.approx(1.0 / 6.0),
    }


def test_prediction_comparison_counts_only_valid_voxels():
    module = load_script_module()
    target = torch.tensor([0, 1, 255, 2])
    reference = torch.tensor([0, 1, 2, 2])
    candidate = torch.tensor([0, 2, 1, 2])

    result = module.compare_predictions(reference, candidate, target, ignore_label=255)

    assert result == {
        "valid_voxels": 3,
        "differing_voxels": 1,
        "disagreement_rate_percent": pytest.approx(100.0 / 3.0),
    }


def test_load_channel_stats_validates_shape_and_positive_std(tmp_path):
    module = load_script_module()
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        json.dumps({"mean": [1.0, 2.0], "std": [0.5, 0.0]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strictly positive"):
        module.load_channel_stats(stats_path, channels=2)


def test_parity_failures_apply_configured_tolerances():
    module = load_script_module()
    summary = {
        "latent": {"max_abs_error": 0.01},
        "decoded_logits": {"max_abs_error": 0.02},
        "decoded_prediction": {"disagreement_rate_percent": 0.03},
    }

    assert module.parity_failures(summary, 0.01, 0.02, 0.03) == []
    failures = module.parity_failures(summary, 0.0, 0.0, 0.0)
    assert len(failures) == 3
    assert "latent max_abs_error" in failures[0]


def test_cli_defaults_match_reproducible_sequence_08_mean_audit():
    module = load_script_module()

    args = module.build_argparser().parse_args([])

    assert args.sequences == "08"
    assert args.frames_per_sequence == 20
    assert args.seed == 20260621
    assert args.max_latent_abs_error == 0.0
    assert args.max_logit_abs_error == 0.0
    assert args.max_prediction_disagreement_percent == 0.0
    assert args.latent_shape == "8,64,64,8"

