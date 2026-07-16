from pathlib import Path
import json
import random

import numpy as np
import pytest
import torch

from utils.full_diffusion_validation import (
    metrics_from_confusion,
    prepare_full_validation_loader_config,
    require_sequence_08_validation_split,
    resolve_best_checkpoint,
    seed_random_generators,
    write_validation_reports,
)


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SCRIPT = (
    ROOT / "scripts" / "evaluation" / "evaluate_diffusion_checkpoint.py"
)
RUNBOOK = ROOT / "README_RUN.md"


def test_resolve_best_checkpoint_uses_miou_then_epoch(tmp_path):
    for name in (
        "best_10_4.5.pth",
        "best_20_4.5.pth",
        "best_5_4.6.pth",
        "best_99_bad.pth",
        "protect_100.pth",
    ):
        (tmp_path / name).touch()

    assert resolve_best_checkpoint("", str(tmp_path)).name == "best_5_4.6.pth"

    (tmp_path / "best_30_4.6.pth").touch()
    assert resolve_best_checkpoint("", str(tmp_path)).name == "best_30_4.6.pth"


def test_resolve_best_checkpoint_honors_explicit_override(tmp_path):
    automatic = tmp_path / "best_20_9.0.pth"
    explicit = tmp_path / "manual_checkpoint.pth"
    automatic.touch()
    explicit.touch()

    assert resolve_best_checkpoint(str(explicit), str(tmp_path)) == explicit


def test_resolve_best_checkpoint_rejects_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="Explicit checkpoint does not exist"):
        resolve_best_checkpoint(str(tmp_path / "missing.pth"), str(tmp_path))

    with pytest.raises(FileNotFoundError, match=r"No valid best_<epoch>_<mIoU>\.pth"):
        resolve_best_checkpoint("", str(tmp_path))


def test_prepare_full_validation_loader_config_removes_sampling_without_mutation():
    original = {
        "imageset": "test",
        "return_ref": True,
        "batch_size": 4,
        "shuffle": True,
        "num_workers": 8,
        "frame_divisor": 10,
    }

    prepared = prepare_full_validation_loader_config(original)

    assert prepared == {
        "imageset": "val",
        "return_ref": True,
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 8,
    }
    assert original["imageset"] == "test"
    assert original["frame_divisor"] == 10


def test_require_sequence_08_validation_split_accepts_only_sequence_08(tmp_path):
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text("split:\n  valid: [8]\n", encoding="utf-8")
    require_sequence_08_validation_split(str(mapping))

    mapping.write_text("split:\n  valid: [8, 9]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly sequence 08"):
        require_sequence_08_validation_split(str(mapping))


def test_seed_random_generators_reproduces_random_inputs():
    seed_random_generators(123)
    first = (random.random(), np.random.rand(), torch.rand(2))

    seed_random_generators(123)
    second = (random.random(), np.random.rand(), torch.rand(2))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_metrics_from_confusion_computes_semantic_and_completion_iou():
    confusion = np.array([[5, 1], [2, 4]], dtype=np.int64)

    metrics = metrics_from_confusion(confusion)

    assert metrics["class_iou_percent"] == pytest.approx([62.5, 400.0 / 7.0])
    assert metrics["semantic_miou_percent"] == pytest.approx(400.0 / 7.0)
    assert metrics["completion_iou_percent"] == pytest.approx(400.0 / 7.0)


def _complete_result():
    return {
        "checkpoint": "/models/best_30_4.6.pth",
        "config_path": "/project/config.yaml",
        "seed": 20260713,
        "frame_count": 4071,
        "class_iou_percent": [0.0, 1.5, 2.5],
        "semantic_miou_percent": 2.0,
        "completion_iou_percent": 12.25,
        "mean_epsilon_mse": 0.125,
        "mean_cross_entropy": 1.25,
    }


def test_write_validation_reports_writes_json_and_readable_text(tmp_path):
    output_dir = tmp_path / "reports"

    json_path, text_path = write_validation_reports(_complete_result(), output_dir)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["checkpoint"] == "/models/best_30_4.6.pth"
    assert saved["frame_count"] == 4071
    report = text_path.read_text(encoding="utf-8")
    assert "Seed: 20260713" in report
    assert "Frames: 4071" in report
    assert "Semantic mIoU: 2.000%" in report
    assert "Completion IoU: 12.250%" in report


def test_write_validation_reports_rejects_non_finite_results_before_writing(tmp_path):
    output_dir = tmp_path / "reports"
    result = _complete_result()
    result["semantic_miou_percent"] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        write_validation_reports(result, output_dir)

    assert not output_dir.exists()


def test_evaluation_parser_has_reproducible_server_defaults():
    from scripts.evaluation.evaluate_diffusion_checkpoint import build_parser

    args = build_parser().parse_args([])

    assert args.config_path == "config/semantickitti_autoencoder.yaml"
    assert args.checkpoint == ""
    assert args.seed == 20260713
    assert args.device == "cuda:0"
    assert args.output_dir == ""


def test_evaluation_entry_point_is_validation_only_and_reuses_live_contracts():
    source = EVALUATION_SCRIPT.read_text(encoding="utf-8")

    required_contracts = (
        "prepare_full_validation_loader_config",
        "require_sequence_08_validation_split",
        "load_model_state",
        "load_autoencoder_state_from_checkpoint",
        "invalid_mask_path",
        "torch.no_grad()",
        "val_pt_dataset.im_idx.sort()",
        "write_validation_reports",
    )
    for contract in required_contracts:
        assert contract in source

    forbidden_training_operations = (
        "restore_training_state",
        "optimizer.step",
        "scheduler.step",
        ".backward(",
    )
    for operation in forbidden_training_operations:
        assert operation not in source


def test_runbook_documents_full_sequence_08_checkpoint_evaluation():
    source = RUNBOOK.read_text(encoding="utf-8")

    assert "scripts/evaluation/evaluate_diffusion_checkpoint.py" in source
    assert "--checkpoint" in source
    assert "sequence 08 全部帧" in source
    assert "不会继续训练" in source
    assert "metrics.json" in source
    assert "report.txt" in source
