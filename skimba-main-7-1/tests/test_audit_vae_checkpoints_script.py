from argparse import Namespace
from pathlib import Path

import numpy as np


def make_label_files(root: Path, sequence: str, count: int) -> None:
    voxel_dir = root / "sequences" / sequence / "voxels"
    voxel_dir.mkdir(parents=True)
    for index in range(count):
        (voxel_dir / f"{index:06d}.label").touch()


def minimal_config(gt_root, configured_checkpoint):
    return {
        "model_params": {
            "num_class": 2,
            "semantic_embed_dim": 8,
            "in_channels": 8,
            "out_channels": 8,
            "latent_channels": 64,
            "autoencoder_num_res_blocks": 1,
            "autoencoder_channels_list": [16, 32, 64, 128],
            "auto_groups": 4,
            "num_input_features": 3,
            "init_size": 8,
            "voxel_channel": 1,
            "dropout_rate": 0.2,
        },
        "dataset_params": {
            "data_root": str(gt_root.parent),
            "gt_root": str(gt_root),
            "label_mapping": "/labels.yaml",
            "ignore_label": 255,
        },
        "train_params": {"vae_checkpoint": str(configured_checkpoint)},
    }


def make_args(checkpoint_dir, output_dir):
    return Namespace(
        config_path="config.yaml",
        checkpoint_dir=str(checkpoint_dir),
        frames_per_sequence=20,
        seed=17,
        output_dir=str(output_dir),
        dataset_root="",
        label_root="",
        label_mapping="",
        sequences="",
        device="cpu",
        rare_class_count=1,
        min_latent_std=1e-4,
        max_latent_std=10.0,
        max_latent_abs=50.0,
        latent_shape="8,64,64,8",
    )


def healthy_frame_row(manifest_row):
    return {
        **manifest_row,
        "ce": 0.1,
        "ce_sum": 0.2,
        "valid_voxels": 2,
        "correct_voxels": 2,
        "confusion": np.eye(2, dtype=np.int64).tolist(),
        "occupancy_intersection": 1,
        "occupancy_union": 1,
        "semantic_miou": 100.0,
        "occupancy_iou": 100.0,
        "voxel_accuracy": 100.0,
        "latent_mean": 0.0,
        "latent_std": 0.7,
        "latent_rms": 0.7,
        "latent_l2": 10.0,
        "latent_abs_max": 2.0,
        "latent_finite": True,
        "logits_finite": True,
    }


def test_discover_checkpoints_returns_only_sorted_best_files(tmp_path):
    from scripts.audit.audit_vae_checkpoints import discover_checkpoints

    for name in ("best_10_99.0.pth", "best_2_90.0.pth", "last.pth", "best_notes.txt"):
        (tmp_path / name).touch()

    checkpoints = discover_checkpoints(tmp_path)

    assert [path.name for path in checkpoints] == ["best_2_90.0.pth", "best_10_99.0.pth"]


def test_discover_checkpoints_rejects_empty_directory(tmp_path):
    from scripts.audit.audit_vae_checkpoints import discover_checkpoints

    try:
        discover_checkpoints(tmp_path)
    except ValueError as exc:
        assert "No best_*.pth" in str(exc)
    else:
        raise AssertionError("Expected empty checkpoint directory to be rejected")


def test_run_audit_isolates_failure_and_shares_manifest(tmp_path):
    from scripts.audit.audit_vae_checkpoints import run_audit

    gt_root = tmp_path / "labels"
    make_label_files(gt_root, "00", 24)
    make_label_files(gt_root, "08", 24)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    good = checkpoint_dir / "best_1_90.0.pth"
    bad = checkpoint_dir / "best_2_91.0.pth"
    good.touch()
    bad.touch()
    calls = []

    def fake_evaluator(checkpoint, manifest, _args, _device):
        calls.append((checkpoint.name, [dict(row) for row in manifest]))
        if checkpoint == bad:
            raise RuntimeError("broken checkpoint")
        return [healthy_frame_row(row) for row in manifest]

    args = make_args(checkpoint_dir, tmp_path / "report")
    config = minimal_config(gt_root, good)
    exit_code, payload = run_audit(
        args,
        evaluator=fake_evaluator,
        config_loader=lambda _path: config,
    )

    assert exit_code == 0
    assert payload["recommended"]["checkpoint"] == str(good)
    assert payload["ranking"][1]["status"] == "error"
    assert payload["ranking"][1]["error"] == "broken checkpoint"
    assert len(calls) == 2
    assert calls[0][1] == calls[1][1] == payload["manifest"]
    assert len(payload["manifest"]) == 40
    assert set(payload["recommended"]["per_sequence"]) == {"00", "08"}
    assert payload["configured_comparison"]["available"] is True
    assert (tmp_path / "report" / "vae_audit_report.md").exists()


def test_run_audit_returns_two_when_every_checkpoint_fails(tmp_path):
    from scripts.audit.audit_vae_checkpoints import run_audit

    gt_root = tmp_path / "labels"
    make_label_files(gt_root, "00", 1)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "best_1.pth"
    checkpoint.touch()

    def failing_evaluator(_checkpoint, _manifest, _args, _device):
        raise RuntimeError("cannot load")

    exit_code, payload = run_audit(
        make_args(checkpoint_dir, tmp_path / "report"),
        evaluator=failing_evaluator,
        config_loader=lambda _path: minimal_config(gt_root, checkpoint),
    )

    assert exit_code == 2
    assert payload["recommended"] is None
    assert payload["ranking"][0]["status"] == "error"


def test_build_evaluation_args_rejects_missing_label_mapping(tmp_path):
    from scripts.audit.audit_vae_checkpoints import build_evaluation_args

    args = make_args(tmp_path, tmp_path / "report")
    config = minimal_config(tmp_path / "labels", tmp_path / "best.pth")
    config["dataset_params"]["label_mapping"] = ""

    try:
        build_evaluation_args(args, config)
    except ValueError as exc:
        assert "label_mapping" in str(exc)
    else:
        raise AssertionError("Expected missing label mapping to be rejected")


def test_shared_frame_evaluator_checks_latent_shape():
    source = Path("scripts/data/select_best_semantic_vae.py").read_text()

    assert "expected_latent_shape = (1, *parse_shape(args.latent_shape))" in source
    assert "tuple(latent.shape) != expected_latent_shape" in source


def test_vae_audit_defaults_to_repository_sibling_results():
    from scripts.audit.audit_vae_checkpoints import (
        build_argparser,
        default_vae_audit_output_dir,
    )

    repository_root = Path("/work/project-master")
    assert default_vae_audit_output_dir(
        repository_root=repository_root,
        project_name="skimba-main-7-1",
    ) == Path("/work/project-master-results/skimba-main-7-1/vae_audit")

    parser = build_argparser()
    args = parser.parse_args(["--checkpoint-dir", "/checkpoints"])
    assert Path(args.output_dir) == default_vae_audit_output_dir()

    overridden = parser.parse_args(
        ["--checkpoint-dir", "/checkpoints", "--output-dir", "/server/artifacts"]
    )
    assert overridden.output_dir == "/server/artifacts"


def test_readme_documents_reproducible_vae_audit_command():
    readme = Path("README_RUN.md").read_text()

    assert "scripts/audit/audit_vae_checkpoints.py" in readme
    assert "--checkpoint-dir" in readme
    assert "--frames-per-sequence 20" in readme
    assert "vae_audit_report.md" in readme
    assert "vae_audit.json" in readme
    assert "vae_audit_summary.csv" in readme
    assert "vae_audit_frames.csv" in readme
    assert "posterior mean" in readme
    assert "same frame manifest" in readme
    assert "project-master-results" in readme
    assert "--output-dir" in readme
    assert "项目同级" in readme
