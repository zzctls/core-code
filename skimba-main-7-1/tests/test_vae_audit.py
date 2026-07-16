from pathlib import Path

import numpy as np


def make_label_files(root: Path, sequence: str, count: int) -> None:
    voxel_dir = root / "sequences" / sequence / "voxels"
    voxel_dir.mkdir(parents=True)
    for index in range(count):
        (voxel_dir / f"{index:06d}.label").touch()


def test_manifest_samples_each_sequence_deterministically(tmp_path):
    from scripts.audit.vae_audit import build_frame_manifest

    make_label_files(tmp_path, "00", 30)
    make_label_files(tmp_path, "08", 30)

    first = build_frame_manifest(tmp_path, frames_per_sequence=20, seed=17)
    second = build_frame_manifest(tmp_path, frames_per_sequence=20, seed=17)

    assert first == second
    assert {row["sequence"] for row in first} == {"00", "08"}
    assert sum(row["sequence"] == "00" for row in first) == 20
    assert sum(row["sequence"] == "08" for row in first) == 20


def test_manifest_keeps_all_frames_in_short_sequence(tmp_path):
    from scripts.audit.vae_audit import build_frame_manifest, summarize_sequence_sampling

    make_label_files(tmp_path, "01", 3)

    manifest = build_frame_manifest(tmp_path, frames_per_sequence=20, seed=0)
    sampling = summarize_sequence_sampling(tmp_path, manifest, frames_per_sequence=20)

    assert [row["frame"] for row in manifest] == ["000000", "000001", "000002"]
    assert sampling == {
        "01": {"available_frames": 3, "sampled_frames": 3, "short_sequence": True}
    }


def test_manifest_rejects_empty_dataset(tmp_path):
    from scripts.audit.vae_audit import build_frame_manifest

    try:
        build_frame_manifest(tmp_path)
    except ValueError as exc:
        assert "No label frames" in str(exc)
    else:
        raise AssertionError("Expected an empty dataset to be rejected")


def test_aggregate_uses_accumulated_confusion_and_voxel_weighted_ce():
    from scripts.audit.vae_audit import aggregate_rows, confusion_from_labels

    rows = [
        {
            "sequence": "00",
            "ce_sum": 2.0,
            "valid_voxels": 2,
            "correct_voxels": 1,
            "confusion": confusion_from_labels([1, 1], [1, 2], 3).tolist(),
            "occupancy_intersection": 1,
            "occupancy_union": 2,
            "latent_mean": 0.0,
            "latent_std": 1.0,
            "latent_rms": 1.0,
            "latent_l2": 2.0,
            "latent_abs_max": 2.0,
            "latent_finite": True,
            "logits_finite": True,
        },
        {
            "sequence": "00",
            "ce_sum": 1.0,
            "valid_voxels": 1,
            "correct_voxels": 1,
            "confusion": confusion_from_labels([2], [2], 3).tolist(),
            "occupancy_intersection": 1,
            "occupancy_union": 1,
            "latent_mean": 2.0,
            "latent_std": 3.0,
            "latent_rms": 3.0,
            "latent_l2": 4.0,
            "latent_abs_max": 3.0,
            "latent_finite": True,
            "logits_finite": True,
        },
    ]

    summary = aggregate_rows(rows, num_classes=3, rare_class_count=1)

    assert summary["valid_voxels"] == 3
    assert summary["mean_ce"] == 1.0
    assert np.isclose(summary["occupancy_iou"], 100.0 * 2 / 3)
    assert np.allclose(summary["per_class_iou"][1:], [50.0, 50.0])
    assert summary["semantic_miou"] == 50.0
    assert summary["rare_class_miou"] == 50.0
    assert summary["latent_mean"] == 1.0
    assert summary["latent_abs_max"] == 3.0


def test_aggregate_counts_nonfinite_frames():
    from scripts.audit.vae_audit import aggregate_rows

    row = {
        "ce_sum": 0.0,
        "valid_voxels": 1,
        "correct_voxels": 1,
        "confusion": np.eye(2, dtype=np.int64).tolist(),
        "occupancy_intersection": 1,
        "occupancy_union": 1,
        "latent_mean": 0.0,
        "latent_std": 1.0,
        "latent_rms": 1.0,
        "latent_l2": 1.0,
        "latent_abs_max": 1.0,
        "latent_finite": False,
        "logits_finite": False,
    }

    summary = aggregate_rows([row], num_classes=2)

    assert summary["nonfinite_latent_frames"] == 1
    assert summary["nonfinite_logit_frames"] == 1


def test_ranking_rejects_failed_or_flagged_candidate_before_miou():
    from scripts.audit.vae_audit import rank_candidates

    rows = [
        {
            "checkpoint": "/c.pth",
            "status": "error",
            "semantic_miou": 100.0,
            "flags": [],
        },
        {
            "checkpoint": "/b.pth",
            "status": "ok",
            "semantic_miou": 99.0,
            "rare_class_miou": 98.0,
            "mean_ce": 0.1,
            "occupancy_iou": 99.0,
            "flags": ["latent_contains_nonfinite"],
        },
        {
            "checkpoint": "/a.pth",
            "status": "ok",
            "semantic_miou": 90.0,
            "rare_class_miou": 80.0,
            "mean_ce": 0.2,
            "occupancy_iou": 95.0,
            "flags": [],
        },
    ]

    ranked = rank_candidates(rows)

    assert [row["checkpoint"] for row in ranked] == ["/a.pth", "/b.pth", "/c.pth"]


def test_ranking_uses_checkpoint_path_as_final_tie_breaker():
    from scripts.audit.vae_audit import rank_candidates

    common = {
        "status": "ok",
        "semantic_miou": 90.0,
        "rare_class_miou": 80.0,
        "mean_ce": 0.2,
        "occupancy_iou": 95.0,
        "flags": [],
    }

    ranked = rank_candidates(
        [{**common, "checkpoint": "/z.pth"}, {**common, "checkpoint": "/a.pth"}]
    )

    assert [row["checkpoint"] for row in ranked] == ["/a.pth", "/z.pth"]


def test_configured_checkpoint_delta_is_explicit():
    from scripts.audit.vae_audit import compare_configured_checkpoint

    ranked = [
        {
            "checkpoint": "/best.pth",
            "semantic_miou": 92.0,
            "rare_class_miou": 70.0,
            "mean_ce": 0.2,
            "occupancy_iou": 96.0,
        },
        {
            "checkpoint": "/current.pth",
            "semantic_miou": 90.0,
            "rare_class_miou": 68.0,
            "mean_ce": 0.3,
            "occupancy_iou": 95.0,
        },
    ]

    comparison = compare_configured_checkpoint(ranked, "/current.pth")

    assert comparison["available"] is True
    assert comparison["deltas"] == {
        "semantic_miou": 2.0,
        "rare_class_miou": 2.0,
        "mean_ce": -0.1,
        "occupancy_iou": 1.0,
    }


def test_configured_checkpoint_comparison_reports_unavailable():
    from scripts.audit.vae_audit import compare_configured_checkpoint

    comparison = compare_configured_checkpoint(
        [{"checkpoint": "/best.pth"}],
        "/not-scanned.pth",
    )

    assert comparison == {
        "available": False,
        "configured_checkpoint": "/not-scanned.pth",
        "reason": "configured checkpoint was not evaluated",
    }


def test_write_reports_preserves_manifest_and_names_weakest_sequence(tmp_path):
    import csv
    import json

    from scripts.audit.vae_audit import write_reports

    recommended = {
        "checkpoint": "/models/best_2.pth",
        "checkpoint_name": "best_2.pth",
        "status": "ok",
        "flags": [],
        "semantic_miou": 92.0,
        "rare_class_miou": 70.0,
        "mean_ce": 0.2,
        "occupancy_iou": 96.0,
        "latent_mean": 0.1,
        "latent_std": 0.7,
        "latent_abs_max": 9.0,
        "per_sequence": {
            "00": {"semantic_miou": 95.0, "occupancy_iou": 97.0, "mean_ce": 0.1},
            "01": {"semantic_miou": 80.0, "occupancy_iou": 90.0, "mean_ce": 0.5},
        },
    }
    payload = {
        "manifest": [
            {"sequence": "00", "frame": "000001", "label_path": "/labels/1.label"},
            {"sequence": "01", "frame": "000002", "label_path": "/labels/2.label"},
        ],
        "sequence_sampling": {
            "00": {"available_frames": 30, "sampled_frames": 1, "short_sequence": False},
            "01": {"available_frames": 2, "sampled_frames": 1, "short_sequence": True},
        },
        "recommended": recommended,
        "configured_comparison": {
            "available": True,
            "configured_checkpoint": "/models/best_1.pth",
            "recommended_checkpoint": "/models/best_2.pth",
            "deltas": {"semantic_miou": 2.0},
        },
        "ranking": [recommended],
        "frames": [
            {
                "checkpoint": "/models/best_2.pth",
                "sequence": "00",
                "frame": "000001",
                "ce": 0.1,
                "semantic_miou": 95.0,
                "occupancy_iou": 97.0,
                "latent_mean": 0.1,
                "latent_std": 0.7,
            }
        ],
    }

    paths = write_reports(payload, tmp_path)

    assert {path.name for path in paths} == {
        "vae_audit.json",
        "vae_audit_summary.csv",
        "vae_audit_frames.csv",
        "vae_audit_report.md",
    }
    saved = json.loads((tmp_path / "vae_audit.json").read_text())
    assert saved["manifest"] == payload["manifest"]
    with (tmp_path / "vae_audit_summary.csv").open(newline="") as handle:
        assert list(csv.DictReader(handle))[0]["checkpoint_name"] == "best_2.pth"
    report = (tmp_path / "vae_audit_report.md").read_text()
    assert "Recommended checkpoint" in report
    assert "Configured checkpoint" in report
    assert "01" in report
    assert "weakest" in report.lower()
    assert "short sequence" in report.lower()


def test_write_reports_serializes_nonfinite_metric_as_json_null(tmp_path):
    import json

    from scripts.audit.vae_audit import write_reports

    candidate = {
        "checkpoint": "/models/best.pth",
        "checkpoint_name": "best.pth",
        "status": "ok",
        "flags": [],
        "semantic_miou": float("nan"),
        "rare_class_miou": 0.0,
        "mean_ce": 1.0,
        "occupancy_iou": 0.0,
        "per_sequence": {},
    }
    payload = {
        "manifest": [],
        "recommended": candidate,
        "configured_comparison": {"available": False, "configured_checkpoint": ""},
        "ranking": [candidate],
        "frames": [],
    }

    write_reports(payload, tmp_path)

    saved = json.loads((tmp_path / "vae_audit.json").read_text())
    assert saved["recommended"]["semantic_miou"] is None
