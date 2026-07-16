"""Pure helpers for reproducible semantic VAE checkpoint audits."""

import csv
import json
import math
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def build_frame_manifest(
    gt_root,
    frames_per_sequence: int = 20,
    seed: int = 0,
    sequences: Optional[Iterable[str]] = None,
):
    """Select one deterministic, shared frame sample from every sequence."""
    if frames_per_sequence <= 0:
        raise ValueError("frames_per_sequence must be positive")

    gt_root = Path(gt_root)
    requested = {str(sequence) for sequence in sequences} if sequences else None
    manifest = []
    for voxel_dir in sorted((gt_root / "sequences").glob("*/voxels")):
        sequence = voxel_dir.parent.name
        if requested is not None and sequence not in requested:
            continue
        frames = sorted(voxel_dir.glob("*.label"))
        if len(frames) > frames_per_sequence:
            rng = random.Random(f"{seed}:{sequence}")
            frames = sorted(rng.sample(frames, frames_per_sequence))
        manifest.extend(
            {
                "sequence": sequence,
                "frame": path.stem,
                "label_path": str(path),
            }
            for path in frames
        )

    if not manifest:
        raise ValueError(f"No label frames found under {gt_root / 'sequences'}")
    return manifest


def summarize_sequence_sampling(
    gt_root,
    manifest,
    frames_per_sequence: int,
    sequences: Optional[Iterable[str]] = None,
):
    """Describe available and selected frames, including short sequences."""
    gt_root = Path(gt_root)
    requested = {str(sequence) for sequence in sequences} if sequences else None
    sampled_counts = {}
    for row in manifest:
        sequence = str(row["sequence"])
        sampled_counts[sequence] = sampled_counts.get(sequence, 0) + 1

    summary = {}
    for voxel_dir in sorted((gt_root / "sequences").glob("*/voxels")):
        sequence = voxel_dir.parent.name
        if requested is not None and sequence not in requested:
            continue
        available = len(list(voxel_dir.glob("*.label")))
        if not available:
            continue
        summary[sequence] = {
            "available_frames": available,
            "sampled_frames": sampled_counts.get(sequence, 0),
            "short_sequence": available < frames_per_sequence,
        }
    return summary


def confusion_from_labels(target, prediction, num_classes: int, ignore_label: int = 255):
    """Build a target-by-prediction confusion matrix for valid class labels."""
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
    if target.shape != prediction.shape:
        raise ValueError("target and prediction must contain the same number of labels")
    valid = (
        (target != ignore_label)
        & (target >= 0)
        & (target < num_classes)
        & (prediction >= 0)
        & (prediction < num_classes)
    )
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    if valid.any():
        encoded = num_classes * target[valid] + prediction[valid]
        confusion += np.bincount(
            encoded,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
    return confusion


def per_class_iou_percent(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    true_positive = np.diag(confusion)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - true_positive
    iou = np.full(confusion.shape[0], math.nan, dtype=np.float64)
    present = union > 0
    iou[present] = true_positive[present] / union[present] * 100.0
    return iou


def _finite_mean(values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else math.nan


def aggregate_rows(rows, num_classes: int, rare_class_count: int = 5):
    """Aggregate frame evidence without averaging frame-level IoUs."""
    if not rows:
        raise ValueError("Cannot aggregate an empty frame list")

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for row in rows:
        frame_confusion = np.asarray(row["confusion"], dtype=np.int64)
        if frame_confusion.shape != confusion.shape:
            raise ValueError(
                f"Confusion shape {frame_confusion.shape} does not match {(num_classes, num_classes)}"
            )
        confusion += frame_confusion

    class_iou = per_class_iou_percent(confusion)
    semantic_iou = class_iou[1:] if num_classes > 1 else class_iou
    present_semantic = [float(value) for value in semantic_iou if math.isfinite(float(value))]
    present_all = [float(value) for value in class_iou if math.isfinite(float(value))]
    rare_count = min(max(rare_class_count, 0), len(present_semantic))

    valid_voxels = int(sum(int(row["valid_voxels"]) for row in rows))
    correct_voxels = int(sum(int(row["correct_voxels"]) for row in rows))
    ce_sum = float(sum(float(row["ce_sum"]) for row in rows))
    occupancy_intersection = int(sum(int(row["occupancy_intersection"]) for row in rows))
    occupancy_union = int(sum(int(row["occupancy_union"]) for row in rows))

    return {
        "num_frames": len(rows),
        "mean_ce": ce_sum / valid_voxels if valid_voxels else math.nan,
        "semantic_miou": _finite_mean(present_semantic),
        "semantic_miou_with_empty": _finite_mean(present_all),
        "per_class_iou": [float(value) for value in class_iou],
        "rare_class_miou": (
            _finite_mean(sorted(present_semantic)[:rare_count]) if rare_count else math.nan
        ),
        "min_class_iou": min(present_semantic) if present_semantic else math.nan,
        "occupancy_iou": (
            100.0 * occupancy_intersection / occupancy_union
            if occupancy_union
            else math.nan
        ),
        "voxel_accuracy": (
            100.0 * correct_voxels / valid_voxels if valid_voxels else math.nan
        ),
        "valid_voxels": valid_voxels,
        "correct_voxels": correct_voxels,
        "occupancy_intersection": occupancy_intersection,
        "occupancy_union": occupancy_union,
        "latent_mean": _finite_mean(row["latent_mean"] for row in rows),
        "latent_std": _finite_mean(row["latent_std"] for row in rows),
        "latent_rms": _finite_mean(row["latent_rms"] for row in rows),
        "latent_l2": _finite_mean(row["latent_l2"] for row in rows),
        "latent_abs_max": max(float(row["latent_abs_max"]) for row in rows),
        "nonfinite_latent_frames": sum(not bool(row["latent_finite"]) for row in rows),
        "nonfinite_logit_frames": sum(not bool(row["logits_finite"]) for row in rows),
    }


def _finite_for_sort(value, default):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def rank_candidates(candidates):
    """Rank checkpoint summaries using reconstruction integrity before quality."""

    def sort_key(row):
        return (
            0 if row.get("status") == "ok" else 1,
            1 if row.get("flags") else 0,
            -_finite_for_sort(row.get("semantic_miou"), -math.inf),
            -_finite_for_sort(row.get("rare_class_miou"), -math.inf),
            _finite_for_sort(row.get("mean_ce"), math.inf),
            -_finite_for_sort(row.get("occupancy_iou"), -math.inf),
            str(row.get("checkpoint", "")),
        )

    return sorted(candidates, key=sort_key)


def _resolved_path(path):
    return str(Path(path).expanduser().resolve())


def compare_configured_checkpoint(ranked, configured_checkpoint):
    """Compare the recommendation with the YAML checkpoint when both were run."""
    configured_checkpoint = str(configured_checkpoint or "")
    configured_resolved = _resolved_path(configured_checkpoint) if configured_checkpoint else ""
    configured = next(
        (
            row
            for row in ranked
            if configured_resolved
            and _resolved_path(row.get("checkpoint", "")) == configured_resolved
        ),
        None,
    )
    if configured is None:
        return {
            "available": False,
            "configured_checkpoint": configured_checkpoint,
            "reason": "configured checkpoint was not evaluated",
        }

    recommended = ranked[0]
    metric_names = (
        "semantic_miou",
        "rare_class_miou",
        "mean_ce",
        "occupancy_iou",
    )
    deltas = {
        name: round(float(recommended[name]) - float(configured[name]), 12)
        for name in metric_names
    }
    return {
        "available": True,
        "configured_checkpoint": configured_checkpoint,
        "recommended_checkpoint": recommended["checkpoint"],
        "deltas": deltas,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _csv_value(value):
    value = _json_safe(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path, rows, preferred_fields):
    rows = list(rows)
    extra_fields = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in preferred_fields and not isinstance(value, dict)
        }
    )
    fieldnames = list(preferred_fields) + extra_fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _format_metric(value, digits=3):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "n/a"


def _render_markdown(payload):
    recommended = payload.get("recommended") or {}
    comparison = payload.get("configured_comparison", {})
    lines = [
        "# Semantic VAE Checkpoint Audit",
        "",
        f"- **Recommended checkpoint:** `{recommended.get('checkpoint', '') or 'unavailable'}`",
        f"- **Configured checkpoint:** `{comparison.get('configured_checkpoint', '') or 'unavailable'}`",
        f"- **Sampled frames:** {len(payload.get('manifest', []))}",
        "",
        "## Checkpoint ranking",
        "",
        "| Rank | Checkpoint | Status | Flags | Semantic mIoU | Rare mIoU | CE | Occupancy IoU |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(payload.get("ranking", []), start=1):
        flags = ", ".join(row.get("flags") or []) or "-"
        lines.append(
            "| {rank} | `{name}` | {status} | {flags} | {miou} | {rare} | {ce} | {occ} |".format(
                rank=index,
                name=row.get("checkpoint_name") or Path(row.get("checkpoint", "")).name,
                status=row.get("status", ""),
                flags=flags,
                miou=_format_metric(row.get("semantic_miou")),
                rare=_format_metric(row.get("rare_class_miou")),
                ce=_format_metric(row.get("mean_ce"), digits=6),
                occ=_format_metric(row.get("occupancy_iou")),
            )
        )

    sequence_sampling = payload.get("sequence_sampling") or {}
    lines.extend(
        [
            "",
            "## Frame sampling by sequence",
            "",
            "| Sequence | Available | Sampled | Note |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for sequence, sampling in sorted(sequence_sampling.items()):
        note = "short sequence" if sampling.get("short_sequence") else "-"
        lines.append(
            f"| {sequence} | {sampling.get('available_frames', 0)} | "
            f"{sampling.get('sampled_frames', 0)} | {note} |"
        )

    per_sequence = recommended.get("per_sequence") or {}
    lines.extend(
        [
            "",
            "## Recommended checkpoint by sequence",
            "",
            "| Sequence | Semantic mIoU | Occupancy IoU | CE |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    sorted_sequences = sorted(
        per_sequence.items(),
        key=lambda item: _finite_for_sort(item[1].get("semantic_miou"), math.inf),
    )
    for sequence, summary in sorted_sequences:
        lines.append(
            f"| {sequence} | {_format_metric(summary.get('semantic_miou'))} | "
            f"{_format_metric(summary.get('occupancy_iou'))} | "
            f"{_format_metric(summary.get('mean_ce'), digits=6)} |"
        )
    if sorted_sequences:
        lines.extend(
            [
                "",
                f"Weakest sequence by semantic mIoU: **{sorted_sequences[0][0]}**.",
            ]
        )

    if comparison.get("available"):
        lines.extend(["", "## Difference from configured checkpoint", ""])
        for name, delta in comparison.get("deltas", {}).items():
            lines.append(f"- `{name}`: {_format_metric(delta, digits=6)}")
    else:
        lines.extend(
            [
                "",
                "Configured-checkpoint comparison is unavailable: "
                + comparison.get("reason", "checkpoint was not evaluated")
                + ".",
            ]
        )
    return "\n".join(lines) + "\n"


def write_reports(payload, output_dir):
    """Write complete machine-readable evidence and a concise human summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "vae_audit.json"
    summary_path = output_dir / "vae_audit_summary.csv"
    frames_path = output_dir / "vae_audit_frames.csv"
    markdown_path = output_dir / "vae_audit_report.md"

    json_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(
        summary_path,
        payload.get("ranking", []),
        (
            "checkpoint",
            "checkpoint_name",
            "status",
            "flags",
            "num_frames",
            "semantic_miou",
            "semantic_miou_with_empty",
            "rare_class_miou",
            "min_class_iou",
            "mean_ce",
            "occupancy_iou",
            "voxel_accuracy",
            "latent_mean",
            "latent_std",
            "latent_rms",
            "latent_l2",
            "latent_abs_max",
            "nonfinite_latent_frames",
            "nonfinite_logit_frames",
            "error",
        ),
    )
    _write_csv(
        frames_path,
        payload.get("frames", []),
        (
            "checkpoint",
            "checkpoint_name",
            "sequence",
            "frame",
            "label_path",
            "ce",
            "semantic_miou",
            "occupancy_iou",
            "voxel_accuracy",
            "valid_voxels",
            "latent_mean",
            "latent_std",
            "latent_rms",
            "latent_l2",
            "latent_abs_max",
            "latent_finite",
            "logits_finite",
        ),
    )
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return [json_path, summary_path, frames_path, markdown_path]
