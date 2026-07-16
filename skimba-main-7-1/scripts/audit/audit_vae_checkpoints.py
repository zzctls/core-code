#!/usr/bin/env python3
"""Audit all semantic VAE best checkpoints on a shared frame manifest."""

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]


def default_vae_audit_output_dir(
    repository_root=REPOSITORY_ROOT,
    project_name=PROJECT_ROOT.name,
):
    repository_root = Path(repository_root)
    return (
        repository_root.parent
        / f"{repository_root.name}-results"
        / project_name
        / "vae_audit"
    )

from scripts.audit.vae_audit import (  # noqa: E402
    aggregate_rows,
    build_frame_manifest,
    compare_configured_checkpoint,
    rank_candidates,
    summarize_sequence_sampling,
    write_reports,
)


BEST_CHECKPOINT_RE = re.compile(r"best_(\d+)_?([0-9.+-eE]*).*\.pth$")


def parse_csv_items(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _checkpoint_sort_key(path):
    match = BEST_CHECKPOINT_RE.match(path.name)
    if not match:
        return (math.inf, math.inf, path.name)
    try:
        score = float(match.group(2)) if match.group(2) else math.inf
    except ValueError:
        score = math.inf
    return (int(match.group(1)), score, path.name)


def discover_checkpoints(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    checkpoints = sorted(checkpoint_dir.glob("best_*.pth"), key=_checkpoint_sort_key)
    if not checkpoints:
        raise ValueError(f"No best_*.pth checkpoints found in {checkpoint_dir}")
    return checkpoints


def _project_relative(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def _config_value(config, section, key, default):
    return config.get(section, {}).get(key, default)


def build_evaluation_args(args, config):
    model = config.get("model_params", {})
    dataset = config.get("dataset_params", {})
    defaults = {
        "num_class": 20,
        "semantic_embed_dim": 8,
        "ignore_label": 255,
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
    }
    values = {
        key: dataset.get(key, default) if key == "ignore_label" else model.get(key, default)
        for key, default in defaults.items()
    }
    channels = values["autoencoder_channels_list"]
    if isinstance(channels, (list, tuple)):
        values["autoencoder_channels_list"] = ",".join(str(value) for value in channels)
    label_mapping = args.label_mapping or dataset.get("label_mapping", "")
    if not label_mapping:
        raise ValueError("Set dataset_params.label_mapping or pass --label-mapping")
    values.update(
        {
            "label_mapping": str(_project_relative(label_mapping)),
            "latent_shape": args.latent_shape,
            "mode": "mean",
            "rare_class_count": args.rare_class_count,
            "min_latent_std": args.min_latent_std,
            "max_latent_std": args.max_latent_std,
            "max_latent_abs": args.max_latent_abs,
        }
    )
    return SimpleNamespace(**values)


def resolve_label_root(args, config):
    dataset = config.get("dataset_params", {})
    if args.label_root:
        return Path(args.label_root)
    gt_root = Path(str(dataset.get("gt_root", "data_odometry_voxels_all")))
    if gt_root.is_absolute():
        return gt_root
    data_root = Path(str(args.dataset_root or dataset.get("data_root", "")))
    return data_root / gt_root if str(data_root) else gt_root


def _health_flags(summary, args):
    flags = []
    if summary["nonfinite_latent_frames"]:
        flags.append("latent_contains_nonfinite")
    if summary["nonfinite_logit_frames"]:
        flags.append("decoded_logits_contain_nonfinite")
    latent_std = float(summary["latent_std"])
    latent_abs_max = float(summary["latent_abs_max"])
    if not math.isfinite(latent_std):
        flags.append("latent_std_nonfinite")
    elif latent_std < args.min_latent_std:
        flags.append(f"latent_std_lt_{args.min_latent_std}")
    elif latent_std > args.max_latent_std:
        flags.append(f"latent_std_gt_{args.max_latent_std}")
    if not math.isfinite(latent_abs_max):
        flags.append("latent_abs_max_nonfinite")
    elif latent_abs_max > args.max_latent_abs:
        flags.append(f"latent_abs_max_gt_{args.max_latent_abs}")
    return flags


def evaluate_checkpoint_frames(checkpoint_path, manifest, args, requested_device):
    """Load one checkpoint once and return auditable evidence for every frame."""
    import torch

    from scripts.data.export_z0_semantic_latents import (
        build_learning_map_lut,
        load_autoencoder_from_checkpoint,
        load_checkpoint,
        load_semantic_embedding_from_checkpoint,
    )
    from scripts.data.select_best_semantic_vae import evaluate_label_file

    if requested_device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested_device}, but CUDA is unavailable")
    device = torch.device(requested_device)
    checkpoint = load_checkpoint(checkpoint_path, torch)
    autoencoder = load_autoencoder_from_checkpoint(args, checkpoint, device)
    semantic_embedding = load_semantic_embedding_from_checkpoint(args, checkpoint, device)
    learning_map_lut = build_learning_map_lut(args.label_mapping)

    rows = []
    for manifest_row in manifest:
        frame_row, confusion, _latent_stats = evaluate_label_file(
            autoencoder,
            semantic_embedding,
            Path(manifest_row["label_path"]),
            learning_map_lut,
            args,
            device,
        )
        frame_row["confusion"] = confusion.tolist()
        frame_row["ce_sum"] = frame_row["ce"] * frame_row["valid_voxels"]
        rows.append(frame_row)
    return rows


def _load_training_config(config_path):
    from scripts.data.config_paths import load_training_config

    return load_training_config(config_path)


def _error_summary(checkpoint, error):
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "status": "error",
        "flags": ["evaluation_error"],
        "error": str(error),
        "semantic_miou": math.nan,
        "rare_class_miou": math.nan,
        "mean_ce": math.nan,
        "occupancy_iou": math.nan,
    }


def run_audit(args, evaluator=None, config_loader=None):
    evaluator = evaluator or evaluate_checkpoint_frames
    config_loader = config_loader or _load_training_config
    config = config_loader(args.config_path)
    evaluation_args = build_evaluation_args(args, config)
    checkpoint_paths = discover_checkpoints(args.checkpoint_dir)
    label_root = resolve_label_root(args, config)
    selected_sequences = parse_csv_items(args.sequences)
    manifest = build_frame_manifest(
        label_root,
        frames_per_sequence=args.frames_per_sequence,
        seed=args.seed,
        sequences=selected_sequences,
    )
    sequence_sampling = summarize_sequence_sampling(
        label_root,
        manifest,
        frames_per_sequence=args.frames_per_sequence,
        sequences=selected_sequences,
    )

    summaries = []
    public_frame_rows = []
    for checkpoint_path in checkpoint_paths:
        try:
            frame_rows = evaluator(
                checkpoint_path,
                manifest,
                evaluation_args,
                args.device,
            )
            if len(frame_rows) != len(manifest):
                raise RuntimeError(
                    f"Evaluator returned {len(frame_rows)} frames for a {len(manifest)}-frame manifest"
                )
            summary = aggregate_rows(
                frame_rows,
                num_classes=evaluation_args.num_class,
                rare_class_count=args.rare_class_count,
            )
            grouped = defaultdict(list)
            for row in frame_rows:
                grouped[row["sequence"]].append(row)
            summary["per_sequence"] = {
                sequence: aggregate_rows(
                    rows,
                    num_classes=evaluation_args.num_class,
                    rare_class_count=args.rare_class_count,
                )
                for sequence, rows in sorted(grouped.items())
            }
            summary.update(
                {
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_name": checkpoint_path.name,
                    "status": "ok",
                }
            )
            summary["flags"] = _health_flags(summary, args)
            for row in frame_rows:
                public_frame_rows.append(
                    {
                        key: value
                        for key, value in {
                            **row,
                            "checkpoint": str(checkpoint_path),
                            "checkpoint_name": checkpoint_path.name,
                        }.items()
                        if key not in {"confusion", "ce_sum"}
                    }
                )
        except Exception as exc:  # Keep one bad candidate from aborting the comparison.
            summary = _error_summary(checkpoint_path, exc)
        summaries.append(summary)

    ranked = rank_candidates(summaries)
    recommended = next((row for row in ranked if row.get("status") == "ok"), None)
    configured_checkpoint = _config_value(
        config,
        "train_params",
        "vae_checkpoint",
        "",
    )
    configured_comparison = (
        compare_configured_checkpoint(ranked, configured_checkpoint)
        if recommended is not None
        else {
            "available": False,
            "configured_checkpoint": str(configured_checkpoint),
            "reason": "no checkpoint evaluated successfully",
        }
    )
    payload = {
        "config_path": str(args.config_path),
        "checkpoint_dir": str(args.checkpoint_dir),
        "label_root": str(label_root),
        "frames_per_sequence": args.frames_per_sequence,
        "seed": args.seed,
        "mode": "mean",
        "manifest": manifest,
        "sequence_sampling": sequence_sampling,
        "recommended": recommended,
        "configured_comparison": configured_comparison,
        "ranking": ranked,
        "frames": public_frame_rows,
        "selection_rule": (
            "successful and unflagged first, then semantic mIoU desc, rare-class mIoU "
            "desc, CE asc, occupancy IoU desc"
        ),
    }
    write_reports(payload, args.output_dir)

    critical_flags = {
        "latent_contains_nonfinite",
        "decoded_logits_contain_nonfinite",
        "latent_std_nonfinite",
        "latent_abs_max_nonfinite",
    }
    exit_code = 2 if recommended is None else 0
    if recommended is not None and critical_flags.intersection(recommended.get("flags", [])):
        exit_code = 2
    return exit_code, payload


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Audit every best_*.pth semantic VAE checkpoint on the same deterministic "
            "sample from every available sequence."
        )
    )
    parser.add_argument("--config-path", default="config/semantickitti_autoencoder.yaml")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--frames-per-sequence", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument(
        "--output-dir",
        default=str(default_vae_audit_output_dir()),
    )
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--label-root", default="")
    parser.add_argument("--label-mapping", default="")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rare-class-count", type=int, default=5)
    parser.add_argument("--min-latent-std", type=float, default=1e-4)
    parser.add_argument("--max-latent-std", type=float, default=10.0)
    parser.add_argument("--max-latent-abs", type=float, default=50.0)
    parser.add_argument("--latent-shape", default="8,64,64,8")
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    try:
        exit_code, payload = run_audit(args)
    except Exception as exc:  # CLI boundary: provide one concise fatal diagnostic.
        print(f"VAE audit failed: {exc}", file=sys.stderr)
        return 2
    recommended = payload.get("recommended")
    if recommended:
        print(f"Recommended checkpoint: {recommended['checkpoint']}")
        print(f"Semantic mIoU: {recommended['semantic_miou']:.3f}")
    else:
        print("No checkpoint evaluated successfully", file=sys.stderr)
    print(f"Reports: {Path(args.output_dir).resolve()}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
