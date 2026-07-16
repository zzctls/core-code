import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPATIAL_SHAPE = (256, 256, 32)
IGNORE_LABEL = 255
BEST_CHECKPOINT_RE = re.compile(r"best_(\d+)_([0-9.+-eE]+).*\.pth$")


def parse_csv_items(value):
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def safe_mean(values):
    clean = [float(value) for value in values if is_finite_number(value)]
    return float(sum(clean) / len(clean)) if clean else math.nan


def is_finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def finite_for_sort(value, default):
    return float(value) if is_finite_number(value) else default


def resolve_project_relative_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    project_path = PROJECT_ROOT / path
    return project_path if project_path.exists() else path


def checkpoint_sort_key(path):
    path = Path(path)
    match = BEST_CHECKPOINT_RE.match(path.name)
    if match:
        return (0, int(match.group(1)), float(match.group(2)), path.name)
    return (1, math.inf, math.nan, path.name)


def parse_yaml_scalar(value):
    value = value.strip()
    if not value:
        return ""
    value = value.split("#", 1)[0].strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value in {"True", "true"}:
        return True
    if value in {"False", "false"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_minimal_config_defaults(config_path):
    """Fallback for environments without strictyaml; reads only scalar defaults this script needs."""
    data = {"dataset_params": {}, "train_params": {}}
    current_section = None
    for raw_line in Path(config_path).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = raw_line.split("#", 1)[0].rstrip()
        if not raw_line.startswith((" ", "\t")) and line.endswith(":"):
            section = line[:-1].strip()
            current_section = section if section in data else None
            continue
        if current_section is None or not raw_line.startswith("  "):
            continue
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        if value.strip():
            data[current_section][key] = parse_yaml_scalar(value)
    return data


def load_config_defaults(config_path):
    try:
        from scripts.data.config_paths import load_training_config

        return load_training_config(config_path)
    except ModuleNotFoundError as exc:
        if exc.name != "strictyaml":
            raise
        return load_minimal_config_defaults(config_path)


def discover_checkpoints(args):
    checkpoints = [Path(path) for path in args.vae_checkpoints]
    if args.vae_checkpoint_dir:
        checkpoint_dir = Path(args.vae_checkpoint_dir)
        checkpoints.extend(checkpoint_dir.glob(args.checkpoint_pattern))

    unique = {}
    for checkpoint in checkpoints:
        unique[str(checkpoint)] = checkpoint

    discovered = sorted(unique.values(), key=checkpoint_sort_key)
    if not discovered:
        raise ValueError("No VAE checkpoints found. Pass --vae-checkpoints or --vae-checkpoint-dir.")
    return discovered


def latent_flags(summary, args):
    flags = []
    latent_std = summary.get("latent_std", math.nan)
    latent_abs_max = summary.get("latent_abs_max", math.nan)
    latent_mean = summary.get("latent_mean", math.nan)

    if not is_finite_number(latent_mean):
        flags.append("latent_mean_nonfinite")
    if not is_finite_number(latent_std):
        flags.append("latent_std_nonfinite")
    elif latent_std < args.min_latent_std:
        flags.append(f"latent_std_lt_{args.min_latent_std}")
    elif latent_std > args.max_latent_std:
        flags.append(f"latent_std_gt_{args.max_latent_std}")

    if not is_finite_number(latent_abs_max):
        flags.append("latent_abs_max_nonfinite")
    elif latent_abs_max > args.max_latent_abs:
        flags.append(f"latent_abs_max_gt_{args.max_latent_abs}")

    if summary.get("nonfinite_latent_frames", 0):
        flags.append("latent_contains_nonfinite")
    if summary.get("nonfinite_logit_frames", 0):
        flags.append("decoded_logits_contain_nonfinite")
    return flags


def rank_rows(rows):
    def sort_key(row):
        return (
            0 if row.get("status") == "ok" else 1,
            1 if row.get("flags") else 0,
            -finite_for_sort(row.get("semantic_miou"), -math.inf),
            -finite_for_sort(row.get("rare_class_miou"), -math.inf),
            finite_for_sort(row.get("mean_ce"), math.inf),
            -finite_for_sort(row.get("occupancy_iou"), -math.inf),
            str(row.get("checkpoint", "")),
        )

    return sorted(rows, key=sort_key)


def confusion_from_prediction(prediction_np, target_np, ignore_label, num_classes):
    target_flat = np.asarray(target_np).reshape(-1).astype(np.int64)
    pred_flat = np.asarray(prediction_np).reshape(-1).astype(np.int64)
    valid = (
        (target_flat != ignore_label)
        & (target_flat >= 0)
        & (target_flat < num_classes)
        & (pred_flat >= 0)
        & (pred_flat < num_classes)
    )
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    if valid.any():
        encoded = num_classes * target_flat[valid] + pred_flat[valid]
        confusion += np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return confusion


def per_class_iou_percent(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    tp = np.diag(confusion)
    target_count = confusion.sum(axis=1)
    pred_count = confusion.sum(axis=0)
    union = target_count + pred_count - tp
    iou = np.full((confusion.shape[0],), math.nan, dtype=np.float64)
    valid = union > 0
    iou[valid] = tp[valid] / union[valid] * 100.0
    return iou


def latent_stats(latent):
    import torch

    latent = latent.detach().float()
    finite = bool(torch.isfinite(latent).all().item())
    return {
        "latent_mean": float(latent.mean().item()),
        "latent_std": float(latent.std(unbiased=False).item()),
        "latent_abs_max": float(latent.abs().max().item()),
        "latent_rms": float(torch.sqrt(torch.mean(latent * latent)).item()),
        "latent_l2": float(torch.linalg.vector_norm(latent).item()),
        "latent_finite": finite,
    }


def aggregate_checkpoint(frame_rows, confusion, latent_stat_rows, args):
    class_iou = per_class_iou_percent(confusion)
    semantic_class_iou = class_iou[1:] if args.num_class > 1 else class_iou
    present_semantic_iou = [value for value in semantic_class_iou if is_finite_number(value)]
    rare_count = min(args.rare_class_count, len(present_semantic_iou))
    rare_class_miou = safe_mean(sorted(present_semantic_iou)[:rare_count]) if rare_count else math.nan

    summary = {
        "num_frames": len(frame_rows),
        "mean_ce": safe_mean(row["ce"] for row in frame_rows),
        "semantic_miou": safe_mean(semantic_class_iou),
        "semantic_miou_with_empty": safe_mean(class_iou),
        "rare_class_miou": rare_class_miou,
        "min_class_iou": min(present_semantic_iou) if present_semantic_iou else math.nan,
        "occupancy_iou": safe_mean(row["occupancy_iou"] for row in frame_rows),
        "voxel_accuracy": safe_mean(row["voxel_accuracy"] for row in frame_rows),
        "valid_voxels": int(sum(row["valid_voxels"] for row in frame_rows)),
        "latent_mean": safe_mean(row["latent_mean"] for row in latent_stat_rows),
        "latent_std": safe_mean(row["latent_std"] for row in latent_stat_rows),
        "latent_abs_max": max(row["latent_abs_max"] for row in latent_stat_rows) if latent_stat_rows else math.nan,
        "latent_rms": safe_mean(row["latent_rms"] for row in latent_stat_rows),
        "latent_l2": safe_mean(row["latent_l2"] for row in latent_stat_rows),
        "nonfinite_latent_frames": int(sum(not row["latent_finite"] for row in latent_stat_rows)),
        "nonfinite_logit_frames": int(sum(not row.get("logits_finite", True) for row in frame_rows)),
    }
    return summary


def evaluate_label_file(autoencoder, semantic_embedding, label_path, learning_map_lut, args, device):
    import torch

    from scripts.data.export_z0_semantic_latents import (
        build_sparse_scene_from_target,
        parse_shape,
        semantic_target_from_label,
    )
    from scripts.data.validate_semantic_vae_latents import compute_content_metrics

    target = semantic_target_from_label(label_path, learning_map_lut, args.ignore_label)
    sparse_scene = build_sparse_scene_from_target(target, semantic_embedding, args, device)
    with torch.no_grad():
        _, posterior = autoencoder.encode(sparse_scene)
        dist = posterior.latent_dist
        latent = dist.sample() if args.mode == "sample" else dist.mean
        expected_latent_shape = (1, *parse_shape(args.latent_shape))
        if tuple(latent.shape) != expected_latent_shape:
            raise ValueError(
                f"Encoded latent shape {tuple(latent.shape)} != expected {expected_latent_shape}."
            )
        logits = autoencoder.decode(latent)

    if tuple(logits.shape) != (1, args.num_class, *SPATIAL_SHAPE):
        raise ValueError(f"Decoded logits shape {tuple(logits.shape)} != expected {(1, args.num_class, *SPATIAL_SHAPE)}.")

    logits_finite = bool(torch.isfinite(logits).all().item())
    metrics = compute_content_metrics(logits, target, args.ignore_label, args.num_class)
    prediction = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)
    confusion = confusion_from_prediction(prediction, target, args.ignore_label, args.num_class)
    stats = latent_stats(latent)
    sequence = Path(label_path).parent.parent.name
    frame = Path(label_path).stem
    return {
        "sequence": sequence,
        "frame": frame,
        "label_path": str(label_path),
        "logits_finite": logits_finite,
        **metrics,
        **stats,
    }, confusion, stats


def evaluate_checkpoint(checkpoint_path, label_files, args, device):
    import torch

    from scripts.data.export_z0_semantic_latents import (
        build_learning_map_lut,
        load_autoencoder_from_checkpoint,
        load_checkpoint,
        load_semantic_embedding_from_checkpoint,
    )

    checkpoint = load_checkpoint(checkpoint_path, torch)
    autoencoder = load_autoencoder_from_checkpoint(args, checkpoint, device)
    semantic_embedding = load_semantic_embedding_from_checkpoint(args, checkpoint, device)
    learning_map_lut = build_learning_map_lut(args.label_mapping)

    frame_rows = []
    latent_stat_rows = []
    confusion = np.zeros((args.num_class, args.num_class), dtype=np.int64)
    for label_path in label_files:
        frame_row, frame_confusion, frame_latent_stats = evaluate_label_file(
            autoencoder,
            semantic_embedding,
            label_path,
            learning_map_lut,
            args,
            device,
        )
        frame_rows.append(frame_row)
        latent_stat_rows.append(frame_latent_stats)
        confusion += frame_confusion

    summary = aggregate_checkpoint(frame_rows, confusion, latent_stat_rows, args)
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_name": Path(checkpoint_path).name,
            "status": "ok",
            "flags": [],
        }
    )
    summary["flags"] = latent_flags(summary, args)
    return summary, frame_rows


def evaluate_checkpoint_or_error(checkpoint_path, label_files, args, device):
    try:
        summary, frame_rows = evaluate_checkpoint(checkpoint_path, label_files, args, device)
        return summary, frame_rows
    except Exception as exc:  # noqa: BLE001 - keep batch ranking alive for other checkpoints.
        return {
            "checkpoint": str(checkpoint_path),
            "checkpoint_name": Path(checkpoint_path).name,
            "status": "error",
            "error": str(exc),
            "flags": ["evaluation_error"],
            "semantic_miou": math.nan,
            "rare_class_miou": math.nan,
            "mean_ce": math.nan,
            "occupancy_iou": math.nan,
        }, []


def apply_config_defaults(args):
    args.config_path = str(resolve_project_relative_path(args.config_path))
    configs = load_config_defaults(args.config_path)
    dataset_config = configs.get("dataset_params", {})
    train_params = configs.get("train_params", {})

    if not args.label_mapping:
        args.label_mapping = dataset_config.get("label_mapping", "")
    if args.label_mapping:
        args.label_mapping = str(resolve_project_relative_path(args.label_mapping))
    if not args.dataset_root:
        args.dataset_root = dataset_config.get("data_root", "")
    if not args.label_root:
        gt_root = dataset_config.get("gt_root", "data_odometry_voxel_all")
        gt_root_path = Path(str(gt_root))
        if gt_root_path.is_absolute():
            args.label_root = str(gt_root_path)
        elif args.dataset_root:
            args.label_root = str(Path(args.dataset_root) / gt_root_path)
        else:
            args.label_root = str(gt_root_path)
    if not args.vae_checkpoints and not args.vae_checkpoint_dir:
        configured_checkpoint = train_params.get("vae_checkpoint", "")
        if configured_checkpoint:
            args.vae_checkpoints = [configured_checkpoint]
    return args


def collect_label_files(args):
    from scripts.data.export_z0_semantic_latents import resolve_voxel_dir, select_label_files

    if not args.label_mapping:
        raise ValueError("Set --label-mapping or dataset_params.label_mapping.")
    if not args.dataset_root and not args.label_root:
        raise ValueError("Set --dataset-root, --label-root, or dataset_params.data_root/gt_root.")

    label_files = []
    frames = parse_csv_items(args.frames)
    for sequence in parse_csv_items(args.sequences):
        voxel_dir = resolve_voxel_dir(args.dataset_root, sequence, args.label_root or None)
        label_files.extend(select_label_files(voxel_dir, frames, args.num_samples))
    if not label_files:
        raise RuntimeError("No label files selected for VAE evaluation.")
    return label_files


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_ranking(rows):
    print("Rank | Checkpoint | Status | Flags | mIoU | Rare mIoU | CE | Occ IoU | z_mean | z_std | z_absmax")
    for index, row in enumerate(rows, start=1):
        flags = ",".join(row.get("flags") or []) or "-"
        print(
            "{rank:>4} | {name} | {status} | {flags} | {miou:.3f} | {rare:.3f} | "
            "{ce:.6f} | {occ:.3f} | {zmean:.5f} | {zstd:.5f} | {zabs:.3f}".format(
                rank=index,
                name=row.get("checkpoint_name", Path(row.get("checkpoint", "")).name),
                status=row.get("status", ""),
                flags=flags,
                miou=finite_for_sort(row.get("semantic_miou"), math.nan),
                rare=finite_for_sort(row.get("rare_class_miou"), math.nan),
                ce=finite_for_sort(row.get("mean_ce"), math.nan),
                occ=finite_for_sort(row.get("occupancy_iou"), math.nan),
                zmean=finite_for_sort(row.get("latent_mean"), math.nan),
                zstd=finite_for_sort(row.get("latent_std"), math.nan),
                zabs=finite_for_sort(row.get("latent_abs_max"), math.nan),
            )
        )


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multiple Code_VAE_SSC semantic VAE checkpoints on the same "
            "SemanticKITTI frames and rank them for skimba latent diffusion use."
        )
    )
    parser.add_argument("--config_path", "--config-path", default="config/semantickitti_autoencoder.yaml")
    parser.add_argument("--vae_checkpoints", "--vae-checkpoints", nargs="*", default=[])
    parser.add_argument("--vae_checkpoint_dir", "--vae-checkpoint-dir", default="")
    parser.add_argument("--checkpoint_pattern", "--checkpoint-pattern", default="best_*.pth")
    parser.add_argument("--dataset_root", "--dataset-root", default="")
    parser.add_argument("--label_root", "--label-root", default="")
    parser.add_argument("--label_mapping", "--label-mapping", default="")
    parser.add_argument("--sequences", default="08")
    parser.add_argument("--frames", default="")
    parser.add_argument("--num_samples", "--num-samples", type=int, default=20)
    parser.add_argument("--mode", choices=["mean", "sample"], default="mean")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_json", "--out-json", default="")
    parser.add_argument("--out_csv", "--out-csv", default="")
    parser.add_argument("--frame_csv", "--frame-csv", default="")
    parser.add_argument("--rare_class_count", "--rare-class-count", type=int, default=5)
    parser.add_argument("--min_latent_std", "--min-latent-std", type=float, default=1e-4)
    parser.add_argument("--max_latent_std", "--max-latent-std", type=float, default=10.0)
    parser.add_argument("--max_latent_abs", "--max-latent-abs", type=float, default=50.0)
    parser.add_argument("--latent_shape", "--latent-shape", default="8,64,64,8")
    parser.add_argument("--num_class", "--num-class", type=int, default=20)
    parser.add_argument("--semantic_embed_dim", "--semantic-embed-dim", type=int, default=8)
    parser.add_argument("--ignore_label", "--ignore-label", type=int, default=IGNORE_LABEL)
    parser.add_argument("--in_channels", "--in-channels", type=int, default=8)
    parser.add_argument("--out_channels", "--out-channels", type=int, default=8)
    parser.add_argument("--latent_channels", "--latent-channels", type=int, default=64)
    parser.add_argument("--autoencoder_num_res_blocks", "--autoencoder-num-res-blocks", type=int, default=1)
    parser.add_argument("--autoencoder_channels_list", "--autoencoder-channels-list", default="16,32,64,128")
    parser.add_argument("--auto_groups", "--auto-groups", type=int, default=4)
    parser.add_argument("--num_input_features", "--num-input-features", type=int, default=3)
    parser.add_argument("--init_size", "--init-size", type=int, default=8)
    parser.add_argument("--voxel_channel", "--voxel-channel", type=int, default=1)
    parser.add_argument("--dropout_rate", "--dropout-rate", type=float, default=0.2)
    return parser


def main():
    args = apply_config_defaults(build_argparser().parse_args())

    import torch

    requested_device = args.device
    if requested_device != "cpu" and not torch.cuda.is_available():
        print(f"Warning: requested device {requested_device}, but CUDA is unavailable; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    checkpoints = discover_checkpoints(args)
    label_files = collect_label_files(args)
    print(f"Checkpoints: {len(checkpoints)} | Frames: {len(label_files)} | mode={args.mode} | device={device}")

    summaries = []
    all_frame_rows = []
    for index, checkpoint_path in enumerate(checkpoints, start=1):
        print(f"[{index}/{len(checkpoints)}] Evaluating {checkpoint_path}")
        summary, frame_rows = evaluate_checkpoint_or_error(checkpoint_path, label_files, args, device)
        summaries.append(summary)
        for frame_row in frame_rows:
            frame_row["checkpoint"] = str(checkpoint_path)
            frame_row["checkpoint_name"] = Path(checkpoint_path).name
        all_frame_rows.extend(frame_rows)

    ranked = rank_rows(summaries)
    print_ranking(ranked)
    recommended = next((row for row in ranked if row.get("status") == "ok" and not row.get("flags")), ranked[0])
    print(f"Recommended checkpoint: {recommended.get('checkpoint')} | flags={recommended.get('flags') or []}")

    payload = {
        "recommended": recommended,
        "ranking": ranked,
        "frames": all_frame_rows,
        "selection_rule": (
            "unflagged checkpoints first, then semantic_miou desc, rare_class_miou desc, "
            "mean_ce asc, occupancy_iou desc"
        ),
    }
    if args.out_json:
        write_json(args.out_json, payload)
    if args.out_csv:
        write_csv(args.out_csv, ranked)
    if args.frame_csv:
        write_csv(args.frame_csv, all_frame_rows)


if __name__ == "__main__":
    main()
