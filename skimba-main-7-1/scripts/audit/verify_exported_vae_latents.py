#!/usr/bin/env python3
"""Verify exported posterior-mean VAE latents against online Encoder output."""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit.vae_audit import confusion_from_labels, per_class_iou_percent
from scripts.data.config_paths import load_training_config, resolve_gt_root, resolve_vae_root
from scripts.data.export_z0_semantic_latents import (
    build_learning_map_lut,
    encode_label_file,
    load_autoencoder_from_checkpoint,
    load_checkpoint,
    load_semantic_embedding_from_checkpoint,
    output_latent_path,
    parse_csv_items,
    parse_shape,
    resolve_voxel_dir,
    select_label_files,
    semantic_target_from_label,
)
from scripts.data.validate_semantic_vae_latents import load_latent_tensor


def compare_tensors(reference, candidate):
    """Return exact and numerical parity evidence for two torch tensors."""
    import torch

    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError(
            f"Tensor shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    if reference.dtype != candidate.dtype:
        raise ValueError(f"Tensor dtype mismatch: {reference.dtype} != {candidate.dtype}")
    element_count = reference.numel()
    if element_count == 0:
        return {
            "element_count": 0,
            "bit_exact": True,
            "differing_elements": 0,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
        }

    difference = (reference - candidate).abs()
    return {
        "element_count": int(element_count),
        "bit_exact": bool(torch.equal(reference, candidate)),
        "differing_elements": int(torch.count_nonzero(reference != candidate).item()),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
    }


def compare_predictions(reference, candidate, target, ignore_label=255):
    """Compare two class grids only over voxels valid for evaluation."""
    if tuple(reference.shape) != tuple(candidate.shape) or tuple(reference.shape) != tuple(target.shape):
        raise ValueError("Prediction and target shapes must match")
    valid = target != ignore_label
    valid_voxels = int(valid.sum().item())
    differing_voxels = int(((reference != candidate) & valid).sum().item())
    disagreement_rate = (
        100.0 * differing_voxels / valid_voxels if valid_voxels else math.nan
    )
    return {
        "valid_voxels": valid_voxels,
        "differing_voxels": differing_voxels,
        "disagreement_rate_percent": disagreement_rate,
    }


def load_channel_stats(path, channels):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mean = [float(value) for value in payload.get("mean", [])]
    std = [float(value) for value in payload.get("std", [])]
    if len(mean) != channels or len(std) != channels:
        raise ValueError(
            f"Latent statistics must contain {channels} mean/std values; "
            f"got {len(mean)}/{len(std)}"
        )
    if not all(math.isfinite(value) for value in mean + std):
        raise ValueError("Latent statistics must be finite")
    if not all(value > 0.0 for value in std):
        raise ValueError("Latent standard deviations must be strictly positive")
    return {"mean": mean, "std": std, "path": str(path.resolve())}


def parity_failures(
    summary,
    max_latent_abs_error,
    max_logit_abs_error,
    max_prediction_disagreement_percent,
):
    checks = (
        (
            "latent max_abs_error",
            summary["latent"]["max_abs_error"],
            max_latent_abs_error,
        ),
        (
            "decoded_logits max_abs_error",
            summary["decoded_logits"]["max_abs_error"],
            max_logit_abs_error,
        ),
        (
            "decoded_prediction disagreement_rate_percent",
            summary["decoded_prediction"]["disagreement_rate_percent"],
            max_prediction_disagreement_percent,
        ),
    )
    failures = []
    for name, actual, limit in checks:
        if not math.isfinite(float(actual)) or float(actual) > float(limit):
            failures.append(f"{name} {actual:.12g} > {limit:.12g}")
    return failures


def _project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _model_args(config, latent_shape):
    model = config.get("model_params", {})
    dataset = config.get("dataset_params", {})
    channels = model.get("autoencoder_channels_list", [16, 32, 64, 128])
    return SimpleNamespace(
        in_channels=int(model.get("in_channels", 8)),
        out_channels=int(model.get("out_channels", 8)),
        latent_channels=int(model.get("latent_channels", 64)),
        autoencoder_num_res_blocks=int(model.get("autoencoder_num_res_blocks", 1)),
        autoencoder_channels_list=",".join(str(value) for value in channels),
        auto_groups=int(model.get("auto_groups", 4)),
        num_input_features=int(model.get("num_input_features", 3)),
        init_size=int(model.get("init_size", 8)),
        voxel_channel=int(model.get("voxel_channel", 1)),
        dropout_rate=float(model.get("dropout_rate", 0.2)),
        num_class=int(model.get("num_class", 20)),
        semantic_embed_dim=int(model.get("semantic_embed_dim", 8)),
        ignore_label=int(dataset.get("ignore_label", 255)),
        latent_shape=latent_shape,
        mode="mean",
    )


def _select_manifest(args):
    sequences = parse_csv_items(args.sequences)
    frames = parse_csv_items(args.frames)
    if not sequences:
        raise ValueError("At least one sequence is required")
    if args.frames_per_sequence <= 0:
        raise ValueError("--frames-per-sequence must be positive")

    manifest = []
    for sequence in sequences:
        voxel_dir = resolve_voxel_dir(args.dataset_root, sequence, args.label_root)
        if frames:
            label_files = select_label_files(voxel_dir, frames, num_samples=0)
        else:
            label_files = sorted(Path(voxel_dir).glob("*.label"))
            if len(label_files) > args.frames_per_sequence:
                rng = random.Random(f"{args.seed}:{str(sequence).zfill(2)}")
                label_files = sorted(rng.sample(label_files, args.frames_per_sequence))
        for label_path in label_files:
            sequence_id = str(sequence).zfill(2)
            manifest.append(
                {
                    "sequence": sequence_id,
                    "frame": label_path.stem,
                    "label_path": str(label_path),
                    "latent_path": str(
                        output_latent_path(args.latent_root, sequence_id, label_path.stem)
                    ),
                }
            )
    if not manifest:
        raise RuntimeError("No SemanticKITTI label frames selected")
    return manifest


def _reconstruction_evidence(logits, target_np, num_classes, ignore_label):
    import torch
    import torch.nn.functional as F

    target = torch.from_numpy(target_np).long().to(logits.device)
    prediction = torch.argmax(logits, dim=1).squeeze(0)
    valid_voxels = int((target != ignore_label).sum().item())
    ce_sum = float(
        F.cross_entropy(
            logits,
            target.unsqueeze(0),
            ignore_index=ignore_label,
            reduction="sum",
        ).item()
    )
    confusion = confusion_from_labels(
        target.detach().cpu().numpy(),
        prediction.detach().cpu().numpy(),
        num_classes,
        ignore_label,
    )
    return {
        "prediction": prediction,
        "confusion": confusion,
        "ce_sum": ce_sum,
        "valid_voxels": valid_voxels,
    }


def _aggregate_reconstruction(frame_rows, branch, num_classes):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    ce_sum = 0.0
    valid_voxels = 0
    for row in frame_rows:
        evidence = row[branch]
        confusion += np.asarray(evidence["confusion"], dtype=np.int64)
        ce_sum += float(evidence["ce_sum"])
        valid_voxels += int(evidence["valid_voxels"])

    class_iou = per_class_iou_percent(confusion)
    semantic = class_iou[1:] if num_classes > 1 else class_iou
    finite_semantic = semantic[np.isfinite(semantic)]
    total = int(confusion.sum())
    occupancy_intersection = int(confusion[1:, 1:].sum())
    occupancy_union = int(total - confusion[0, 0])
    return {
        "mean_ce": ce_sum / valid_voxels if valid_voxels else math.nan,
        "semantic_miou_percent": (
            float(finite_semantic.mean()) if finite_semantic.size else math.nan
        ),
        "occupancy_iou_percent": (
            100.0 * occupancy_intersection / occupancy_union
            if occupancy_union
            else math.nan
        ),
        "voxel_accuracy_percent": (
            100.0 * float(np.diag(confusion).sum()) / total if total else math.nan
        ),
        "valid_voxels": valid_voxels,
        "per_class_iou_percent": [
            float(value) if math.isfinite(float(value)) else None for value in class_iou
        ],
        "confusion": confusion.tolist(),
    }


def _aggregate_tensor_comparisons(frame_rows, key):
    comparisons = [row[key] for row in frame_rows]
    elements = sum(item["element_count"] for item in comparisons)
    differing = sum(item["differing_elements"] for item in comparisons)
    absolute_error_sum = sum(
        item["mean_abs_error"] * item["element_count"] for item in comparisons
    )
    return {
        "frames_bit_exact": sum(bool(item["bit_exact"]) for item in comparisons),
        "num_frames": len(comparisons),
        "element_count": int(elements),
        "differing_elements": int(differing),
        "max_abs_error": max(item["max_abs_error"] for item in comparisons),
        "mean_abs_error": absolute_error_sum / elements if elements else 0.0,
    }


def _aggregate_prediction_comparisons(frame_rows, key):
    comparisons = [row[key] for row in frame_rows]
    valid = sum(item["valid_voxels"] for item in comparisons)
    differing = sum(item["differing_voxels"] for item in comparisons)
    return {
        "valid_voxels": int(valid),
        "differing_voxels": int(differing),
        "disagreement_rate_percent": 100.0 * differing / valid if valid else math.nan,
    }


def _public_frame_row(row):
    public = {key: value for key, value in row.items() if key not in {"online", "saved", "roundtrip"}}
    for branch in ("online", "saved", "roundtrip"):
        if branch not in row:
            continue
        evidence = row[branch]
        public[branch] = {
            "ce_sum": evidence["ce_sum"],
            "valid_voxels": evidence["valid_voxels"],
            "confusion": evidence["confusion"].tolist(),
        }
    return public


def _evaluate_frame(
    manifest_row,
    autoencoder,
    semantic_embedding,
    learning_map_lut,
    model_args,
    device,
    normalization,
):
    import torch

    label_path = Path(manifest_row["label_path"])
    latent_path = Path(manifest_row["latent_path"])
    target_np = semantic_target_from_label(
        label_path,
        learning_map_lut,
        model_args.ignore_label,
    )
    online_cpu = encode_label_file(
        autoencoder,
        semantic_embedding,
        label_path,
        learning_map_lut,
        model_args,
        device,
    )
    saved_cpu = load_latent_tensor(latent_path, parse_shape(model_args.latent_shape))
    online = online_cpu.unsqueeze(0).to(device)
    saved = saved_cpu.unsqueeze(0).to(device)

    with torch.no_grad():
        online_logits = autoencoder.decode(online)
        saved_logits = autoencoder.decode(saved)
        online_evidence = _reconstruction_evidence(
            online_logits,
            target_np,
            model_args.num_class,
            model_args.ignore_label,
        )
        saved_evidence = _reconstruction_evidence(
            saved_logits,
            target_np,
            model_args.num_class,
            model_args.ignore_label,
        )

        target = torch.from_numpy(target_np).long().to(device)
        row = {
            **manifest_row,
            "latent": compare_tensors(online, saved),
            "decoded_logits": compare_tensors(online_logits, saved_logits),
            "decoded_prediction": compare_predictions(
                online_evidence["prediction"],
                saved_evidence["prediction"],
                target,
                model_args.ignore_label,
            ),
            "online": online_evidence,
            "saved": saved_evidence,
        }

        if normalization is not None:
            channels = saved.shape[1]
            mean = torch.tensor(
                normalization["mean"], dtype=saved.dtype, device=device
            ).reshape(1, channels, 1, 1, 1)
            std = torch.tensor(
                normalization["std"], dtype=saved.dtype, device=device
            ).reshape(1, channels, 1, 1, 1)
            normalized = (saved - mean) / std
            roundtrip = normalized * std + mean
            roundtrip_logits = autoencoder.decode(roundtrip)
            roundtrip_evidence = _reconstruction_evidence(
                roundtrip_logits,
                target_np,
                model_args.num_class,
                model_args.ignore_label,
            )
            row.update(
                {
                    "normalization_roundtrip_latent": compare_tensors(saved, roundtrip),
                    "normalization_roundtrip_logits": compare_tensors(
                        saved_logits, roundtrip_logits
                    ),
                    "normalization_roundtrip_prediction": compare_predictions(
                        saved_evidence["prediction"],
                        roundtrip_evidence["prediction"],
                        target,
                        model_args.ignore_label,
                    ),
                    "roundtrip": roundtrip_evidence,
                }
            )
    return row


def _resolve_runtime(args):
    args.config_path = str(_project_path(args.config_path))
    config = load_training_config(args.config_path)
    model = config.get("model_params", {})
    dataset = config.get("dataset_params", {})
    train = config.get("train_params", {})

    args.vae_checkpoint = str(args.vae_checkpoint or train.get("vae_checkpoint", ""))
    args.dataset_root = str(args.dataset_root or dataset.get("data_root", ""))
    args.label_root = str(args.label_root or resolve_gt_root(config))
    args.latent_root = str(args.latent_root or resolve_vae_root(config))
    args.label_mapping = str(args.label_mapping or dataset.get("label_mapping", ""))
    configured_stats = model.get("latent_normalization", {}).get("stats_path", "")
    args.stats_path = str(args.stats_path or configured_stats)
    if args.label_mapping:
        args.label_mapping = str(_project_path(args.label_mapping))
    for name in ("vae_checkpoint", "dataset_root", "label_root", "latent_root", "label_mapping"):
        if not getattr(args, name):
            raise ValueError(f"Missing --{name.replace('_', '-')} and no config default is available")
    return config


def run(args):
    import torch

    config = _resolve_runtime(args)
    model_args = _model_args(config, args.latent_shape)
    device_name = args.device
    if device_name != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device_name}, but CUDA is unavailable")
    device = torch.device(device_name)

    manifest = _select_manifest(args)
    for row in manifest:
        if not Path(row["latent_path"]).is_file():
            raise FileNotFoundError(f"Missing exported latent: {row['latent_path']}")

    normalization = None
    if not args.skip_normalization_roundtrip:
        if not args.stats_path:
            raise ValueError(
                "Normalization roundtrip is enabled but no --stats-path/config stats_path is set"
            )
        normalization = load_channel_stats(
            args.stats_path,
            channels=parse_shape(args.latent_shape)[0],
        )

    checkpoint = load_checkpoint(args.vae_checkpoint, torch)
    autoencoder = load_autoencoder_from_checkpoint(model_args, checkpoint, device)
    semantic_embedding = load_semantic_embedding_from_checkpoint(
        model_args, checkpoint, device
    )
    learning_map_lut = build_learning_map_lut(args.label_mapping)

    frame_rows = []
    for index, manifest_row in enumerate(manifest, start=1):
        row = _evaluate_frame(
            manifest_row,
            autoencoder,
            semantic_embedding,
            learning_map_lut,
            model_args,
            device,
            normalization,
        )
        frame_rows.append(row)
        print(
            f"[{index}/{len(manifest)}] {row['sequence']}/{row['frame']} "
            f"latent_max={row['latent']['max_abs_error']:.9g} "
            f"logit_max={row['decoded_logits']['max_abs_error']:.9g} "
            f"pred_diff={row['decoded_prediction']['differing_voxels']}"
        )

    parity = {
        "latent": _aggregate_tensor_comparisons(frame_rows, "latent"),
        "decoded_logits": _aggregate_tensor_comparisons(frame_rows, "decoded_logits"),
        "decoded_prediction": _aggregate_prediction_comparisons(
            frame_rows, "decoded_prediction"
        ),
    }
    failures = parity_failures(
        parity,
        args.max_latent_abs_error,
        args.max_logit_abs_error,
        args.max_prediction_disagreement_percent,
    )
    result = {
        "status": "pass" if not failures else "fail",
        "config_path": args.config_path,
        "vae_checkpoint": str(Path(args.vae_checkpoint).resolve()),
        "latent_root": str(Path(args.latent_root).resolve()),
        "label_root": str(Path(args.label_root).resolve()),
        "mode": "mean",
        "device": str(device),
        "seed": args.seed,
        "num_frames": len(frame_rows),
        "manifest": manifest,
        "parity": parity,
        "online_reconstruction": _aggregate_reconstruction(
            frame_rows, "online", model_args.num_class
        ),
        "saved_reconstruction": _aggregate_reconstruction(
            frame_rows, "saved", model_args.num_class
        ),
        "threshold_failures": failures,
        "frames": [_public_frame_row(row) for row in frame_rows],
    }
    if normalization is not None:
        result["normalization"] = normalization
        result["normalization_roundtrip"] = {
            "latent": _aggregate_tensor_comparisons(
                frame_rows, "normalization_roundtrip_latent"
            ),
            "decoded_logits": _aggregate_tensor_comparisons(
                frame_rows, "normalization_roundtrip_logits"
            ),
            "decoded_prediction": _aggregate_prediction_comparisons(
                frame_rows, "normalization_roundtrip_prediction"
            ),
            "reconstruction": _aggregate_reconstruction(
                frame_rows, "roundtrip", model_args.num_class
            ),
        }

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {output.resolve()}")

    print(
        "Parity: latent_max={:.9g}, logit_max={:.9g}, prediction_diff={}/{}".format(
            parity["latent"]["max_abs_error"],
            parity["decoded_logits"]["max_abs_error"],
            parity["decoded_prediction"]["differing_voxels"],
            parity["decoded_prediction"]["valid_voxels"],
        )
    )
    print(
        "Saved reconstruction: semantic_mIoU={:.9f}% occupancy_IoU={:.9f}%".format(
            result["saved_reconstruction"]["semantic_miou_percent"],
            result["saved_reconstruction"]["occupancy_iou_percent"],
        )
    )
    if failures:
        print("Verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 2, result
    return 0, result


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare online posterior-mean Encoder latents with exported float32 .bin "
            "latents, decode both, and optionally audit normalization roundtrip error."
        )
    )
    parser.add_argument("--config-path", default="config/semantickitti_autoencoder.yaml")
    parser.add_argument("--vae-checkpoint", default="")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--label-root", default="")
    parser.add_argument("--latent-root", default="")
    parser.add_argument("--label-mapping", default="")
    parser.add_argument("--stats-path", default="")
    parser.add_argument("--sequences", default="08")
    parser.add_argument("--frames", default="")
    parser.add_argument("--frames-per-sequence", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--latent-shape", default="8,64,64,8")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--skip-normalization-roundtrip", action="store_true")
    parser.add_argument("--max-latent-abs-error", type=float, default=0.0)
    parser.add_argument("--max-logit-abs-error", type=float, default=0.0)
    parser.add_argument(
        "--max-prediction-disagreement-percent", type=float, default=0.0
    )
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    try:
        exit_code, _ = run(args)
        return exit_code
    except Exception as exc:  # CLI boundary: concise failure for batch jobs.
        print(f"VAE latent verification failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
