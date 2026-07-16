import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.config_paths import (
    load_training_config,
    resolve_gt_root,
    resolve_vae_root,
)

SPATIAL_SHAPE = (256, 256, 32)
IGNORE_LABEL = 255


def parse_csv_items(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_shape(value):
    parts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 4:
        raise ValueError(f"Expected C,W,L,H latent shape, got: {value}")
    return tuple(parts)


def parse_int_list(value):
    return [int(item) for item in parse_csv_items(value)]


def normalize_ext(value):
    return value if value.startswith(".") else f".{value}"


def apply_latent_format_defaults(args):
    if args.skimba_bin:
        args.latent_folder = "voxels"
        args.latent_ext = ".bin"
    args.latent_ext = normalize_ext(args.latent_ext)
    return args


def latent_path_for(root, sequence, frame, latent_folder, latent_ext):
    return (
        Path(root)
        / "sequences"
        / str(sequence).zfill(2)
        / latent_folder
        / f"{str(frame).zfill(6)}{normalize_ext(latent_ext)}"
    )


def label_path_for(dataset_root, sequence, frame):
    return (
        Path(dataset_root)
        / "sequences"
        / str(sequence).zfill(2)
        / "voxels"
        / f"{str(frame).zfill(6)}.label"
    )


def load_checkpoint(path, torch_module):
    try:
        return torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("autoencoder", "state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def normalize_vae_key(key):
    prefixes = (
        "model_part.module.autoencoder.",
        "model_part.autoencoder.",
        "module.model_part.autoencoder.",
        "module.autoencoder.",
        "autoencoder.",
        "module.",
        "model.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def adapt_vae_state_dict(checkpoint, target_state):
    source_state = extract_state_dict(checkpoint)
    adapted = {}
    mismatched = {}
    skipped = []

    for key, value in source_state.items():
        if not isinstance(key, str):
            continue
        target_key = normalize_vae_key(key)
        if target_key not in target_state:
            skipped.append(target_key)
            continue
        source_shape = tuple(value.shape) if hasattr(value, "shape") else None
        target_shape = tuple(target_state[target_key].shape)
        if source_shape != target_shape:
            mismatched[target_key] = (source_shape, target_shape)
            continue
        adapted[target_key] = value

    missing = [key for key in target_state.keys() if key not in adapted]
    return adapted, missing, mismatched, skipped


def build_skimba_autoencoder(args, device):
    import torch

    from stable_diffusion.models.autoencoder import AutoEncoderKL

    model = AutoEncoderKL(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        latent_channels=args.latent_channels,
        autoencoder_num_res_blocks=args.autoencoder_num_res_blocks,
        autoencoder_channels_list=parse_int_list(args.autoencoder_channels_list),
        groups=args.auto_groups,
        num_input_features=args.num_input_features,
        init_size=args.init_size,
        voxel_channel=args.voxel_channel,
        dropout_rate=args.dropout_rate,
        num_class=args.num_class,
        semantic_embed_dim=args.semantic_embed_dim,
    )
    checkpoint = load_checkpoint(args.vae_checkpoint, torch)
    target_state = model.state_dict()
    adapted, missing, mismatched, skipped = adapt_vae_state_dict(checkpoint, target_state)

    if missing or mismatched:
        messages = []
        if missing:
            messages.append("missing keys:\n  " + "\n  ".join(missing[:50]))
        if mismatched:
            lines = [
                f"{key}: checkpoint{src} != model{dst}"
                for key, (src, dst) in sorted(mismatched.items())
            ]
            messages.append("shape mismatches:\n  " + "\n  ".join(lines[:50]))
        raise RuntimeError("Could not load VAE checkpoint into skimba AutoEncoderKL:\n" + "\n".join(messages))

    load_result = model.load_state_dict(adapted, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Adapted VAE state did not load cleanly: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    if skipped:
        print(f"Skipped {len(skipped)} non-autoencoder checkpoint keys.")
    model.to(device)
    model.eval()
    return model


def build_learning_map_lut(label_mapping_path):
    import yaml

    with Path(label_mapping_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    learning_map = {int(key): int(value) for key, value in data["learning_map"].items()}
    max_key = max(learning_map.keys())
    lut = np.zeros((max_key + 100,), dtype=np.uint8)
    for raw_id, train_id in learning_map.items():
        lut[raw_id] = train_id
    return lut


def read_dense_label(label_path, spatial_shape=SPATIAL_SHAPE):
    label_path = Path(label_path)
    expected = int(np.prod(spatial_shape))
    file_size = label_path.stat().st_size
    if file_size == expected:
        dtype = np.uint8
    elif file_size == expected * 2:
        dtype = np.uint16
    elif file_size == expected * 4:
        dtype = np.uint32
    else:
        raise ValueError(
            f"{label_path} has {file_size} bytes, expected {expected}, "
            f"{expected * 2}, or {expected * 4}."
        )
    labels = np.fromfile(label_path, dtype=dtype)
    if labels.size != expected:
        raise ValueError(f"{label_path} has {labels.size} voxels, expected {expected}.")
    return labels.reshape(spatial_shape)


def read_invalid_mask(invalid_path, spatial_shape=SPATIAL_SHAPE):
    invalid_path = Path(invalid_path)
    if not invalid_path.exists():
        return None
    expected = int(np.prod(spatial_shape))
    raw = np.fromfile(invalid_path, dtype=np.uint8)
    if raw.size == expected:
        return raw.reshape(spatial_shape).astype(np.uint8)
    unpacked = np.unpackbits(raw)
    if unpacked.size < expected:
        raise ValueError(f"{invalid_path} expands to {unpacked.size} voxels, expected {expected}.")
    return unpacked[:expected].reshape(spatial_shape).astype(np.uint8)


def semantic_target_from_label(label_path, learning_map_lut, ignore_label=IGNORE_LABEL):
    raw = read_dense_label(label_path)
    lower = (raw.astype(np.uint32) & 0xFFFF).reshape(-1)
    max_label = int(lower.max()) if lower.size else 0
    if max_label >= learning_map_lut.shape[0]:
        raise ValueError(f"{label_path} contains raw label {max_label}, outside learning map LUT.")
    target = learning_map_lut[lower].reshape(SPATIAL_SHAPE).astype(np.uint8)

    invalid = read_invalid_mask(Path(label_path).with_suffix(".invalid"))
    if invalid is not None:
        target = target.copy()
        target[invalid == 1] = ignore_label
    return target


def load_latent_tensor(latent_path, expected_shape):
    import torch

    latent_path = Path(latent_path)
    if latent_path.suffix.lower() in {".pt", ".pth"}:
        payload = load_checkpoint(latent_path, torch)
        if isinstance(payload, dict):
            for key in ("latent", "latents", "z"):
                if key in payload:
                    payload = payload[key]
                    break
        latent = payload.detach().cpu() if hasattr(payload, "detach") else torch.as_tensor(payload)
    else:
        data = np.fromfile(latent_path, dtype=np.float32)
        latent = torch.from_numpy(data.reshape(expected_shape))

    latent = latent.float().contiguous()
    if latent.ndim == 5 and latent.shape[0] == 1:
        latent = latent.squeeze(0).contiguous()
    if tuple(latent.shape) != tuple(expected_shape):
        raise ValueError(f"{latent_path} latent shape {tuple(latent.shape)} != expected {expected_shape}.")
    if not torch.isfinite(latent).all():
        raise ValueError(f"{latent_path} contains non-finite latent values.")
    return latent


def safe_mean(values):
    values = [float(value) for value in values if not math.isnan(float(value))]
    return float(sum(values) / len(values)) if values else math.nan


def compute_content_metrics(logits, target_np, ignore_label, num_classes):
    import torch
    import torch.nn.functional as F

    target = torch.from_numpy(target_np[None]).long().to(logits.device)
    ce = F.cross_entropy(logits, target, ignore_index=ignore_label).item()
    prediction = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)
    target_flat = target_np.reshape(-1).astype(np.int64)
    pred_flat = prediction.reshape(-1).astype(np.int64)

    valid = (target_flat != ignore_label) & (target_flat >= 0) & (target_flat < num_classes)
    valid_pred = valid & (pred_flat >= 0) & (pred_flat < num_classes)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    if valid_pred.any():
        encoded = num_classes * target_flat[valid_pred] + pred_flat[valid_pred]
        confusion += np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    tp = np.diag(confusion).astype(np.float64)
    target_count = confusion.sum(axis=1).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    union = target_count + pred_count - tp
    iou = np.full((num_classes,), math.nan, dtype=np.float64)
    has_union = union > 0
    iou[has_union] = tp[has_union] / union[has_union] * 100.0

    pred_occ = (pred_flat > 0) & valid
    target_occ = (target_flat > 0) & valid
    occ_intersection = int(np.logical_and(pred_occ, target_occ).sum())
    occ_union = int(np.logical_or(pred_occ, target_occ).sum())
    correct = int(((pred_flat == target_flat) & valid).sum())
    valid_count = int(valid.sum())

    return {
        "ce": float(ce),
        "semantic_miou": safe_mean(iou[1:] if num_classes > 1 else iou),
        "semantic_miou_with_empty": safe_mean(iou),
        "occupancy_iou": (occ_intersection / occ_union * 100.0) if occ_union else math.nan,
        "voxel_accuracy": (correct / valid_count * 100.0) if valid_count else math.nan,
        "valid_voxels": valid_count,
        "correct_voxels": correct,
        "occupancy_intersection": occ_intersection,
        "occupancy_union": occ_union,
    }


def evaluate_pair(model, latent_path, label_path, learning_map_lut, args, device):
    import torch

    expected_shape = parse_shape(args.latent_shape)
    latent = load_latent_tensor(latent_path, expected_shape).unsqueeze(0).to(device)
    target = semantic_target_from_label(label_path, learning_map_lut, args.ignore_label)

    with torch.no_grad():
        logits = model.decode(latent)
    expected_logits_shape = (1, args.num_class, *SPATIAL_SHAPE)
    if tuple(logits.shape) != expected_logits_shape:
        raise ValueError(f"Decoded logits shape {tuple(logits.shape)} != expected {expected_logits_shape}.")
    if not torch.isfinite(logits).all():
        raise ValueError(f"Decoded logits for {latent_path} contain non-finite values.")

    metrics = compute_content_metrics(logits, target, args.ignore_label, args.num_class)
    sequence = Path(label_path).parent.parent.name
    frame = Path(label_path).stem
    return {
        "sequence": sequence,
        "frame": frame,
        "latent_path": str(latent_path),
        "label_path": str(label_path),
        **metrics,
    }


def collect_pairs(args):
    if args.latent_path or args.label_path:
        if not args.latent_path or not args.label_path:
            raise ValueError("--latent_path and --label_path must be provided together.")
        return [(Path(args.latent_path), Path(args.label_path))]

    if not args.latent_root or not args.dataset_root:
        raise ValueError("Provide either --latent_path/--label_path or --latent_root/--dataset_root.")

    pairs = []
    sequences = parse_csv_items(args.sequences)
    frames = parse_csv_items(args.frames)
    for sequence in sequences:
        if frames:
            latent_files = [
                latent_path_for(args.latent_root, sequence, frame, args.latent_folder, args.latent_ext)
                for frame in frames
            ]
        else:
            latent_dir = Path(args.latent_root) / "sequences" / str(sequence).zfill(2) / args.latent_folder
            latent_files = sorted(latent_dir.glob(f"*{normalize_ext(args.latent_ext)}"))
            if args.num_samples > 0:
                latent_files = latent_files[: args.num_samples]
        for latent_file in latent_files:
            frame = latent_file.stem
            label_file = label_path_for(args.dataset_root, sequence, frame)
            pairs.append((latent_file, label_file))
    return pairs


def aggregate_rows(rows):
    if not rows:
        return {}
    return {
        "num_frames": len(rows),
        "mean_ce": safe_mean(row["ce"] for row in rows),
        "semantic_miou": safe_mean(row["semantic_miou"] for row in rows),
        "semantic_miou_with_empty": safe_mean(row["semantic_miou_with_empty"] for row in rows),
        "occupancy_iou": safe_mean(row["occupancy_iou"] for row in rows),
        "voxel_accuracy": safe_mean(row["voxel_accuracy"] for row in rows),
        "valid_voxels": int(sum(row["valid_voxels"] for row in rows)),
    }


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def check_thresholds(summary, args):
    failures = []
    if args.min_miou is not None and summary["semantic_miou"] < args.min_miou:
        failures.append(f"semantic_miou {summary['semantic_miou']:.4f} < {args.min_miou:.4f}")
    if args.min_occ_iou is not None and summary["occupancy_iou"] < args.min_occ_iou:
        failures.append(f"occupancy_iou {summary['occupancy_iou']:.4f} < {args.min_occ_iou:.4f}")
    if args.max_ce is not None and summary["mean_ce"] > args.max_ce:
        failures.append(f"mean_ce {summary['mean_ce']:.6f} > {args.max_ce:.6f}")
    if failures:
        raise SystemExit("Latent content validation failed:\n  " + "\n  ".join(failures))


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Validate precomputed semantic VAE latents by decoding them with skimba and comparing to labels."
    )
    parser.add_argument("--config_path", "--config-path", default="config/semantickitti_autoencoder.yaml")
    parser.add_argument("--vae_checkpoint", "--vae-checkpoint", default="")
    parser.add_argument("--label_mapping", "--label-mapping", default="")
    parser.add_argument("--latent_path", "--latent-path", default="")
    parser.add_argument("--label_path", "--label-path", default="")
    parser.add_argument("--latent_root", "--latent-root", default="")
    parser.add_argument("--dataset_root", "--dataset-root", default="")
    parser.add_argument("--sequences", default="08")
    parser.add_argument("--frames", default="")
    parser.add_argument("--num_samples", "--num-samples", type=int, default=20)
    parser.add_argument("--latent_folder", "--latent-folder", default="latents")
    parser.add_argument("--latent_ext", "--latent-ext", default=".pt")
    parser.add_argument(
        "--skimba_bin",
        "--skimba-bin",
        action="store_true",
        help="Read skimba-compatible raw float32 latents from sequences/XX/voxels/*.bin.",
    )
    parser.add_argument("--latent_shape", "--latent-shape", default="8,64,64,8")
    parser.add_argument("--num_class", "--num-class", type=int, default=20)
    parser.add_argument("--semantic_embed_dim", "--semantic-embed-dim", type=int, default=8)
    parser.add_argument("--ignore_label", "--ignore-label", type=int, default=IGNORE_LABEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_json", "--out-json", default="")
    parser.add_argument("--out_csv", "--out-csv", default="")
    parser.add_argument("--min_miou", "--min-miou", type=float, default=None)
    parser.add_argument("--min_occ_iou", "--min-occ-iou", type=float, default=None)
    parser.add_argument("--max_ce", "--max-ce", type=float, default=None)
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
    args = build_argparser().parse_args()
    configs = load_training_config(args.config_path)
    args.vae_checkpoint = args.vae_checkpoint or configs.get("train_params", {}).get("vae_checkpoint", "")
    args.label_mapping = args.label_mapping or configs["dataset_params"].get("label_mapping", "")
    args.latent_root = args.latent_root or str(resolve_vae_root(configs))
    args.dataset_root = args.dataset_root or str(resolve_gt_root(configs))
    if not args.vae_checkpoint:
        raise ValueError("Set train_params.vae_checkpoint in the config or pass --vae_checkpoint.")
    if not args.latent_root:
        raise ValueError("Set dataset_params.vae_feature_root/data_root in the config or pass --latent_root.")
    if not args.dataset_root:
        raise ValueError("Set dataset_params.gt_root/data_root in the config or pass --dataset_root.")
    apply_latent_format_defaults(args)

    import torch

    requested_device = args.device
    if requested_device != "cpu" and not torch.cuda.is_available():
        print(f"Warning: requested device {requested_device}, but CUDA is unavailable; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    pairs = collect_pairs(args)
    if not pairs:
        raise RuntimeError("No latent/label pairs selected.")
    for latent_path, label_path in pairs:
        if not Path(latent_path).exists():
            raise FileNotFoundError(f"Missing latent file: {latent_path}")
        if not Path(label_path).exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")

    learning_map_lut = build_learning_map_lut(args.label_mapping)
    model = build_skimba_autoencoder(args, device)

    rows = []
    for index, (latent_path, label_path) in enumerate(pairs, start=1):
        row = evaluate_pair(model, latent_path, label_path, learning_map_lut, args, device)
        rows.append(row)
        print(
            f"[{index}/{len(pairs)}] seq={row['sequence']} frame={row['frame']} "
            f"CE={row['ce']:.6f} semantic_miou={row['semantic_miou']:.3f}% "
            f"occupancy_iou={row['occupancy_iou']:.3f}% voxel_acc={row['voxel_accuracy']:.3f}%"
        )

    summary = aggregate_rows(rows)
    print(
        "Summary | "
        f"frames={summary['num_frames']} "
        f"mean_CE={summary['mean_ce']:.6f} "
        f"semantic_miou={summary['semantic_miou']:.3f}% "
        f"occupancy_iou={summary['occupancy_iou']:.3f}% "
        f"voxel_acc={summary['voxel_accuracy']:.3f}%"
    )

    if args.out_csv:
        write_csv(args.out_csv, rows)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps({"summary": summary, "frames": rows}, indent=2), encoding="utf-8")

    check_thresholds(summary, args)


if __name__ == "__main__":
    main()
