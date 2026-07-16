import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.config_paths import (
    load_training_config,
    resolve_image_condition_root,
    resolve_partial_condition_root,
)


DEFAULT_INPUT_CHANNELS = 64
DEFAULT_OUTPUT_CHANNELS = 8
DEFAULT_SPATIAL_SHAPE = (64, 64, 8)
OUTPUT_ROOT_SUFFIX = "_8ch"

SEMANTIC_GROUPS = (
    (9, 10, 12),      # road, parking, other-ground
    (11,),            # sidewalk
    (1, 4, 5),        # car, truck, other-vehicle
    (2, 3, 6, 7, 8),  # bicycle, motorcycle, person, bicyclist, motorcyclist
    (13, 14, 18, 19), # building, fence, pole, traffic-sign
    (15, 16, 17),     # vegetation, trunk, terrain
)


def parse_csv_items(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_tuple(value):
    return tuple(int(item) for item in parse_csv_items(value))


def normalize_ext(value):
    return value if value.startswith(".") else f".{value}"


def list_sequences(input_root):
    sequences_root = Path(input_root) / "sequences"
    if not sequences_root.exists():
        return []
    return sorted(path.name for path in sequences_root.iterdir() if path.is_dir())


def list_condition_files(input_root, sequences=None, frames=None, condition_folder="voxels", ext=".bin", num_samples=0):
    input_root = Path(input_root)
    sequences = sequences or list_sequences(input_root)
    frames = frames or []
    ext = normalize_ext(ext)
    files = []

    for sequence in sequences:
        sequence = str(sequence).zfill(2)
        condition_dir = input_root / "sequences" / sequence / condition_folder
        if frames:
            for frame in frames:
                frame_id = Path(str(frame)).stem.zfill(6)
                files.append(condition_dir / f"{frame_id}{ext}")
        else:
            files.extend(sorted(condition_dir.glob(f"*{ext}")))

    if num_samples > 0:
        files = files[:num_samples]
    return files


def output_path_for(input_path, input_root, output_root, condition_folder="voxels", ext=".bin"):
    input_path = Path(input_path)
    parts = input_path.parts
    if "sequences" not in parts:
        raise ValueError(f"Could not infer sequence/frame from condition path: {input_path}")
    index = parts.index("sequences")
    sequence = parts[index + 1]
    frame = input_path.stem.zfill(6)
    return Path(output_root) / "sequences" / sequence.zfill(2) / condition_folder / f"{frame}{normalize_ext(ext)}"


def expected_value_count(channels, spatial_shape):
    return int(channels * np.prod(spatial_shape))


def load_condition(path, input_channels=DEFAULT_INPUT_CHANNELS, spatial_shape=DEFAULT_SPATIAL_SHAPE):
    path = Path(path)
    values = np.fromfile(path, dtype=np.float32)
    expected = expected_value_count(input_channels, spatial_shape)
    if values.size != expected:
        raise ValueError(
            f"Condition feature {path} has {values.size} float32 values; expected {expected} "
            f"for shape ({input_channels}, {', '.join(str(item) for item in spatial_shape)})"
        )
    return values.reshape((input_channels,) + tuple(spatial_shape))


def build_semantic_projection_conv1x1(
    input_channels=DEFAULT_INPUT_CHANNELS,
    output_channels=DEFAULT_OUTPUT_CHANNELS,
    uncertainty_mode="entropy",
    device="cpu",
):
    import torch

    if input_channels < 24:
        raise ValueError("input_channels must include at least the first 24 semantic channels")
    if output_channels != DEFAULT_OUTPUT_CHANNELS:
        raise ValueError(f"output_channels must be {DEFAULT_OUTPUT_CHANNELS}")

    conv = torch.nn.Conv3d(
        input_channels,
        output_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
    )
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        for output_channel, source_channels in enumerate(SEMANTIC_GROUPS):
            for source_channel in source_channels:
                conv.weight[output_channel, source_channel, 0, 0, 0] = 1.0

        if uncertainty_mode == "entropy":
            conv.weight[6, 21, 0, 0, 0] = 1.0
        elif uncertainty_mode == "low_confidence":
            conv.weight[6, 20, 0, 0, 0] = -1.0
            conv.bias[6] = 1.0
        else:
            raise ValueError("uncertainty_mode must be 'entropy' or 'low_confidence'")

        conv.weight[7, 23, 0, 0, 0] = 1.0

    conv.requires_grad_(False)
    conv.eval()
    return conv.to(device)


def compress_condition_channels(condition, uncertainty_mode="entropy"):
    import torch

    condition = np.asarray(condition, dtype=np.float32)
    if condition.ndim != 4:
        raise ValueError("condition must have shape [C, W, L, H]")
    if condition.shape[0] < 24:
        raise ValueError("condition must contain at least the first 24 semantic channels")

    conv = build_semantic_projection_conv1x1(
        input_channels=condition.shape[0],
        uncertainty_mode=uncertainty_mode,
        device="cpu",
    )
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(condition)).unsqueeze(0)
        compressed = conv(tensor).squeeze(0).cpu().numpy()
    return compressed.astype(np.float32, copy=False)


def compress_condition_file(
    input_path,
    output_path,
    spatial_shape=DEFAULT_SPATIAL_SHAPE,
    uncertainty_mode="entropy",
    overwrite=True,
    dry_run=False,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    result = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "",
        "input_channels": DEFAULT_INPUT_CHANNELS,
        "output_channels": DEFAULT_OUTPUT_CHANNELS,
        "num_values": expected_value_count(DEFAULT_OUTPUT_CHANNELS, spatial_shape),
    }

    if not input_path.exists():
        raise FileNotFoundError(f"Missing condition feature: {input_path}")
    if output_path.exists() and not overwrite:
        result["status"] = "skipped_exists"
        return result

    condition = load_condition(input_path, spatial_shape=spatial_shape)
    compressed = compress_condition_channels(condition, uncertainty_mode=uncertainty_mode)

    if dry_run:
        result["status"] = "dry_run"
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(compressed.astype(np.float32)).tofile(output_path)
    result["status"] = "written"
    return result


def compress_condition_features(
    input_root,
    output_root,
    sequences=None,
    frames=None,
    num_samples=0,
    condition_folder="voxels",
    ext=".bin",
    spatial_shape=DEFAULT_SPATIAL_SHAPE,
    uncertainty_mode="entropy",
    overwrite=True,
    dry_run=False,
):
    input_files = list_condition_files(
        input_root,
        sequences=sequences,
        frames=frames,
        condition_folder=condition_folder,
        ext=ext,
        num_samples=num_samples,
    )
    if not input_files:
        raise FileNotFoundError(f"No condition files found under {input_root}")

    results = []
    for input_path in input_files:
        output_path = output_path_for(
            input_path,
            input_root,
            output_root,
            condition_folder=condition_folder,
            ext=ext,
        )
        results.append(
            compress_condition_file(
                input_path,
                output_path,
                spatial_shape=spatial_shape,
                uncertainty_mode=uncertainty_mode,
                overwrite=overwrite,
                dry_run=dry_run,
            )
        )
    return results


def default_output_root(input_root):
    input_root = Path(input_root)
    return input_root.with_name(f"{input_root.name}{OUTPUT_ROOT_SUFFIX}")


def default_input_root(configured_root):
    configured_root = Path(configured_root)
    if configured_root.name.endswith(OUTPUT_ROOT_SUFFIX):
        raw_name = configured_root.name[:-len(OUTPUT_ROOT_SUFFIX)]
        if raw_name:
            return configured_root.with_name(raw_name)
    return configured_root


def summarize_results(results):
    status_counts = {}
    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "ok": True,
        "total": len(results),
        "status_counts": status_counts,
        "input_channels": DEFAULT_INPUT_CHANNELS,
        "output_channels": DEFAULT_OUTPUT_CHANNELS,
    }


def write_json(path, summary, results):
    payload = {
        "summary": summary,
        "results": results,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path, results):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["input_path", "output_path", "status", "input_channels", "output_channels", "num_values"]
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(summary, results, max_items=10):
    print("condition compression to 8 channels: OK")
    print(f"total: {summary['total']}")
    print(f"status_counts: {summary['status_counts']}")
    print(f"input_channels: {summary['input_channels']}")
    print(f"output_channels: {summary['output_channels']}")
    for result in results[:max_items]:
        print(f"{result['status']}: {result['input_path']} -> {result['output_path']}")
    if len(results) > max_items:
        print(f"... {len(results) - max_items} more files")


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Compress semantic condition files from 64 channels to 8 channels with a fixed "
            "offline 1x1x1 Conv3d projection over semantic/confidence/entropy/FOV channels."
        )
    )
    parser.add_argument("--config_path", "--config-path", default="config/semantickitti_autoencoder.yaml")
    parser.add_argument("--condition_root", "--condition-root", choices=["partial", "image"], default="partial")
    parser.add_argument("--input_root", "--input-root", default="")
    parser.add_argument("--output_root", "--output-root", default="")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--frames", default="")
    parser.add_argument("--num_samples", "--num-samples", type=int, default=0)
    parser.add_argument("--condition_folder", "--condition-folder", default="voxels")
    parser.add_argument("--ext", default=".bin")
    parser.add_argument("--spatial_shape", "--spatial-shape", default="64,64,8")
    parser.add_argument("--uncertainty_mode", "--uncertainty-mode", choices=["entropy", "low_confidence"], default="entropy")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=True)
    parser.add_argument("--no_overwrite", "--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--out_json", "--out-json", default="")
    parser.add_argument("--out_csv", "--out-csv", default="")
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    configs = load_training_config(args.config_path)
    configured_root = (
        resolve_image_condition_root(configs)
        if args.condition_root == "image"
        else resolve_partial_condition_root(configs)
    )
    input_root = Path(args.input_root) if args.input_root else default_input_root(configured_root)
    output_root = Path(args.output_root) if args.output_root else default_output_root(input_root)
    results = compress_condition_features(
        input_root=input_root,
        output_root=output_root,
        sequences=parse_csv_items(args.sequences),
        frames=parse_csv_items(args.frames),
        num_samples=args.num_samples,
        condition_folder=args.condition_folder,
        ext=args.ext,
        spatial_shape=parse_int_tuple(args.spatial_shape),
        uncertainty_mode=args.uncertainty_mode,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    summary = summarize_results(results)
    print_summary(summary, results)
    if args.out_json:
        write_json(args.out_json, summary, results)
    if args.out_csv:
        write_csv(args.out_csv, results)


if __name__ == "__main__":
    main()
