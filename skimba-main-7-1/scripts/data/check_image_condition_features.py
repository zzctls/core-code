import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.config_paths import (
    load_training_config,
    resolve_image_condition_root,
    resolve_partial_condition_root,
    resolve_vae_root,
)

DEFAULT_SPATIAL_SHAPE = (64, 64, 8)
DEFAULT_CHANNEL_CANDIDATES = (24, 64)
KNOWN_VAE_ROOT_NAMES = (
    "VAE_Encoder_Features_One_To_One",
    "VAE_Encoder_Features_Semantic20",
)
IMAGE_CONDITION_ROOT_NAME = "Image_transform_Voxel_Condition_Features"


@dataclass
class ConditionCheckResult:
    path: str
    sequence: str = ""
    frame: str = ""
    channels: int = 0
    num_values: int = 0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    zero_ratio: Optional[float] = None
    finite: bool = False
    ok: bool = False
    issues: list[str] = field(default_factory=list)


def parse_csv_items(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_tuple(value):
    return tuple(int(item) for item in parse_csv_items(value))


def normalize_ext(value):
    return value if value.startswith(".") else f".{value}"


def expected_value_count(channels, spatial_shape):
    return int(channels * np.prod(spatial_shape))


def detect_channels(num_values, spatial_shape=DEFAULT_SPATIAL_SHAPE, channel_candidates=DEFAULT_CHANNEL_CANDIDATES):
    for channels in channel_candidates:
        if num_values == expected_value_count(channels, spatial_shape):
            return channels
    return 0


def sequence_frame_from_path(path):
    path = Path(path)
    parts = path.parts
    if "sequences" in parts:
        index = parts.index("sequences")
        if index + 1 < len(parts):
            return parts[index + 1], path.stem.zfill(6)
    return "", path.stem.zfill(6)


def infer_image_condition_path(vae_path):
    vae_path = Path(vae_path)
    path_text = str(vae_path)
    for root_name in KNOWN_VAE_ROOT_NAMES:
        if root_name in path_text:
            return Path(path_text.replace(root_name, IMAGE_CONDITION_ROOT_NAME))
    raise ValueError(
        f"Could not infer image condition path from {vae_path}. "
        f"Expected one of {KNOWN_VAE_ROOT_NAMES} in the VAE root path."
    )


def image_condition_path_for(vae_path, image_condition_root=None, image_folder="voxels", ext=".bin"):
    if image_condition_root:
        sequence, frame = sequence_frame_from_path(vae_path)
        if not sequence:
            raise ValueError(f"Could not infer sequence id from VAE path: {vae_path}")
        return (
            Path(image_condition_root)
            / "sequences"
            / sequence.zfill(2)
            / image_folder
            / f"{frame}{normalize_ext(ext)}"
        )
    return infer_image_condition_path(vae_path)


def list_sequences(vae_root):
    sequences_root = Path(vae_root) / "sequences"
    if not sequences_root.exists():
        return []
    return sorted(path.name for path in sequences_root.iterdir() if path.is_dir())


def list_vae_files(vae_root, sequences=None, frames=None, latent_folder="voxels", ext=".bin", num_samples=0):
    vae_root = Path(vae_root)
    sequences = sequences or list_sequences(vae_root)
    frames = frames or []
    ext = normalize_ext(ext)
    files = []

    for sequence in sequences:
        sequence = str(sequence).zfill(2)
        latent_dir = vae_root / "sequences" / sequence / latent_folder
        if frames:
            for frame in frames:
                frame_id = Path(str(frame)).stem.zfill(6)
                files.append(latent_dir / f"{frame_id}{ext}")
        else:
            files.extend(sorted(latent_dir.glob(f"*{ext}")))

    if num_samples > 0:
        files = files[:num_samples]
    return files


def analyze_condition_file(
    path,
    spatial_shape=DEFAULT_SPATIAL_SHAPE,
    channel_candidates=DEFAULT_CHANNEL_CANDIDATES,
    expected_channels=0,
    allow_all_zero=False,
    min_std=0.0,
):
    path = Path(path)
    result = ConditionCheckResult(path=str(path))
    result.sequence, result.frame = sequence_frame_from_path(path)

    if not path.exists():
        result.issues.append("missing")
        return result
    if path.stat().st_size % np.dtype(np.float32).itemsize != 0:
        result.issues.append("not_float32_sized")
        return result

    values = np.fromfile(path, dtype=np.float32)
    result.num_values = int(values.size)
    result.channels = detect_channels(values.size, spatial_shape, channel_candidates)

    if result.channels == 0:
        expected_counts = [
            expected_value_count(channels, spatial_shape)
            for channels in channel_candidates
        ]
        result.issues.append(f"bad_size_expected_{expected_counts}")
    elif expected_channels and result.channels != expected_channels:
        result.issues.append(f"unexpected_channels_{result.channels}_expected_{expected_channels}")

    if values.size == 0:
        result.issues.append("empty")
        return result

    finite_mask = np.isfinite(values)
    result.finite = bool(finite_mask.all())
    if not result.finite:
        result.issues.append("non_finite")

    finite_values = values[finite_mask]
    if finite_values.size:
        result.min_value = float(finite_values.min())
        result.max_value = float(finite_values.max())
        result.mean = float(finite_values.mean())
        result.std = float(finite_values.std())
        result.zero_ratio = float(np.count_nonzero(finite_values == 0) / finite_values.size)

        if np.count_nonzero(finite_values) == 0 and not allow_all_zero:
            result.issues.append("all_zero")
        if min_std > 0 and result.std is not None and result.std < min_std:
            result.issues.append(f"low_std_below_{min_std}")

    result.ok = not result.issues
    return result


def summarize_results(results, allow_mixed_channels=False):
    channel_counts = {}
    for result in results:
        if result.channels:
            key = str(result.channels)
            channel_counts[key] = channel_counts.get(key, 0) + 1

    mixed_channels = len(channel_counts) > 1
    failed_results = [result for result in results if not result.ok]
    ok = not failed_results and (allow_mixed_channels or not mixed_channels)

    return {
        "ok": ok,
        "checked": len(results),
        "failed": len(failed_results),
        "mixed_channels": mixed_channels,
        "channel_counts": channel_counts,
        "failures": [asdict(result) for result in failed_results],
    }


def write_json(path, summary, results):
    payload = {
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path, results):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(ConditionCheckResult(path="")).keys())
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["issues"] = ";".join(result.issues)
            writer.writerow(row)


def print_summary(summary, max_failures=10):
    status = "OK" if summary["ok"] else "FAILED"
    print(f"image condition check: {status}")
    print(f"checked: {summary['checked']}")
    print(f"failed: {summary['failed']}")
    print(f"channel_counts: {summary['channel_counts']}")
    print(f"mixed_channels: {summary['mixed_channels']}")
    for failure in summary["failures"][:max_failures]:
        print(f"failure: {failure['path']} issues={failure['issues']}")
    if len(summary["failures"]) > max_failures:
        print(f"... {len(summary['failures']) - max_failures} more failures")


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Check whether skimba image_condition features can be used by the current dataloader/model."
    )
    parser.add_argument("--config_path", "--config-path", default="config/semantickitti_autoencoder.yaml")
    parser.add_argument("--condition_root", "--condition-root", choices=["image", "partial"], default="image")
    parser.add_argument("--vae_root", "--vae-root", default="")
    parser.add_argument("--image_condition_root", "--image-condition-root", default="")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--frames", default="")
    parser.add_argument("--num_samples", "--num-samples", type=int, default=0)
    parser.add_argument("--latent_folder", "--latent-folder", default="voxels")
    parser.add_argument("--image_folder", "--image-folder", default="voxels")
    parser.add_argument("--ext", default=".bin")
    parser.add_argument("--spatial_shape", "--spatial-shape", default="64,64,8")
    parser.add_argument("--channel_candidates", "--channel-candidates", default="24,64")
    parser.add_argument("--expected_channels", "--expected-channels", type=int, default=0)
    parser.add_argument("--allow_all_zero", "--allow-all-zero", action="store_true")
    parser.add_argument("--allow_mixed_channels", "--allow-mixed-channels", action="store_true")
    parser.add_argument("--min_std", "--min-std", type=float, default=0.0)
    parser.add_argument("--out_json", "--out-json", default="")
    parser.add_argument("--out_csv", "--out-csv", default="")
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    configs = load_training_config(args.config_path)
    vae_root = args.vae_root or str(resolve_vae_root(configs))
    default_condition_root = (
        resolve_partial_condition_root(configs)
        if args.condition_root == "partial"
        else resolve_image_condition_root(configs)
    )
    image_condition_root = args.image_condition_root or str(default_condition_root)
    spatial_shape = parse_int_tuple(args.spatial_shape)
    channel_candidates = parse_int_tuple(args.channel_candidates)
    sequences = parse_csv_items(args.sequences)
    frames = parse_csv_items(args.frames)

    vae_files = list_vae_files(
        vae_root,
        sequences=sequences,
        frames=frames,
        latent_folder=args.latent_folder,
        ext=args.ext,
        num_samples=args.num_samples,
    )
    if not vae_files:
        raise SystemExit(f"No VAE latent files found under {args.vae_root}")

    results = []
    for vae_file in vae_files:
        condition_path = image_condition_path_for(
            vae_file,
            image_condition_root=image_condition_root or None,
            image_folder=args.image_folder,
            ext=args.ext,
        )
        results.append(
            analyze_condition_file(
                condition_path,
                spatial_shape=spatial_shape,
                channel_candidates=channel_candidates,
                expected_channels=args.expected_channels,
                allow_all_zero=args.allow_all_zero,
                min_std=args.min_std,
            )
        )

    summary = summarize_results(results, allow_mixed_channels=args.allow_mixed_channels)
    print_summary(summary)

    if args.out_json:
        write_json(args.out_json, summary, results)
    if args.out_csv:
        write_csv(args.out_csv, results)

    raise SystemExit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
