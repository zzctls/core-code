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
    resolve_partial_condition_root,
    resolve_vae_root,
)

DEFAULT_SPATIAL_SHAPE = (64, 64, 8)
DEFAULT_CHANNELS = 8
KNOWN_VAE_ROOT_NAMES = (
    "VAE_Encoder_Features_One_To_One",
    "VAE_Encoder_Features_Semantic20",
)
PARTIAL_CONDITION_ROOT_NAME = "Condition_Features_2"


def parse_csv_items(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_tuple(value):
    return tuple(int(item) for item in parse_csv_items(value))


def normalize_ext(value):
    return value if value.startswith(".") else f".{value}"


def sequence_frame_from_path(path):
    path = Path(path)
    parts = path.parts
    if "sequences" in parts:
        index = parts.index("sequences")
        if index + 1 < len(parts):
            return parts[index + 1], path.stem.zfill(6)
    return "", path.stem.zfill(6)


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


def infer_partial_condition_path(vae_path):
    vae_path = Path(vae_path)
    path_text = str(vae_path)
    for root_name in KNOWN_VAE_ROOT_NAMES:
        if root_name in path_text:
            return Path(path_text.replace(root_name, PARTIAL_CONDITION_ROOT_NAME))
    raise ValueError(
        f"Could not infer partial condition path from {vae_path}. "
        f"Expected one of {KNOWN_VAE_ROOT_NAMES} in the VAE root path."
    )


def partial_condition_path_for(vae_path, output_root=None, partial_folder="voxels", ext=".bin"):
    if output_root:
        sequence, frame = sequence_frame_from_path(vae_path)
        if not sequence:
            raise ValueError(f"Could not infer sequence id from VAE path: {vae_path}")
        return (
            Path(output_root)
            / "sequences"
            / sequence.zfill(2)
            / partial_folder
            / f"{frame}{normalize_ext(ext)}"
        )
    return infer_partial_condition_path(vae_path)


def expected_value_count(channels, spatial_shape):
    return int(channels * np.prod(spatial_shape))


def write_zero_partial_file(path, channels=DEFAULT_CHANNELS, spatial_shape=DEFAULT_SPATIAL_SHAPE, overwrite=True, dry_run=False):
    path = Path(path)
    num_values = expected_value_count(channels, spatial_shape)
    result = {
        "path": str(path),
        "channels": int(channels),
        "spatial_shape": list(spatial_shape),
        "num_values": int(num_values),
        "status": "",
    }

    if path.exists() and not overwrite:
        result["status"] = "skipped_exists"
        return result

    if dry_run:
        result["status"] = "dry_run"
        return result

    path.parent.mkdir(parents=True, exist_ok=True)
    np.zeros(num_values, dtype=np.float32).tofile(path)
    result["status"] = "written"
    return result


def create_zero_partial_features(
    vae_root,
    output_root=None,
    sequences=None,
    frames=None,
    num_samples=0,
    latent_folder="voxels",
    partial_folder="voxels",
    ext=".bin",
    channels=DEFAULT_CHANNELS,
    spatial_shape=DEFAULT_SPATIAL_SHAPE,
    overwrite=True,
    dry_run=False,
):
    vae_files = list_vae_files(
        vae_root,
        sequences=sequences,
        frames=frames,
        latent_folder=latent_folder,
        ext=ext,
        num_samples=num_samples,
    )
    if not vae_files:
        raise FileNotFoundError(f"No VAE latent files found under {vae_root}")

    results = []
    for vae_file in vae_files:
        if not vae_file.exists():
            raise FileNotFoundError(f"Missing VAE latent file: {vae_file}")
        partial_path = partial_condition_path_for(
            vae_file,
            output_root=output_root,
            partial_folder=partial_folder,
            ext=ext,
        )
        results.append(
            write_zero_partial_file(
                partial_path,
                channels=channels,
                spatial_shape=spatial_shape,
                overwrite=overwrite,
                dry_run=dry_run,
            )
        )
    return results


def summarize_results(results):
    status_counts = {}
    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "ok": True,
        "total": len(results),
        "status_counts": status_counts,
        "channels": sorted({result["channels"] for result in results}),
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
    fieldnames = ["path", "channels", "spatial_shape", "num_values", "status"]
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["spatial_shape"] = ",".join(str(item) for item in result["spatial_shape"])
            writer.writerow(row)


def print_summary(summary, results, max_items=10):
    print("zero partial condition generation: OK")
    print(f"total: {summary['total']}")
    print(f"status_counts: {summary['status_counts']}")
    print(f"channels: {summary['channels']}")
    for result in results[:max_items]:
        print(f"{result['status']}: {result['path']}")
    if len(results) > max_items:
        print(f"... {len(results) - max_items} more files")


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Create zero partial-condition features aligned with skimba VAE latent files."
    )
    parser.add_argument("--config_path", "--config-path", default="config/semantickitti_autoencoder.yaml")
    parser.add_argument("--vae_root", "--vae-root", default="")
    parser.add_argument("--output_root", "--output-root", default="")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--frames", default="")
    parser.add_argument("--num_samples", "--num-samples", type=int, default=0)
    parser.add_argument("--latent_folder", "--latent-folder", default="voxels")
    parser.add_argument("--partial_folder", "--partial-folder", default="voxels")
    parser.add_argument("--ext", default=".bin")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--spatial_shape", "--spatial-shape", default="64,64,8")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=True)
    parser.add_argument("--no_overwrite", "--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--out_json", "--out-json", default="")
    parser.add_argument("--out_csv", "--out-csv", default="")
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    configs = load_training_config(args.config_path)
    vae_root = args.vae_root or str(resolve_vae_root(configs))
    output_root = args.output_root or str(resolve_partial_condition_root(configs))
    results = create_zero_partial_features(
        vae_root=vae_root,
        output_root=output_root or None,
        sequences=parse_csv_items(args.sequences),
        frames=parse_csv_items(args.frames),
        num_samples=args.num_samples,
        latent_folder=args.latent_folder,
        partial_folder=args.partial_folder,
        ext=args.ext,
        channels=args.channels,
        spatial_shape=parse_int_tuple(args.spatial_shape),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    summary = summarize_results(results)
    print_summary(summary, results)

    if args.out_json:
        write_json(args.out_json, summary, results)
    if args.out_csv:
        write_csv(args.out_csv, results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
