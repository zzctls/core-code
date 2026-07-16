import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml


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
DEFAULT_MAX_VOXELS_PER_FILE = 2048
DEFAULT_SEED = 20260702
OUTPUT_ROOT_SUFFIX = "_pca8ch"


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


def split_sequences_from_config(configs, split_name="train"):
    label_mapping = Path(configs["dataset_params"]["label_mapping"])
    if not label_mapping.is_absolute():
        label_mapping = PROJECT_ROOT / label_mapping
    with label_mapping.open("r", encoding="utf-8") as stream:
        mapping = yaml.safe_load(stream)
    return [str(sequence).zfill(2) for sequence in mapping["split"][split_name]]


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


def output_path_for(input_path, output_root, condition_folder="voxels", ext=".bin"):
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


def sample_voxels(condition, max_voxels_per_file=0, rng=None):
    flat = np.asarray(condition, dtype=np.float32).reshape(condition.shape[0], -1).T
    if max_voxels_per_file and flat.shape[0] > max_voxels_per_file:
        rng = rng or np.random.default_rng(DEFAULT_SEED)
        indices = rng.choice(flat.shape[0], size=max_voxels_per_file, replace=False)
        flat = flat[np.sort(indices)]
    return flat.astype(np.float64, copy=False)


def orient_components(components):
    oriented = components.copy()
    for row in oriented:
        anchor = int(np.argmax(np.abs(row)))
        if row[anchor] < 0:
            row *= -1.0
    return oriented


def finalize_pca(sum_channels, gram_channels, sample_count, output_channels, metadata=None):
    if sample_count < 2:
        raise ValueError("PCA requires at least two sampled voxels")
    mean = sum_channels / sample_count
    covariance = (gram_channels - sample_count * np.outer(mean, mean)) / (sample_count - 1)
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    components = orient_components(eigenvectors[:, :output_channels].T)
    total_variance = float(eigenvalues.sum())
    explained = eigenvalues[:output_channels]
    explained_ratio = explained / total_variance if total_variance > 0 else np.zeros_like(explained)

    model = {
        "input_channels": int(sum_channels.shape[0]),
        "output_channels": int(output_channels),
        "sample_count": int(sample_count),
        "mean": mean.astype(np.float64),
        "components": components.astype(np.float64),
        "explained_variance": explained.astype(np.float64),
        "explained_variance_ratio": explained_ratio.astype(np.float64),
    }
    if metadata:
        model.update(metadata)
    return model


def fit_pca_model(
    conditions,
    output_channels=DEFAULT_OUTPUT_CHANNELS,
    input_channels=DEFAULT_INPUT_CHANNELS,
    spatial_shape=DEFAULT_SPATIAL_SHAPE,
    max_voxels_per_file=0,
    seed=DEFAULT_SEED,
):
    if output_channels > input_channels:
        raise ValueError("output_channels cannot exceed input_channels")

    rng = np.random.default_rng(seed)
    sum_channels = np.zeros(input_channels, dtype=np.float64)
    gram_channels = np.zeros((input_channels, input_channels), dtype=np.float64)
    sample_count = 0

    for condition in conditions:
        if isinstance(condition, (str, Path)):
            condition = load_condition(condition, input_channels=input_channels, spatial_shape=spatial_shape)
        condition = np.asarray(condition, dtype=np.float32)
        if condition.shape != (input_channels,) + tuple(spatial_shape):
            raise ValueError(
                f"Condition has shape {condition.shape}; expected "
                f"{(input_channels,) + tuple(spatial_shape)}"
            )
        samples = sample_voxels(condition, max_voxels_per_file=max_voxels_per_file, rng=rng)
        sum_channels += samples.sum(axis=0)
        gram_channels += samples.T @ samples
        sample_count += samples.shape[0]

    return finalize_pca(
        sum_channels,
        gram_channels,
        sample_count,
        output_channels,
        metadata={
            "spatial_shape": tuple(spatial_shape),
            "max_voxels_per_file": int(max_voxels_per_file),
            "seed": int(seed),
        },
    )


def fit_pca_model_from_files(
    input_root,
    sequences=None,
    frames=None,
    num_samples=0,
    condition_folder="voxels",
    ext=".bin",
    input_channels=DEFAULT_INPUT_CHANNELS,
    output_channels=DEFAULT_OUTPUT_CHANNELS,
    spatial_shape=DEFAULT_SPATIAL_SHAPE,
    max_voxels_per_file=DEFAULT_MAX_VOXELS_PER_FILE,
    seed=DEFAULT_SEED,
):
    files = list_condition_files(
        input_root,
        sequences=sequences,
        frames=frames,
        condition_folder=condition_folder,
        ext=ext,
        num_samples=num_samples,
    )
    if not files:
        raise FileNotFoundError(f"No condition files found under {input_root}")
    model = fit_pca_model(
        files,
        output_channels=output_channels,
        input_channels=input_channels,
        spatial_shape=spatial_shape,
        max_voxels_per_file=max_voxels_per_file,
        seed=seed,
    )
    model["fit_file_count"] = len(files)
    model["fit_input_root"] = str(input_root)
    return model


def transform_condition_with_pca(condition, pca_model):
    condition = np.asarray(condition, dtype=np.float32)
    input_channels = int(pca_model.get("input_channels", condition.shape[0]))
    components = np.asarray(pca_model["components"], dtype=np.float64)
    mean = np.asarray(pca_model["mean"], dtype=np.float64)
    if condition.shape[0] != input_channels:
        raise ValueError(f"Condition has {condition.shape[0]} channels; expected {input_channels}")
    flat = condition.reshape(input_channels, -1).astype(np.float64, copy=False)
    projected = components @ (flat - mean[:, None])
    return projected.reshape((components.shape[0],) + condition.shape[1:]).astype(np.float32)


def transform_condition_file(
    input_path,
    output_path,
    pca_model,
    spatial_shape=DEFAULT_SPATIAL_SHAPE,
    overwrite=True,
    dry_run=False,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    input_channels = int(pca_model["input_channels"])
    output_channels = int(pca_model["output_channels"])
    result = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "",
        "input_channels": input_channels,
        "output_channels": output_channels,
        "num_values": expected_value_count(output_channels, spatial_shape),
    }

    if not input_path.exists():
        raise FileNotFoundError(f"Missing condition feature: {input_path}")
    if output_path.exists() and not overwrite:
        result["status"] = "skipped_exists"
        return result

    condition = load_condition(input_path, input_channels=input_channels, spatial_shape=spatial_shape)
    projected = transform_condition_with_pca(condition, pca_model)

    if dry_run:
        result["status"] = "dry_run"
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(projected).tofile(output_path)
    result["status"] = "written"
    return result


def transform_condition_features(
    input_root,
    output_root,
    pca_model,
    sequences=None,
    frames=None,
    num_samples=0,
    condition_folder="voxels",
    ext=".bin",
    overwrite=True,
    dry_run=False,
):
    spatial_shape = tuple(int(item) for item in pca_model.get("spatial_shape", DEFAULT_SPATIAL_SHAPE))
    files = list_condition_files(
        input_root,
        sequences=sequences,
        frames=frames,
        condition_folder=condition_folder,
        ext=ext,
        num_samples=num_samples,
    )
    if not files:
        raise FileNotFoundError(f"No condition files found under {input_root}")

    results = []
    for input_path in files:
        output_path = output_path_for(
            input_path,
            output_root,
            condition_folder=condition_folder,
            ext=ext,
        )
        results.append(
            transform_condition_file(
                input_path,
                output_path,
                pca_model,
                spatial_shape=spatial_shape,
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


def save_pca_model(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "mean": np.asarray(model["mean"], dtype=np.float64),
        "components": np.asarray(model["components"], dtype=np.float64),
        "explained_variance": np.asarray(model["explained_variance"], dtype=np.float64),
        "explained_variance_ratio": np.asarray(model["explained_variance_ratio"], dtype=np.float64),
        "spatial_shape": np.asarray(model.get("spatial_shape", DEFAULT_SPATIAL_SHAPE), dtype=np.int64),
        "input_channels": np.asarray(model["input_channels"], dtype=np.int64),
        "output_channels": np.asarray(model["output_channels"], dtype=np.int64),
        "sample_count": np.asarray(model["sample_count"], dtype=np.int64),
        "max_voxels_per_file": np.asarray(model.get("max_voxels_per_file", 0), dtype=np.int64),
        "seed": np.asarray(model.get("seed", DEFAULT_SEED), dtype=np.int64),
        "fit_file_count": np.asarray(model.get("fit_file_count", 0), dtype=np.int64),
    }
    np.savez(path, **arrays)


def load_pca_model(path):
    with np.load(path) as data:
        return {
            "mean": data["mean"].astype(np.float64),
            "components": data["components"].astype(np.float64),
            "explained_variance": data["explained_variance"].astype(np.float64),
            "explained_variance_ratio": data["explained_variance_ratio"].astype(np.float64),
            "spatial_shape": tuple(int(item) for item in data["spatial_shape"].tolist()),
            "input_channels": int(data["input_channels"]),
            "output_channels": int(data["output_channels"]),
            "sample_count": int(data["sample_count"]),
            "max_voxels_per_file": int(data["max_voxels_per_file"]),
            "seed": int(data["seed"]),
            "fit_file_count": int(data["fit_file_count"]),
        }


def summarize_model(model):
    return {
        "input_channels": int(model["input_channels"]),
        "output_channels": int(model["output_channels"]),
        "sample_count": int(model["sample_count"]),
        "fit_file_count": int(model.get("fit_file_count", 0)),
        "explained_variance_ratio": np.asarray(model["explained_variance_ratio"]).tolist(),
        "explained_variance_ratio_sum": float(np.asarray(model["explained_variance_ratio"]).sum()),
    }


def summarize_results(results):
    status_counts = {}
    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "ok": True,
        "total": len(results),
        "status_counts": status_counts,
    }


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path, results):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["input_path", "output_path", "status", "input_channels", "output_channels", "num_values"]
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_fit_summary(model, model_path):
    summary = summarize_model(model)
    print("condition PCA fit: OK")
    print(f"model_path: {model_path}")
    print(f"fit_file_count: {summary['fit_file_count']}")
    print(f"sample_count: {summary['sample_count']}")
    print(f"explained_variance_ratio_sum: {summary['explained_variance_ratio_sum']:.6f}")


def print_transform_summary(summary, results, max_items=10):
    print("condition PCA transform: OK")
    print(f"total: {summary['total']}")
    print(f"status_counts: {summary['status_counts']}")
    for result in results[:max_items]:
        print(f"{result['status']}: {result['input_path']} -> {result['output_path']}")
    if len(results) > max_items:
        print(f"... {len(results) - max_items} more files")


def add_common_args(parser):
    parser.add_argument("--config_path", "--config-path", default="config/semantickitti_autoencoder.yaml")
    parser.add_argument("--condition_root", "--condition-root", choices=["image", "partial"], default="image")
    parser.add_argument("--input_root", "--input-root", default="")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--frames", default="")
    parser.add_argument("--num_samples", "--num-samples", type=int, default=0)
    parser.add_argument("--condition_folder", "--condition-folder", default="voxels")
    parser.add_argument("--ext", default=".bin")
    parser.add_argument("--spatial_shape", "--spatial-shape", default="64,64,8")
    parser.add_argument("--input_channels", "--input-channels", type=int, default=DEFAULT_INPUT_CHANNELS)
    parser.add_argument("--output_channels", "--output-channels", type=int, default=DEFAULT_OUTPUT_CHANNELS)


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Fit and apply a fixed PCA/SVD 1x1x1 channel projection for raw 64-channel "
            "MonoScene/FLoSP condition features."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit")
    add_common_args(fit_parser)
    fit_parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    fit_parser.add_argument("--max_voxels_per_file", "--max-voxels-per-file", type=int, default=DEFAULT_MAX_VOXELS_PER_FILE)
    fit_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    fit_parser.add_argument("--pca_model", "--pca-model", default="")
    fit_parser.add_argument("--out_json", "--out-json", default="")

    transform_parser = subparsers.add_parser("transform")
    add_common_args(transform_parser)
    transform_parser.add_argument("--output_root", "--output-root", default="")
    transform_parser.add_argument("--pca_model", "--pca-model", required=True)
    transform_parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=True)
    transform_parser.add_argument("--no_overwrite", "--no-overwrite", dest="overwrite", action="store_false")
    transform_parser.add_argument("--dry_run", "--dry-run", action="store_true")
    transform_parser.add_argument("--out_json", "--out-json", default="")
    transform_parser.add_argument("--out_csv", "--out-csv", default="")

    fit_transform_parser = subparsers.add_parser("fit-transform")
    add_common_args(fit_transform_parser)
    fit_transform_parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    fit_transform_parser.add_argument("--max_voxels_per_file", "--max-voxels-per-file", type=int, default=DEFAULT_MAX_VOXELS_PER_FILE)
    fit_transform_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    fit_transform_parser.add_argument("--output_root", "--output-root", default="")
    fit_transform_parser.add_argument("--pca_model", "--pca-model", default="")
    fit_transform_parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=True)
    fit_transform_parser.add_argument("--no_overwrite", "--no-overwrite", dest="overwrite", action="store_false")
    fit_transform_parser.add_argument("--dry_run", "--dry-run", action="store_true")
    fit_transform_parser.add_argument("--out_json", "--out-json", default="")
    fit_transform_parser.add_argument("--out_csv", "--out-csv", default="")
    return parser


def resolve_input_root(configs, args):
    if args.input_root:
        return Path(args.input_root)
    configured_root = (
        resolve_image_condition_root(configs)
        if args.condition_root == "image"
        else resolve_partial_condition_root(configs)
    )
    return default_input_root(configured_root)


def resolve_fit_sequences(configs, args):
    explicit = parse_csv_items(args.sequences)
    if explicit:
        return explicit
    if args.split == "all":
        return []
    return split_sequences_from_config(configs, args.split)


def run_fit(args, configs, input_root):
    sequences = resolve_fit_sequences(configs, args)
    model = fit_pca_model_from_files(
        input_root=input_root,
        sequences=sequences,
        frames=parse_csv_items(args.frames),
        num_samples=args.num_samples,
        condition_folder=args.condition_folder,
        ext=args.ext,
        input_channels=args.input_channels,
        output_channels=args.output_channels,
        spatial_shape=parse_int_tuple(args.spatial_shape),
        max_voxels_per_file=args.max_voxels_per_file,
        seed=args.seed,
    )
    model_path = Path(args.pca_model) if args.pca_model else Path(input_root).with_name(f"{Path(input_root).name}_pca{args.output_channels}.npz")
    save_pca_model(model_path, model)
    print_fit_summary(model, model_path)
    if args.out_json:
        write_json(args.out_json, {"summary": summarize_model(model), "model_path": str(model_path)})
    return model, model_path


def run_transform(args, input_root, pca_model):
    output_root = Path(args.output_root) if args.output_root else default_output_root(input_root)
    results = transform_condition_features(
        input_root=input_root,
        output_root=output_root,
        pca_model=pca_model,
        sequences=parse_csv_items(args.sequences),
        frames=parse_csv_items(args.frames),
        num_samples=args.num_samples,
        condition_folder=args.condition_folder,
        ext=args.ext,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    summary = summarize_results(results)
    print_transform_summary(summary, results)
    if args.out_json:
        write_json(args.out_json, {"summary": summary, "results": results})
    if args.out_csv:
        write_csv(args.out_csv, results)
    return results


def main(argv=None):
    args = build_argparser().parse_args(argv)
    configs = load_training_config(args.config_path)
    input_root = resolve_input_root(configs, args)

    if args.command == "fit":
        run_fit(args, configs, input_root)
    elif args.command == "transform":
        pca_model = load_pca_model(args.pca_model)
        run_transform(args, input_root, pca_model)
    elif args.command == "fit-transform":
        pca_model, model_path = run_fit(args, configs, input_root)
        args.pca_model = str(model_path)
        run_transform(args, input_root, pca_model)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
