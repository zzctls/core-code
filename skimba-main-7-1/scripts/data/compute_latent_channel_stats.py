"""Compute reproducible train-split per-channel statistics for raw VAE latents."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.config_paths import load_training_config, resolve_vae_root


DEFAULT_LATENT_SHAPE = (8, 64, 64, 8)


def parse_shape(value):
    if isinstance(value, str):
        shape = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        shape = tuple(int(item) for item in value)
    if len(shape) != 4 or any(size <= 0 for size in shape):
        raise ValueError(f"Expected positive C,W,L,H latent shape, got {value}")
    return shape


def _sequence_ids(values):
    return [str(int(value)).zfill(2) for value in values]


def _parse_split_without_yaml(label_mapping):
    split = {}
    in_split = False
    active_name = None
    for raw_line in Path(label_mapping).read_text(encoding="utf-8").splitlines():
        content = raw_line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip())
        stripped = content.strip()
        if indent == 0:
            in_split = stripped == "split:"
            active_name = None
            continue
        if not in_split:
            continue
        if indent == 2 and ":" in stripped:
            active_name, inline = stripped.split(":", 1)
            active_name = active_name.strip()
            split[active_name] = []
            inline = inline.strip()
            if inline.startswith("[") and inline.endswith("]"):
                split[active_name] = [
                    item.strip().strip("'\"")
                    for item in inline[1:-1].split(",")
                    if item.strip()
                ]
            continue
        if indent >= 4 and active_name and stripped.startswith("-"):
            split[active_name].append(stripped[1:].strip().strip("'\""))
    if not split:
        raise ValueError(f"Could not parse SemanticKITTI split from {label_mapping}")
    return split


def load_split_mapping(label_mapping):
    try:
        import yaml

        with Path(label_mapping).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle).get("split", {})
    except ModuleNotFoundError:
        return _parse_split_without_yaml(label_mapping)


def discover_train_latent_files(latent_root, label_mapping):
    latent_root = Path(latent_root)
    label_mapping = Path(label_mapping)
    split = load_split_mapping(label_mapping)

    train_sequences = _sequence_ids(split.get("train", []))
    valid_sequences = set(_sequence_ids(split.get("valid", split.get("validation", []))))
    test_sequences = set(_sequence_ids(split.get("test", [])))
    if not train_sequences:
        raise ValueError(f"SemanticKITTI train split is empty in {label_mapping}")
    overlap = set(train_sequences) & (valid_sequences | test_sequences)
    if overlap:
        raise ValueError(
            f"SemanticKITTI split overlap: train also contains {sorted(overlap)}"
        )

    files = []
    for sequence in train_sequences:
        sequence_files = sorted(
            (latent_root / "sequences" / sequence / "voxels").glob("*.bin")
        )
        if not sequence_files:
            raise FileNotFoundError(
                f"No raw latent .bin files found for train sequence {sequence} under {latent_root}"
            )
        files.extend(sequence_files)
    return train_sequences, files


def compute_channel_statistics(latent_files, latent_shape=DEFAULT_LATENT_SHAPE):
    latent_shape = parse_shape(latent_shape)
    channels = latent_shape[0]
    expected_elements = int(np.prod(latent_shape))
    mean = np.zeros(channels, dtype=np.float64)
    m2 = np.zeros(channels, dtype=np.float64)
    elements_per_channel = 0
    sample_count = 0

    for latent_path in latent_files:
        latent_path = Path(latent_path)
        raw = np.fromfile(latent_path, dtype=np.float32)
        if raw.size != expected_elements:
            raise ValueError(
                f"{latent_path} has {raw.size} float32 values; expected {expected_elements} "
                f"for latent shape {latent_shape}"
            )
        latent = raw.reshape(latent_shape).astype(np.float64)
        if not np.isfinite(latent).all():
            raise ValueError(f"{latent_path} contains non-finite latent values")

        flat = latent.reshape(channels, -1)
        batch_count = flat.shape[1]
        batch_mean = flat.mean(axis=1, dtype=np.float64)
        centered = flat - batch_mean[:, None]
        batch_m2 = np.sum(centered * centered, axis=1, dtype=np.float64)

        combined_count = elements_per_channel + batch_count
        delta = batch_mean - mean
        mean += delta * (batch_count / combined_count)
        m2 += batch_m2 + delta * delta * (
            elements_per_channel * batch_count / combined_count
        )
        elements_per_channel = combined_count
        sample_count += 1

    if sample_count == 0:
        raise ValueError("No raw latent files were provided")

    variance = np.maximum(m2 / elements_per_channel, 0.0)
    std = np.sqrt(variance)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "sample_count": sample_count,
        "elements_per_channel": int(elements_per_channel),
        "total_element_count": int(elements_per_channel * channels),
        "accumulation_dtype": "float64",
    }


def build_report(
    statistics,
    *,
    latent_root,
    label_mapping,
    train_sequences,
    latent_shape=DEFAULT_LATENT_SHAPE,
    export_mode="mean",
):
    latent_shape = parse_shape(latent_shape)
    return {
        "format_version": 1,
        "mean": list(statistics["mean"]),
        "std": list(statistics["std"]),
        "sample_count": int(statistics["sample_count"]),
        "input_file_count": int(statistics["sample_count"]),
        "elements_per_channel": int(statistics["elements_per_channel"]),
        "total_element_count": int(statistics["total_element_count"]),
        "split": "train",
        "train_sequences": list(train_sequences),
        "latent_shape": list(latent_shape),
        "export_mode": export_mode,
        "source_dtype": "float32",
        "accumulation_dtype": statistics["accumulation_dtype"],
        "std_definition": "population",
        "algorithm": "parallel_welford_per_channel",
        "latent_root": str(Path(latent_root).resolve()),
        "label_mapping": str(Path(label_mapping).resolve()),
    }


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Compute float64 streaming per-channel mean/std from only the "
            "SemanticKITTI train-split raw posterior-mean latents."
        )
    )
    parser.add_argument(
        "--config_path",
        "--config-path",
        default="config/semantickitti_autoencoder.yaml",
    )
    parser.add_argument("--latent_root", "--latent-root", default="")
    parser.add_argument("--label_mapping", "--label-mapping", default="")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--latent_shape",
        "--latent-shape",
        default=",".join(map(str, DEFAULT_LATENT_SHAPE)),
    )
    parser.add_argument("--export_mode", "--export-mode", choices=["mean"], default="mean")
    return parser


def run(args):
    configs = load_training_config(args.config_path)
    latent_root = Path(args.latent_root) if args.latent_root else resolve_vae_root(configs)
    label_mapping = Path(
        args.label_mapping or configs["dataset_params"].get("label_mapping", "")
    )
    if not str(latent_root):
        raise ValueError("Set --latent-root or configure dataset_params.vae_feature_root")
    if not str(label_mapping):
        raise ValueError("Set --label-mapping or configure dataset_params.label_mapping")

    latent_shape = parse_shape(args.latent_shape)
    train_sequences, latent_files = discover_train_latent_files(
        latent_root,
        label_mapping,
    )
    statistics = compute_channel_statistics(latent_files, latent_shape)
    report = build_report(
        statistics,
        latent_root=latent_root,
        label_mapping=label_mapping,
        train_sequences=train_sequences,
        latent_shape=latent_shape,
        export_mode=args.export_mode,
    )

    configured_output = configs["model_params"].get("latent_normalization", {}).get(
        "stats_path", ""
    )
    output = Path(
        args.output
        or configured_output
        or (latent_root / "latent_channel_stats.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote train-only latent statistics for {report['sample_count']} samples "
        f"to {output}"
    )
    return report


def main():
    run(build_argparser().parse_args())


if __name__ == "__main__":
    main()
