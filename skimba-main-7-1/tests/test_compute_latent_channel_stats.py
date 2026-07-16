import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data" / "compute_latent_channel_stats.py"


def load_script_module():
    assert SCRIPT.exists(), "statistics script is not implemented"
    spec = importlib.util.spec_from_file_location("compute_latent_channel_stats", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_mapping(path, train=(0,), valid=(8,), test=(11,)):
    path.write_text(
        "split:\n"
        + "  train: [" + ", ".join(map(str, train)) + "]\n"
        + "  valid: [" + ", ".join(map(str, valid)) + "]\n"
        + "  test: [" + ", ".join(map(str, test)) + "]\n",
        encoding="utf-8",
    )


def write_latent(root, sequence, frame, values):
    path = root / "sequences" / f"{sequence:02d}" / "voxels" / f"{frame:06d}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype=np.float32).tofile(path)
    return path


def test_file_discovery_uses_train_split_and_excludes_validation(tmp_path):
    module = load_script_module()
    mapping = tmp_path / "semantic-kitti.yaml"
    write_mapping(mapping)
    train_file = write_latent(tmp_path, 0, 0, np.zeros((2, 2, 1, 1)))
    write_latent(tmp_path, 8, 0, np.full((2, 2, 1, 1), 999.0))

    sequences, files = module.discover_train_latent_files(tmp_path, mapping)

    assert sequences == ["00"]
    assert files == [train_file]
    assert all("/08/" not in str(path) for path in files)


def test_overlapping_train_and_validation_split_is_rejected(tmp_path):
    module = load_script_module()
    mapping = tmp_path / "semantic-kitti.yaml"
    write_mapping(mapping, train=(0, 8), valid=(8,))

    with pytest.raises(ValueError, match="overlap"):
        module.discover_train_latent_files(tmp_path, mapping)


def test_float64_streaming_channel_statistics_match_numpy(tmp_path):
    module = load_script_module()
    first = np.array([[[[1.0]], [[3.0]]], [[[10.0]], [[14.0]]]], dtype=np.float32)
    second = np.array([[[[5.0]], [[7.0]]], [[[18.0]], [[22.0]]]], dtype=np.float32)
    paths = [
        write_latent(tmp_path, 0, 0, first),
        write_latent(tmp_path, 0, 1, second),
    ]

    result = module.compute_channel_statistics(paths, latent_shape=(2, 2, 1, 1))
    expected = np.stack([first, second]).astype(np.float64)

    np.testing.assert_allclose(result["mean"], expected.mean(axis=(0, 2, 3, 4)))
    np.testing.assert_allclose(result["std"], expected.std(axis=(0, 2, 3, 4), ddof=0))
    assert result["sample_count"] == 2
    assert result["elements_per_channel"] == 4
    assert result["total_element_count"] == 8
    assert result["accumulation_dtype"] == "float64"


def test_report_contains_reproducible_train_mean_export_metadata(tmp_path):
    module = load_script_module()
    mapping = tmp_path / "semantic-kitti.yaml"
    write_mapping(mapping)
    latent = write_latent(tmp_path, 0, 0, np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1))
    result = module.compute_channel_statistics([latent], latent_shape=(2, 2, 2, 1))

    report = module.build_report(
        result,
        latent_root=tmp_path,
        label_mapping=mapping,
        train_sequences=["00"],
        latent_shape=(2, 2, 2, 1),
        export_mode="mean",
    )
    encoded = json.loads(json.dumps(report))

    assert encoded["split"] == "train"
    assert encoded["train_sequences"] == ["00"]
    assert encoded["latent_shape"] == [2, 2, 2, 1]
    assert encoded["export_mode"] == "mean"
    assert encoded["source_dtype"] == "float32"
    assert encoded["accumulation_dtype"] == "float64"
    assert encoded["std_definition"] == "population"

