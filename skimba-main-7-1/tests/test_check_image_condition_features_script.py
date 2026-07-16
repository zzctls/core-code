import importlib.util
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data" / "check_image_condition_features.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("check_image_condition_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_infers_image_condition_path_from_semantic_vae_root():
    module = load_script_module()

    vae_path = Path("/data/VAE_Encoder_Features_Semantic20/sequences/08/voxels/000123.bin")

    assert module.infer_image_condition_path(vae_path) == (
        Path("/data/Image_transform_Voxel_Condition_Features/sequences/08/voxels/000123.bin")
    )


def test_script_detects_valid_channels_and_bad_content():
    module = load_script_module()
    spatial_shape = (2, 2, 1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        valid_path = tmp_path / "valid.bin"
        np.arange(8 * 4, dtype=np.float32).tofile(valid_path)
        valid = module.analyze_condition_file(
            valid_path,
            spatial_shape=spatial_shape,
            channel_candidates=(8, 64),
        )

        assert valid.channels == 8
        assert valid.ok
        assert not valid.issues

        zero_path = tmp_path / "zero.bin"
        np.zeros(8 * 4, dtype=np.float32).tofile(zero_path)
        zero = module.analyze_condition_file(
            zero_path,
            spatial_shape=spatial_shape,
            channel_candidates=(8, 64),
        )

        assert zero.channels == 8
        assert not zero.ok
        assert "all_zero" in zero.issues

        nan_path = tmp_path / "nan.bin"
        nan_values = np.zeros(8 * 4, dtype=np.float32)
        nan_values[0] = np.nan
        nan_values.tofile(nan_path)
        nan = module.analyze_condition_file(
            nan_path,
            spatial_shape=spatial_shape,
            channel_candidates=(8, 64),
        )

        assert not nan.ok
        assert "non_finite" in nan.issues


def test_script_rejects_mixed_channel_batches_by_default():
    module = load_script_module()

    summary = module.summarize_results(
        [
            module.ConditionCheckResult(path="a.bin", channels=8, ok=True),
            module.ConditionCheckResult(path="b.bin", channels=64, ok=True),
        ],
        allow_mixed_channels=False,
    )

    assert not summary["ok"]
    assert summary["mixed_channels"]
    assert summary["channel_counts"] == {"8": 1, "64": 1}


def test_script_accepts_64_channel_semantic_partial_condition():
    module = load_script_module()
    spatial_shape = (2, 2, 1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "semantic_partial.bin"
        values = np.zeros(64 * 4, dtype=np.float32)
        values[1::64] = 0.8
        values[20::64] = 0.8
        values[21::64] = 0.2
        values[23::64] = 1.0
        values.tofile(path)

        result = module.analyze_condition_file(
            path,
            spatial_shape=spatial_shape,
            channel_candidates=(8, 64),
            expected_channels=64,
            min_std=0.001,
        )

        assert result.channels == 64
        assert result.ok
