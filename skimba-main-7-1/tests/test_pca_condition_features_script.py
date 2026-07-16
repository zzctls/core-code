import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data" / "pca_condition_features_to_8.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("pca_condition_features_to_8", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fits_channel_pca_from_condition_arrays():
    module = load_script_module()
    spatial_shape = (2, 2, 1)
    condition_a = np.zeros((4,) + spatial_shape, dtype=np.float32)
    condition_b = np.zeros((4,) + spatial_shape, dtype=np.float32)
    condition_a[0] = [[1.0], [2.0]]
    condition_a[1] = [[2.0], [4.0]]
    condition_b[0] = [[3.0], [4.0]]
    condition_b[1] = [[6.0], [8.0]]
    condition_a[2] = 5.0
    condition_b[2] = 5.0

    model = module.fit_pca_model(
        [condition_a, condition_b],
        output_channels=2,
        input_channels=4,
        spatial_shape=spatial_shape,
    )

    assert model["mean"].shape == (4,)
    assert model["components"].shape == (2, 4)
    assert model["explained_variance"].shape == (2,)
    assert np.all(model["explained_variance"][:-1] >= model["explained_variance"][1:])
    assert np.allclose(model["components"] @ model["components"].T, np.eye(2), atol=1e-5)


def test_transforms_condition_with_fixed_pca_projection():
    module = load_script_module()
    spatial_shape = (2, 1, 1)
    condition = np.arange(4 * np.prod(spatial_shape), dtype=np.float32).reshape((4,) + spatial_shape)
    model = {
        "mean": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "components": np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5, 0.0],
            ],
            dtype=np.float64,
        ),
    }

    projected = module.transform_condition_with_pca(condition, model)
    flat = condition.reshape(4, -1).astype(np.float64)
    expected = (model["components"] @ (flat - model["mean"][:, None])).reshape((2,) + spatial_shape)

    assert projected.shape == (2,) + spatial_shape
    assert projected.dtype == np.float32
    assert np.allclose(projected, expected)


def test_writes_pca_compressed_files_with_matching_sequence_layout(tmp_path):
    module = load_script_module()
    spatial_shape = (2, 2, 1)
    input_root = tmp_path / "Image_transform_Voxel_Condition_Features"
    input_path = input_root / "sequences" / "08" / "voxels" / "000123.bin"
    output_root = tmp_path / "Image_transform_Voxel_Condition_Features_pca8ch"
    input_path.parent.mkdir(parents=True)

    condition = np.arange(4 * np.prod(spatial_shape), dtype=np.float32).reshape((4,) + spatial_shape)
    condition.tofile(input_path)
    model = {
        "input_channels": 4,
        "output_channels": 2,
        "spatial_shape": spatial_shape,
        "mean": np.zeros(4, dtype=np.float64),
        "components": np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    }

    results = module.transform_condition_features(
        input_root=input_root,
        output_root=output_root,
        pca_model=model,
        sequences=["08"],
        frames=["000123"],
        condition_folder="voxels",
        ext=".bin",
        overwrite=True,
        dry_run=False,
    )

    output_path = output_root / "sequences" / "08" / "voxels" / "000123.bin"
    values = np.fromfile(output_path, dtype=np.float32).reshape((2,) + spatial_shape)

    assert results == [
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "status": "written",
            "input_channels": 4,
            "output_channels": 2,
            "num_values": 8,
        }
    ]
    assert np.allclose(values[0], condition[0])
    assert np.allclose(values[1], condition[1])


def test_default_input_root_uses_raw_root_when_config_points_to_pca8ch_output(tmp_path):
    module = load_script_module()
    configured_output = tmp_path / "Image_transform_Voxel_Condition_Features_pca8ch"

    assert module.default_input_root(configured_output) == (
        tmp_path / "Image_transform_Voxel_Condition_Features"
    )
