import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data" / "compress_condition_features_to_8.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("compress_condition_features_to_8", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compresses_only_first_24_channels_into_semantic_groups():
    module = load_script_module()
    spatial_shape = (2, 1, 1)
    condition = np.zeros((64,) + spatial_shape, dtype=np.float32)
    for channel in range(20):
        condition[channel] = channel + 1
    condition[20] = 0.75
    condition[21] = 0.25
    condition[22] = 0.5
    condition[23] = 1.0
    condition[24:] = 99.0

    compressed = module.compress_condition_channels(
        condition,
        uncertainty_mode="entropy",
    )

    assert compressed.shape == (8,) + spatial_shape
    assert compressed.dtype == np.float32
    assert np.allclose(compressed[0], condition[[9, 10, 12]].sum(axis=0))
    assert np.allclose(compressed[1], condition[11])
    assert np.allclose(compressed[2], condition[[1, 4, 5]].sum(axis=0))
    assert np.allclose(compressed[3], condition[[2, 3, 6, 7, 8]].sum(axis=0))
    assert np.allclose(compressed[4], condition[[13, 14, 18, 19]].sum(axis=0))
    assert np.allclose(compressed[5], condition[[15, 16, 17]].sum(axis=0))
    assert np.allclose(compressed[6], condition[21])
    assert np.allclose(compressed[7], condition[23])
    assert not np.any(compressed == 99.0)


def test_offline_compression_uses_fixed_1x1x1_conv_projection():
    module = load_script_module()
    spatial_shape = (2, 1, 1)
    condition = np.zeros((64,) + spatial_shape, dtype=np.float32)
    condition[9] = 0.2
    condition[10] = 0.3
    condition[12] = 0.4
    condition[21] = 0.5
    condition[23] = 1.0

    conv = module.build_semantic_projection_conv1x1(uncertainty_mode="entropy")
    compressed = module.compress_condition_channels(condition, uncertainty_mode="entropy")

    assert tuple(conv.weight.shape) == (8, 64, 1, 1, 1)
    assert conv.weight.requires_grad is False
    assert conv.bias.requires_grad is False
    assert np.allclose(compressed[0], 0.9)
    assert np.allclose(compressed[6], 0.5)
    assert np.allclose(compressed[7], 1.0)


def test_writes_compressed_condition_file_with_matching_sequence_layout(tmp_path):
    module = load_script_module()
    spatial_shape = (2, 2, 1)
    input_root = tmp_path / "Mask2Former_Partial_Condition_Features"
    input_path = input_root / "sequences" / "08" / "voxels" / "000123.bin"
    output_root = tmp_path / "Mask2Former_Partial_Condition_Features_8ch"
    input_path.parent.mkdir(parents=True)

    condition = np.zeros((64,) + spatial_shape, dtype=np.float32)
    condition[9] = 0.2
    condition[10] = 0.3
    condition[21] = 0.4
    condition[23] = 1.0
    condition[24:] = 42.0
    condition.tofile(input_path)

    results = module.compress_condition_features(
        input_root=input_root,
        output_root=output_root,
        sequences=["08"],
        frames=["000123"],
        condition_folder="voxels",
        ext=".bin",
        spatial_shape=spatial_shape,
        overwrite=True,
        dry_run=False,
    )

    output_path = output_root / "sequences" / "08" / "voxels" / "000123.bin"
    values = np.fromfile(output_path, dtype=np.float32).reshape((8,) + spatial_shape)

    assert results == [
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "status": "written",
            "input_channels": 64,
            "output_channels": 8,
            "num_values": 32,
        }
    ]
    assert np.allclose(values[0], 0.5)
    assert np.allclose(values[6], 0.4)
    assert np.allclose(values[7], 1.0)
    assert not np.any(values == 42.0)


def test_default_input_root_uses_raw_root_when_config_points_to_8ch_output(tmp_path):
    module = load_script_module()
    configured_output = tmp_path / "Mask2Former_Partial_Condition_Features_8ch"

    assert module.default_input_root(configured_output) == (
        tmp_path / "Mask2Former_Partial_Condition_Features"
    )
