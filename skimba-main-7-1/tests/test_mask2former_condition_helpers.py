import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "data" / "mask2former_condition.py"


def load_module():
    spec = importlib.util.spec_from_file_location("mask2former_condition", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builds_fixed_64_channel_semantic_condition():
    module = load_module()
    probs = np.zeros((3, 4, 5), dtype=np.float32)
    probs[0, :, :] = 0.1
    probs[1, :, :] = 0.7
    probs[2, :, :] = 0.2
    model_to_semkitti = {0: 9, 1: 10, 2: 18}

    condition = module.build_semantic_condition_2d(
        probs,
        model_to_semkitti=model_to_semkitti,
        output_channels=64,
        semantic_channels=20,
    )

    assert condition.shape == (64, 4, 5)
    assert condition.dtype == np.float32
    assert np.allclose(condition[9], 0.1)
    assert np.allclose(condition[10], 0.7)
    assert np.allclose(condition[18], 0.2)
    assert np.allclose(condition[20], 0.7)
    assert np.all(condition[21] >= 0.0)
    assert np.all(condition[21] <= 1.0)
    assert set(np.unique(condition[23])).issubset({0.0, 1.0})
    assert np.count_nonzero(condition[24:]) == 0


def test_unmapped_classes_accumulate_in_unknown_channel():
    module = load_module()
    probs = np.zeros((2, 2, 2), dtype=np.float32)
    probs[0, :, :] = 0.25
    probs[1, :, :] = 0.75

    condition = module.build_semantic_condition_2d(
        probs,
        model_to_semkitti={0: 5},
        output_channels=64,
        semantic_channels=20,
    )

    assert np.allclose(condition[5], 0.25)
    assert np.allclose(condition[0], 0.75)


def test_depth_surface_weights_peak_at_depth_and_cut_off_far_voxels():
    module = load_module()
    projected_pix = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [0, 1],
        ],
        dtype=np.int64,
    )
    fov_mask = np.array([True, True, True, False])
    voxel_depth = np.array([10.0, 10.35, 11.2, 10.0], dtype=np.float32)
    depth_map = np.full((2, 3), 10.0, dtype=np.float32)

    weights = module.compute_depth_surface_weights(
        projected_pix,
        fov_mask,
        voxel_depth,
        depth_map,
    )

    assert weights.dtype == np.float32
    assert np.isclose(weights[0], 1.0)
    assert 0.0 < weights[1] < 1.0
    assert weights[2] == 0.0
    assert weights[3] == 0.0


def test_depth_surface_weights_reject_invalid_depth_values():
    module = load_module()
    projected_pix = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.int64)
    fov_mask = np.array([True, True, True])
    voxel_depth = np.array([5.0, 5.0, np.nan], dtype=np.float32)
    depth_map = np.array([[5.0, 0.0, 5.0]], dtype=np.float32)

    weights = module.compute_depth_surface_weights(
        projected_pix,
        fov_mask,
        voxel_depth,
        depth_map,
    )

    assert np.isclose(weights[0], 1.0)
    assert weights[1] == 0.0
    assert weights[2] == 0.0


def test_resolves_mobilestereonet_depth_layouts(tmp_path):
    module = load_module()
    expected = tmp_path / "sequences" / "08" / "depth2" / "000123.npy"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"depth")

    assert module.resolve_depth_path(tmp_path, "08", "000123") == expected


def test_loads_model_to_semantickitti_mapping(tmp_path):
    module = load_module()
    mapping_path = tmp_path / "mapping.yaml"
    mapping_path.write_text(
        "model_to_semantickitti:\n"
        "  0: 9\n"
        "  1: 10\n",
        encoding="utf-8",
    )

    mapping = module.load_model_to_semkitti_mapping(mapping_path)

    assert mapping == {0: 9, 1: 10}


def test_default_mapping_matches_cityscapes_semantic_order():
    module = load_module()
    mapping = module.load_model_to_semkitti_mapping(
        ROOT / "config" / "mask2former_to_semantickitti.yaml"
    )

    assert mapping == {
        0: 9,   # road
        1: 11,  # sidewalk
        2: 13,  # building
        3: 0,   # wall
        4: 14,  # fence
        5: 18,  # pole
        6: 19,  # traffic light
        7: 19,  # traffic sign
        8: 15,  # vegetation
        9: 17,  # terrain
        10: 0,  # sky
        11: 6,  # person
        12: 7,  # rider
        13: 1,  # car
        14: 4,  # truck
        15: 5,  # bus
        16: 5,  # train
        17: 3,  # motorcycle
        18: 2,  # bicycle
    }
