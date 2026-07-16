from pathlib import Path

import pytest

from utils.frame_filtering import (
    filter_dataset_frames_by_divisor,
    frame_number_from_path,
    keep_divisible_frame_paths,
)


class DummyDataset:
    def __init__(self, im_idx):
        self.im_idx = im_idx


def test_frame_number_from_path_reads_zero_padded_stem():
    assert frame_number_from_path("/data/sequences/08/voxels/000120.bin") == 120


def test_keep_divisible_frame_paths_keeps_only_requested_stride():
    paths = [
        Path("/data/sequences/00/voxels/000000.bin"),
        Path("/data/sequences/00/voxels/000001.bin"),
        Path("/data/sequences/00/voxels/000005.bin"),
        Path("/data/sequences/00/voxels/000010.bin"),
    ]

    assert keep_divisible_frame_paths(paths, 5) == [paths[0], paths[2], paths[3]]


def test_keep_divisible_frame_paths_rejects_nonpositive_divisor():
    with pytest.raises(ValueError, match="positive integer"):
        keep_divisible_frame_paths(["/data/000000.bin"], 0)


def test_filter_dataset_frames_by_divisor_updates_im_idx_in_place():
    dataset = DummyDataset(
        [
            "/data/sequences/08/voxels/000000.bin",
            "/data/sequences/08/voxels/000009.bin",
            "/data/sequences/08/voxels/000010.bin",
        ]
    )

    assert filter_dataset_frames_by_divisor(dataset, 10) is dataset
    assert dataset.im_idx == [
        "/data/sequences/08/voxels/000000.bin",
        "/data/sequences/08/voxels/000010.bin",
    ]
