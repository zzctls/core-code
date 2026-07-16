from pathlib import Path


def frame_number_from_path(path):
    return int(Path(path).stem)


def keep_divisible_frame_paths(paths, divisor):
    if divisor <= 0:
        raise ValueError("divisor must be a positive integer")
    return [
        path
        for path in paths
        if frame_number_from_path(path) % divisor == 0
    ]


def filter_dataset_frames_by_divisor(dataset, divisor):
    dataset.im_idx = keep_divisible_frame_paths(dataset.im_idx, divisor)
    return dataset
