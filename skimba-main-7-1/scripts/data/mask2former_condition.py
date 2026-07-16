import math
from pathlib import Path

import numpy as np
import yaml


DEFAULT_OUTPUT_CHANNELS = 64
DEFAULT_SEMANTIC_CHANNELS = 20
DEFAULT_SPATIAL_SHAPE = (64, 64, 8)


def parse_csv_items(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_ext(value):
    return value if value.startswith(".") else f".{value}"


def sequence_frame_from_path(path):
    path = Path(path)
    parts = path.parts
    if "sequences" in parts:
        index = parts.index("sequences")
        if index + 1 < len(parts):
            return parts[index + 1].zfill(2), path.stem.zfill(6)
    return "", path.stem.zfill(6)


def condition_path_for(vae_path, output_root, condition_folder="voxels", ext=".bin"):
    sequence, frame = sequence_frame_from_path(vae_path)
    if not sequence:
        raise ValueError(f"Could not infer sequence id from VAE path: {vae_path}")
    return Path(output_root) / "sequences" / sequence / condition_folder / f"{frame}{normalize_ext(ext)}"


def resolve_depth_path(depth_root, sequence, frame):
    root = Path(depth_root)
    candidates = (
        root / "sequences" / sequence / "depth2" / f"{frame}.npy",
        root / sequence / "depth2" / f"{frame}.npy",
        root / sequence / f"{frame}.npy",
        root / "sequences" / sequence / "voxels" / f"{frame}.npy",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Depth map not found. Checked: " + ", ".join(str(path) for path in candidates)
    )


def load_model_to_semkitti_mapping(path):
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    mapping = data.get("model_to_semantickitti", {})
    return {int(source): int(target) for source, target in mapping.items()}


def normalized_entropy(probabilities, eps=1e-8):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    class_count = probabilities.shape[0]
    if class_count <= 1:
        return np.zeros(probabilities.shape[1:], dtype=np.float32)
    entropy = -(probabilities * np.log(np.clip(probabilities, eps, 1.0))).sum(axis=0)
    return (entropy / math.log(class_count)).astype(np.float32)


def semantic_boundary(probabilities):
    labels = np.argmax(probabilities, axis=0).astype(np.int32)
    boundary = np.zeros(labels.shape, dtype=np.float32)
    boundary[:, 1:] = np.maximum(boundary[:, 1:], labels[:, 1:] != labels[:, :-1])
    boundary[1:, :] = np.maximum(boundary[1:, :], labels[1:, :] != labels[:-1, :])
    return boundary


def compute_depth_surface_weights(
    projected_pix,
    fov_mask,
    voxel_depth,
    depth_map,
    relative_tolerance=0.1,
    min_tolerance=0.7,
    max_tolerance=1.5,
):
    projected_pix = np.asarray(projected_pix, dtype=np.int64)
    fov_mask = np.asarray(fov_mask, dtype=np.bool_)
    voxel_depth = np.asarray(voxel_depth, dtype=np.float32)
    depth_map = np.asarray(depth_map, dtype=np.float32)

    if depth_map.ndim == 3 and depth_map.shape[0] == 1:
        depth_map = depth_map[0]
    if depth_map.ndim != 2:
        raise ValueError(f"Expected depth map [H,W] or [1,H,W], got {depth_map.shape}")
    if projected_pix.shape != (len(voxel_depth), 2):
        raise ValueError("Projected pixels and voxel depths must describe the same voxels")
    if fov_mask.shape != voxel_depth.shape:
        raise ValueError("FOV mask and voxel depths must have the same shape")

    valid = fov_mask.copy()
    valid &= projected_pix[:, 0] >= 0
    valid &= projected_pix[:, 0] < depth_map.shape[1]
    valid &= projected_pix[:, 1] >= 0
    valid &= projected_pix[:, 1] < depth_map.shape[0]

    sampled_depth = np.zeros_like(voxel_depth)
    valid_indices = np.flatnonzero(valid)
    sampled_depth[valid_indices] = depth_map[
        projected_pix[valid_indices, 1],
        projected_pix[valid_indices, 0],
    ]
    valid &= np.isfinite(sampled_depth) & (sampled_depth > 0)
    valid &= np.isfinite(voxel_depth) & (voxel_depth > 0)

    tolerance = np.clip(
        relative_tolerance * sampled_depth,
        min_tolerance,
        max_tolerance,
    )
    sigma = np.maximum(tolerance * 0.5, 1e-6)
    delta = voxel_depth - sampled_depth
    weights = np.exp(-0.5 * np.square(delta / sigma))
    weights[~valid | (np.abs(delta) > tolerance)] = 0.0
    return np.ascontiguousarray(weights, dtype=np.float32)


def build_semantic_condition_2d(
    probabilities,
    model_to_semkitti,
    output_channels=DEFAULT_OUTPUT_CHANNELS,
    semantic_channels=DEFAULT_SEMANTIC_CHANNELS,
):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 3:
        raise ValueError("Mask2Former probabilities must be [classes,H,W]")
    if output_channels < semantic_channels + 4:
        raise ValueError("output_channels must fit semantic, confidence, entropy, boundary, and fov channels")

    condition = np.zeros((output_channels, probabilities.shape[1], probabilities.shape[2]), dtype=np.float32)
    for source_index in range(probabilities.shape[0]):
        target_index = int(model_to_semkitti.get(source_index, 0))
        if target_index < 0 or target_index >= semantic_channels:
            target_index = 0
        condition[target_index] += probabilities[source_index]

    semantic_sum = condition[:semantic_channels].sum(axis=0, keepdims=True)
    condition[:semantic_channels] = condition[:semantic_channels] / np.clip(semantic_sum, 1e-6, None)
    condition[20] = condition[:semantic_channels].max(axis=0)
    condition[21] = normalized_entropy(condition[:semantic_channels])
    condition[22] = semantic_boundary(condition[:semantic_channels])
    condition[23] = (semantic_sum[0] > 0).astype(np.float32)
    return condition
