import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.config_paths import load_training_config, resolve_kitti_root, resolve_partial_condition_root, resolve_vae_root
from scripts.data.mask2former_condition import (
    build_semantic_condition_2d,
    compute_depth_surface_weights,
    condition_path_for,
    load_model_to_semkitti_mapping,
    normalize_ext,
    parse_csv_items,
    resolve_depth_path,
)
from train_joint_vae_monoscene_condition import (
    IMAGE_SIZE,
    LATENT_PROJECT_SCALE,
    LATENT_SHAPE,
    SCENE_SIZE_METERS,
    SPATIAL_SHAPE,
    VOXEL_SIZE,
    VOX_ORIGIN,
    add_monoscene_to_path,
    read_calib,
    sample_from_vae_path,
)


def list_sequences(vae_root):
    sequences_root = Path(vae_root) / "sequences"
    if not sequences_root.exists():
        return []
    return sorted(path.name for path in sequences_root.iterdir() if path.is_dir())


def list_vae_files(vae_root, sequences=None, frames=None, latent_folder="voxels", ext=".bin", num_samples=0):
    vae_root = Path(vae_root)
    sequences = sequences or list_sequences(vae_root)
    frames = frames or []
    ext = normalize_ext(ext)
    files = []
    for sequence in sequences:
        latent_dir = vae_root / "sequences" / str(sequence).zfill(2) / latent_folder
        if frames:
            for frame in frames:
                files.append(latent_dir / f"{Path(str(frame)).stem.zfill(6)}{ext}")
        else:
            files.extend(sorted(latent_dir.glob(f"*{ext}")))
    return files[:num_samples] if num_samples > 0 else files


def semantic_probability_path(semantic_root, sequence, frame):
    return Path(semantic_root) / "sequences" / sequence / "image_2" / f"{frame}.npz"


def load_semantic_probabilities(path):
    payload = np.load(path)
    if "probabilities" not in payload:
        raise KeyError(f"{path} must contain a 'probabilities' array")
    probabilities = payload["probabilities"].astype(np.float32)
    if probabilities.ndim != 3:
        raise ValueError(f"{path} probabilities must be [classes,H,W]")
    return probabilities


def resize_condition_2d(condition_2d, target_hw=IMAGE_SIZE):
    import torch

    tensor = torch.from_numpy(condition_2d).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor,
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )[0]
    return resized.numpy().astype(np.float32)


def project_semantic_condition(flosp, condition_2d, projected_pix, fov_mask):
    import torch

    feature = torch.from_numpy(condition_2d).to(projected_pix.device)
    return flosp(feature, projected_pix, fov_mask)


def apply_depth_surface_weights(
    projected,
    projected_pix,
    fov_mask,
    voxel_depth,
    depth_map,
    spatial_shape=LATENT_SHAPE,
):
    import torch

    surface_weights = compute_depth_surface_weights(
        projected_pix,
        fov_mask,
        voxel_depth,
        depth_map,
    ).reshape(spatial_shape)
    weights = torch.from_numpy(surface_weights).to(
        device=projected.device,
        dtype=projected.dtype,
    )
    return projected * weights.unsqueeze(0), surface_weights


def summarize_projected(projected, fov_mask):
    values = projected.detach().cpu().numpy()
    fov_values = fov_mask.detach().cpu().numpy() if hasattr(fov_mask, "detach") else np.asarray(fov_mask)
    finite = np.isfinite(values)
    return {
        "finite": bool(finite.all()),
        "num_values": int(values.size),
        "min": float(values[finite].min()) if finite.any() else None,
        "max": float(values[finite].max()) if finite.any() else None,
        "mean": float(values[finite].mean()) if finite.any() else None,
        "std": float(values[finite].std()) if finite.any() else None,
        "zero_ratio": float(np.count_nonzero(values == 0) / values.size),
        "fov_voxel_ratio": float(fov_values.astype(np.float32).mean()),
    }


def export_mask2former_partial_conditions(args):
    import torch
    from tqdm import tqdm

    configs = load_training_config(args.config_path)
    vae_root = Path(args.vae_root) if args.vae_root else resolve_vae_root(configs)
    kitti_root = Path(args.kitti_root) if args.kitti_root else resolve_kitti_root(configs)
    output_root = Path(args.output_root) if args.output_root else resolve_partial_condition_root(configs)
    mapping = load_model_to_semkitti_mapping(args.mapping_path)
    depth_root = Path(args.depth_root) if args.depth_root else None
    if args.projection_mode == "surface" and depth_root is None:
        raise ValueError("Surface projection requires --depth-root with MobileStereoNet depth maps")

    add_monoscene_to_path(args.monoscene_root)
    from monoscene.data.utils.helpers import vox2pix
    from monoscene.models.flosp import FLoSP

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    flosp = FLoSP(scene_size=SPATIAL_SHAPE, dataset="kitti", project_scale=args.latent_project_scale).to(device)
    vae_files = list_vae_files(
        vae_root,
        sequences=parse_csv_items(args.sequences),
        frames=parse_csv_items(args.frames),
        latent_folder=args.latent_folder,
        ext=args.ext,
        num_samples=args.num_samples,
    )
    if not vae_files:
        raise RuntimeError(f"No VAE latent files found under {vae_root}")

    projection_cache = {}
    rows = []
    written = 0
    skipped = 0

    for vae_path in tqdm(vae_files, desc="Export Mask2Former partial_condition"):
        sample = sample_from_vae_path(vae_path, kitti_root)
        sequence = sample["sequence"]
        frame = sample["frame"]
        out_path = condition_path_for(vae_path, output_root, condition_folder=args.condition_folder, ext=args.ext)
        semantic_path = semantic_probability_path(args.semantic_root, sequence, frame)
        row = {"sequence": sequence, "frame": frame, "path": str(out_path), "semantic_path": str(semantic_path)}

        if out_path.exists() and not args.overwrite:
            row["status"] = "skipped_exists"
            rows.append(row)
            skipped += 1
            continue
        if not semantic_path.exists():
            row["status"] = "missing_semantic_probability"
            rows.append(row)
            continue

        if sequence not in projection_cache:
            calib = read_calib(sample["calib"])
            projected_pix, fov_mask, voxel_depth = vox2pix(
                calib["Tr"],
                calib["P2"][:3, :3],
                VOX_ORIGIN,
                VOXEL_SIZE * args.latent_project_scale,
                IMAGE_SIZE[1],
                IMAGE_SIZE[0],
                SCENE_SIZE_METERS,
            )
            projection_cache[sequence] = (
                projected_pix.astype(np.int64),
                fov_mask.astype(np.bool_),
                voxel_depth.astype(np.float32),
            )

        probabilities = load_semantic_probabilities(semantic_path)
        condition_2d = build_semantic_condition_2d(probabilities, mapping)
        condition_2d = resize_condition_2d(condition_2d, target_hw=IMAGE_SIZE)
        projected_pix, fov_mask, voxel_depth = projection_cache[sequence]
        projected = project_semantic_condition(
            flosp,
            condition_2d,
            torch.from_numpy(projected_pix).to(device),
            torch.from_numpy(fov_mask).to(device),
        )

        if args.projection_mode == "surface":
            depth_path = resolve_depth_path(depth_root, sequence, frame)
            depth_map = np.load(depth_path).astype(np.float32)
            projected, surface_weights = apply_depth_surface_weights(
                projected,
                projected_pix,
                fov_mask,
                voxel_depth,
                depth_map,
            )
            row["depth_path"] = str(depth_path)
            row["surface_weight_mean"] = float(surface_weights.mean())
            row["surface_weight_nonzero_ratio"] = float(
                np.count_nonzero(surface_weights) / surface_weights.size
            )

        expected_shape = (64, *LATENT_SHAPE)
        if tuple(projected.shape) != expected_shape:
            raise RuntimeError(f"Unexpected projected shape {tuple(projected.shape)}, expected {expected_shape}")

        row.update(summarize_projected(projected, fov_mask))
        row["status"] = "dry_run" if args.dry_run else "written"
        if not args.dry_run:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.ascontiguousarray(projected.detach().cpu().numpy().astype(np.float32)).tofile(out_path)
        rows.append(row)
        written += 1

    print(f"Mask2Former partial_condition output_root: {output_root}")
    print(f"projection_mode: {args.projection_mode}")
    if depth_root is not None:
        print(f"depth_root: {depth_root}")
    print(f"written: {written}")
    print(f"skipped_existing: {skipped}")
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.out_csv and rows:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def build_argparser():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Project Mask2Former 2D semantic probabilities into skimba partial-condition voxels.")
    parser.add_argument("--config_path", "--config-path", default="config/semantickitti_autoencoder.yaml")
    parser.add_argument("--vae_root", "--vae-root", default="")
    parser.add_argument("--kitti_root", "--kitti-root", default="")
    parser.add_argument("--semantic_root", "--semantic-root", required=True)
    parser.add_argument("--mapping_path", "--mapping-path", default="config/mask2former_to_semantickitti.yaml")
    parser.add_argument("--output_root", "--output-root", default="")
    parser.add_argument("--monoscene_root", "--monoscene-root", default=str(project_root / "MonoScene-master"))
    parser.add_argument("--sequences", default="")
    parser.add_argument("--frames", default="")
    parser.add_argument("--num_samples", "--num-samples", type=int, default=0)
    parser.add_argument("--latent_folder", "--latent-folder", default="voxels")
    parser.add_argument("--condition_folder", "--condition-folder", default="voxels")
    parser.add_argument("--ext", default=".bin")
    parser.add_argument("--latent_project_scale", "--latent-project-scale", type=int, default=LATENT_PROJECT_SCALE)
    parser.add_argument("--projection_mode", "--projection-mode", choices=("ray", "surface"), default="ray")
    parser.add_argument("--depth_root", "--depth-root", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_json", "--out-json", default="")
    parser.add_argument("--out_csv", "--out-csv", default="")
    return parser


if __name__ == "__main__":
    export_mask2former_partial_conditions(build_argparser().parse_args())
