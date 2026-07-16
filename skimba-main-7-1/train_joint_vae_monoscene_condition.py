import argparse
import sys
from pathlib import Path

import numpy as np

from scripts.data.config_paths import (
    load_training_config,
    resolve_kitti_root,
    resolve_vae_root,
)


SPATIAL_SHAPE = (256, 256, 32)
LATENT_PROJECT_SCALE = 4
LATENT_SHAPE = (64, 64, 8)
VOX_ORIGIN = np.array([0.0, -25.6, -2.0], dtype=np.float32)
SCENE_SIZE_METERS = (51.2, 51.2, 6.4)
VOXEL_SIZE = 0.2
IMAGE_SIZE = (370, 1220)
IMAGE_CONDITION_ROOT_NAME = "Image_transform_Voxel_Condition_Features"
KNOWN_VAE_ROOT_NAMES = (
    "VAE_Encoder_Features_One_To_One",
    "VAE_Encoder_Features_Semantic20",
)


def parse_csv_items(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_ext(value):
    return value if value.startswith(".") else f".{value}"


def add_monoscene_to_path(monoscene_root):
    root_path = Path(monoscene_root).resolve()
    if root_path.name == "monoscene" and (root_path.parent / "setup.py").exists():
        root_path = root_path.parent
    root = str(root_path)
    if root not in sys.path:
        sys.path.insert(0, root)


def read_calib(calib_path):
    calib_all = {}
    with open(calib_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "":
                break
            key, value = line.split(":", 1)
            calib_all[key] = np.array([float(x) for x in value.split()], dtype=np.float32)

    calib_out = {}
    calib_out["P2"] = calib_all["P2"].reshape(3, 4)
    calib_out["Tr"] = np.eye(4, dtype=np.float32)
    calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
    return calib_out


def load_checkpoint(path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def image_normalization_tensors():
    import torch

    image_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return image_mean, image_std


def load_image(path):
    import torch
    from PIL import Image

    image_mean, image_std = image_normalization_tensors()
    img = Image.open(path).convert("RGB")
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = img[: IMAGE_SIZE[0], : IMAGE_SIZE[1], :]
    tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    return (tensor - image_mean) / image_std


def infer_image_condition_root(vae_root):
    vae_root = Path(vae_root)
    text = str(vae_root)
    for root_name in KNOWN_VAE_ROOT_NAMES:
        if root_name in text:
            return Path(text.replace(root_name, IMAGE_CONDITION_ROOT_NAME))
    return vae_root.parent / IMAGE_CONDITION_ROOT_NAME


def sequence_frame_from_path(path):
    path = Path(path)
    parts = path.parts
    if "sequences" in parts:
        index = parts.index("sequences")
        if index + 1 < len(parts):
            return parts[index + 1].zfill(2), path.stem.zfill(6)
    return "", path.stem.zfill(6)


def image_condition_path_for(vae_path, output_root, image_folder="voxels", ext=".bin"):
    sequence, frame = sequence_frame_from_path(vae_path)
    if not sequence:
        raise ValueError(f"Could not infer sequence id from VAE path: {vae_path}")
    return (
        Path(output_root)
        / "sequences"
        / sequence
        / image_folder
        / f"{frame}{normalize_ext(ext)}"
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
        sequence = str(sequence).zfill(2)
        latent_dir = vae_root / "sequences" / sequence / latent_folder
        if frames:
            for frame in frames:
                frame_id = Path(str(frame)).stem.zfill(6)
                files.append(latent_dir / f"{frame_id}{ext}")
        else:
            files.extend(sorted(latent_dir.glob(f"*{ext}")))

    if num_samples > 0:
        files = files[:num_samples]
    return files


def resolve_sequence_dir(kitti_root, sequence):
    root = Path(kitti_root)
    candidates = [
        root / "dataset" / "sequences" / sequence,
        root / "sequences" / sequence,
        root / sequence,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def sample_from_vae_path(vae_path, kitti_root):
    sequence, frame = sequence_frame_from_path(vae_path)
    if not sequence:
        raise ValueError(f"Could not infer sequence id from VAE path: {vae_path}")

    seq_dir = resolve_sequence_dir(kitti_root, sequence)
    sample = {
        "sequence": sequence,
        "frame": frame,
        "image": seq_dir / "image_2" / f"{frame}.png",
        "calib": seq_dir / "calib.txt",
    }
    missing = [key for key in ("image", "calib") if not sample[key].exists()]
    if missing:
        missing_paths = ", ".join(f"{key}={sample[key]}" for key in missing)
        raise FileNotFoundError(f"Missing KITTI files for {sequence}/{frame}: {missing_paths}")
    return sample


def build_image_backbone(args):
    add_monoscene_to_path(args.monoscene_root)
    from monoscene.models.unet2d import UNet2D

    model = UNet2D.build(out_feature=args.project_feature_dim, use_decoder=True)
    checkpoint_path = Path(args.monoscene_checkpoint)
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        state = checkpoint.get("state_dict", checkpoint)
        rgb_state = {
            key.replace("net_rgb.", "", 1): value
            for key, value in state.items()
            if key.startswith("net_rgb.")
        }
        missing, unexpected = model.load_state_dict(rgb_state, strict=False)
        print(
            f"Loaded MonoScene net_rgb weights: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
    else:
        print(f"Warning: MonoScene checkpoint not found, using initialized net_rgb: {checkpoint_path}")
    return model


def project_features(flosp, rgb_features, projected_pix, fov_mask, project_res):
    projected_sum = None
    for scale_2d in project_res:
        key = f"1_{scale_2d}"
        if key not in rgb_features:
            raise KeyError(
                f"Backbone did not return {key}. Available keys: {list(rgb_features.keys())}"
            )
        projected = flosp(
            rgb_features[key][0],
            projected_pix // scale_2d,
            fov_mask,
        )
        projected_sum = projected if projected_sum is None else projected_sum + projected
    return projected_sum


def export_image_conditions(args):
    import torch
    from tqdm import tqdm

    configs = load_training_config(args.config_path)
    vae_root = Path(args.vae_root) if args.vae_root else resolve_vae_root(configs)
    kitti_root = Path(args.kitti_root) if args.kitti_root else resolve_kitti_root(configs)

    add_monoscene_to_path(args.monoscene_root)
    from monoscene.data.utils.helpers import vox2pix
    from monoscene.models.flosp import FLoSP

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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

    output_root = Path(args.output_root) if args.output_root else infer_image_condition_root(vae_root)
    project_res = [int(item) for item in parse_csv_items(args.project_res)]
    backbone = build_image_backbone(args).to(device)
    backbone.eval()
    flosp = FLoSP(
        scene_size=SPATIAL_SHAPE,
        dataset="kitti",
        project_scale=args.latent_project_scale,
    ).to(device)

    projection_cache = {}
    written = 0
    skipped = 0

    with torch.no_grad():
        for vae_path in tqdm(vae_files, desc="Export MonoScene image_condition"):
            if not Path(vae_path).exists():
                raise FileNotFoundError(f"Missing VAE latent file: {vae_path}")
            sample = sample_from_vae_path(vae_path, kitti_root)
            sequence = sample["sequence"]
            out_path = image_condition_path_for(
                vae_path,
                output_root,
                image_folder=args.image_folder,
                ext=args.ext,
            )
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            if sequence not in projection_cache:
                calib = read_calib(sample["calib"])
                projected_pix, fov_mask, _ = vox2pix(
                    calib["Tr"],
                    calib["P2"][:3, :3],
                    VOX_ORIGIN,
                    VOXEL_SIZE * args.latent_project_scale,
                    IMAGE_SIZE[1],
                    IMAGE_SIZE[0],
                    SCENE_SIZE_METERS,
                )
                projection_cache[sequence] = (
                    torch.from_numpy(projected_pix.astype(np.int64)).to(device),
                    torch.from_numpy(fov_mask.astype(np.bool_)).to(device),
                )

            projected_pix, fov_mask = projection_cache[sequence]
            image = load_image(sample["image"]).unsqueeze(0).to(device)
            rgb_features = backbone(image)
            projected = project_features(flosp, rgb_features, projected_pix, fov_mask, project_res)
            expected_shape = (args.project_feature_dim, *LATENT_SHAPE)
            if tuple(projected.shape) != expected_shape:
                raise RuntimeError(
                    f"Unexpected image condition shape {tuple(projected.shape)}, expected {expected_shape}"
                )

            if not args.dry_run:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                output = projected.detach().cpu().numpy().astype(np.float32)
                np.ascontiguousarray(output).tofile(out_path)
            written += 1

    print(f"image_condition output_root: {output_root}")
    print(f"written: {written}")
    print(f"skipped_existing: {skipped}")


def build_argparser():
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Export MonoScene FLoSP image_condition features for the current skimba training path. "
            "This legacy-named script writes raw 64-channel condition .bin files; "
            "the current default diffusion config expects separate 8-channel condition files."
        )
    )
    parser.add_argument("--config_path", "--config-path", default="config/semantickitti_autoencoder.yaml")
    parser.add_argument("--vae_root", "--vae-root", default="")
    parser.add_argument("--kitti_root", "--kitti-root", default="")
    parser.add_argument("--output_root", "--output-root", default="")
    parser.add_argument("--monoscene_root", "--monoscene-root", default=str(project_root / "MonoScene-master"))
    parser.add_argument(
        "--monoscene_checkpoint",
        "--monoscene-checkpoint",
        default=str(project_root / "MonoScene-master" / "trained_models" / "monoscene_kitti.ckpt"),
    )
    parser.add_argument("--sequences", default="")
    parser.add_argument("--frames", default="")
    parser.add_argument("--num_samples", "--num-samples", type=int, default=0)
    parser.add_argument("--latent_folder", "--latent-folder", default="voxels")
    parser.add_argument("--image_folder", "--image-folder", default="voxels")
    parser.add_argument("--ext", default=".bin")
    parser.add_argument("--project_res", "--project-res", default="1,2,4,8")
    parser.add_argument("--project_feature_dim", "--project-feature-dim", type=int, default=64)
    parser.add_argument("--latent_project_scale", "--latent-project-scale", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    export_image_conditions(build_argparser().parse_args())
