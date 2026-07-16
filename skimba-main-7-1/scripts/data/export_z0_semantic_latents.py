import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.config_paths import (
    load_training_config,
    resolve_gt_root,
    resolve_vae_root,
)

SPATIAL_SHAPE = (256, 256, 32)
DEFAULT_TRAIN_SEQUENCES = "00,01,02,03,04,05,06,07,09,10"
IGNORE_LABEL = 255


def parse_csv_items(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value):
    return [int(item) for item in parse_csv_items(value)]


def parse_shape(value):
    parts = parse_int_list(value)
    if len(parts) != 4:
        raise ValueError(f"Expected C,W,L,H latent shape, got: {value}")
    return tuple(parts)


def output_latent_path(output_root, sequence, frame):
    return (
        Path(output_root)
        / "sequences"
        / str(sequence).zfill(2)
        / "voxels"
        / f"{str(frame).zfill(6)}.bin"
    )


def label_root_candidates(dataset_root):
    root = Path(dataset_root)
    return [
        root,
        root / "sequences",
        root / "dataset" / "sequences",
        root / "data_odometry_voxels_all_with_one" / "sequences",
        root / "data_odometry_voxel_all_with_one" / "sequences",
        root / "data_odometry_voxels_all" / "dataset" / "sequences",
        root / "data_odometry_voxel_all" / "dataset" / "sequences",
        root / "data_odometry_voxels_all" / "sequences",
        root / "data_odometry_voxel_all" / "sequences",
    ]


def resolve_voxel_dir(dataset_root, sequence, label_root=None):
    sequence = str(sequence).zfill(2)
    roots = [Path(label_root)] if label_root else label_root_candidates(dataset_root)
    checked = []
    for root in roots:
        candidates = [
            root if root.name == "voxels" else None,
            root / sequence / "voxels",
            root / "sequences" / sequence / "voxels",
            root / "dataset" / "sequences" / sequence / "voxels",
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            checked.append(str(candidate))
            if candidate.exists() and any(candidate.glob("*.label")):
                return candidate
    raise FileNotFoundError(
        "Could not find SemanticKITTI voxel labels for sequence "
        f"{sequence}. Checked:\n  " + "\n  ".join(checked)
    )


def select_label_files(voxel_dir, frames=None, num_samples=0):
    voxel_dir = Path(voxel_dir)
    frames = frames or []
    if frames:
        files = []
        for frame in frames:
            frame_id = Path(str(frame)).stem.zfill(6)
            path = voxel_dir / f"{frame_id}.label"
            if not path.exists():
                raise FileNotFoundError(f"Missing label file: {path}")
            files.append(path)
        return files
    files = sorted(voxel_dir.glob("*.label"))
    if num_samples > 0:
        files = files[:num_samples]
    return files


def load_checkpoint(path, torch_module):
    try:
        return torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("autoencoder", "state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def strip_prefixes(key, prefixes):
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def normalize_autoencoder_key(key):
    return strip_prefixes(
        key,
        (
            "module.model_part.autoencoder.",
            "model_part.autoencoder.",
            "module.model_part.",
            "model_part.",
            "module.autoencoder.",
            "autoencoder.",
            "module.",
            "model.",
        ),
    )


def normalize_embedding_key(key):
    return strip_prefixes(key, ("module.", "model."))


def adapt_autoencoder_state_dict(checkpoint, target_state):
    source_state = extract_state_dict(checkpoint)
    adapted = {}
    mismatched = {}
    skipped = []
    for key, value in source_state.items():
        if not isinstance(key, str):
            continue
        target_key = normalize_autoencoder_key(key)
        if target_key not in target_state:
            skipped.append(target_key)
            continue
        source_shape = tuple(value.shape) if hasattr(value, "shape") else None
        target_shape = tuple(target_state[target_key].shape)
        if source_shape != target_shape:
            mismatched[target_key] = (source_shape, target_shape)
            continue
        adapted[target_key] = value
    missing = [key for key in target_state.keys() if key not in adapted]
    return adapted, missing, mismatched, skipped


def load_autoencoder_from_checkpoint(args, checkpoint, device):
    from stable_diffusion.models.autoencoder import AutoEncoderKL

    model = AutoEncoderKL(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        latent_channels=args.latent_channels,
        autoencoder_num_res_blocks=args.autoencoder_num_res_blocks,
        autoencoder_channels_list=parse_int_list(args.autoencoder_channels_list),
        groups=args.auto_groups,
        num_input_features=args.num_input_features,
        init_size=args.init_size,
        voxel_channel=args.voxel_channel,
        dropout_rate=args.dropout_rate,
        num_class=args.num_class,
        semantic_embed_dim=args.semantic_embed_dim,
    )
    target_state = model.state_dict()
    adapted, missing, mismatched, skipped = adapt_autoencoder_state_dict(checkpoint, target_state)
    if missing or mismatched:
        messages = []
        if missing:
            messages.append("missing autoencoder keys:\n  " + "\n  ".join(missing[:60]))
        if mismatched:
            lines = [
                f"{key}: checkpoint{source_shape} != model{target_shape}"
                for key, (source_shape, target_shape) in sorted(mismatched.items())
            ]
            messages.append("autoencoder shape mismatches:\n  " + "\n  ".join(lines[:60]))
        raise RuntimeError("Could not load Code_VAE_SSC autoencoder weights:\n" + "\n".join(messages))
    result = model.load_state_dict(adapted, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Adapted autoencoder state did not load cleanly: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    if skipped:
        print(f"Skipped {len(skipped)} non-autoencoder checkpoint keys.")
    model.to(device)
    model.eval()
    return model


def load_semantic_embedding_from_checkpoint(args, checkpoint, device):
    import torch
    from torch import nn

    source_state = extract_state_dict(checkpoint)
    embedding_weight = None
    for key, value in source_state.items():
        if isinstance(key, str) and normalize_embedding_key(key) == "semantic_embedding.weight":
            embedding_weight = value
            break
    if embedding_weight is None:
        raise RuntimeError(
            "Checkpoint does not contain semantic_embedding.weight. "
            "Use a full Code_VAE_SSC VAE checkpoint saved from the wrapper model, "
            "not an autoencoder-only subset."
        )

    expected_shape = (args.num_class, args.semantic_embed_dim)
    if tuple(embedding_weight.shape) != expected_shape:
        raise RuntimeError(
            f"semantic_embedding.weight shape {tuple(embedding_weight.shape)} != expected {expected_shape}."
        )
    embedding = nn.Embedding(args.num_class, args.semantic_embed_dim)
    with torch.no_grad():
        embedding.weight.copy_(embedding_weight.float())
    embedding.to(device)
    embedding.eval()
    return embedding


def load_learning_map(label_mapping_path):
    try:
        import yaml

        with Path(label_mapping_path).open("r", encoding="utf-8") as handle:
            return {int(key): int(value) for key, value in yaml.safe_load(handle)["learning_map"].items()}
    except ModuleNotFoundError:
        return load_learning_map_without_yaml(label_mapping_path)


def load_learning_map_without_yaml(label_mapping_path):
    mapping = {}
    in_learning_map = False
    for raw_line in Path(label_mapping_path).read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not raw_line.startswith((" ", "\t")):
            in_learning_map = line == "learning_map:"
            continue
        if not in_learning_map:
            continue
        stripped = line.split("#", 1)[0].strip()
        if ":" not in stripped:
            continue
        key_text, value_text = stripped.split(":", 1)
        key_text = key_text.strip().strip("'\"")
        value_text = value_text.strip().strip("'\"")
        if key_text.isdigit() and value_text.lstrip("-").isdigit():
            mapping[int(key_text)] = int(value_text)
    if not mapping:
        raise ValueError(f"Could not parse learning_map from {label_mapping_path}.")
    return mapping


def build_learning_map_lut(label_mapping_path):
    mapping = load_learning_map(label_mapping_path)
    max_key = max(mapping.keys())
    lut = np.zeros((max_key + 100,), dtype=np.uint8)
    for raw_id, mapped_id in mapping.items():
        lut[raw_id] = mapped_id
    return lut


def read_dense_label(label_path, spatial_shape=SPATIAL_SHAPE):
    label_path = Path(label_path)
    expected = int(np.prod(spatial_shape))
    file_size = label_path.stat().st_size
    if file_size == expected:
        dtype = np.uint8
    elif file_size == expected * 2:
        dtype = np.uint16
    elif file_size == expected * 4:
        dtype = np.uint32
    else:
        raise ValueError(
            f"{label_path} has {file_size} bytes, expected {expected}, "
            f"{expected * 2}, or {expected * 4}."
        )
    labels = np.fromfile(label_path, dtype=dtype)
    if labels.size != expected:
        raise ValueError(f"{label_path} has {labels.size} voxels, expected {expected}.")
    return labels.reshape(spatial_shape)


def read_invalid_mask(invalid_path, spatial_shape=SPATIAL_SHAPE):
    invalid_path = Path(invalid_path)
    if not invalid_path.exists():
        return None
    expected = int(np.prod(spatial_shape))
    raw = np.fromfile(invalid_path, dtype=np.uint8)
    if raw.size == expected:
        return raw.reshape(spatial_shape).astype(np.uint8)
    unpacked = np.unpackbits(raw)
    if unpacked.size < expected:
        raise ValueError(f"{invalid_path} expands to {unpacked.size} voxels, expected {expected}.")
    return unpacked[:expected].reshape(spatial_shape).astype(np.uint8)


def semantic_target_from_label(label_path, learning_map_lut, ignore_label=IGNORE_LABEL):
    raw = read_dense_label(label_path)
    lower = (raw.astype(np.uint32) & 0xFFFF).reshape(-1)
    max_label = int(lower.max()) if lower.size else 0
    if max_label >= learning_map_lut.shape[0]:
        raise ValueError(f"{label_path} contains raw label {max_label}, outside learning map LUT.")
    target = learning_map_lut[lower].reshape(SPATIAL_SHAPE).astype(np.uint8)

    invalid = read_invalid_mask(Path(label_path).with_suffix(".invalid"))
    if invalid is not None:
        target = target.copy()
        target[invalid == 1] = ignore_label
    return target


def build_sparse_scene_from_target(target_np, semantic_embedding, args, device):
    import torch
    import spconv.pytorch as spconv

    valid_mask = target_np != args.ignore_label
    coords_np = np.argwhere(valid_mask).astype(np.int32)
    if coords_np.size == 0:
        batch_coords_np = np.zeros((0, 4), dtype=np.int32)
        features = torch.zeros((0, args.semantic_embed_dim), dtype=torch.float32, device=device)
    else:
        labels_np = target_np[valid_mask].astype(np.int64)
        labels = torch.from_numpy(labels_np).long().to(device).clamp(min=0, max=args.num_class - 1)
        with torch.no_grad():
            features = semantic_embedding(labels).float()
        batch_ids = np.zeros((coords_np.shape[0], 1), dtype=np.int32)
        batch_coords_np = np.concatenate([batch_ids, coords_np], axis=1)

    coords = torch.from_numpy(batch_coords_np).int().to(device)
    return spconv.SparseConvTensor(
        features=features,
        indices=coords,
        spatial_shape=list(SPATIAL_SHAPE),
        batch_size=1,
    )


def encode_label_file(autoencoder, semantic_embedding, label_path, learning_map_lut, args, device):
    import torch

    target = semantic_target_from_label(label_path, learning_map_lut, args.ignore_label)
    sparse_scene = build_sparse_scene_from_target(target, semantic_embedding, args, device)
    with torch.no_grad():
        _, posterior = autoencoder.encode(sparse_scene)
        dist = posterior.latent_dist
        latent = dist.sample() if args.mode == "sample" else dist.mean
    latent = latent.squeeze(0).detach().cpu().contiguous()
    expected_shape = parse_shape(args.latent_shape)
    if tuple(latent.shape) != expected_shape:
        raise ValueError(f"Encoded latent shape {tuple(latent.shape)} != expected {expected_shape}.")
    if not torch.isfinite(latent).all():
        raise ValueError(f"Encoded latent for {label_path} contains non-finite values.")
    return latent


def write_skimba_latent_bin(latent, output_path, overwrite=False):
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latent.numpy().astype(np.float32).tofile(output_path)
    return True


def frame_info_from_label_path(label_path):
    label_path = Path(label_path)
    return label_path.parent.parent.name, label_path.stem


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Encode GT SemanticKITTI voxel labels into clean z0 semantic VAE "
            "latents as skimba-compatible raw float32 .bin files."
        )
    )
    parser.add_argument("--config_path", "--config-path", default="config/semantickitti_autoencoder.yaml")
    parser.add_argument("--vae_checkpoint", "--vae-checkpoint", default="")
    parser.add_argument("--dataset_root", "--dataset-root", default="")
    parser.add_argument("--output_root", "--output-root", default="")
    parser.add_argument("--label_mapping", "--label-mapping", default="")
    parser.add_argument("--label_root", "--label-root", default="")
    parser.add_argument("--sequences", default=DEFAULT_TRAIN_SEQUENCES)
    parser.add_argument("--frames", default="")
    parser.add_argument("--num_samples", "--num-samples", type=int, default=0)
    parser.add_argument("--mode", choices=["mean", "sample"], default="mean")
    parser.add_argument("--latent_shape", "--latent-shape", default="8,64,64,8")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_progress", "--no-progress", action="store_true")
    parser.add_argument("--num_class", "--num-class", type=int, default=20)
    parser.add_argument("--semantic_embed_dim", "--semantic-embed-dim", type=int, default=8)
    parser.add_argument("--ignore_label", "--ignore-label", type=int, default=IGNORE_LABEL)
    parser.add_argument("--in_channels", "--in-channels", type=int, default=8)
    parser.add_argument("--out_channels", "--out-channels", type=int, default=8)
    parser.add_argument("--latent_channels", "--latent-channels", type=int, default=64)
    parser.add_argument("--autoencoder_num_res_blocks", "--autoencoder-num-res-blocks", type=int, default=1)
    parser.add_argument("--autoencoder_channels_list", "--autoencoder-channels-list", default="16,32,64,128")
    parser.add_argument("--auto_groups", "--auto-groups", type=int, default=4)
    parser.add_argument("--num_input_features", "--num-input-features", type=int, default=3)
    parser.add_argument("--init_size", "--init-size", type=int, default=8)
    parser.add_argument("--voxel_channel", "--voxel-channel", type=int, default=1)
    parser.add_argument("--dropout_rate", "--dropout-rate", type=float, default=0.2)
    return parser


def run(args):
    configs = load_training_config(args.config_path)
    args.vae_checkpoint = args.vae_checkpoint or configs.get("train_params", {}).get("vae_checkpoint", "")
    args.dataset_root = args.dataset_root or str(configs["dataset_params"].get("data_root", ""))
    args.output_root = args.output_root or str(resolve_vae_root(configs))
    args.label_mapping = args.label_mapping or configs["dataset_params"].get("label_mapping", "")
    args.label_root = args.label_root or str(resolve_gt_root(configs))
    if not args.vae_checkpoint:
        raise ValueError("Set train_params.vae_checkpoint in the config or pass --vae_checkpoint.")
    if not args.dataset_root:
        raise ValueError("Set dataset_params.data_root in the config or pass --dataset_root.")
    if not args.output_root:
        raise ValueError("Set dataset_params.vae_feature_root/data_root in the config or pass --output_root.")

    import torch
    from tqdm import tqdm

    requested_device = args.device
    if requested_device != "cpu" and not torch.cuda.is_available():
        print(f"Warning: requested device {requested_device}, but CUDA is unavailable; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    checkpoint = load_checkpoint(args.vae_checkpoint, torch)
    autoencoder = load_autoencoder_from_checkpoint(args, checkpoint, device)
    semantic_embedding = load_semantic_embedding_from_checkpoint(args, checkpoint, device)
    learning_map_lut = build_learning_map_lut(args.label_mapping)

    label_files = []
    for sequence in parse_csv_items(args.sequences):
        voxel_dir = resolve_voxel_dir(args.dataset_root, sequence, args.label_root or None)
        label_files.extend(select_label_files(voxel_dir, parse_csv_items(args.frames), args.num_samples))
    if not label_files:
        raise RuntimeError("No SemanticKITTI label files selected for export.")

    print(f"Loading Code_VAE_SSC semantic VAE checkpoint: {args.vae_checkpoint}")
    print(f"Writing skimba latent .bin files to: {Path(args.output_root) / 'sequences'}")
    print(f"Frames: {len(label_files)} | device={device} | mode={args.mode}")

    written = 0
    skipped = 0
    iterator = tqdm(label_files, desc="Export skimba semantic VAE latents", disable=args.no_progress)
    for label_path in iterator:
        sequence, frame = frame_info_from_label_path(label_path)
        output_path = output_latent_path(args.output_root, sequence, frame)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            iterator.set_postfix({"written": written, "skipped": skipped, "last": output_path.name})
            continue
        latent = encode_label_file(autoencoder, semantic_embedding, label_path, learning_map_lut, args, device)
        if write_skimba_latent_bin(latent, output_path, overwrite=True):
            written += 1
        iterator.set_postfix({"written": written, "skipped": skipped, "last": output_path.name})

    print(f"Export complete: written={written}, skipped={skipped}, output_root={args.output_root}")


def main():
    run(build_argparser().parse_args())


if __name__ == "__main__":
    main()
