import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.config_paths import load_training_config, resolve_kitti_root
from scripts.data.mask2former_condition import normalize_ext, parse_csv_items, sequence_frame_from_path


DEFAULT_MASK2FORMER_ROOT = PROJECT_ROOT.parent / "Mask2Former-main"
DEFAULT_CITYSCAPES_CONFIG = "configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml"
DEFAULT_CITYSCAPES_CHECKPOINT = "weights/cityscapes_semantic_R50.pkl"


def resolve_under_root(path, root):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(root) / path


def resolve_sequence_dir(kitti_root, sequence):
    root = Path(kitti_root)
    sequence = str(sequence).zfill(2)
    candidates = [
        root / "dataset" / "sequences" / sequence,
        root / "sequences" / sequence,
        root / sequence,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def list_sequences(kitti_root):
    root = Path(kitti_root)
    candidates = [
        root / "dataset" / "sequences",
        root / "sequences",
    ]
    for sequences_root in candidates:
        if sequences_root.exists():
            return sorted(path.name for path in sequences_root.iterdir() if path.is_dir())
    return []


def list_kitti_images(kitti_root, sequences=None, frames=None, image_folder="image_2", ext=".png", num_samples=0):
    sequences = sequences or list_sequences(kitti_root)
    frames = frames or []
    ext = normalize_ext(ext)
    files = []
    for sequence in sequences:
        image_dir = resolve_sequence_dir(kitti_root, sequence) / image_folder
        if frames:
            for frame in frames:
                files.append(image_dir / f"{Path(str(frame)).stem.zfill(6)}{ext}")
        else:
            files.extend(sorted(image_dir.glob(f"*{ext}")))
    return files[:num_samples] if num_samples > 0 else files


def semantic_probability_output_path(image_path, output_root, image_folder="image_2", ext=".npz"):
    sequence, frame = sequence_frame_from_path(image_path)
    if not sequence:
        raise ValueError(f"Could not infer sequence id from image path: {image_path}")
    return Path(output_root) / "sequences" / sequence / image_folder / f"{frame}{normalize_ext(ext)}"


def semantic_scores_to_probabilities(scores, eps=1e-8):
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 3:
        raise ValueError("Mask2Former semantic scores must be [classes,H,W]")
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.maximum(scores, 0.0)
    sums = scores.sum(axis=0, keepdims=True)
    probabilities = scores / np.clip(sums, eps, None)
    zero_mask = sums[0] <= eps
    if np.any(zero_mask):
        probabilities[:, zero_mask] = 1.0 / scores.shape[0]
    return probabilities.astype(np.float32)


def add_mask2former_to_path(mask2former_root):
    root = str(Path(mask2former_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def build_predictor(mask2former_root, config_file, checkpoint, device):
    add_mask2former_to_path(mask2former_root)
    from detectron2.config import get_cfg
    from detectron2.engine.defaults import DefaultPredictor
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(config_file))
    cfg.defrost()
    cfg.MODEL.WEIGHTS = str(checkpoint)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.freeze()
    return DefaultPredictor(cfg)


def class_names_from_predictor(predictor):
    metadata = getattr(predictor, "metadata", None)
    if metadata is None:
        return []
    for attribute in ("stuff_classes", "thing_classes"):
        if hasattr(metadata, attribute):
            return list(getattr(metadata, attribute))
    return []


def write_json(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def export_mask2former_semantic_probabilities(args):
    from tqdm import tqdm

    configs = load_training_config(args.config_path)
    kitti_root = Path(args.kitti_root) if args.kitti_root else resolve_kitti_root(configs)
    output_root = Path(args.output_root)
    mask2former_root = Path(args.mask2former_root)
    config_file = resolve_under_root(args.config_file, mask2former_root)
    checkpoint = resolve_under_root(args.checkpoint, mask2former_root)

    image_files = list_kitti_images(
        kitti_root,
        sequences=parse_csv_items(args.sequences),
        frames=parse_csv_items(args.frames),
        image_folder=args.image_folder,
        ext=args.image_ext,
        num_samples=args.num_samples,
    )
    if not image_files:
        raise RuntimeError(f"No KITTI images found under {kitti_root}")

    predictor = None
    read_image = None
    class_names = []
    if not args.dry_run:
        if not config_file.exists():
            raise FileNotFoundError(f"Missing Mask2Former config: {config_file}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing Mask2Former checkpoint: {checkpoint}")
        predictor = build_predictor(mask2former_root, config_file, checkpoint, args.device)
        class_names = class_names_from_predictor(predictor)
        from detectron2.data.detection_utils import read_image

    rows = []
    written = 0
    skipped = 0
    for image_path in tqdm(image_files, desc="Export Mask2Former semantic probabilities"):
        out_path = semantic_probability_output_path(
            image_path,
            output_root,
            image_folder=args.output_image_folder,
            ext=args.output_ext,
        )
        row = {
            "sequence": sequence_frame_from_path(image_path)[0],
            "frame": sequence_frame_from_path(image_path)[1],
            "image_path": str(image_path),
            "path": str(out_path),
        }
        if out_path.exists() and not args.overwrite:
            row["status"] = "skipped_exists"
            rows.append(row)
            skipped += 1
            continue
        if not Path(image_path).exists():
            row["status"] = "missing_image"
            rows.append(row)
            continue
        if args.dry_run:
            row["status"] = "dry_run"
            rows.append(row)
            written += 1
            continue

        image = read_image(str(image_path), format="BGR")
        predictions = predictor(image)
        if "sem_seg" not in predictions:
            raise KeyError(f"Mask2Former did not return sem_seg for {image_path}")
        probabilities = semantic_scores_to_probabilities(predictions["sem_seg"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            probabilities=probabilities.astype(np.float32),
            class_names=np.asarray(class_names, dtype=str),
            source_image=str(image_path),
            config_file=str(config_file),
            checkpoint=str(checkpoint),
        )
        row.update(
            {
                "status": "written",
                "classes": int(probabilities.shape[0]),
                "height": int(probabilities.shape[1]),
                "width": int(probabilities.shape[2]),
                "min": float(probabilities.min()),
                "max": float(probabilities.max()),
                "mean": float(probabilities.mean()),
            }
        )
        rows.append(row)
        written += 1

    print(f"Mask2Former semantic probabilities output_root: {output_root}")
    print(f"written: {written}")
    print(f"skipped_existing: {skipped}")
    if args.out_json:
        write_json(args.out_json, rows)
    if args.out_csv:
        write_csv(args.out_csv, rows)
    return rows


def build_argparser():
    parser = argparse.ArgumentParser(description="Export Mask2Former 2D semantic probabilities for KITTI images.")
    parser.add_argument("--config_path", "--config-path", default="config/semantickitti_autoencoder.yaml")
    parser.add_argument("--kitti_root", "--kitti-root", default="")
    parser.add_argument("--output_root", "--output-root", required=True)
    parser.add_argument("--mask2former_root", "--mask2former-root", default=str(DEFAULT_MASK2FORMER_ROOT))
    parser.add_argument("--config_file", "--config-file", default=DEFAULT_CITYSCAPES_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CITYSCAPES_CHECKPOINT)
    parser.add_argument("--sequences", default="")
    parser.add_argument("--frames", default="")
    parser.add_argument("--num_samples", "--num-samples", type=int, default=0)
    parser.add_argument("--image_folder", "--image-folder", default="image_2")
    parser.add_argument("--output_image_folder", "--output-image-folder", default="image_2")
    parser.add_argument("--image_ext", "--image-ext", default=".png")
    parser.add_argument("--output_ext", "--output-ext", default=".npz")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out_json", "--out-json", default="")
    parser.add_argument("--out_csv", "--out-csv", default="")
    return parser


if __name__ == "__main__":
    export_mask2former_semantic_probabilities(build_argparser().parse_args())
