import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
import yaml


BEST_CHECKPOINT_PATTERN = re.compile(
    r"^best_(?P<epoch>\d+)_(?P<miou>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\.pth$"
)


def resolve_best_checkpoint(explicit_checkpoint, model_save_path):
    if explicit_checkpoint:
        checkpoint = Path(explicit_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "Explicit checkpoint does not exist: {}".format(checkpoint)
            )
        return checkpoint

    checkpoint_dir = Path(model_save_path)
    candidates = []
    for checkpoint in checkpoint_dir.glob("best_*.pth"):
        match = BEST_CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
        if match is None:
            continue
        candidates.append(
            (
                float(match.group("miou")),
                int(match.group("epoch")),
                checkpoint,
            )
        )

    if not candidates:
        raise FileNotFoundError(
            "No valid best_<epoch>_<mIoU>.pth checkpoint found in {}".format(
                checkpoint_dir
            )
        )

    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def prepare_full_validation_loader_config(config):
    prepared = dict(config)
    prepared.pop("frame_divisor", None)
    prepared["imageset"] = "val"
    prepared["batch_size"] = 1
    prepared["shuffle"] = False
    return prepared


def require_sequence_08_validation_split(label_mapping_path):
    with open(label_mapping_path, "r", encoding="utf-8") as stream:
        mapping = yaml.safe_load(stream)

    try:
        valid_split = mapping["split"]["valid"]
        normalized = ["{:02d}".format(int(sequence)) for sequence in valid_split]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Label mapping must define split.valid as exactly sequence 08"
        ) from exc

    if normalized != ["08"]:
        raise ValueError(
            "Label mapping split.valid must be exactly sequence 08, got {}".format(
                valid_split
            )
        )


def seed_random_generators(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics_from_confusion(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("Confusion matrix must be square")
    if confusion.shape[0] < 2:
        raise ValueError("Confusion matrix must contain empty and semantic classes")
    if np.any(confusion < 0):
        raise ValueError("Confusion matrix counts must be non-negative")

    true_positive = np.diag(confusion).astype(np.float64)
    false_positive = confusion.sum(axis=1) - true_positive
    false_negative = confusion.sum(axis=0) - true_positive
    union = true_positive + false_positive + false_negative
    class_iou = np.divide(
        true_positive,
        union,
        out=np.zeros_like(true_positive),
        where=union > 0,
    )

    occupancy_union = int(confusion.sum() - confusion[0, 0])
    if occupancy_union <= 0:
        raise ValueError("Completion IoU is undefined because occupancy union is zero")
    occupancy_intersection = int(confusion[1:, 1:].sum())

    return {
        "class_iou_percent": (class_iou * 100.0).tolist(),
        "semantic_miou_percent": float(class_iou[1:].mean() * 100.0),
        "completion_iou_percent": float(
            occupancy_intersection / occupancy_union * 100.0
        ),
    }


def _require_finite_result(result):
    scalar_fields = (
        "semantic_miou_percent",
        "completion_iou_percent",
        "mean_epsilon_mse",
        "mean_cross_entropy",
    )
    for field in scalar_fields:
        value = result.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Validation result contains non-finite {}".format(field))

    class_iou = result.get("class_iou_percent")
    if not isinstance(class_iou, list) or not class_iou:
        raise ValueError("Validation result must contain class_iou_percent")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in class_iou):
        raise ValueError("Validation result contains non-finite class_iou_percent")


def write_validation_reports(result, output_dir):
    _require_finite_result(result)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "metrics.json"
    text_path = output_dir / "report.txt"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    class_lines = [
        "Class {:02d} IoU: {:.3f}%".format(index, value)
        for index, value in enumerate(result["class_iou_percent"])
    ]
    report_lines = [
        "Full Sequence 08 Diffusion Validation",
        "Checkpoint: {}".format(result["checkpoint"]),
        "Config: {}".format(result["config_path"]),
        "Seed: {}".format(result["seed"]),
        "Frames: {}".format(result["frame_count"]),
        "Semantic mIoU: {:.3f}%".format(result["semantic_miou_percent"]),
        "Completion IoU: {:.3f}%".format(result["completion_iou_percent"]),
        "Mean epsilon MSE: {:.8f}".format(result["mean_epsilon_mse"]),
        "Mean cross entropy: {:.8f}".format(result["mean_cross_entropy"]),
    ]
    text_path.write_text(
        "\n".join(report_lines + [""] + class_lines) + "\n",
        encoding="utf-8",
    )
    return json_path, text_path
