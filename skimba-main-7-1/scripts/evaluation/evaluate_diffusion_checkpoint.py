#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.full_diffusion_validation import (  # noqa: E402
    metrics_from_confusion,
    prepare_full_validation_loader_config,
    require_sequence_08_validation_split,
    resolve_best_checkpoint,
    seed_random_generators,
    write_validation_reports,
)


DEFAULT_CONFIG_PATH = "config/semantickitti_autoencoder.yaml"
DEFAULT_SEED = 20260713


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one diffusion checkpoint on every SemanticKITTI "
            "validation frame in sequence 08."
        )
    )
    parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="Diffusion training configuration.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help=(
            "Checkpoint to evaluate. By default, select the greatest mIoU from "
            "best_<epoch>_<mIoU>.pth in train_params.model_save_path."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed controlling Python, NumPy, PyTorch, and CUDA random inputs.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device used for evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional report directory override.",
    )
    return parser


def _require_available_device(device):
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA evaluation was requested but torch.cuda.is_available() is False"
        )
    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device {} is unavailable; detected {} device(s)".format(
                    device.index,
                    torch.cuda.device_count(),
                )
            )


def _accumulate_confusion(confusion, prediction, target):
    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes differ: {} versus {}".format(
                prediction.shape,
                target.shape,
            )
        )
    if prediction.size == 0:
        return
    if prediction.min() < 0 or prediction.max() >= confusion.shape[0]:
        raise ValueError("Prediction contains a class outside the configured range")
    if target.min() < 0 or target.max() >= confusion.shape[1]:
        raise ValueError("Target contains a class outside the configured range")
    np.add.at(confusion, (prediction.reshape(-1), target.reshape(-1)), 1)


def evaluate(args):
    from builder import (
        data_builder_autoencoder,
        loss_builder,
        model_builder_3D_Voxel_unet_diffusion,
    )
    from config.config import load_config_data
    from dataloader.pc_dataset_gai import get_eval_mask, unpack
    from train_diffusion_network_2 import (
        batch_item,
        checkpoint_model_state,
        invalid_mask_path,
        load_autoencoder_state_from_checkpoint,
        load_model_state,
    )

    configs = load_config_data(args.config_path)
    dataset_config = configs["dataset_params"]
    model_config = configs["model_params"]
    train_hypers = configs["train_params"]
    val_loader_config = prepare_full_validation_loader_config(
        configs["val_data_loader"]
    )
    require_sequence_08_validation_split(dataset_config["label_mapping"])

    checkpoint_path = resolve_best_checkpoint(
        args.checkpoint,
        train_hypers["model_save_path"],
    )
    seed_random_generators(args.seed)
    device = torch.device(args.device)
    _require_available_device(device)

    model = model_builder_3D_Voxel_unet_diffusion.build(model_config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_model_state(model, checkpoint_model_state(checkpoint))

    vae_checkpoint = train_hypers.get("vae_checkpoint", "")
    freeze_autoencoder = train_hypers.get("freeze_autoencoder", True)
    if vae_checkpoint:
        load_autoencoder_state_from_checkpoint(
            model.model_part.autoencoder,
            vae_checkpoint,
            device,
            strict=train_hypers.get("strict_vae_load", True),
        )
    elif freeze_autoencoder:
        raise FileNotFoundError(
            "train_params.vae_checkpoint must be set when freeze_autoencoder is True"
        )

    if freeze_autoencoder:
        for parameter in model.model_part.autoencoder.parameters():
            parameter.requires_grad = False

    model.to(device)
    model.eval()

    _, val_loader, val_pt_dataset = data_builder_autoencoder.build(
        dataset_config,
        configs["train_data_loader"],
        val_loader_config,
        grid_size=model_config["output_shape"],
        use_tta=False,
        use_multiscan=True,
        distributed_context=None,
    )
    val_pt_dataset.im_idx.sort()
    loss_func, _ = loss_builder.build(
        wce=True,
        lovasz=True,
        num_class=model_config["num_class"],
        ignore_label=dataset_config["ignore_label"],
    )

    num_class = int(model_config["num_class"])
    grid_shape = tuple(int(value) for value in model_config["output_shape"])
    confusion = np.zeros((num_class, num_class), dtype=np.int64)
    epsilon_losses = []
    cross_entropy_losses = []
    frame_count = 0

    from tqdm import tqdm

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Full validation sequence 08"):
            (
                val_gt,
                val_latent,
                val_partial,
                val_image,
                _origin_len,
                dir_idx,
                number_idx,
            ) = batch
            if val_latent.shape[0] != 1:
                raise RuntimeError(
                    "Full validation requires batch_size=1, got {}".format(
                        val_latent.shape[0]
                    )
                )

            val_latent = val_latent.to(device=device, dtype=torch.float32)
            val_partial = val_partial.to(device=device, dtype=torch.float32)
            val_image = val_image.to(device=device, dtype=torch.float32)
            val_gt = val_gt.to(device=device, dtype=torch.long)

            epsilon_loss, logits = model(
                val_latent.shape[0],
                val_latent,
                val_partial,
                val_image,
                train=False,
            )
            cross_entropy = loss_func(logits.detach(), val_gt)
            prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]
            target = val_gt.cpu().numpy()[0]

            sequence = batch_item(dir_idx, 0)
            frame = batch_item(number_idx, 0)
            invalid_file = invalid_mask_path(
                dataset_config,
                sequence,
                frame,
            )
            invalid = unpack(np.fromfile(invalid_file, dtype=np.uint8))
            if invalid.size != int(np.prod(grid_shape)):
                raise ValueError(
                    "Invalid mask {} has {} voxels; expected {}".format(
                        invalid_file,
                        invalid.size,
                        int(np.prod(grid_shape)),
                    )
                )
            invalid = invalid.reshape(grid_shape)
            mask = get_eval_mask(target, invalid)
            _accumulate_confusion(confusion, prediction[mask], target[mask])

            epsilon_losses.append(float(epsilon_loss.item()))
            cross_entropy_losses.append(float(cross_entropy.item()))
            frame_count += 1

    if frame_count == 0:
        raise RuntimeError("Full sequence 08 validation loader yielded zero frames")

    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "config_path": str(Path(args.config_path).resolve()),
        "seed": int(args.seed),
        "frame_count": frame_count,
        "mean_epsilon_mse": float(np.mean(epsilon_losses)),
        "mean_cross_entropy": float(np.mean(cross_entropy_losses)),
    }
    result.update(metrics_from_confusion(confusion))

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path(train_hypers["model_save_path"])
            / "full_validation"
            / checkpoint_path.stem
            / "seed_{}".format(args.seed)
        )
    json_path, text_path = write_validation_reports(result, output_dir)
    result["metrics_path"] = str(json_path)
    result["report_path"] = str(text_path)
    return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = evaluate(args)
    print("Checkpoint: {}".format(result["checkpoint"]))
    print("Frames: {}".format(result["frame_count"]))
    print("Semantic mIoU: {:.3f}%".format(result["semantic_miou_percent"]))
    print("Completion IoU: {:.3f}%".format(result["completion_iou_percent"]))
    print("Metrics: {}".format(result["metrics_path"]))
    print("Report: {}".format(result["report_path"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
