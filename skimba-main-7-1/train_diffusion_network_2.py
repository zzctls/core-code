import os
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from dataloader.pc_dataset_gai import get_eval_mask, unpack
from builder import data_builder_autoencoder, model_builder_3D_Voxel_unet_diffusion, loss_builder
from config.config import load_config_data
from torch.utils.tensorboard import SummaryWriter
import warnings
from utils.np_ioueval import iouEval
import yaml
from utils.warmup_lr import *
import torch.nn.functional as F
from collections import OrderedDict
from pathlib import Path
from utils.distributed_training import (
    barrier,
    cleanup_distributed,
    initialize_distributed,
    unwrap_model,
)
from utils.validation_metrics import mask_invalid_voxels
from utils.validation_schedule import should_validate_epoch

def freeze_model(model, to_free_dict):
    for name, param in model.named_parameters():
        if name in to_free_dict:
            param.requires_grad = False
        else:
            pass
    return model


def load_model_state(model, state_dict):
    model.model_part.validate_checkpoint_normalization(state_dict)
    return model.load_state_dict(state_dict)


def checkpoint_model_state(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def normalize_autoencoder_key(key):
    for prefix in ("module.", "model_part.autoencoder.", "autoencoder.", "model_part."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def load_autoencoder_state_from_checkpoint(autoencoder, checkpoint_path, torch_device, strict=True):
    checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    state_dict = checkpoint_model_state(checkpoint)
    autoencoder_state_dict = {}
    autoencoder_prefixes = ("encoder.", "decoder.", "quant_conv.", "post_quant_conv.")

    for name, param in state_dict.items():
        normalized_name = normalize_autoencoder_key(name)
        if normalized_name.startswith(autoencoder_prefixes):
            autoencoder_state_dict[normalized_name] = param

    if not autoencoder_state_dict:
        sample_keys = list(state_dict.keys())[:10]
        raise RuntimeError(
            "No autoencoder weights were found in VAE checkpoint. "
            f"Sample checkpoint keys: {sample_keys}"
        )

    return autoencoder.load_state_dict(autoencoder_state_dict, strict=strict)


def resolve_configured_root(dataset_config, root_key, default_root_name=""):
    root = dataset_config.get(root_key, default_root_name)
    if not root:
        return ""
    root_path = Path(root)
    if root_path.is_absolute():
        return root_path
    data_root = dataset_config.get("data_root", "")
    if data_root:
        return Path(data_root) / root
    return root_path


def invalid_mask_path(dataset_config, sequence, frame):
    invalid_root = resolve_configured_root(
        dataset_config,
        "invalid_root",
        "data_odometry_voxels_all",
    )
    if not invalid_root:
        raise KeyError("Set dataset_params.invalid_root or dataset_params.data_root in the config.")
    return invalid_root / "sequences" / f"{int(sequence):02d}" / "voxels" / f"{int(frame):06d}.invalid"


def resolve_validation_output_path(train_hypers):
    configured_path = train_hypers.get('validation_output_path', '')
    if configured_path:
        return configured_path
    return os.path.join(train_hypers['model_save_path'], 'validation_predictions')


def resolve_model_checkpoint_path(checkpoint_path, model_load_path, configured_checkpoint_path=""):
    if checkpoint_path:
        return checkpoint_path
    if configured_checkpoint_path:
        return configured_checkpoint_path
    return os.path.join(model_load_path, '0.pth')


def restore_training_state(optimizer, scheduler, checkpoint, steps_per_epoch):
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Training resume requires a checkpoint dict with optimizer and scheduler state. "
            "Use a protect_*.pth checkpoint for full training resume."
        )
    required_keys = ("optimizer_state_dict", "scheduler_state_dict")
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise RuntimeError(
            "Training resume checkpoint is missing required key(s): {}".format(
                ", ".join(missing_keys)
            )
        )

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    completed_epoch = int(checkpoint.get("epoch", 0))
    start_epoch = completed_epoch + 1 if completed_epoch > 0 else 1
    best_val_miou = float(checkpoint.get("best_val_miou", checkpoint.get("Loss", 0)))
    if "global_iter" in checkpoint:
        global_iter = int(checkpoint["global_iter"])
    else:
        global_iter = max(completed_epoch, 0) * int(steps_per_epoch)
    return start_epoch, best_val_miou, global_iter


def format_sequence_id(sequence):
    return "{:02d}".format(int(sequence))


def format_frame_id(frame):
    return "{:06d}".format(int(frame))


def batch_item(values, index):
    if torch.is_tensor(values):
        return int(values[index].item())
    if isinstance(values, np.ndarray):
        return int(values[index])
    return int(values[index])


def target_frame_matches(dir_idx, number_idx, sample_index, target_sequence, target_frame):
    return (
        format_sequence_id(batch_item(dir_idx, sample_index)) == target_sequence
        and format_frame_id(batch_item(number_idx, sample_index)) == target_frame
    )


def resolve_export_output_root(export_output_root, model_save_path, export_name):
    if export_output_root:
        return export_output_root
    return os.path.join(model_save_path, "visualization_exports", export_name)


def export_prediction_path(output_root, sequence, frame):
    return os.path.join(output_root, format_sequence_id(sequence), "{}.label".format(format_frame_id(frame)))


def save_prediction_sample(recon_voxel, sample_index, output_path, remap_lut):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    recon_voxel_occupy = torch.argmax(recon_voxel[sample_index : sample_index + 1], dim=1)
    pred = recon_voxel_occupy.cpu().numpy().astype(np.uint32).reshape((-1))
    upper_half = pred >> 16
    lower_half = pred & 0xFFFF
    lower_half = remap_lut[lower_half]
    pred = (upper_half << 16) + lower_half
    final_preds = pred.astype(np.uint16)
    final_preds.tofile(output_path)
    return output_path


def export_target_validation_frame(
    model,
    val_dataset_loader,
    pytorch_device,
    remap_lut,
    output_root,
    target_sequence,
    target_frame,
):
    model.eval()
    normalized_sequence = format_sequence_id(target_sequence)
    normalized_frame = format_frame_id(target_frame)

    with torch.no_grad():
        for (
            val_GT_features,
            val_VAE_features,
            val_partial_features,
            val_image_features,
            origin_len,
            dir_idx,
            number_idx,
        ) in val_dataset_loader:
            matching_sample_index = None
            for sample_index in range(val_VAE_features.shape[0]):
                if target_frame_matches(
                    dir_idx,
                    number_idx,
                    sample_index,
                    normalized_sequence,
                    normalized_frame,
                ):
                    matching_sample_index = sample_index
                    break
            if matching_sample_index is None:
                continue

            batch_slice = slice(matching_sample_index, matching_sample_index + 1)
            val_VAE_features_change = val_VAE_features[batch_slice].to(
                device=pytorch_device,
                dtype=torch.float32,
            )
            val_partial_features_change = val_partial_features[batch_slice].to(
                device=pytorch_device,
                dtype=torch.float32,
            )
            val_image_features_change = val_image_features[batch_slice].to(
                device=pytorch_device,
                dtype=torch.float32,
            )

            _, recon_voxel = model(
                val_VAE_features_change.shape[0],
                val_VAE_features_change,
                val_partial_features_change,
                val_image_features_change,
                train=False,
            )
            output_path = export_prediction_path(
                output_root,
                normalized_sequence,
                normalized_frame,
            )
            return save_prediction_sample(
                recon_voxel,
                0,
                output_path,
                remap_lut,
            )

    raise FileNotFoundError(
        "Validation frame not found: sequence {} frame {}".format(
            normalized_sequence,
            normalized_frame,
        )
    )


def _run_training(args, context):
    rank_print = print if context.is_main else lambda *args, **kwargs: None
    writer = None
    warnings.filterwarnings("ignore")
    pytorch_device = context.device
    rank_print(' '.join(sys.argv))
    rank_print(args)

    config_path = args.config_path
    configs = load_config_data(config_path)
    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']
    if args.export_only:
        val_dataloader_config = dict(val_dataloader_config)
        val_dataloader_config.pop("frame_divisor", None)

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']
    configured_resume_checkpoint = train_hypers.get('resume_checkpoint', '')
    checkpoint_path = resolve_model_checkpoint_path(
        args.resume,
        model_load_path,
        configured_resume_checkpoint,
    )

    with open(dataset_config['label_mapping'], 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    learning_map_inv = semkittiyaml["learning_map_inv"]
    maxkey = max(learning_map_inv.keys())
    remap_lut_First = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut_First[list(learning_map_inv.keys())] = list(learning_map_inv.values())

    my_model = model_builder_3D_Voxel_unet_diffusion.build(model_config)
    total_params = sum(p.numel() for p in my_model.parameters())
    rank_print(f"总参数量: {total_params}")
    checkpoint = None
    if os.path.exists(checkpoint_path):
        rank_print('Load model from: %s' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=pytorch_device)
        load_model_state(
            my_model,
            checkpoint_model_state(checkpoint),
        )
    elif args.resume or configured_resume_checkpoint:
        raise FileNotFoundError("Resume checkpoint does not exist: %s" % checkpoint_path)
    elif args.export_only:
        raise FileNotFoundError(
            "Export-only requires --resume or a checkpoint at {}".format(checkpoint_path)
        )
    else:
        rank_print('No existing model, training model from scratch...')

    vae_checkpoint = train_hypers.get('vae_checkpoint', '')
    freeze_autoencoder = train_hypers.get('freeze_autoencoder', True)
    strict_vae_load = train_hypers.get('strict_vae_load', True)
    if vae_checkpoint:
        rank_print('Load VAE from: %s' % vae_checkpoint)
        load_autoencoder_state_from_checkpoint(
            my_model.model_part.autoencoder,
            vae_checkpoint,
            pytorch_device,
            strict=strict_vae_load,
        )
    elif freeze_autoencoder:
        raise FileNotFoundError("train_params.vae_checkpoint must be set when freeze_autoencoder is True")

    if freeze_autoencoder:
        for param in my_model.model_part.autoencoder.parameters():
            param.requires_grad = False
    else:
        rank_print('Autoencoder is trainable.')

    if context.is_main and not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    rank_print(model_save_path)
    tensorboard_log_dir = os.path.join(model_save_path, 'logs')
    if context.is_main:
        writer = SummaryWriter(tensorboard_log_dir)
        rank_print(f"TensorBoard logs: {tensorboard_log_dir}")

    my_model.to(pytorch_device)
    if context.distributed:
        my_model = DistributedDataParallel(
            my_model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            find_unused_parameters=True,
        )
    raw_model = unwrap_model(my_model)
    optimizer = optim.AdamW(my_model.parameters(), lr=train_hypers["learning_rate"], weight_decay=1e-4)

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True, num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader, val_pt_dataset = data_builder_autoencoder.build(dataset_config,
                                                                                  train_dataloader_config,
                                                                                  val_dataloader_config,
                                                                                  grid_size=grid_size,
                                                                                  use_tta=False,
                                                                                  use_multiscan=True,
                                                                                  distributed_context=context)

    if args.export_only:
        if context.is_main:
            output_root = resolve_export_output_root(
                args.export_output_root,
                model_save_path,
                "completion",
            )
            output_path = export_target_validation_frame(
                raw_model,
                val_dataset_loader,
                pytorch_device,
                remap_lut_First,
                output_root,
                args.export_sequence,
                args.export_frame,
            )
            rank_print("Exported completion validation frame to: {}".format(output_path))
        barrier(context)
        return

    scheduler = WarmupConstantLearningRate(
        optimizer=optimizer,
        lr=train_hypers["learning_rate"],
        start_lr=train_hypers["warmup_start_lr"],
        warmup_steps=train_hypers["warmup_epochs"] * len(train_dataset_loader),
    )

    # training
    start_epoch = 1
    best_val_miou = 0
    global_iter = 0
    if checkpoint is not None:
        rank_print('Load model from: %s' % checkpoint_path)
        start_epoch, best_val_miou, global_iter = restore_training_state(
            optimizer,
            scheduler,
            checkpoint,
            steps_per_epoch=len(train_dataset_loader),
        )
    else:
        rank_print('No existing model, training model from scratch...')

    check_iter = train_hypers['eval_every_n_steps']
    max_epoch = train_hypers['max_num_epochs']

    # learning map
    with open(dataset_config['label_mapping'], 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    class_strings = semkittiyaml["labels"]
    class_inv_remap = semkittiyaml["learning_map_inv"]

    bin_file_path = resolve_validation_output_path(train_hypers)
    if context.is_main:
        os.makedirs(bin_file_path, exist_ok=True)
    rank_print(bin_file_path)

    while start_epoch < max_epoch+1:
        if context.distributed:
            train_dataset_loader.sampler.set_epoch(start_epoch)
        pbar = tqdm(total=len(train_dataset_loader), disable=not context.is_main)
        my_model.train()
        for module in raw_model.model_part.autoencoder.modules():
            if isinstance(module, nn.BatchNorm3d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()
        train_loss_mse = []
        for i_iter, (train_GT_features, train_VAE_features, train_partial_features, train_image_features, origin_len, dir_idx, number_idx) in enumerate(train_dataset_loader):  # origin_len
            train_VAE_features_change = train_VAE_features.type(torch.FloatTensor).to(pytorch_device)
            train_partial_features_change = train_partial_features.type(torch.FloatTensor).to(pytorch_device)
            train_image_features_change = train_image_features.type(torch.FloatTensor).to(pytorch_device)

            # forward + backward + optimize
            noise, pred_noise = my_model(
                                                     train_VAE_features_change.shape[0],
                                                     train_VAE_features_change,
                                                     train_partial_features_change,
                                                     train_image_features_change,
                                                     train = True
                                                     )

            ###------------------------------------acquire recover feature --------------------------------------------------#####
            #----mse----
            loss_train = F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")
            if context.is_main:
                print('epoch %d iter %d total train loss %.8f  \n' %(start_epoch,i_iter,loss_train))
            train_loss_mse.append(loss_train.item())
            loss_train.backward()
            optimizer.step()
            scheduler.step()
            if global_iter % 1000 == 0:
                if len(train_loss_mse) > 0:
                    if context.is_main:
                        print('epoch %d iter %5d, loss: %.3f\n' %
                              (start_epoch, i_iter, np.mean(train_loss_mse)))
                else:
                    if context.is_main:
                        print('loss error')

            optimizer.zero_grad()
            pbar.update(1)
            global_iter += 1
            if global_iter % check_iter == 0:
                if len(train_loss_mse) > 0:
                    if context.is_main:
                        print('epoch %d iter %5d, loss: %.3f\n' %
                              (start_epoch, i_iter, np.mean(train_loss_mse)))
                else:
                    if context.is_main:
                        print('loss error')

        should_validate = should_validate_epoch(start_epoch, max_epoch, train_hypers)
        if should_validate:
            barrier(context)
        if should_validate and context.is_main:  # and epoch > 0
            raw_model.eval()
            val_loss_conditional_list = []
            val_loss_cfg_list = []
            val_reconstruction_loss = []
            evaluator = iouEval(num_class, [])
            with torch.no_grad():
                for i_iter_val, (val_GT_features, val_VAE_features, val_partial_features, val_image_features, origin_len, dir_idx, number_idx) in enumerate(val_dataset_loader):

                    val_VAE_features_change = val_VAE_features.type(torch.FloatTensor).to(pytorch_device)
                    val_partial_features_change = val_partial_features.type(torch.FloatTensor).to(pytorch_device)
                    val_image_features_change = val_image_features.type(torch.FloatTensor).to(pytorch_device)

                    val_GT_features_change = val_GT_features.type(torch.LongTensor).to(pytorch_device)

                    loss_val_conditional_mse, loss_val_cfg_mse, recon_voxel = raw_model(
                                                        val_VAE_features_change.shape[0],
                                                        val_VAE_features_change,
                                                        val_partial_features_change,
                                                        val_image_features_change,
                                                        train = False
                                                                               )
                    invalid_name = invalid_mask_path(dataset_config, dir_idx[0], number_idx[0])
                    invalid_voxels = unpack(np.fromfile(invalid_name, dtype=np.uint8))
                    invalid_voxels = invalid_voxels.reshape((256, 256, 32))
                    reconstruction_target = mask_invalid_voxels(
                        val_GT_features_change,
                        invalid_voxels,
                        ignore_label,
                    )
                    reconstruction_loss = loss_func(
                        recon_voxel.detach(), reconstruction_target
                    )
                    

                    print(
                        'epoch %d iter %d loss_val_conditional_mse %.8f '
                        'loss_val_cfg_mse %.8f reconstruction_loss %.8f \n'
                        % (
                            start_epoch,
                            i_iter_val,
                            loss_val_conditional_mse,
                            loss_val_cfg_mse,
                            reconstruction_loss,
                        )
                    )

                    val_vox_label0 = val_GT_features_change.cpu().detach().numpy()
                    val_vox_label0 = np.squeeze(val_vox_label0)
                    recon_voxel = F.softmax(recon_voxel)
                    predict_labels = torch.argmax(recon_voxel, dim=1)
                    predict_labels = predict_labels.cpu().detach().numpy()
                    predict_labels = np.squeeze(predict_labels)
                    masks = get_eval_mask(val_vox_label0, invalid_voxels)
                    predict_labels = predict_labels[masks]
                    val_vox_label0 = val_vox_label0[masks]

                    evaluator.addBatch(predict_labels.astype(int), val_vox_label0.astype(int))
                    val_loss_conditional_list.append(loss_val_conditional_mse.item())
                    val_loss_cfg_list.append(loss_val_cfg_mse.item())
                    val_reconstruction_loss.append(reconstruction_loss.item())
                    if start_epoch % 43 ==0:
                        ###--------------------------------------------save the validation result to visualization--------------------------------------###

                        fout_pred_voxel_path = os.path.join(bin_file_path, "sequences_{}".format(start_epoch), "{:02}".format(dir_idx[0]),'pred')
                        if not os.path.exists(fout_pred_voxel_path):
                            os.makedirs(fout_pred_voxel_path)

                        fout_pred_voxel = os.path.join(fout_pred_voxel_path, "{:06}.label".format(number_idx[0]))
                        recon_voxel_occupy = torch.argmax(recon_voxel, dim=1)
                        recon_voxel_1 = recon_voxel_occupy
                        recon_voxel_1 = recon_voxel_1.cpu().numpy()

                        pred = recon_voxel_1.astype(np.uint32)
                        pred = pred.reshape((-1))
                        upper_half = pred >> 16  # get upper half for instances
                        lower_half = pred & 0xFFFF  # get lower half for semantics
                        lower_half = remap_lut_First[lower_half]  # do the remapping of semantics
                        pred = (upper_half << 16) + lower_half  # reconstruct full label
                        pred = pred.astype(np.uint32)
                        final_preds = pred.astype(np.uint16)
                        final_preds.tofile(fout_pred_voxel)

            print('Validation per class iou: ')
            _, class_jaccard = evaluator.getIoU()
            m_jaccard = class_jaccard[1:].mean()
            class_iou_percent = class_jaccard * 100.0
            val_miou_percent = m_jaccard * 100.0
            ignore = [0]
            outpath = model_save_path + 'output_{}.txt'.format(start_epoch)
            # print also classwise
            for i, _ in enumerate(class_jaccard):
                if i not in ignore:
                    class_iou_line = (
                        f'IoU class {i} [{class_strings[class_inv_remap[i]]}] = '
                        f'{class_iou_percent[i]:.3f}%'
                    )
                    print(class_iou_line)
                    with open(outpath, 'a', encoding='utf-8') as file:
                        file.write(class_iou_line + "\n")
            # compute remaining metrics.
            conf = evaluator.get_confusion()
            acc_completion = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0])
            completion_iou_percent = acc_completion * 100.0
            validation_iou_line = (
                f'Current val completion iou is {completion_iou_percent:.3f}% '
                f'and Current val miou is {val_miou_percent:.3f}%'
            )
            print(validation_iou_line)
            with open(outpath, 'a', encoding='utf-8') as file:
                file.write(validation_iou_line + "\n")

            del val_GT_features, val_VAE_features, val_partial_features, val_image_features, val_VAE_features_change, val_partial_features_change, val_image_features_change, val_GT_features_change
                #val_vox_label, point_label_tensor, point_label_tensor_1

            if best_val_miou < val_miou_percent:
                best_val_miou = val_miou_percent
                # model_save_name = model_save_path + ('iou%.4f_epoch%d.pth' % (val_loss, epoch))
                model_save_name = model_save_path + ('{}_{}_{}.pth'.format('best', start_epoch, best_val_miou))
                torch.save(raw_model.state_dict(), model_save_name)
                model_save_name_protect = model_save_path + ('{}_{}.pth'.format('protect', start_epoch))
                torch.save({'epoch': start_epoch,
                            'model_state_dict': raw_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'Loss': best_val_miou,
                            'best_val_miou': best_val_miou,
                            'global_iter': global_iter},
                           model_save_name_protect)
            print(
                f'Current val miou is {val_miou_percent:.3f}% while the best val miou is '
                f'{best_val_miou:.3f}%'
            )
            print(
                'Current conditional val loss is %.3f and CFG-guided val loss is %.3f'
                % (
                    np.mean(val_loss_conditional_list),
                    np.mean(val_loss_cfg_list),
                )
            )

            val_loss_conditional_mean = np.mean(val_loss_conditional_list)
            val_loss_cfg_mean = np.mean(val_loss_cfg_list)
            val_reconstruction_loss_mean = np.mean(val_reconstruction_loss)

            writer.add_scalar("val_loss_conditional", val_loss_conditional_mean, start_epoch)
            writer.add_scalar("val_loss", val_loss_cfg_mean, start_epoch)
            writer.add_scalar("mIoU_val_mean", val_miou_percent, start_epoch)
            writer.add_scalar("iou_val_mean", completion_iou_percent, start_epoch)
            writer.add_scalar("val_reconstruction_loss_mean", val_reconstruction_loss_mean, start_epoch)

            raw_model.train()

        if should_validate:
            barrier(context)

        pbar.close()

        train_loss_mse_mean = np.mean(train_loss_mse)

        for g in optimizer.param_groups:
            lr = g["lr"]
            break
        if writer is not None:
            writer.add_scalar("train_loss_mse_mean", train_loss_mse_mean, start_epoch)
            writer.add_scalar("lr_rate", lr, start_epoch)
        start_epoch += 1

    if writer is not None:
        writer.close()


def main(args):
    context = None
    try:
        context = initialize_distributed()
        _run_training(args, context)
    finally:
        cleanup_distributed(context)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti_autoencoder.yaml')
    parser.add_argument("--resume", default="", help="Optional checkpoint path to resume from.")
    parser.add_argument("--export-only", action="store_true", help="Export one validation frame and exit.")
    parser.add_argument("--export-sequence", default="08", help="Validation sequence to export.")
    parser.add_argument("--export-frame", default="000000", help="Validation frame to export.")
    parser.add_argument("--export-output-root", default="", help="Root directory for exported .label files.")
    args = parser.parse_args()

    main(args)

#
