import torch
from pathlib import Path
from dataloader.dataset_semantickitti_gai import get_model_class, collate_fn_BEV, collate_fn_BEV_tta, collate_fn_BEV_ms, collate_fn_BEV_ms_tta
from dataloader.pc_dataset_gai import get_pc_model_class
from utils.distributed_training import build_train_sampler
from utils.frame_filtering import filter_dataset_frames_by_divisor


def resolve_feature_root(dataset_config, root_key, default_root_name=""):
    root = dataset_config.get(root_key, default_root_name)
    if not root:
        return ""
    root_path = Path(root)
    if root_path.is_absolute():
        return str(root_path)
    data_root = dataset_config.get("data_root", "")
    if data_root:
        return str(Path(data_root) / root)
    return str(root_path)


def resolve_dataloader_data_path(dataset_config, dataloader_config):
    data_path = dataloader_config.get("data_path", "")
    if data_path:
        return data_path
    data_path = resolve_feature_root(
        dataset_config,
        "vae_feature_root",
        "VAE_Encoder_Features_Semantic20",
    )
    if not data_path:
        raise KeyError(
            "Set either train/val data_loader.data_path or dataset_params.data_root "
            "with dataset_params.vae_feature_root."
        )
    return data_path


def build(dataset_config,
          train_dataloader_config,
          val_dataloader_config,
          grid_size=[480, 360, 32],
          use_tta=False,
          use_multiscan=False,
          use_waymo=False,
          train_frame_divisor=None,
          val_frame_divisor=None,
          distributed_context=None):
    data_path = resolve_dataloader_data_path(dataset_config, train_dataloader_config)
    val_data_path = resolve_dataloader_data_path(dataset_config, val_dataloader_config)
    train_imageset = train_dataloader_config["imageset"]
    val_imageset = val_dataloader_config["imageset"]
    train_ref = train_dataloader_config["return_ref"]
    val_ref = val_dataloader_config["return_ref"]

    label_mapping = dataset_config["label_mapping"]

    SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])
    feature_roots = {
        "image_condition_path": resolve_feature_root(
            dataset_config,
            "image_condition_root",
            "Image_transform_Voxel_Condition_Features",
        ),
        "partial_condition_path": resolve_feature_root(
            dataset_config,
            "partial_condition_root",
            "Condition_Features_2",
        ),
        "gt_path": resolve_feature_root(
            dataset_config,
            "gt_root",
            "data_odometry_voxel_all_with_one",
        ),
    }

    nusc=None
    if "nusc" in dataset_config['pc_dataset_type']:
        from nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)

    train_pt_dataset = SemKITTI(data_path, imageset=train_imageset,
                                return_ref=train_ref, label_mapping=label_mapping, nusc=nusc, **feature_roots)
    val_pt_dataset = SemKITTI(val_data_path, imageset=val_imageset,
                              return_ref=val_ref, label_mapping=label_mapping, nusc=nusc, **feature_roots)
    if train_frame_divisor is None:
        train_frame_divisor = train_dataloader_config.get("frame_divisor")
    if val_frame_divisor is None:
        val_frame_divisor = val_dataloader_config.get("frame_divisor")
    if train_frame_divisor is not None:
        filter_dataset_frames_by_divisor(train_pt_dataset, train_frame_divisor)
    if val_frame_divisor is not None:
        filter_dataset_frames_by_divisor(val_pt_dataset, val_frame_divisor)
    dataAug = 0
    if dataAug:
        train_dataset = get_model_class(dataset_config['dataset_type'])(
        train_pt_dataset,
        grid_size=grid_size,
        rotate_aug=True,
        flip_aug=True,
        ignore_label=dataset_config["ignore_label"],
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        return_test=True,
        )
    else:
        train_dataset = get_model_class(dataset_config['dataset_type'])(
            train_pt_dataset,
            grid_size=grid_size,
            rotate_aug=False,
            flip_aug=False,
            ignore_label=dataset_config["ignore_label"],
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            return_test=True,
        )
    if use_tta:
        if dataAug:
            val_dataset = get_model_class(dataset_config['dataset_type'])(
                val_pt_dataset,
                grid_size=grid_size,
                rotate_aug=False,  # True
                flip_aug=False,  # True
                ignore_label=dataset_config["ignore_label"],
                fixed_volume_space=dataset_config['fixed_volume_space'],
                max_volume_space=dataset_config['max_volume_space'],
                min_volume_space=dataset_config['min_volume_space'],
                return_test=True,
            )
        else:
            val_dataset = get_model_class(dataset_config['dataset_type'])(
                val_pt_dataset,
                grid_size=grid_size,
                rotate_aug=False,
                flip_aug=False,
                ignore_label=dataset_config["ignore_label"],
                fixed_volume_space=dataset_config['fixed_volume_space'],
                max_volume_space=dataset_config['max_volume_space'],
                min_volume_space=dataset_config['min_volume_space'],
                return_test=True,
            )
        if use_multiscan:
            collate_fn_BEV_tmp = collate_fn_BEV_ms_tta
        else:
            collate_fn_BEV_tmp = collate_fn_BEV_tta
    else:
        val_dataset = get_model_class(dataset_config['dataset_type'])(
            val_pt_dataset,
            grid_size=grid_size,
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            ignore_label=dataset_config["ignore_label"],
            return_test=True,
        )
        if use_multiscan or use_waymo:
            collate_fn_BEV_tmp = collate_fn_BEV_ms
        else:
            collate_fn_BEV_tmp = collate_fn_BEV

    train_sampler = build_train_sampler(
        train_dataset,
        distributed_context,
        shuffle=train_dataloader_config["shuffle"],
    )
    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_dataloader_config["batch_size"],
                                                       collate_fn=collate_fn_BEV_tmp,
                                                       shuffle=train_dataloader_config["shuffle"] if train_sampler is None else False,
                                                       sampler=train_sampler,
                                                       num_workers=train_dataloader_config["num_workers"],
                                                       drop_last = True)
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=collate_fn_BEV_tmp,
                                                     shuffle=val_dataloader_config["shuffle"],
                                                     num_workers=val_dataloader_config["num_workers"],
                                                     drop_last = True)

    if use_tta:
        return train_dataset_loader, val_dataset_loader, val_pt_dataset
    else:
        return train_dataset_loader, val_dataset_loader, val_pt_dataset
