from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_training_config(config_path):
    from config.config import load_config_data

    return load_config_data(config_path)


def resolve_dataset_root(dataset_config, root_key, default_root_name=""):
    root = dataset_config.get(root_key, default_root_name)
    if not root:
        return Path("")

    root_path = Path(root)
    if root_path.is_absolute():
        return root_path

    data_root = dataset_config.get("data_root", "")
    if data_root:
        return Path(data_root) / root_path

    return root_path


def resolve_vae_root(configs):
    train_loader = configs.get("train_data_loader", {})
    data_path = train_loader.get("data_path", "")
    if data_path:
        return Path(data_path)
    return resolve_dataset_root(
        configs["dataset_params"],
        "vae_feature_root",
        "VAE_Encoder_Features_Semantic20",
    )


def resolve_image_condition_root(configs):
    return resolve_dataset_root(
        configs["dataset_params"],
        "image_condition_root",
        "Image_transform_Voxel_Condition_Features",
    )


def resolve_partial_condition_root(configs):
    return resolve_dataset_root(
        configs["dataset_params"],
        "partial_condition_root",
        "Condition_Features_2",
    )


def resolve_gt_root(configs):
    return resolve_dataset_root(
        configs["dataset_params"],
        "gt_root",
        "data_odometry_voxel_all",
    )


def resolve_kitti_root(configs):
    dataset_config = configs["dataset_params"]
    kitti_root = dataset_config.get("kitti_root", "")
    if kitti_root:
        return Path(kitti_root)
    data_root = dataset_config.get("data_root", "")
    return Path(data_root) if data_root else Path("")
