from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NETWORK_SOURCE = ROOT / "network" / "cylinder_3D_Unet_mamba_diffusion.py"
BUILDER_SOURCE = ROOT / "builder" / "model_builder_3D_Voxel_unet_diffusion.py"
DATA_BUILDER_SOURCE = ROOT / "builder" / "data_builder_autoencoder.py"
CONFIG_SOURCE = ROOT / "config" / "config.py"
YAML_SOURCE = ROOT / "config" / "semantickitti_autoencoder.yaml"
DATALOADER_SOURCE = ROOT / "dataloader" / "pc_dataset_gai.py"
TRAIN_SOURCE = ROOT / "train_diffusion_network_2.py"


def read(path):
    return path.read_text(encoding="utf-8")


def test_diffusion_wrapper_concats_raw_conditions_before_compression():
    source = read(NETWORK_SOURCE)

    assert "condition_encoder" not in source
    assert "ImageConditionEncoder" not in source
    assert "PartialConditionEncoder" not in source
    assert "class ConditionFusionCompressor(nn.Module)" in source
    assert "torch.cat((image_condition.float(), partial_condition.float()), dim=1)" in source
    assert "nn.Conv3d(self.in_channels, hidden_channels, kernel_size=1" in source
    assert "nn.Conv3d(hidden_channels, out_channels, kernel_size=1" in source
    assert "kernel_size=3" not in source
    assert "self.condition_mid_channels = condition_mid_channels" in source
    assert "self.condition_channels = condition_channels" in source
    assert "self.condition_compressor = ConditionFusionCompressor" in source
    assert "def compress_conditions" in source
    assert "self.compress_conditions(" in source
    assert "self.model_part.pred_noise" in source
    assert "self.model_part.sample" in source


def test_condition_compressor_defaults_to_raw_monoscene_and_mask2former_channels():
    source = read(NETWORK_SOURCE)

    assert "condition_in_channels=88" in source
    assert "image_condition_channels=64" in source
    assert "partial_condition_channels=24" in source
    assert "condition_mid_channels=44" in source
    assert "condition_channels=16" in source
    assert "condition.dim() != 5" in source
    assert "condition.shape[1] != expected_channels" in source
    assert "image_condition.shape[2:] != partial_condition.shape[2:]" in source
    assert "return self.projection(condition)" in source


def test_builder_and_config_default_to_raw_condition_fusion():
    builder_source = read(BUILDER_SOURCE)
    config_source = read(CONFIG_SOURCE)
    yaml_source = read(YAML_SOURCE)

    forbidden = (
        "condition_latent_channels",
        "train_condition_encoder",
        "condition_features_preencoded",
    )
    for text in forbidden:
        assert text not in builder_source
        assert text not in config_source
        assert text not in yaml_source

    assert 'Optional("condition_in_channels"): Int()' in config_source
    assert 'Optional("image_condition_channels"): Int()' in config_source
    assert 'Optional("partial_condition_channels"): Int()' in config_source
    assert 'Optional("condition_mid_channels"): Int()' in config_source
    assert 'Optional("condition_channels"): Int()' in config_source
    assert "condition_in_channels: 88" in yaml_source
    assert "image_condition_channels: 64" in yaml_source
    assert "partial_condition_channels: 24" in yaml_source
    assert "condition_mid_channels: 44" in yaml_source
    assert "condition_channels: 16" in yaml_source
    assert "condition_in_channels = model_config.get('condition_in_channels', 88)" in builder_source
    assert "image_condition_channels = model_config.get('image_condition_channels', 64)" in builder_source
    assert "partial_condition_channels = model_config.get('partial_condition_channels', 24)" in builder_source
    assert "condition_mid_channels = model_config.get('condition_mid_channels', condition_in_channels // 2)" in builder_source
    assert "condition_channels = model_config.get('condition_channels', 16)" in builder_source
    assert "condition_in_channels=condition_in_channels" in builder_source
    assert "image_condition_channels=image_condition_channels" in builder_source
    assert "partial_condition_channels=partial_condition_channels" in builder_source
    assert "condition_mid_channels=condition_mid_channels" in builder_source
    assert "condition_channels=condition_channels" in builder_source


def test_dataloader_accepts_raw_condition_features():
    source = read(DATALOADER_SOURCE)

    assert "VAE_Encoder_Features_Semantic20" in source
    assert "def replace_known_feature_root" in source
    assert "def configured_feature_path" in source
    assert "image_condition_path=\"\"" in source
    assert "partial_condition_path=\"\"" in source
    assert "gt_path=\"\"" in source
    assert "def reshape_condition_feature" in source
    assert "def reshape_image_condition_feature" in source
    assert "channel_candidates=(64,)" in source
    assert "def reshape_partial_condition_feature" in source
    assert "channel_candidates=(24, 64)" in source
    assert "return condition[:24]" in source
    assert "spatial_shape=(64, 64, 8)" in source
    assert "annotated_data_image = reshape_image_condition_feature(image_data, path_image)" in source
    assert "annotated_data_partial = reshape_partial_condition_feature(partial_data, path_partial)" in source


def test_frame_divisor_sampling_uses_full_training_set_and_sampled_validation():
    data_builder_source = read(DATA_BUILDER_SOURCE)
    config_source = read(CONFIG_SOURCE)
    yaml_source = read(YAML_SOURCE)
    train_section = yaml_source.split("train_data_loader:", 1)[1].split(
        "val_data_loader:",
        1,
    )[0]
    val_section = yaml_source.split("val_data_loader:", 1)[1].split(
        "###################",
        1,
    )[0]

    assert 'Optional("frame_divisor"): Int()' in config_source
    assert "frame_divisor" not in train_section
    assert "frame_divisor: 10" in val_section
    assert "from utils.frame_filtering import filter_dataset_frames_by_divisor" in data_builder_source
    assert "train_frame_divisor = train_dataloader_config.get(\"frame_divisor\")" in data_builder_source
    assert "val_frame_divisor = val_dataloader_config.get(\"frame_divisor\")" in data_builder_source
    assert "filter_dataset_frames_by_divisor(train_pt_dataset, train_frame_divisor)" in data_builder_source
    assert "filter_dataset_frames_by_divisor(val_pt_dataset, val_frame_divisor)" in data_builder_source


def test_dataloader_remaps_completion_gt_labels_before_loss():
    source = read(DATALOADER_SOURCE)

    assert "def remap_completion_labels" in source
    assert "np.fromfile(path_GT, dtype=np.uint16)" in source
    assert "remap_completion_labels(GT_data, self.comletion_remap_lut)" in source
    assert "annotated_data_GT.astype(np.uint8)" in source


def test_training_loads_current_checkpoint_state_strictly():
    source = read(TRAIN_SOURCE)

    assert "def load_model_state(model, state_dict)" in source
    assert "load_model_state(" in source
    assert "model.load_state_dict(state_dict)" in source
    assert "optimizer.load_state_dict(checkpoint['optimizer_state_dict'])" in source
    assert "scheduler.load_state_dict(checkpoint['scheduler_state_dict'])" in source
    assert "load_model_state_compatible" not in source
    assert "load_training_state_compatible" not in source
    assert "strict=False" not in source
    assert "condition_encoder" not in source
    assert "missing condition encoder keys" not in source


def test_training_validates_latent_normalization_before_loading_diffusion_checkpoint():
    source = read(TRAIN_SOURCE)

    validation = "model.model_part.validate_checkpoint_normalization(state_dict)"
    assert validation in source
    assert source.index(validation) < source.index("model.load_state_dict(state_dict)")


def test_training_validation_prediction_dir_is_not_empty():
    source = read(TRAIN_SOURCE)
    config_source = read(CONFIG_SOURCE)
    yaml_source = read(YAML_SOURCE)

    assert "def resolve_validation_output_path" in source
    assert "train_hypers.get('validation_output_path', '')" in source
    assert "os.path.join(train_hypers['model_save_path'], 'validation_predictions')" in source
    assert 'bin_file_path = ""' not in source
    assert "os.makedirs(bin_file_path, exist_ok=True)" in source

    assert "Optional(\"validation_output_path\"): Str()" in config_source
    assert "validation_output_path:" in yaml_source


def test_training_validation_epoch_interval_is_configurable():
    source = read(TRAIN_SOURCE)
    config_source = read(CONFIG_SOURCE)
    yaml_source = read(YAML_SOURCE)

    assert "from utils.validation_schedule import should_validate_epoch" in source
    assert "should_validate = should_validate_epoch(start_epoch, max_epoch, train_hypers)" in source
    assert "Optional(\"eval_every_n_epochs\"): Int()" in config_source
    assert "Optional(\"eval_first_phase_epochs\"): Int()" in config_source
    assert "Optional(\"eval_first_phase_every_n_epochs\"): Int()" in config_source
    assert "Optional(\"eval_after_phase_every_n_epochs\"): Int()" in config_source
    assert "eval_every_n_epochs: 10" in yaml_source
    assert "eval_first_phase_epochs: 450" in yaml_source
    assert "eval_first_phase_every_n_epochs: 30" in yaml_source
    assert "eval_after_phase_every_n_epochs: 10" in yaml_source


def test_training_does_not_force_cuda_launch_blocking():
    source = read(TRAIN_SOURCE)

    assert "CUDA_LAUNCH_BLOCKING'] = '1'" not in source
    assert 'CUDA_LAUNCH_BLOCKING"] = "1"' not in source


def test_cfg_parameters_are_yaml_configured_and_wired_to_live_paths():
    network_source = read(NETWORK_SOURCE)
    builder_source = read(BUILDER_SOURCE)
    config_source = read(CONFIG_SOURCE)
    yaml_source = read(YAML_SOURCE)

    assert 'Optional("condition_dropout_prob"): Float()' in config_source
    assert 'Optional("guidance_scale"): Float()' in config_source
    assert 'Optional("num_inference_steps"): Int()' in config_source
    assert "condition_dropout_prob: 0.1" in yaml_source
    assert "guidance_scale: 3.0" in yaml_source
    assert "num_inference_steps: 100" in yaml_source

    assert "condition_dropout_prob = model_config.get('condition_dropout_prob', 0.1)" in builder_source
    assert "guidance_scale = model_config.get('guidance_scale', 3.0)" in builder_source
    assert "num_inference_steps = model_config.get('num_inference_steps', 100)" in builder_source
    assert "condition_dropout_prob=condition_dropout_prob" in builder_source
    assert "guidance_scale=guidance_scale" in builder_source
    assert "num_inference_steps=num_inference_steps" in builder_source

    assert "self.condition_dropout_prob = condition_dropout_prob" in network_source
    assert "self.guidance_scale = guidance_scale" in network_source
    assert "self.num_inference_steps = num_inference_steps" in network_source
    assert "guidance_scale=1.0" in network_source
    assert "condition_dropout_prob=self.condition_dropout_prob" in network_source
    assert "guidance_scale=self.guidance_scale" in network_source
    assert "num_inference_steps=self.num_inference_steps" in network_source
    assert "guidance_scale=3.0,\n                                                    train=train" not in network_source
    assert "\n            num_inference_steps = 100\n" not in network_source


def test_training_logs_conditional_and_cfg_validation_noise_mse_separately():
    source = read(TRAIN_SOURCE)

    assert (
        "loss_val_conditional_mse, loss_val_cfg_mse, recon_voxel = raw_model("
        in source
    )
    assert (
        'writer.add_scalar("val_loss_conditional", '
        "val_loss_conditional_mean, start_epoch)"
        in source
    )
    assert 'writer.add_scalar("val_loss", val_loss_cfg_mean, start_epoch)' in source


def test_training_uses_configured_codevae_checkpoint_and_freezes_autoencoder():
    source = read(TRAIN_SOURCE)
    config_source = read(CONFIG_SOURCE)
    yaml_source = read(YAML_SOURCE)

    assert "def load_autoencoder_state_from_checkpoint" in source
    assert "vae_checkpoint = train_hypers.get('vae_checkpoint', '')" in source
    assert "freeze_autoencoder = train_hypers.get('freeze_autoencoder', True)" in source
    assert "strict_vae_load = train_hypers.get('strict_vae_load', True)" in source
    assert "for param in my_model.model_part.autoencoder.parameters()" in source
    assert "Code_Li_bev_one_to_one" not in source
    assert "dataset_config['label_mapping']" in source
    assert "def invalid_mask_path" in source
    assert "'/home/SSC_CODE/Data/data_odometry_voxel_all" not in source

    assert "Optional(\"vae_checkpoint\"): Str()" in config_source
    assert "Optional(\"freeze_autoencoder\"): Bool()" in config_source
    assert "Optional(\"strict_vae_load\"): Bool()" in config_source
    assert "vae_checkpoint:" in yaml_source
    assert "freeze_autoencoder: True" in yaml_source
    assert "strict_vae_load: True" in yaml_source


def test_dataset_roots_are_configurable_from_yaml():
    builder_source = read(DATA_BUILDER_SOURCE)
    config_source = read(CONFIG_SOURCE)
    yaml_source = read(YAML_SOURCE)

    assert "Optional(\"data_root\"): Str()" in config_source
    assert "Optional(\"kitti_root\"): Str()" in config_source
    assert "Optional(\"vae_feature_root\"): Str()" in config_source
    assert "Optional(\"image_condition_root\"): Str()" in config_source
    assert "Optional(\"partial_condition_root\"): Str()" in config_source
    assert "Optional(\"gt_root\"): Str()" in config_source
    assert "Optional(\"invalid_root\"): Str()" in config_source

    assert "def resolve_dataloader_data_path" in builder_source
    assert "resolve_feature_root" in builder_source
    assert "image_condition_path" in builder_source
    assert "partial_condition_path" in builder_source
    assert "gt_path" in builder_source

    assert "data_root:" in yaml_source
    assert "kitti_root:" in yaml_source
    assert "vae_feature_root:" in yaml_source
    assert "image_condition_root: \"Image_transform_Voxel_Condition_Features\"" in yaml_source
    assert "partial_condition_root: \"Mask2Former_Partial_Condition_Features\"" in yaml_source
    assert "gt_root: \"data_odometry_voxels_all\"" in yaml_source
    assert "invalid_root: \"data_odometry_voxels_all\"" in yaml_source
