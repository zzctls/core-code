import json
from pathlib import Path

from network.cylinder_3D_Unet_mamba_diffusion import get_model_class
from stable_diffusion.models.autoencoder import AutoEncoderKL
from stable_diffusion.models.skimba import SegMamba
from stable_diffusion.models.latent_diffusion import LatentDiffusion
from diffusers import DDPMScheduler
from diffusers import DDIMScheduler


EXPECTED_LATENT_SHAPE = [8, 64, 64, 8]


def load_latent_normalization_config(model_config):
    normalization = model_config.get('latent_normalization', {})
    enabled = normalization.get('enabled', False)
    min_std = normalization.get('min_std', 1e-6)
    if not enabled:
        return {
            'enabled': False,
            'mean': None,
            'std': None,
            'min_std': min_std,
        }

    stats_path = normalization.get('stats_path', '')
    if not stats_path:
        raise ValueError(
            "model_params.latent_normalization.stats_path is required when enabled"
        )
    path = Path(stats_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Latent normalization stats file not found: {path}")
    with path.open('r', encoding='utf-8') as handle:
        stats = json.load(handle)

    if stats.get('split') != 'train':
        raise ValueError("Latent normalization stats must be computed from split=train")
    if stats.get('export_mode') != 'mean':
        raise ValueError("Latent normalization stats must use export_mode=mean")
    if stats.get('latent_shape') != EXPECTED_LATENT_SHAPE:
        raise ValueError(
            f"Latent normalization stats shape {stats.get('latent_shape')} "
            f"!= expected {EXPECTED_LATENT_SHAPE}"
        )
    return {
        'enabled': True,
        'mean': stats.get('mean'),
        'std': stats.get('std'),
        'min_std': min_std,
    }

def build(model_config):
    output_shape = model_config['output_shape']
    num_class = model_config['num_class']
    semantic_embed_dim = model_config.get('semantic_embed_dim', None)
    num_input_features = model_config['num_input_features']
    init_size = model_config['init_size']
    channels_list = model_config['channels_list']
    latent_channels = model_config['latent_channels']
    condition_in_channels = model_config.get('condition_in_channels', 88)
    image_condition_channels = model_config.get('image_condition_channels', 64)
    partial_condition_channels = model_config.get('partial_condition_channels', 24)
    condition_mid_channels = model_config.get('condition_mid_channels', condition_in_channels // 2)
    condition_channels = model_config.get('condition_channels', 16)
    condition_dropout_prob = model_config.get('condition_dropout_prob', 0.1)
    guidance_scale = model_config.get('guidance_scale', 3.0)
    num_inference_steps = model_config.get('num_inference_steps', 100)
    in_channels = model_config['in_channels']
    out_channels = model_config['out_channels']
    autoencoder_num_res_blocks = model_config['autoencoder_num_res_blocks']
    autoencoder_channels_list = model_config['autoencoder_channels_list']
    auto_groups = model_config['auto_groups']
    voxel_channel = model_config['voxel_channel']
    latent_normalization = load_latent_normalization_config(model_config)

    unet = SegMamba()
    autoencoder = AutoEncoderKL(in_channels=in_channels, latent_channels=latent_channels, out_channels=out_channels,
                                autoencoder_num_res_blocks=autoencoder_num_res_blocks,
                                autoencoder_channels_list=autoencoder_channels_list, groups=auto_groups,
                                num_class=num_class, semantic_embed_dim=semantic_embed_dim)

    SchedulerDDPM = DDPMScheduler(num_train_timesteps = 1000, beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', clip_sample=False, steps_offset=1)
    SchedulerDDIM = DDIMScheduler(num_train_timesteps = 1000, beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', clip_sample=False, steps_offset=1, set_alpha_to_one=False)
    # 0.00085   0.012
    model_part = LatentDiffusion(
        unet,
        autoencoder,
        SchedulerDDPM,
        SchedulerDDIM,
        latent_channels=in_channels,
        latent_normalization_enabled=latent_normalization['enabled'],
        latent_mean=latent_normalization['mean'],
        latent_std=latent_normalization['std'],
        latent_min_std=latent_normalization['min_std'],
    )

    model = get_model_class(model_config["model_architecture"])(model_part = model_part,
                                                                sparse_shape=output_shape,
                                                                channels_list = channels_list,
                                                                num_input_features =num_input_features,
                                                                init_size = init_size,
                                                                voxel_channel = voxel_channel,
                                                                condition_in_channels=condition_in_channels,
                                                                image_condition_channels=image_condition_channels,
                                                                partial_condition_channels=partial_condition_channels,
                                                                condition_mid_channels=condition_mid_channels,
                                                                condition_channels=condition_channels,
                                                                condition_dropout_prob=condition_dropout_prob,
                                                                guidance_scale=guidance_scale,
                                                                num_inference_steps=num_inference_steps,
                                                                )
    return model
