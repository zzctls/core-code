from pathlib import Path

import pytest
import torch
from torch import nn

from stable_diffusion.models.latent_diffusion import LatentDiffusion


ROOT = Path(__file__).resolve().parents[1]
NETWORK_SOURCE = ROOT / "network" / "cylinder_3D_Unet_mamba_diffusion.py"
BUILDER_SOURCE = ROOT / "builder" / "model_builder_3D_Voxel_unet_diffusion.py"
CONFIG_SCHEMA_SOURCE = ROOT / "config" / "config.py"
CONFIG_YAML_SOURCE = ROOT / "config" / "semantickitti_autoencoder.yaml"


class RecordingNoiseScheduler:
    def __init__(self):
        self.original_samples = None

    def add_noise(self, original_samples, timesteps, noise):
        self.original_samples = original_samples.detach().clone()
        return original_samples + noise


def make_model(*, enabled=True, mean=None, std=None, min_std=1e-6, channels=8):
    scheduler = RecordingNoiseScheduler()
    model = LatentDiffusion(
        nn.Identity(),
        nn.Identity(),
        scheduler,
        object(),
        latent_channels=channels,
        latent_normalization_enabled=enabled,
        latent_mean=mean,
        latent_std=std,
        latent_min_std=min_std,
    )
    return model, scheduler


def test_normalize_denormalize_round_trip_and_buffer_shape():
    mean = torch.arange(8, dtype=torch.float32)
    std = torch.arange(1, 9, dtype=torch.float32)
    model, _ = make_model(mean=mean, std=std)
    z_raw = torch.randn((3, 8, 2, 3, 4), dtype=torch.float32)

    recovered = model.denormalize_latent(model.normalize_latent(z_raw))

    assert model.latent_mean.shape == (1, 8, 1, 1, 1)
    assert model.latent_std.shape == (1, 8, 1, 1, 1)
    assert "latent_mean" in model.state_dict()
    assert "latent_std" in model.state_dict()
    torch.testing.assert_close(recovered, z_raw)


def test_disabled_normalization_is_identity():
    model, scheduler = make_model(enabled=False, mean=None, std=None)
    z_raw = torch.randn((2, 8, 2, 2, 2))
    noise = torch.randn_like(z_raw)

    normalized = model.normalize_latent(z_raw)
    denormalized = model.denormalize_latent(z_raw)
    model.add_noise(z_raw, torch.tensor([1, 2]), noise)

    torch.testing.assert_close(normalized, z_raw)
    torch.testing.assert_close(denormalized, z_raw)
    torch.testing.assert_close(scheduler.original_samples, z_raw)


def test_channel_statistics_broadcast_across_batch_and_space():
    mean = [float(channel) for channel in range(8)]
    std = [float(channel + 1) for channel in range(8)]
    model, _ = make_model(mean=mean, std=std)
    z_raw = torch.zeros((2, 8, 2, 1, 3))
    for channel in range(8):
        z_raw[:, channel] = mean[channel] + 2.0 * std[channel]

    z_norm = model.normalize_latent(z_raw)

    torch.testing.assert_close(z_norm, torch.full_like(z_norm, 2.0))


def test_add_noise_normalizes_raw_latent_before_scheduler():
    model, scheduler = make_model(mean=[2.0] * 8, std=[4.0] * 8)
    z_raw = torch.full((2, 8, 1, 1, 1), 10.0)
    noise = torch.zeros_like(z_raw)

    model.add_noise(z_raw, torch.tensor([1, 2]), noise)

    torch.testing.assert_close(scheduler.original_samples, torch.full_like(z_raw, 2.0))


def test_small_nonnegative_std_is_floored():
    model, _ = make_model(
        mean=[0.0] * 8,
        std=[0.0] + [1e-12] * 7,
        min_std=1e-6,
    )

    torch.testing.assert_close(model.latent_std, torch.full_like(model.latent_std, 1e-6))


@pytest.mark.parametrize(("mean", "std"), [(None, [1.0] * 8), ([0.0] * 8, None)])
def test_enabled_normalization_requires_mean_and_std(mean, std):
    with pytest.raises(ValueError, match="requires both mean and std"):
        make_model(mean=mean, std=std)


@pytest.mark.parametrize(
    ("mean", "std", "min_std", "message"),
    [
        ([0.0] * 7, [1.0] * 8, 1e-6, "mean"),
        ([0.0] * 8, [1.0] * 7, 1e-6, "std"),
        ([0.0] * 7 + [float("nan")], [1.0] * 8, 1e-6, "finite"),
        ([0.0] * 8, [1.0] * 7 + [float("inf")], 1e-6, "finite"),
        ([0.0] * 8, [1.0] * 7 + [-1.0], 1e-6, "non-negative"),
        ([0.0] * 8, [1.0] * 8, 0.0, "min_std"),
    ],
)
def test_invalid_statistics_are_rejected(mean, std, min_std, message):
    with pytest.raises(ValueError, match=message):
        make_model(mean=mean, std=std, min_std=min_std)


def test_checkpoint_normalization_must_match_configured_model():
    normalized_model, _ = make_model(mean=[1.0] * 8, std=[2.0] * 8)
    raw_model, _ = make_model(enabled=False)

    raw_model.validate_checkpoint_normalization({})
    normalized_model.validate_checkpoint_normalization(normalized_model.state_dict())
    with pytest.raises(RuntimeError, match="raw-latent diffusion checkpoint"):
        normalized_model.validate_checkpoint_normalization({})
    with pytest.raises(RuntimeError, match="normalization setting"):
        raw_model.validate_checkpoint_normalization(
            {"model_part.latent_normalization_enabled": torch.tensor(True)}
        )


def test_normalized_checkpoint_statistics_must_match_configuration():
    model, _ = make_model(mean=[1.0] * 8, std=[2.0] * 8)
    state_dict = {
        "model_part.latent_normalization_enabled": torch.tensor(True),
        "model_part.latent_mean": torch.zeros((1, 8, 1, 1, 1)),
        "model_part.latent_std": torch.full((1, 8, 1, 1, 1), 2.0),
    }

    with pytest.raises(RuntimeError, match="statistics differ"):
        model.validate_checkpoint_normalization(state_dict)


def test_training_and_validation_normalize_before_add_noise_and_decode_raw_latent():
    source = NETWORK_SOURCE.read_text(encoding="utf-8")

    assert source.count("self.model_part.add_noise(") == 2
    assert "self.model_part.noise_scheduel_ddpm.add_noise(" not in source
    denormalize_index = source.index("self.model_part.denormalize_latent(")
    decode_index = source.index("self.model_part.autoencoder.decode(")
    assert denormalize_index < decode_index


def test_builder_and_default_config_expose_enabled_normalization():
    builder_source = BUILDER_SOURCE.read_text(encoding="utf-8")
    schema_source = CONFIG_SCHEMA_SOURCE.read_text(encoding="utf-8")
    yaml_source = CONFIG_YAML_SOURCE.read_text(encoding="utf-8")

    assert 'Optional("latent_normalization")' in schema_source
    assert '"enabled": Bool()' in schema_source
    assert '"stats_path": Str()' in schema_source
    assert 'Optional("min_std"): Float()' in schema_source
    assert "latent_normalization:" in yaml_source
    assert "enabled: True" in yaml_source
    assert 'stats_path: "/mnt/data/projects/skimba-main-7-1/latent_channel_stats.json"' in yaml_source
    assert "min_std: 1e-6" in yaml_source
    assert "load_latent_normalization_config" in builder_source
