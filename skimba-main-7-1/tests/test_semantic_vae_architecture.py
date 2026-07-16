from pathlib import Path

import torch

from stable_diffusion.modules.distributions import GaussianDistribution


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_autoencoder_exposes_codevae_semantic_constructor_boundary():
    source = read_source("stable_diffusion/models/autoencoder.py")

    assert "num_class=2" in source
    assert "semantic_embed_dim=None" in source
    assert "self.num_class = num_class" in source
    assert "self.semantic_embed_dim = semantic_embed_dim or num_class" in source
    assert "num_class=self.num_class" in source
    assert "semantic_embed_dim=self.semantic_embed_dim" in source


def test_autoencoder_encoder_and_decoder_match_codevae_semantic_shapes():
    source = read_source("stable_diffusion/models/autoencoder.py")

    assert "input_channels = semantic_embed_dim + 3" in source
    assert "nn.Linear(input_channels, 16)" in source
    assert "FinalConv(num_outs=num_class" in source
    assert "FinalConv(num_outs=2" not in source
    assert "nn.Linear(4, 16)" not in source


def test_diffusion_builder_passes_semantic_vae_config_to_autoencoder():
    source = read_source("builder/model_builder_3D_Voxel_unet_diffusion.py")

    assert "num_class = model_config['num_class']" in source
    assert "semantic_embed_dim = model_config.get('semantic_embed_dim', None)" in source
    assert "num_class=num_class" in source
    assert "semantic_embed_dim=semantic_embed_dim" in source


def test_config_schema_and_default_config_accept_semantic_embedding():
    schema_source = read_source("config/config.py")
    config_source = read_source("config/semantickitti_autoencoder.yaml")

    assert "Optional(\"semantic_embed_dim\"): Int()" in schema_source
    assert "num_class: 20" in config_source
    assert "semantic_embed_dim: 8" in config_source
    assert "VAE_Encoder_Features_Semantic20" in config_source
    assert "VAE_Encoder_Features_One_To_One" not in config_source


def test_gaussian_distribution_preserves_unclamped_log_variance():
    moments = torch.tensor(
        [1.0, 2.0, -25.0, 12.0],
        dtype=torch.float32,
    ).reshape(1, 4, 1, 1, 1)

    distribution = GaussianDistribution(moments)

    torch.testing.assert_close(
        distribution.mean.flatten(),
        torch.tensor([1.0, 2.0]),
    )
    torch.testing.assert_close(
        distribution.log_var.flatten(),
        torch.tensor([-25.0, 12.0]),
    )
