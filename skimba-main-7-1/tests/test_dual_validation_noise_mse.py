import torch

import network.cylinder_3D_Unet_mamba_diffusion as diffusion_network


class FakeAutoencoder(torch.nn.Module):
    def decode(self, latent):
        batch, _, width, length, height = latent.shape
        return latent.new_zeros((batch, 2, width, length, height))


class FakeDiffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoder = FakeAutoencoder()
        self.pred_noise_call_count = 0
        self.return_cfg_components = None
        self.sample_guidance_scale = None

    def add_noise(self, latent, timesteps, noise):
        return latent

    def pred_noise(
        self,
        noised_sample,
        context_emb_all,
        time_step,
        guidance_scale,
        train,
        return_cfg_components=False,
    ):
        self.pred_noise_call_count += 1
        self.return_cfg_components = return_cfg_components
        unconditional = torch.full_like(noised_sample, 2.0)
        conditional = torch.full_like(noised_sample, 5.0)
        guided = torch.full_like(noised_sample, 11.0)
        return guided, unconditional, conditional

    def sample(
        self,
        noised_sample,
        condition_features_change,
        guidance_scale,
        num_inference_steps,
        train,
    ):
        self.sample_guidance_scale = guidance_scale
        return torch.zeros_like(noised_sample)

    def denormalize_latent(self, latent):
        return latent


class ZeroCondition(torch.nn.Module):
    def forward(self, partial_condition, image_condition):
        batch, _, width, length, height = partial_condition.shape
        return partial_condition.new_zeros((batch, 1, width, length, height))


def test_validation_reports_conditional_and_cfg_mse_from_one_prediction_call(
    monkeypatch,
):
    fake_diffusion = FakeDiffusion()
    model = diffusion_network.cylinder_asym(
        model_part=fake_diffusion,
        sparse_shape=None,
        channels_list=None,
        num_input_features=None,
        init_size=None,
        voxel_channel=None,
        condition_in_channels=2,
        image_condition_channels=1,
        partial_condition_channels=1,
        condition_mid_channels=1,
        condition_channels=1,
        guidance_scale=3.0,
        num_inference_steps=4,
    )
    model.condition_compressor = ZeroCondition()
    monkeypatch.setattr(
        diffusion_network.torch,
        "randn_like",
        lambda tensor: torch.ones_like(tensor),
    )

    latent = torch.zeros((1, 1, 2, 2, 2))
    partial_condition = torch.zeros_like(latent)
    image_condition = torch.zeros_like(latent)

    conditional_mse, cfg_mse, reconstruction = model(
        batch_size=1,
        val_VAE_features_change=latent,
        val_partial_features_change=partial_condition,
        val_image_features_change=image_condition,
        train=False,
    )

    torch.testing.assert_close(conditional_mse, torch.tensor(16.0))
    torch.testing.assert_close(cfg_mse, torch.tensor(100.0))
    assert reconstruction.shape == (1, 2, 2, 2, 2)
    assert fake_diffusion.pred_noise_call_count == 1
    assert fake_diffusion.return_cfg_components is True
    assert fake_diffusion.sample_guidance_scale == model.guidance_scale
