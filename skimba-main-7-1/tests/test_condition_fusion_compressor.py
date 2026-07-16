import torch

from network.cylinder_3D_Unet_mamba_diffusion import ConditionFusionCompressor


def test_condition_fusion_compressor_reduces_raw_conditions_to_denoiser_channels():
    compressor = ConditionFusionCompressor(
        image_channels=64,
        partial_channels=24,
        hidden_channels=44,
        out_channels=16,
    )
    image = torch.ones((2, 64, 4, 3, 2))
    partial = torch.ones((2, 24, 4, 3, 2))

    fused = compressor(partial, image)

    assert compressor.projection[0].in_channels == 88
    assert compressor.projection[0].out_channels == 44
    assert compressor.projection[0].kernel_size == (1, 1, 1)
    assert compressor.projection[0].padding == (0, 0, 0)
    assert compressor.projection[2].in_channels == 44
    assert compressor.projection[2].out_channels == 16
    assert compressor.projection[2].kernel_size == (1, 1, 1)
    assert compressor.projection[2].padding == (0, 0, 0)
    assert fused.shape == (2, 16, 4, 3, 2)
