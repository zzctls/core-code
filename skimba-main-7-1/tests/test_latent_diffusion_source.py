from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "stable_diffusion" / "models" / "latent_diffusion.py"
VARIANT = ROOT / "stable_diffusion" / "models" / "latent_diffusion_cfg_dropout_variant.py"


def test_cfg_dropout_logic_is_in_main_latent_diffusion_file():
    source = SOURCE.read_text(encoding="utf-8")

    assert not VARIANT.exists()
    assert "class LatentDiffusion(nn.Module)" in source
    assert "condition_dropout_prob" in source
    assert "element_dropout = torch.rand_like(context_emb_all)" not in source
    assert "mask_shape = (context_emb_all.shape[0], 1, 1, 1, 1)" in source
    assert "training_context_emb_all = context_emb_all * condition_keep_mask" in source
    assert "return self.unet(noised_sample, training_context_emb_all, time_step)" in source
    assert "pred_noise_dropped, pred_noise_full = torch.chunk(" not in source
    assert "return 0.5 * (pred_noise_full + pred_noise_dropped)" not in source
    assert "torch.zeros_like(context_emb_all)" in source
    assert "x_in = torch.cat([noised_sample] * 2)" in source
    assert "guidance_scale: float = 7.5" not in source
    assert "to('cuda:0')" not in source


def test_classifier_free_guidance_uses_unconditional_prediction_as_baseline():
    source = SOURCE.read_text(encoding="utf-8")

    assert (
        "pred_noise_uncond + guidance_scale * "
        "(pred_noise_cond - pred_noise_uncond)"
    ) in source
    assert (
        "pred_noise_cond + guidance_scale * "
        "(pred_noise_cond - pred_noise_uncond)"
    ) not in source
