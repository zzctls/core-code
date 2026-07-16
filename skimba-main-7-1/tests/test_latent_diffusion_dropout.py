import torch

from stable_diffusion.models.latent_diffusion import LatentDiffusion


class RecordingDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, noised_sample, fused_condition, time_step):
        self.calls.append(
            (
                noised_sample.detach().clone(),
                fused_condition.detach().clone(),
                time_step.detach().clone(),
            )
        )
        return torch.zeros_like(noised_sample)


class CFGFormulaDenoiser(RecordingDenoiser):
    def forward(self, noised_sample, fused_condition, time_step):
        super().forward(noised_sample, fused_condition, time_step)
        is_unconditional = torch.all(fused_condition == 0, dim=(1, 2, 3, 4))
        branch_value = torch.where(
            is_unconditional,
            noised_sample.new_tensor(2.0),
            noised_sample.new_tensor(5.0),
        )
        return branch_value[:, None, None, None, None].expand_as(noised_sample)


def make_diffusion(denoiser):
    return LatentDiffusion(denoiser, None, None, None)


def test_condition_dropout_samples_one_broadcastable_mask_value_per_sample():
    torch.manual_seed(7)
    context = torch.ones((64, 16, 4, 3, 2))

    mask = LatentDiffusion._sample_condition_keep_mask(None, context, 0.5)

    assert mask.dtype == torch.bool
    assert mask.shape == (64, 1, 1, 1, 1)
    broadcast_mask = mask.expand_as(context)
    assert all(sample.unique().numel() == 1 for sample in broadcast_mask)


def test_condition_dropout_drops_about_ten_percent_of_large_batch_samples():
    torch.manual_seed(20260622)
    context = torch.ones((20_000, 2, 1, 1, 1))

    mask = LatentDiffusion._sample_condition_keep_mask(None, context, 0.1)

    dropped_fraction = (~mask).float().mean().item()
    assert 0.085 < dropped_fraction < 0.115


def test_condition_dropout_probability_boundaries_use_broadcastable_masks():
    context = torch.ones((2, 3, 2, 2, 2))

    keep_all = LatentDiffusion._sample_condition_keep_mask(None, context, 0.0)
    drop_all = LatentDiffusion._sample_condition_keep_mask(None, context, 1.0)

    assert keep_all.shape == (2, 1, 1, 1, 1)
    assert drop_all.shape == (2, 1, 1, 1, 1)
    assert torch.all(keep_all)
    assert not torch.any(drop_all)


def test_training_drops_both_conditions_together_without_rescaling_or_latent_change(
    monkeypatch,
):
    denoiser = RecordingDenoiser()
    diffusion = make_diffusion(denoiser)
    noised_sample = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2, 1)
    fused_condition = torch.full((2, 3, 2, 2, 1), 7.0)
    time_step = torch.tensor([4, 9])
    keep_mask = torch.tensor([False, True]).reshape(2, 1, 1, 1, 1)

    monkeypatch.setattr(
        diffusion,
        "_sample_condition_keep_mask",
        lambda context, probability: keep_mask,
    )

    diffusion.pred_noise(
        noised_sample,
        fused_condition,
        time_step,
        guidance_scale=1.0,
        train=True,
        condition_dropout_prob=0.1,
    )

    assert len(denoiser.calls) == 1
    seen_latent, seen_condition, seen_time = denoiser.calls[0]
    torch.testing.assert_close(seen_latent, noised_sample)
    torch.testing.assert_close(seen_condition[0], torch.zeros_like(fused_condition[0]))
    torch.testing.assert_close(seen_condition[1], fused_condition[1])
    torch.testing.assert_close(seen_time, time_step)


def test_inference_uses_zero_and_full_conditions_on_the_same_noisy_latent_and_cfg_formula():
    denoiser = CFGFormulaDenoiser()
    diffusion = make_diffusion(denoiser)
    noised_sample = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2, 1)
    fused_condition = torch.full((2, 3, 2, 2, 1), 7.0)
    time_step = torch.tensor([4, 9])

    prediction = diffusion.pred_noise(
        noised_sample,
        fused_condition,
        time_step,
        guidance_scale=3.0,
        train=False,
    )

    assert len(denoiser.calls) == 1
    seen_latent, seen_condition, seen_time = denoiser.calls[0]
    torch.testing.assert_close(seen_latent[:2], noised_sample)
    torch.testing.assert_close(seen_latent[2:], noised_sample)
    torch.testing.assert_close(seen_condition[:2], torch.zeros_like(fused_condition))
    torch.testing.assert_close(seen_condition[2:], fused_condition)
    torch.testing.assert_close(seen_time, torch.cat((time_step, time_step)))
    torch.testing.assert_close(prediction, torch.full_like(noised_sample, 11.0))


def test_inference_guidance_scale_one_still_evaluates_both_cfg_branches():
    denoiser = CFGFormulaDenoiser()
    diffusion = make_diffusion(denoiser)
    noised_sample = torch.ones((1, 1, 1, 1, 1))
    fused_condition = torch.full((1, 2, 1, 1, 1), 7.0)
    time_step = torch.tensor([4])

    prediction = diffusion.pred_noise(
        noised_sample,
        fused_condition,
        time_step,
        guidance_scale=1.0,
        train=False,
    )

    assert len(denoiser.calls) == 1
    seen_latent, seen_condition, seen_time = denoiser.calls[0]
    torch.testing.assert_close(seen_latent, torch.cat((noised_sample, noised_sample)))
    torch.testing.assert_close(seen_condition[0], torch.zeros_like(seen_condition[0]))
    torch.testing.assert_close(seen_condition[1], fused_condition[0])
    torch.testing.assert_close(seen_time, torch.cat((time_step, time_step)))
    torch.testing.assert_close(prediction, torch.full_like(noised_sample, 5.0))


def test_inference_can_return_cfg_components_without_extra_denoiser_call():
    denoiser = CFGFormulaDenoiser()
    diffusion = make_diffusion(denoiser)
    noised_sample = torch.randn(2, 1, 2, 2, 2)
    fused_condition = torch.randn(2, 3, 2, 2, 2)
    time_step = torch.tensor([1, 2])

    guided, unconditional, conditional = diffusion.pred_noise(
        noised_sample=noised_sample,
        context_emb_all=fused_condition,
        time_step=time_step,
        guidance_scale=3.0,
        train=False,
        return_cfg_components=True,
    )

    assert len(denoiser.calls) == 1
    torch.testing.assert_close(unconditional, torch.full_like(unconditional, 2.0))
    torch.testing.assert_close(conditional, torch.full_like(conditional, 5.0))
    torch.testing.assert_close(guided, torch.full_like(guided, 11.0))
