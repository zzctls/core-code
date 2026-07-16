import math
from typing import Optional

import torch
from torch import nn
from tqdm import tqdm


class LatentDiffusion(nn.Module):
    """Latent diffusion wrapper with CFG-compatible condition dropout training."""

    def __init__(
        self,
        unet,
        autoencoder,
        noise_scheduel_ddpm,
        noise_scheduel_ddim,
        latent_channels: int = 8,
        latent_normalization_enabled: bool = False,
        latent_mean=None,
        latent_std=None,
        latent_min_std: float = 1e-6,
    ):
        super().__init__()
        self.unet = unet
        self.autoencoder = autoencoder
        self.noise_scheduel_ddpm = noise_scheduel_ddpm
        self.noise_scheduel_ddim = noise_scheduel_ddim

        if not math.isfinite(latent_min_std) or latent_min_std <= 0:
            raise ValueError("latent min_std must be finite and positive")
        if latent_normalization_enabled and (latent_mean is None or latent_std is None):
            raise ValueError("Enabled latent normalization requires both mean and std")
        mean = self._channel_vector(latent_mean, latent_channels, "mean", default=0.0)
        std = self._channel_vector(latent_std, latent_channels, "std", default=1.0)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Latent normalization mean/std must contain only finite values")
        if torch.any(std < 0):
            raise ValueError("Latent normalization std values must be non-negative")
        std = std.clamp_min(float(latent_min_std))

        buffer_shape = (1, latent_channels, 1, 1, 1)
        self.register_buffer(
            "latent_normalization_enabled",
            torch.tensor(bool(latent_normalization_enabled), dtype=torch.bool),
        )
        self.register_buffer("latent_mean", mean.reshape(buffer_shape))
        self.register_buffer("latent_std", std.reshape(buffer_shape))

    @staticmethod
    def _channel_vector(value, channels: int, name: str, default: float) -> torch.Tensor:
        if channels <= 0:
            raise ValueError("latent_channels must be positive")
        if value is None:
            return torch.full((channels,), default, dtype=torch.float32)
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim != 1 or tensor.numel() != channels:
            raise ValueError(
                f"Latent normalization {name} must contain exactly {channels} channel values"
            )
        return tensor

    @property
    def uses_latent_normalization(self) -> bool:
        return bool(self.latent_normalization_enabled.item())

    def _validate_latent_shape(self, latent: torch.Tensor) -> None:
        if latent.dim() != 5:
            raise ValueError("Latent tensor must be [B, C, W, L, H]")
        if latent.shape[1] != self.latent_mean.shape[1]:
            raise ValueError(
                f"Latent tensor has {latent.shape[1]} channels; "
                f"expected {self.latent_mean.shape[1]}"
            )

    def normalize_latent(self, z_raw: torch.Tensor) -> torch.Tensor:
        if not self.uses_latent_normalization:
            return z_raw
        self._validate_latent_shape(z_raw)
        return (z_raw - self.latent_mean) / self.latent_std

    def denormalize_latent(self, z_norm: torch.Tensor) -> torch.Tensor:
        if not self.uses_latent_normalization:
            return z_norm
        self._validate_latent_shape(z_norm)
        return z_norm * self.latent_std + self.latent_mean

    def add_noise(
        self,
        z_raw: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        z_norm = self.normalize_latent(z_raw)
        return self.noise_scheduel_ddpm.add_noise(
            original_samples=z_norm,
            timesteps=timesteps,
            noise=noise,
        )

    @staticmethod
    def _checkpoint_value(state_dict, suffix):
        matches = [value for key, value in state_dict.items() if key.endswith(suffix)]
        if len(matches) > 1:
            raise RuntimeError(f"Diffusion checkpoint contains duplicate {suffix} entries")
        return matches[0] if matches else None

    def validate_checkpoint_normalization(self, state_dict) -> None:
        marker = self._checkpoint_value(state_dict, "latent_normalization_enabled")
        checkpoint_enabled = bool(marker.item()) if marker is not None else False
        configured_enabled = self.uses_latent_normalization

        if configured_enabled and marker is None:
            raise RuntimeError(
                "Cannot resume normalized training from a legacy raw-latent diffusion checkpoint. "
                "Start a new diffusion run or disable latent normalization."
            )
        if checkpoint_enabled != configured_enabled:
            raise RuntimeError(
                "Diffusion checkpoint normalization setting does not match the configured model."
            )
        if not configured_enabled:
            return

        checkpoint_mean = self._checkpoint_value(state_dict, "latent_mean")
        checkpoint_std = self._checkpoint_value(state_dict, "latent_std")
        if checkpoint_mean is None or checkpoint_std is None:
            raise RuntimeError("Normalized diffusion checkpoint is missing mean/std buffers")
        statistics_match = (
            checkpoint_mean.shape == self.latent_mean.shape
            and checkpoint_std.shape == self.latent_std.shape
            and torch.equal(checkpoint_mean.cpu(), self.latent_mean.detach().cpu())
            and torch.equal(checkpoint_std.cpu(), self.latent_std.detach().cpu())
        )
        if not statistics_match:
            raise RuntimeError(
                "Diffusion checkpoint latent normalization statistics differ from configuration."
            )

    def _sample_condition_keep_mask(
        self,
        context_emb_all: torch.Tensor,
        condition_dropout_prob: float,
    ) -> torch.Tensor:
        mask_shape = (context_emb_all.shape[0], 1, 1, 1, 1)
        if condition_dropout_prob <= 0:
            return torch.ones(mask_shape, device=context_emb_all.device, dtype=torch.bool)
        if condition_dropout_prob >= 1:
            return torch.zeros(mask_shape, device=context_emb_all.device, dtype=torch.bool)

        sample_dropout = torch.rand(mask_shape, device=context_emb_all.device)
        condition_keep_mask = sample_dropout >= condition_dropout_prob
        return condition_keep_mask

    def pred_noise(
        self,
        noised_sample: torch.Tensor,
        context_emb_all: torch.Tensor,
        time_step: torch.Tensor,
        guidance_scale: float = 1.0,
        train: bool = True,
        condition_dropout_prob: float = 0.1,
        return_cfg_components: bool = False,
    ):
        """Predict noise for training or CFG sampling.

        Training:
        - drop the complete fused condition for randomly selected samples in one
          denoiser pass; dropped samples use the same zero condition as CFG.

        Sampling:
        - evaluate zero and full conditions on the same noisy latent, then combine
          their predicted noise with the standard CFG equation.
        """
        if context_emb_all.dim() != 5:
            raise ValueError("Condition tensor must be [B, C, W, L, H]")

        if train:
            if return_cfg_components:
                raise ValueError(
                    "CFG components are only available when train=False"
                )
            condition_keep_mask = self._sample_condition_keep_mask(
                context_emb_all,
                condition_dropout_prob,
            )
            training_context_emb_all = context_emb_all * condition_keep_mask
            return self.unet(noised_sample, training_context_emb_all, time_step)

        t_in = torch.cat([time_step] * 2)
        x_in = torch.cat([noised_sample] * 2)
        c_in_empty = torch.zeros_like(context_emb_all)
        c_in_all = torch.cat([c_in_empty, context_emb_all], dim=0)
        pred_noise_uncond, pred_noise_cond = torch.chunk(
            self.unet(x_in, c_in_all, t_in),
            2,
            dim=0,
        )
        pred_noise_guided = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
        if return_cfg_components:
            return pred_noise_guided, pred_noise_uncond, pred_noise_cond
        return pred_noise_guided

    def sample(
        self,
        noised_sample: torch.Tensor,
        condition_features_change: torch.Tensor,
        guidance_scale: float,
        num_inference_steps: Optional[int],
        train: bool = False,
    ):
        bsz = noised_sample.shape[0]
        x = noised_sample

        pytorch_device = x.device
        self.noise_scheduel_ddim.set_timesteps(num_inference_steps, pytorch_device)
        time_steps = self.noise_scheduel_ddim.timesteps

        progress_bar = tqdm(total=len(time_steps))
        for _, timestep in enumerate(time_steps):
            time_step = x.new_full((bsz,), timestep, dtype=torch.long)
            pred_noise = self.pred_noise(
                noised_sample=x,
                time_step=time_step,
                context_emb_all=condition_features_change,
                guidance_scale=guidance_scale,
                train=train,
            )
            x = self.noise_scheduel_ddim.step(
                model_output=pred_noise,
                timestep=timestep,
                sample=x,
            )
            x = x["prev_sample"]
            progress_bar.update(1)
        progress_bar.close()
        return x
