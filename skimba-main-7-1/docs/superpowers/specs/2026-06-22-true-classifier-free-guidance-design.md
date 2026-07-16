# True Classifier-Free Guidance Design

## Status

This design supersedes `2026-06-20-cfg-per-sample-element-dropout-design.md`.
The older document remains unchanged as historical evidence of the element-wise
condition-corruption approach.

## Goal

Align training and inference with standard classifier-free guidance (CFG): train
the denoiser on a mixture of complete fused conditions and the same all-zero fused
condition used by the inference-time unconditional branch.

## Data Flow

1. Encode partial and image conditions independently.
2. Concatenate the encoded conditions along the channel dimension.
3. During training, sample one Bernoulli keep/drop decision per sample with a
   broadcastable `[B, 1, 1, 1, 1]` mask. A dropped sample receives an all-zero
   fused condition; a kept sample is unchanged and is not rescaled.
4. Pass the original noisy VAE latent and the masked fused condition to the
   denoiser once. Training always uses `guidance_scale=1.0`.
5. During inference, duplicate the same noisy VAE latent for an all-zero fused
   condition and the complete fused condition. Combine denoiser outputs as
   `epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)`.

The condition mask never changes the VAE latent. CFG does not change network
dropout, the VAE, schedulers, optimization, or data configuration.

## Configuration

The active YAML model configuration owns these values:

- `condition_dropout_prob: 0.1`
- `guidance_scale: 3.0`
- `num_inference_steps: 100`

The builder passes them to the training/validation wrapper. The training path uses
only `condition_dropout_prob` and fixes guidance to `1.0`; validation and sampling
use `guidance_scale` and `num_inference_steps`.

## Verification

Behavioral tests cover mask shape and within-sample consistency, approximate drop
frequency, probability boundaries, one training denoiser call, joint partial/image
dropping, unchanged noisy latents, the zero unconditional branch, latent reuse,
the CFG equation, and live YAML-to-wrapper parameter wiring.

Local macOS CPU tests establish only Python/PyTorch behavior available in this
workspace. Linux/CUDA runtime, memory use, convergence, and performance remain
unavailable until server validation.
