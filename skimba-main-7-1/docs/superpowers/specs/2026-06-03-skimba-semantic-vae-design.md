# Skimba Semantic VAE Architecture Design

## Goal

Adapt `skimba-main` so its diffusion model can load and freeze the 20-class
semantic VAE pretrained in `Code_VAE_SSC`, while leaving the current diffusion
training flow unchanged.

This first step changes only the VAE architecture boundary in `skimba-main`.
It does not replace the precomputed diffusion training features, does not add
online semantic-label encoding during diffusion training, and does not change
the dataloader contract used by `train_diffusion_network_2.py`.

## Current Boundary

`skimba-main` builds a latent diffusion model through
`builder/model_builder_3D_Voxel_unet_diffusion.py`. That builder constructs:

- `SegMamba` as the denoising network
- `stable_diffusion.models.autoencoder.AutoEncoderKL` as the latent decoder
- `LatentDiffusion` as the wrapper that owns both modules
- `network/cylinder_3D_Unet_mamba_diffusion.py` as the outer training wrapper

The training script already loads a pretrained VAE into
`my_model.model_part.autoencoder` and freezes its parameters. This behavior
should stay in place.

The current VAE is still occupancy-shaped:

- decoder output is hard-coded to `FinalConv(num_outs=2)`
- encoder input is hard-coded to `Linear(4, 16)`, meaning one occupancy feature
  plus three coordinates
- `num_class` and `semantic_embed_dim` are not passed into the VAE constructor

## Target Boundary

`skimba-main` should expose the same VAE architecture knobs used by
`Code_VAE_SSC`:

- `num_class: 20` for semantic logits
- optional `semantic_embed_dim`, defaulting to `num_class` when omitted
- encoder input width equal to `semantic_embed_dim + 3`
- decoder output width equal to `num_class`

The VAE checkpoint loading path should continue to use `strict=True`, assuming
the `skimba-main` VAE module now matches the corresponding `Code_VAE_SSC`
semantic VAE state dict.

## Design

Update `stable_diffusion/models/autoencoder.py` in `skimba-main` to mirror the
semantic VAE boundary from `Code_VAE_SSC`:

- add `num_class` and `semantic_embed_dim` parameters to `AutoEncoderKL`
- pass those values through `build_encoder` and `build_decoder`
- store `self.num_class` and `self.semantic_embed_dim`
- make `Encoder.conv_input` use `semantic_embed_dim + 3`
- make `Decoder.out_conv` use `FinalConv(num_outs=num_class)`

Update `builder/model_builder_3D_Voxel_unet_diffusion.py` so the diffusion
builder reads `num_class` and optional `semantic_embed_dim` from config and
passes both into `AutoEncoderKL`.

Update `config/semantickitti_autoencoder.yaml` minimally so the VAE architecture
declares `num_class: 20` and `semantic_embed_dim: 8`. The existing data paths
and diffusion-specific settings remain unchanged.

Update `config/config.py` so `semantic_embed_dim` is an optional model config
field. This keeps the older occupancy-style config shape valid while allowing
the semantic VAE configuration to load through the strict schema.

Apply the numerical stability fix already present in `Code_VAE_SSC` by
clamping `GaussianDistribution.log_var` to `[-20, 10]` in
`stable_diffusion/modules/distributions.py`.

## Tests

Add lightweight source or unit tests in `skimba-main/tests` to verify:

- the VAE builder passes `num_class` and `semantic_embed_dim`
- `AutoEncoderKL` stores the semantic settings
- encoder input uses `semantic_embed_dim + 3`
- decoder output uses `num_class`
- the strict config schema accepts optional `semantic_embed_dim`
- `GaussianDistribution` clamps `log_var`

These tests should not require CUDA, `spconv`, the dataset, or external
checkpoints. If imports are too heavy for the local environment, source-level
tests are acceptable for this first architecture-boundary change.

## Out Of Scope

- changing the diffusion dataloader
- changing `train_diffusion_network_2.py` data tuple structure
- adding online semantic VAE pretraining inside `skimba-main`
- changing `LatentDiffusion` sampling or denoising logic
- loading old binary occupancy checkpoints with `strict=True`
