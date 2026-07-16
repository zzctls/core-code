# Dual Validation Noise MSE Design

## Objective

Record and plot two validation noise-prediction losses without changing model
training or DDIM sampling:

- conditional validation MSE using the full condition without CFG
  extrapolation (`guidance_scale = 1.0`);
- CFG-guided validation MSE using the configured inference scale (currently
  `guidance_scale = 3.0`).

The implementation applies only to the local `skimba-main-7-1` tree. It must
not modify, stop, or restart the current server training process.

## Data Flow

During validation, the existing CFG forward pass already evaluates the
unconditional and conditional branches together. The diffusion wrapper will
optionally expose those branch predictions while preserving its existing
default return value.

The validation wrapper will use the same sampled noise, timestep, noised VAE
latent, and single CFG model evaluation to calculate:

1. `conditional_mse = mse(pred_noise_cond, sampled_noise)`;
2. `cfg_mse = mse(pred_noise_guided, sampled_noise)`.

The DDIM sampling path remains unchanged and continues to use the configured
guidance scale.

## Logging and Compatibility

- Add `val_loss_conditional` for the scale-1 conditional validation MSE.
- Preserve `val_loss` as the existing CFG-guided validation MSE so historical
  event files and plotting remain compatible.
- Existing event files without `val_loss_conditional` remain readable; their
  plots show only the available training and CFG-guided series.
- Historical conditional MSE values will not be fabricated or interpolated.

## Plotting

The noise-loss plots will use these labels:

- `Train conditional noise MSE` from `train_loss_mse_mean`;
- `Validation conditional noise MSE` from `val_loss_conditional`;
- `Validation CFG-guided noise MSE (s=<configured value>)` from `val_loss` when
  the scale is available, otherwise a scale-neutral CFG-guided label.

Validation series will retain sparse markers at actual validation epochs.

## Non-goals

- No change to the training objective, gradients, optimizer, scheduler,
  condition dropout, latent normalization, or checkpoint format.
- No extra denoiser forward pass during validation.
- No change to DDIM sampling, semantic cross-entropy, Completion IoU, or mIoU.
- No backfilling of historical conditional validation MSE in this change.

## Verification

Tests will establish that:

1. default CFG prediction behavior remains unchanged;
2. validation can obtain conditional and guided predictions from one CFG
   denoiser evaluation;
3. the validation wrapper calculates both MSE values from the same noise;
4. TensorBoard receives both validation scalar tags;
5. plotting accepts both new logs and legacy logs;
6. DDIM sampling still uses the configured guidance scale.

