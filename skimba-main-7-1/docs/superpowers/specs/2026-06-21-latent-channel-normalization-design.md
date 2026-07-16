# Reversible Per-Channel Latent Normalization Design

## Scope and invariants

The diffusion model may optionally operate on normalized semantic VAE latents while the pretrained VAE and existing raw float32 latent files remain unchanged. The supported latent contract is posterior-mean tensors with shape `[8, 64, 64, 8]`; batched tensors are `[B, 8, 64, 64, 8]`. Condition features are outside this feature and must never be normalized.

The default configuration is disabled and therefore numerically identical to the existing raw-latent path.

## Components

1. `scripts/data/compute_latent_channel_stats.py` reads only sequence IDs listed in `split.train` from the configured SemanticKITTI label-mapping YAML. It reads raw float32 latent files, converts each chunk to float64, and merges per-channel count, mean, and second central moment using the parallel Welford equations. It emits JSON containing mean, population standard deviation, sample count, per-channel and total element counts, split identity and sequence list, latent shape, dtype, export mode, variance convention, source paths, and algorithm.
2. `LatentDiffusion` owns persistent buffers shaped `[1, 8, 1, 1, 1]` for mean/std and a persistent normalization-enabled marker. Construction validates shape, finiteness, and strictly positive standard deviations after applying a configured positive minimum floor. `normalize_latent`, `denormalize_latent`, and `add_noise` are the only latent-affine boundaries.
3. The diffusion builder loads the JSON only when normalization is enabled and passes statistics to `LatentDiffusion`. Disabled configuration supplies identity statistics and does not require a stats file.
4. Both training and validation call `LatentDiffusion.add_noise`, so normalization occurs immediately before DDPM noise addition. DDIM sampling remains in normalized space. The generated sample is denormalized exactly once immediately before `autoencoder.decode`.
5. Diffusion checkpoint loading compares the checkpoint's persistent normalization marker with the configured model before loading. Legacy checkpoints without the marker are classified as raw-latent checkpoints. A mismatch raises a clear `RuntimeError`; matching normalized checkpoints must also carry matching mean/std buffers. VAE-only checkpoint loading remains independent.

## Failure handling

The statistics script rejects missing files, malformed latent sizes, non-finite values, an empty train split, and overlap between train and validation/test sequence declarations. Model construction rejects missing/malformed stats JSON, non-finite means/stds, wrong channel counts, non-positive `min_std`, and negative raw std values; zero or very small nonnegative std values are floored. Checkpoint compatibility is checked before `load_state_dict` can overwrite configured buffers.

## Test strategy

CPU tests cover affine round-trip, disabled identity behavior, `[B,C,W,L,H]` broadcasting, invalid statistics, both forward branches normalizing before their scheduler call, decoder-boundary denormalization, train-only file selection, float64 streaming statistics, schema/default configuration, and raw-versus-normalized checkpoint rejection. Source-level tests are used only for the CUDA-bound wrapper call-order assertions; numeric helpers use real PyTorch/NumPy execution.
