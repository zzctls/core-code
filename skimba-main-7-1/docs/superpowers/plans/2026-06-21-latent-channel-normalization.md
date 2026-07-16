# Reversible Per-Channel Latent Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional, reversible per-channel affine normalization around diffusion while preserving the pretrained VAE and raw latent files.

**Architecture:** A train-only offline script produces reproducible JSON statistics. `LatentDiffusion` owns validated persistent affine buffers and centralizes normalization around DDPM/DDIM boundaries; the training loader rejects checkpoint normalization mismatches before state loading.

**Tech Stack:** Python, NumPy, PyTorch, StrictYAML, pytest

---

### Task 1: Specify the affine and split contracts

**Files:**
- Create: `tests/test_latent_normalization.py`
- Create: `tests/test_compute_latent_channel_stats.py`
- Modify: `tests/test_joint_condition_training_source.py`

- [ ] **Step 1: Write failing numeric tests**

Instantiate `LatentDiffusion` with lightweight `nn.Identity` dependencies and assert round-trip, disabled identity, channel broadcast, buffer shape, and rejection of malformed/non-finite/non-positive statistics.

- [ ] **Step 2: Write failing integration/source tests**

Assert both `cylinder_asym.forward` branches call `model_part.add_noise`, assert `model_part.denormalize_latent` occurs before `autoencoder.decode`, and assert checkpoint loading rejects a missing/false normalization marker when the configured model is enabled.

- [ ] **Step 3: Write failing statistics tests**

Build temporary sequence directories where train and validation have distinguishable values. Assert file discovery returns only train sequence files and the float64 streaming result matches NumPy population mean/std.

- [ ] **Step 4: Verify RED**

Run: `python -m pytest tests/test_latent_normalization.py tests/test_compute_latent_channel_stats.py tests/test_joint_condition_training_source.py -q`

Expected: failures because normalization APIs, stats script, config keys, and call-site boundaries do not exist.

### Task 2: Implement the model boundary

**Files:**
- Modify: `stable_diffusion/models/latent_diffusion.py`
- Modify: `network/cylinder_3D_Unet_mamba_diffusion.py`

- [ ] **Step 1: Add validated buffers**

Extend `LatentDiffusion.__init__` with `latent_channels`, `latent_normalization_enabled`, `latent_mean`, `latent_std`, and `latent_min_std`. Validate positive floor, exact channel count, finite values, and nonnegative raw std; register floored mean/std as `[1,C,1,1,1]` float32 buffers plus a scalar bool marker.

- [ ] **Step 2: Add reversible APIs**

Implement `normalize_latent(z_raw)` as `(z_raw - mean) / std` and `denormalize_latent(z_norm)` as `z_norm * std + mean`, returning inputs unchanged when disabled. Validate five-dimensional tensors and channel count when enabled.

- [ ] **Step 3: Centralize noise addition and decoding boundary**

Implement `add_noise(z_raw, timesteps, noise)` to normalize immediately before delegating to DDPM. Replace both direct scheduler calls in `cylinder_asym.forward`; denormalize the DDIM result immediately before `autoencoder.decode`.

- [ ] **Step 4: Verify GREEN for affine/call paths**

Run: `python -m pytest tests/test_latent_normalization.py tests/test_joint_condition_training_source.py -q`

Expected: all selected tests pass.

### Task 3: Wire configuration and checkpoint safety

**Files:**
- Modify: `config/config.py`
- Modify: `config/semantickitti_autoencoder.yaml`
- Modify: `builder/model_builder_3D_Voxel_unet_diffusion.py`
- Modify: `train_diffusion_network_2.py`
- Test: `tests/test_latent_normalization.py`

- [ ] **Step 1: Add default-off StrictYAML configuration**

Define optional `model_params.latent_normalization` with required `enabled`, required `stats_path`, and optional `min_std`. Add the default YAML block with `enabled: False`, empty path, and `min_std: 1e-6`.

- [ ] **Step 2: Load reproducible JSON stats in the builder**

When enabled, require a stats path, parse JSON, require `split == "train"`, `export_mode == "mean"`, and latent shape `[8,64,64,8]`, then pass its mean/std to `LatentDiffusion`. When disabled, use identity settings without reading a file.

- [ ] **Step 3: Protect checkpoint resume**

Before `load_state_dict`, classify absent markers as legacy raw checkpoints, require enabled markers to match, and for normalized checkpoints require checkpoint mean/std buffers to equal configured values. Keep VAE-only loading unchanged.

- [ ] **Step 4: Verify GREEN for configuration and compatibility**

Run: `python -m pytest tests/test_latent_normalization.py tests/test_joint_condition_training_source.py tests/test_semantic_vae_architecture.py -q`

Expected: all selected tests pass.

### Task 4: Implement train-only float64 statistics

**Files:**
- Create: `scripts/data/compute_latent_channel_stats.py`
- Test: `tests/test_compute_latent_channel_stats.py`
- Modify: `README_RUN.md`

- [ ] **Step 1: Discover only train files**

Read the mapping YAML, reject overlap of `split.train` with valid/test, and enumerate only `<latent-root>/sequences/<train-id>/voxels/*.bin` in deterministic order.

- [ ] **Step 2: Stream float64 moments**

For each raw float32 file, validate exact shape and finiteness, convert to float64, reduce spatial elements per channel, and merge count/mean/M2 using the parallel Welford formula. Report population std as `sqrt(M2 / count)`.

- [ ] **Step 3: Emit provenance JSON**

Write mean/std, sample count, elements per channel, total elements, split/sequences, latent shape, export mode, source dtype, accumulation dtype, variance convention, algorithm, source roots, and label mapping.

- [ ] **Step 4: Document commands**

Add the stats command, YAML enablement example, and training command to `README_RUN.md`, including the warning that raw diffusion checkpoints are incompatible with enabled normalization.

- [ ] **Step 5: Verify GREEN for statistics**

Run: `python -m pytest tests/test_compute_latent_channel_stats.py -q`

Expected: all selected tests pass.

### Task 5: Full verification and evidence memory

**Files:**
- Modify if durable: `docs/agent-memory/PROJECT.md` or `docs/agent-memory/DECISIONS.md`

- [ ] **Step 1: Run focused and full CPU tests**

Run: `python -m pytest tests/test_latent_normalization.py tests/test_compute_latent_channel_stats.py -q`

Run: `python -m pytest tests -q`

- [ ] **Step 2: Run static syntax checks**

Run: `python -m compileall -q stable_diffusion/models/latent_diffusion.py network/cylinder_3D_Unet_mamba_diffusion.py builder/model_builder_3D_Voxel_unet_diffusion.py scripts/data/compute_latent_channel_stats.py train_diffusion_network_2.py config/config.py`

- [ ] **Step 3: Review the exact diff and requirements**

Run: `git diff --check`

Run: `git diff -- core-code/skimba-main-7-1 docs/agent-memory`

Confirm every numbered user requirement is either covered by code/tests or explicitly reported as requiring Linux/CUDA experiments.

- [ ] **Step 4: Update evidence memory only if durable**

Merge a concise verified entry with exact files and CPU test command; label CUDA training/performance as unavailable. Do not overwrite unrelated memory changes and do not commit.
