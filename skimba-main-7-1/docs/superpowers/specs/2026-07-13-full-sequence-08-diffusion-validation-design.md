# Full Sequence 08 Diffusion Validation Design

## Goal

Add a standalone evaluation entry point that loads the currently best diffusion
checkpoint and evaluates every frame in SemanticKITTI validation sequence 08.
The command must never resume training and must not inherit the active
`val_data_loader.frame_divisor: 10` sampling rule.

## Command Contract

The new entry point is
`scripts/evaluation/evaluate_diffusion_checkpoint.py`.

It reads `config/semantickitti_autoencoder.yaml` by default and accepts:

- `--config-path` to select another compatible configuration;
- `--checkpoint` to explicitly select a diffusion checkpoint;
- `--seed`, defaulting to `20260713`, to control Python, NumPy, PyTorch, and CUDA
  random-number generators;
- `--device`, defaulting to `cuda:0`;
- `--output-dir` to override the report destination.

When `--checkpoint` is absent, the script scans
`train_params.model_save_path/best_*.pth`. It accepts the training filename form
`best_<epoch>_<semantic-mIoU>.pth`, selects the greatest encoded mIoU, and uses
the greatest epoch as the tie-breaker. Missing, malformed, or ambiguous input
must produce a clear error rather than silently evaluating another file.

## Data and Model Setup

The script reuses the active project builders and checkpoint-loading contract.
Before constructing the validation loader, it copies the validation data-loader
configuration, removes `frame_divisor`, and enforces `imageset: val`,
`batch_size: 1`, and `shuffle: false`. The label-mapping `valid` split must resolve
to sequence 08; otherwise the script fails instead of evaluating an unexpected
split.

The diffusion checkpoint is loaded strictly, including latent-normalization
metadata validation. The configured semantic VAE checkpoint is then loaded using
the same key normalization and strictness rules as the training entry point. The
model is moved to the requested device, switched to evaluation mode, and run
under `torch.no_grad()`.

The initial latent noise and validation timestep remain random samples, matching
the current model's inference path. Seeding makes those samples repeatable across
runs with the same command. This controls random inputs but does not promise
bitwise equality for CUDA kernels that are nondeterministic.

## Evaluation Flow

For each sequence 08 frame, the script:

1. loads the raw VAE latent, 24-channel partial condition, 64-channel image
   condition, dense ground truth, and invalid mask;
2. runs the existing validation branch with the configured CFG scale and DDIM
   inference-step count;
3. computes the diagnostic epsilon MSE and decoded-logit cross entropy;
4. converts logits to semantic predictions with channel-wise `argmax`;
5. excludes ground-truth label 255 and invalid voxels;
6. accumulates one global 20-class confusion matrix.

After all frames, it reports class IoU for classes 1 through 19, semantic mIoU
over those classes, and occupancy completion IoU computed as occupied
intersection divided by occupied union. Checkpoint ranking is not performed by
this script; it evaluates exactly one resolved checkpoint.

## Outputs

Unless `--output-dir` is supplied, outputs are written under

`<model_save_path>/full_validation/<checkpoint-stem>/seed_<seed>/`.

The directory contains:

- `metrics.json`, with the checkpoint path, configuration path, seed, frame
  count, per-class IoU, semantic mIoU, completion IoU, mean epsilon MSE, and mean
  cross entropy;
- `report.txt`, with the same core results in a concise human-readable form.

Per-frame `.label` predictions are not saved by default because a complete
sequence export would consume substantial storage and is not required for metric
revalidation.

## Error Handling

The command exits nonzero when no valid checkpoint exists, an explicit
checkpoint is missing, the validation split is not exactly sequence 08, required
data or invalid masks are missing, the validation loader is empty, checkpoint
loading is incompatible, or any final aggregate metric is non-finite. Reports
are written only after a complete successful evaluation so partial runs cannot
be mistaken for final evidence.

## Verification

Test-driven implementation will cover checkpoint discovery and tie-breaking,
explicit checkpoint override, removal of `frame_divisor`, sequence-08 enforcement,
seed setup, confusion-derived metrics, and report serialization. Local
verification will include focused unit tests, the relevant project test suite,
syntax compilation, and whitespace checks. End-to-end sequence 08 results,
CUDA/Mamba execution, runtime, and model-quality conclusions remain unavailable
until the script is run on the Linux GPU server with external data and
checkpoints.
