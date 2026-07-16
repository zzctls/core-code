# VAE Latent Reconstruction Evaluation Design

## Objective

Add a minimal, evaluation-only path that measures how much SemanticKITTI
semantic information is preserved by the pretrained VAE and its exported
posterior-mean latent files. The evaluation compares raw latent decoding with a
per-channel normalization/de-normalization round trip on the complete
SemanticKITTI validation sequence 08.

This evaluation does not train a model. It does not modify or invoke the
diffusion training loop, and it does not claim scene generation or completion
from partial observations.

## Evidence Boundary

The experiment can support two claims:

1. Decoding saved raw VAE latents reconstructs semantic voxel labels with the
   measured validation metrics.
2. Applying the configured train-only channel standardization and its inverse
   before decoding preserves reconstruction within the measured numerical and
   prediction differences.

It cannot establish that an unconditional model generates scene-specific
latents, that condition features are useful, or that diffusion is useful. The
input latent was produced from the complete semantic target, so reconstruction
metrics are information-retention evidence rather than a semantic scene
completion result.

## Scope and Constraints

- Add `scripts/evaluation/evaluate_vae_latent_reconstruction.py` as an
  independent command-line entry point.
- Leave `train_diffusion_network_2.py`, its model builder, and its training
  behavior unchanged.
- Evaluate every available frame in SemanticKITTI validation sequence 08.
- Use one shared, deterministic frame manifest for both branches.
- Read no image-condition or partial-condition files.
- Construct no SegMamba denoiser, latent-diffusion wrapper, Diffusers scheduler,
  optimizer, or training component.
- Use the same pretrained VAE decoder, ground truth, invalid mask, and metric
  definitions for both branches.
- Treat Linux/CUDA sequence-08 execution as performance evidence. Local CPU
  tests establish implementation behavior only.

## Architecture

The evaluator constructs only the semantic `AutoEncoderKL` required to load the
configured VAE checkpoint and decode an `(8, 64, 64, 8)` latent tensor into
20-class voxel logits.

For each manifest row, it loads the saved raw float32 posterior-mean latent,
the corresponding remapped ground-truth voxel labels, and the SemanticKITTI
invalid mask. It then runs two branches:

### Raw branch

```text
saved raw latent -> VAE decoder -> 20-class logits -> masked metrics
```

### Normalization round-trip branch

```text
saved raw latent
  -> (latent - train_mean) / train_std
  -> normalized_latent * train_std + train_mean
  -> VAE decoder
  -> 20-class logits
  -> masked metrics
```

The normalized tensor is not passed directly to the VAE decoder. The decoder
was trained on the raw latent distribution, so direct standardized decoding
would test a distribution mismatch rather than the reversible diffusion-boundary
transform.

## Inputs and Configuration

The CLI reads defaults from the existing training YAML and permits explicit
server-path overrides. Every run resolves the following inputs before
evaluation:

- VAE checkpoint;
- raw exported latent root;
- SemanticKITTI ground-truth voxel root;
- SemanticKITTI invalid-mask root;
- label-mapping YAML;
- train-only latent channel statistics JSON;
- output directory, using the default below when it is not specified.

Sequence defaults to `08`. The output directory defaults to a sibling artifact
tree outside the repository, consistent with existing VAE audit artifacts.
The report records resolved input paths, checkpoint identity, actual execution
device, frame count, latent shape, and statistics provenance.

## Manifest and Masking

The evaluator discovers the complete sorted sequence-08 ground-truth manifest.
For every frame it requires exactly one matching latent and invalid mask. Missing
or ambiguous counterparts fail the run; frames are never silently skipped.

Raw SemanticKITTI labels are remapped with the configured learning map. Voxels
with the configured ignore label or an invalid-mask value of one are excluded
from cross-entropy, confusion matrices, prediction disagreement, and every
reported aggregate metric.

## Metrics

Each branch reports:

- semantic mIoU, calculated from the aggregate confusion matrix and excluding
  the empty class;
- occupancy IoU;
- voxel accuracy;
- cross-entropy over valid voxels;
- IoU for all 20 classes;
- valid-voxel count;
- per-frame metrics for diagnosis.

Dataset cross-entropy is calculated as the sum of per-voxel cross-entropy over
all valid voxels divided by the total valid-voxel count. It is not the unweighted
mean of per-frame means, because frames contain different numbers of valid
voxels.

The branch comparison additionally reports:

- latent maximum and mean absolute round-trip error;
- decoded-logit maximum and mean absolute difference;
- prediction disagreement count and percentage over valid voxels.

All dataset-level IoU values come from one accumulated confusion matrix, not an
unweighted average of frame-level IoUs.

## Output Artifacts

One run writes:

- `summary.json`, containing resolved configuration, provenance, validation
  checks, aggregate metrics, per-class IoUs, and branch-comparison evidence;
- `frames.csv`, containing frame identifiers, valid-voxel counts, per-branch
  metrics, and branch disagreements;
- `report.md`, containing a concise human-readable comparison and the explicit
  evidence boundary.

Artifacts are written only after the manifest and global inputs validate. A
runtime failure returns a nonzero status and must not produce a success claim.

## Validation and Failure Handling

The evaluator fails fast when:

- any required path or frame counterpart is missing;
- a latent does not contain exactly `8 * 64 * 64 * 8` float32 values;
- latent values, statistics, decoded logits, or aggregate metrics contain
  unexpected non-finite values;
- statistics do not declare `split=train` and `export_mode=mean`;
- statistics do not declare latent shape `[8, 64, 64, 8]`;
- mean/std do not each contain eight finite values;
- any standard deviation is not strictly positive;
- prediction and target shapes differ;
- the two branches do not use the same ordered manifest.

The runtime automatically selects CUDA when available and otherwise uses CPU.
The selected device is evidence metadata, not a performance equivalence claim.

## Testing Strategy

Implementation follows test-first development. Focused tests cover:

1. raw latents are passed unchanged to the decoder;
2. the normalized branch applies the exact standardize/invert equations;
3. both branches consume the same ordered manifest;
4. invalid and ignored voxels do not enter any metric;
5. semantic mIoU excludes the empty class;
6. cross-entropy is weighted by valid-voxel count;
7. aggregate IoU is derived from the accumulated confusion matrix;
8. statistics provenance, channel count, positive standard deviation, and
   latent-shape checks reject invalid inputs;
9. missing latent or invalid-mask counterparts fail instead of skipping;
10. JSON, CSV, and Markdown artifacts agree on frame counts and aggregate
    values;
11. the evaluation entry point imports no condition, diffusion, SegMamba,
    scheduler, optimizer, or training module;
12. the existing diffusion training script remains unchanged by this work.

Local verification includes focused tests, the relevant existing VAE audit
tests, Python syntax compilation, and the project test suite to the extent its
Apple Silicon environment supports. The final performance evidence requires a
fresh full-sequence run against the server dataset and checkpoint.

## Server Result Interpretation

The report compares raw and round-trip branches without declaring a preferred
branch merely because of floating-point-scale differences. A defensible result
states the measured reconstruction quality and the measured round-trip changes.
If either branch has poor reconstruction, non-finite values, missing frames, or
material prediction disagreement, the result is reported as observed rather
than interpreted as proof of effectiveness.
