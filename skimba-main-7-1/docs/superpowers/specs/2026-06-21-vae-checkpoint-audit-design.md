# VAE Checkpoint Audit Design

## Goal

Provide one reproducible command that audits every `best_*.pth` VAE checkpoint in a user-supplied directory, measures semantic reconstruction quality on 20 frames from every available dataset sequence, and recommends the strongest checkpoint using defensible metrics.

Classifier-free guidance analysis is explicitly outside this task's scope.

## Command Interface

Add `scripts/audit/audit_vae_checkpoints.py` with these primary inputs:

- `--config-path`: training YAML; defaults to `config/semantickitti_autoencoder.yaml`.
- `--checkpoint-dir`: directory scanned non-recursively for `best_*.pth`.
- `--frames-per-sequence`: defaults to `20`.
- `--seed`: controls deterministic sampling.
- `--output-dir`: directory for the Markdown, JSON, summary CSV, and per-frame CSV reports.
- Optional path overrides for the dataset root, label mapping, and sequence selection.

The command must fail clearly when no checkpoints, label files, or usable sequences are found. A sequence with fewer than 20 frames contributes all available frames and is identified in the report.

## Sampling

Discover sequences from the configured ground-truth root. Sort sequence and frame paths before sampling. For each sequence, select at most 20 frames using a seeded deterministic sampler, so identical inputs and seed produce identical manifests.

Write the selected sequence/frame manifest into the JSON report. Every checkpoint is evaluated on exactly the same manifest. This paired design prevents checkpoint rankings from being confounded by different frame samples.

## Checkpoint and Reconstruction Audit

For every checkpoint:

1. Load the checkpoint and semantic embedding with the existing semantic VAE loading functions.
2. Require architecture-compatible strict loading and record a checkpoint-level error without aborting the remaining candidates when loading fails.
3. Build the semantic input scene using the existing label mapping and semantic embedding path.
4. Encode and decode with posterior mean, not posterior sampling, to make reconstruction comparisons deterministic.
5. Check logits and latents for non-finite values and verify expected tensor shapes.

The script reuses existing project reconstruction helpers where their behavior matches this design. New orchestration and reporting logic remains independently testable without loading the full GPU model.

## Metrics and Aggregation

Collect these metrics per frame, per sequence, and globally:

- semantic mIoU excluding the empty class;
- semantic mIoU including the empty class;
- per-class IoU;
- rare-class mIoU and minimum present-class IoU;
- occupancy IoU;
- cross-entropy;
- voxel accuracy;
- valid voxel count;
- latent mean, standard deviation, RMS, L2 norm, and absolute maximum;
- counts of frames containing non-finite logits or latents.

Global and per-sequence IoU values are computed from accumulated confusion matrices, not by averaging per-frame IoU. Cross-entropy, accuracy, and latent statistics retain explicit aggregation definitions in the JSON output.

## Ranking and Recommendation

Rank candidates lexicographically by:

1. successful, architecture-compatible evaluation;
2. absence of non-finite values and configured latent-health flags;
3. descending global semantic mIoU;
4. descending rare-class mIoU;
5. ascending mean cross-entropy;
6. descending occupancy IoU;
7. checkpoint path as a deterministic tie-breaker.

The report identifies both the recommended checkpoint and the checkpoint currently configured in YAML. When the configured checkpoint was evaluated, report absolute metric deltas. When it lies outside the scanned directory or is unavailable, state that comparison is unavailable instead of inferring a result.

## Outputs and Exit Status

The output directory contains:

- `vae_audit_report.md`: concise human-readable findings, ranking, per-sequence weaknesses, and recommendation;
- `vae_audit.json`: complete manifest, configuration, metrics, flags, errors, and ranking;
- `vae_audit_summary.csv`: one row per checkpoint;
- `vae_audit_frames.csv`: one row per checkpoint and sampled frame.

Exit status is `0` when at least one checkpoint evaluates successfully and the recommended checkpoint has no critical integrity flags. Exit status is `2` when no checkpoint can be evaluated or the recommended result contains non-finite tensors. Individual broken candidates are recorded but do not prevent comparison of valid candidates.

## Testing

Use test-driven development for the new orchestration logic:

- deterministic, per-sequence sampling and short-sequence behavior;
- identical manifests shared by every checkpoint;
- confusion-matrix aggregation and semantic/occupancy metrics;
- ranking, deterministic tie-breaking, and configured-checkpoint deltas;
- isolation of a failed checkpoint from valid candidates;
- JSON, CSV, and Markdown report contents;
- CLI validation and exit statuses.

Run focused unit tests first, then the existing project test suite that is available in the local environment. The real GPU audit remains a documented server command because the local workspace does not contain the `/mnt/data/...` datasets and checkpoints.

## Scope Boundaries

This task does not change VAE training, diffusion training, CFG behavior, dataset labels, model architecture, or checkpoints. It only adds a read-only VAE audit command, tests, and usage documentation.
