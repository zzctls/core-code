# VAE Checkpoint Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible command that evaluates every `best_*.pth` VAE checkpoint on the same deterministic sample of up to 20 frames per sequence, recommends a checkpoint, and emits evidence-rich reports.

**Architecture:** Keep dataset-independent sampling, aggregation, ranking, and report serialization in `scripts/audit/vae_audit.py`. Keep project/GPU integration and CLI handling in `scripts/audit/audit_vae_checkpoints.py`, reusing the existing semantic VAE loader and frame evaluator. Tests exercise pure logic directly and inject a fake evaluator into the orchestration layer so they do not require CUDA or real checkpoints.

**Tech Stack:** Python 3, pathlib, argparse, csv/json, NumPy, PyTorch project helpers, pytest.

---

## File Map

- Create `scripts/audit/__init__.py`: package marker.
- Create `scripts/audit/vae_audit.py`: manifests, confusion aggregation, summaries, ranking, configured-checkpoint comparison, and report writers.
- Create `scripts/audit/audit_vae_checkpoints.py`: CLI, config/path resolution, checkpoint and label discovery, GPU evaluator adapter, orchestration, and exit code.
- Create `tests/test_vae_audit.py`: pure behavioral tests.
- Create `tests/test_audit_vae_checkpoints_script.py`: CLI/orchestration regression tests.
- Modify `README_RUN.md`: server command and report interpretation.

### Task 1: Deterministic per-sequence manifest

**Files:**
- Create: `scripts/audit/__init__.py`
- Create: `scripts/audit/vae_audit.py`
- Create: `tests/test_vae_audit.py`

- [ ] **Step 1: Write failing manifest tests**

```python
from pathlib import Path

from scripts.audit.vae_audit import build_frame_manifest


def test_manifest_samples_each_sequence_deterministically(tmp_path):
    for sequence in ("00", "08"):
        voxel_dir = tmp_path / "sequences" / sequence / "voxels"
        voxel_dir.mkdir(parents=True)
        for index in range(30):
            (voxel_dir / f"{index:06d}.label").touch()

    first = build_frame_manifest(tmp_path, frames_per_sequence=20, seed=17)
    second = build_frame_manifest(tmp_path, frames_per_sequence=20, seed=17)

    assert first == second
    assert {row["sequence"] for row in first} == {"00", "08"}
    assert sum(row["sequence"] == "00" for row in first) == 20
    assert sum(row["sequence"] == "08" for row in first) == 20


def test_manifest_keeps_all_frames_in_short_sequence(tmp_path):
    voxel_dir = tmp_path / "sequences" / "01" / "voxels"
    voxel_dir.mkdir(parents=True)
    for index in range(3):
        (voxel_dir / f"{index:06d}.label").touch()

    manifest = build_frame_manifest(tmp_path, frames_per_sequence=20, seed=0)
    assert [row["frame"] for row in manifest] == ["000000", "000001", "000002"]
```

- [ ] **Step 2: Verify RED**

Run: `cd core-code/skimba-main-7-1 && pytest -q tests/test_vae_audit.py`

Expected: collection fails because `scripts.audit.vae_audit` does not exist.

- [ ] **Step 3: Implement the minimal manifest API**

```python
def build_frame_manifest(gt_root, frames_per_sequence=20, seed=0, sequences=None):
    gt_root = Path(gt_root)
    requested = set(sequences) if sequences else None
    manifest = []
    for voxel_dir in sorted((gt_root / "sequences").glob("*/voxels")):
        sequence = voxel_dir.parent.name
        if requested is not None and sequence not in requested:
            continue
        frames = sorted(voxel_dir.glob("*.label"))
        if len(frames) > frames_per_sequence:
            rng = random.Random(f"{seed}:{sequence}")
            frames = sorted(rng.sample(frames, frames_per_sequence))
        manifest.extend(
            {"sequence": sequence, "frame": path.stem, "label_path": str(path)}
            for path in frames
        )
    if not manifest:
        raise ValueError(f"No label frames found under {gt_root / 'sequences'}")
    return manifest
```

- [ ] **Step 4: Verify GREEN**

Run: `cd core-code/skimba-main-7-1 && pytest -q tests/test_vae_audit.py`

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add core-code/skimba-main-7-1/scripts/audit core-code/skimba-main-7-1/tests/test_vae_audit.py
git commit -m "feat: add deterministic VAE audit manifest"
```

### Task 2: Confusion-based metric aggregation

**Files:**
- Modify: `scripts/audit/vae_audit.py`
- Modify: `tests/test_vae_audit.py`

- [ ] **Step 1: Write failing aggregation tests**

```python
from scripts.audit.vae_audit import aggregate_rows, confusion_from_labels


def test_aggregate_uses_accumulated_confusion_not_frame_miou_average():
    rows = [
        {"sequence": "00", "ce_sum": 2.0, "valid_voxels": 2, "correct_voxels": 1,
         "confusion": confusion_from_labels([1, 1], [1, 2], 3).tolist(),
         "occupancy_intersection": 1, "occupancy_union": 2,
         "latent_mean": 0.0, "latent_std": 1.0, "latent_rms": 1.0,
         "latent_l2": 2.0, "latent_abs_max": 2.0, "latent_finite": True, "logits_finite": True},
        {"sequence": "00", "ce_sum": 1.0, "valid_voxels": 1, "correct_voxels": 1,
         "confusion": confusion_from_labels([2], [2], 3).tolist(),
         "occupancy_intersection": 1, "occupancy_union": 1,
         "latent_mean": 0.0, "latent_std": 1.0, "latent_rms": 1.0,
         "latent_l2": 2.0, "latent_abs_max": 3.0, "latent_finite": True, "logits_finite": True},
    ]
    summary = aggregate_rows(rows, num_classes=3, rare_class_count=1)
    assert summary["valid_voxels"] == 3
    assert summary["mean_ce"] == 1.0
    assert summary["occupancy_iou"] == 100.0 * 2 / 3
    assert summary["per_class_iou"][1:] == [50.0, 50.0]
    assert summary["semantic_miou"] == 50.0
```

- [ ] **Step 2: Verify RED**

Run: `cd core-code/skimba-main-7-1 && pytest -q tests/test_vae_audit.py::test_aggregate_uses_accumulated_confusion_not_frame_miou_average`

Expected: FAIL because aggregation functions are absent.

- [ ] **Step 3: Implement aggregation**

Implement `confusion_from_labels`, `per_class_iou_percent`, and `aggregate_rows`. Sum confusion matrices and occupancy counts. Calculate `mean_ce` as total `ce_sum / valid_voxels`, accuracy as total correct/valid, and latent scalar fields as frame means while using the maximum `latent_abs_max`. Include `nonfinite_latent_frames` and `nonfinite_logit_frames`.

```python
def confusion_from_labels(target, prediction, num_classes):
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
    encoded = num_classes * target + prediction
    return np.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)
```

- [ ] **Step 4: Verify GREEN**

Run: `cd core-code/skimba-main-7-1 && pytest -q tests/test_vae_audit.py`

Expected: all manifest and metric tests pass.

- [ ] **Step 5: Commit**

```bash
git add core-code/skimba-main-7-1/scripts/audit/vae_audit.py core-code/skimba-main-7-1/tests/test_vae_audit.py
git commit -m "feat: aggregate VAE reconstruction evidence"
```

### Task 3: Ranking and configured-checkpoint comparison

**Files:**
- Modify: `scripts/audit/vae_audit.py`
- Modify: `tests/test_vae_audit.py`

- [ ] **Step 1: Write failing ranking tests**

```python
from scripts.audit.vae_audit import compare_configured_checkpoint, rank_candidates


def test_ranking_rejects_failed_or_nonfinite_candidate_before_miou():
    rows = [
        {"checkpoint": "/c.pth", "status": "error", "semantic_miou": 100.0, "flags": []},
        {"checkpoint": "/b.pth", "status": "ok", "semantic_miou": 99.0,
         "rare_class_miou": 98.0, "mean_ce": 0.1, "occupancy_iou": 99.0,
         "flags": ["latent_contains_nonfinite"]},
        {"checkpoint": "/a.pth", "status": "ok", "semantic_miou": 90.0,
         "rare_class_miou": 80.0, "mean_ce": 0.2, "occupancy_iou": 95.0, "flags": []},
    ]
    assert rank_candidates(rows)[0]["checkpoint"] == "/a.pth"


def test_configured_checkpoint_delta_is_explicit():
    ranked = [
        {"checkpoint": "/best.pth", "semantic_miou": 92.0, "rare_class_miou": 70.0,
         "mean_ce": 0.2, "occupancy_iou": 96.0},
        {"checkpoint": "/current.pth", "semantic_miou": 90.0, "rare_class_miou": 68.0,
         "mean_ce": 0.3, "occupancy_iou": 95.0},
    ]
    comparison = compare_configured_checkpoint(ranked, "/current.pth")
    assert comparison["available"] is True
    assert comparison["deltas"]["semantic_miou"] == 2.0
```

- [ ] **Step 2: Verify RED**

Run: `cd core-code/skimba-main-7-1 && pytest -q tests/test_vae_audit.py -k 'ranking or configured'`

Expected: FAIL because ranking and comparison functions are absent.

- [ ] **Step 3: Implement exact ranking and deltas**

Implement the design's lexicographic order with finite-safe fallbacks and a resolved-path comparison for the configured checkpoint. Produce deltas for semantic mIoU, rare-class mIoU, mean CE, and occupancy IoU; define each delta as `recommended - configured`.

- [ ] **Step 4: Verify GREEN**

Run: `cd core-code/skimba-main-7-1 && pytest -q tests/test_vae_audit.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add core-code/skimba-main-7-1/scripts/audit/vae_audit.py core-code/skimba-main-7-1/tests/test_vae_audit.py
git commit -m "feat: rank audited VAE checkpoints"
```

### Task 4: JSON, CSV, and Markdown reports

**Files:**
- Modify: `scripts/audit/vae_audit.py`
- Modify: `tests/test_vae_audit.py`

- [ ] **Step 1: Write a failing report test**

Create a two-checkpoint audit result containing global and per-sequence summaries, call `write_reports`, and assert that these files exist: `vae_audit.json`, `vae_audit_summary.csv`, `vae_audit_frames.csv`, and `vae_audit_report.md`. Assert JSON preserves the sampling manifest and Markdown contains `Recommended checkpoint`, `Configured checkpoint`, and the weakest sequence.

- [ ] **Step 2: Verify RED**

Run: `cd core-code/skimba-main-7-1 && pytest -q tests/test_vae_audit.py -k report`

Expected: FAIL because `write_reports` is absent.

- [ ] **Step 3: Implement report serialization**

Add JSON-safe conversion for NumPy values, fixed CSV field lists, and Markdown rendering. The Markdown ranking table must include status, flags, semantic mIoU, rare-class mIoU, CE, occupancy IoU, and checkpoint name. Add a per-sequence table for the recommended checkpoint sorted by semantic mIoU ascending.

- [ ] **Step 4: Verify GREEN**

Run: `cd core-code/skimba-main-7-1 && pytest -q tests/test_vae_audit.py`

Expected: all tests pass and `git diff --check` is clean.

- [ ] **Step 5: Commit**

```bash
git add core-code/skimba-main-7-1/scripts/audit/vae_audit.py core-code/skimba-main-7-1/tests/test_vae_audit.py
git commit -m "feat: write VAE audit reports"
```

### Task 5: CLI and isolated checkpoint evaluation

**Files:**
- Create: `scripts/audit/audit_vae_checkpoints.py`
- Create: `tests/test_audit_vae_checkpoints_script.py`

- [ ] **Step 1: Write failing CLI/orchestration tests**

Test that `discover_checkpoints(directory)` returns only sorted `best_*.pth` files and errors on an empty directory. Test `run_audit(args, evaluate_checkpoint=fake)` where the fake returns valid rows for one checkpoint and raises for another; assert the valid checkpoint is recommended, the failure is recorded, and every fake invocation receives the identical manifest.

- [ ] **Step 2: Verify RED**

Run: `cd core-code/skimba-main-7-1 && pytest -q tests/test_audit_vae_checkpoints_script.py`

Expected: collection fails because the CLI module does not exist.

- [ ] **Step 3: Implement CLI and orchestration**

Define arguments from the design, including comma-separated `--sequences`. Resolve YAML paths through `scripts.data.config_paths.load_training_config`. Discover the GT root from `data_root / gt_root` unless overridden. Build the manifest once before looping checkpoints. Catch each checkpoint exception and add `{"status": "error", "error": str(exc)}` without stopping valid candidates.

The production evaluator must reuse the existing checkpoint/model loaders and the existing single-frame evaluator:

```python
from scripts.data.export_skimba_semantic_vae_latents import (
    build_learning_map_lut,
    load_autoencoder_from_checkpoint,
    load_checkpoint,
    load_semantic_embedding_from_checkpoint,
)
from scripts.data.select_best_semantic_vae import evaluate_label_file
```

Load each model once, call `evaluate_label_file` for every manifest row, and attach `frame_confusion.tolist()` as `confusion`. Attach `ce_sum = frame_row["ce"] * frame_row["valid_voxels"]` so global CE is voxel-weighted. Preserve the returned occupancy intersection/union, correct/valid voxel counts, and latent statistics. Aggregate all rows globally and again after grouping by `sequence`. This leaves the existing selection script's published behavior unchanged.

- [ ] **Step 4: Add strict integrity and exit behavior**

Return `2` when no candidate succeeds or the recommended checkpoint has non-finite tensor flags; otherwise return `0`. Validate expected latent/logit shapes and run checkpoint loading with the existing strict semantic VAE loader.

- [ ] **Step 5: Verify GREEN**

Run: `cd core-code/skimba-main-7-1 && pytest -q tests/test_audit_vae_checkpoints_script.py tests/test_vae_audit.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add core-code/skimba-main-7-1/scripts/audit/audit_vae_checkpoints.py core-code/skimba-main-7-1/tests/test_audit_vae_checkpoints_script.py
git commit -m "feat: add VAE checkpoint audit command"
```

### Task 6: Server usage documentation

**Files:**
- Modify: `README_RUN.md`
- Modify: `tests/test_audit_vae_checkpoints_script.py`

- [ ] **Step 1: Write a failing documentation regression test**

Assert `README_RUN.md` names `scripts/audit/audit_vae_checkpoints.py`, `--checkpoint-dir`, `--frames-per-sequence 20`, all four report filenames, and explains that posterior mean and the same frame manifest are used for every checkpoint.

- [ ] **Step 2: Verify RED**

Run: `cd core-code/skimba-main-7-1 && pytest -q tests/test_audit_vae_checkpoints_script.py -k documentation`

Expected: FAIL because the command is not documented.

- [ ] **Step 3: Add the server command**

```bash
python scripts/audit/audit_vae_checkpoints.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --checkpoint-dir /mnt/data/datasets/kitti/odometry/skimba_data/model_save_path/VAE \
  --frames-per-sequence 20 \
  --seed 20260621 \
  --output-dir results/vae_audit
```

Document GPU/runtime expectations, short-sequence behavior, report files, selection priority, and how to compare `recommended` with the YAML checkpoint.

- [ ] **Step 4: Verify GREEN and commit**

Run: `cd core-code/skimba-main-7-1 && pytest -q tests/test_audit_vae_checkpoints_script.py`

Expected: all tests pass.

```bash
git add core-code/skimba-main-7-1/README_RUN.md core-code/skimba-main-7-1/tests/test_audit_vae_checkpoints_script.py
git commit -m "docs: document VAE checkpoint audit"
```

### Task 7: Full verification and durable memory

**Files:**
- Inspect and conditionally modify: `docs/agent-memory/LESSONS.md`
- Inspect and conditionally modify: `docs/agent-memory/EXPERIMENTS.md`

- [ ] **Step 1: Run focused tests**

Run: `cd core-code/skimba-main-7-1 && pytest -q tests/test_vae_audit.py tests/test_audit_vae_checkpoints_script.py`

Expected: all focused tests pass with no warnings caused by the new code.

- [ ] **Step 2: Run the available project suite**

Run: `cd core-code/skimba-main-7-1 && pytest -q`

Expected: all locally runnable tests pass. If optional CUDA/spconv dependencies prevent collection, record the exact command and traceback; do not claim those tests passed.

- [ ] **Step 3: Run CLI smoke checks**

Run: `cd core-code/skimba-main-7-1 && python scripts/audit/audit_vae_checkpoints.py --help`

Expected: exit `0` and display every documented option.

Run the unit-backed fake orchestration test again to verify a failed checkpoint cannot abort valid candidates.

- [ ] **Step 4: Inspect changes**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only intended audit, test, documentation, plan, and justified memory files are changed.

- [ ] **Step 5: Update memory only if evidence is durable**

If implementation establishes a reusable pitfall or the real server audit produces reproducible checkpoint results, merge a concise evidence-labelled entry into `LESSONS.md` or `EXPERIMENTS.md`. Do not record planned or locally unavailable server metrics as verified.

- [ ] **Step 6: Commit final memory changes, if any**

```bash
git add docs/agent-memory/LESSONS.md docs/agent-memory/EXPERIMENTS.md
git commit -m "docs: record VAE audit evidence"
```

Skip this commit when the task produces no durable new knowledge.
