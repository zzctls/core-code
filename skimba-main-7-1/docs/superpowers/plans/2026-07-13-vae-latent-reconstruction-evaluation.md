# VAE Latent Reconstruction Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a training-free full-sequence-08 evaluator that compares raw saved VAE latent reconstruction with a train-statistics normalization/de-normalization round trip.

**Architecture:** Put pure manifest, validation, dual-branch, metric, and artifact helpers in `scripts/evaluation/vae_latent_reconstruction.py`. Keep YAML resolution, decoder-only VAE loading, device selection, and orchestration in `scripts/evaluation/evaluate_vae_latent_reconstruction.py`; both branches share one validated manifest and one decoder instance.

**Tech Stack:** Python 3.9, PyTorch, NumPy, existing StrictYAML/PyYAML config helpers, pytest, JSON, CSV, Markdown.

## Global Constraints

- Do not modify `train_diffusion_network_2.py`, its builder, or its behavior.
- Do not read condition files or import condition, SegMamba, LatentDiffusion, Diffusers scheduler, optimizer, or training paths.
- Evaluate every available SemanticKITTI validation sequence `08` frame by default.
- Decode the standardized branch only after the inverse transform.
- Aggregate CE by total valid-voxel CE sum divided by total valid-voxel count.
- Aggregate IoU from one accumulated confusion matrix.
- Preserve unrelated worktree changes and commit only task-owned paths.
- Treat local tests as implementation evidence; full metrics require the Linux/CUDA server run.

---

## File Map

- Create `scripts/evaluation/vae_latent_reconstruction.py`: pure input validation, branch comparison, metrics, and artifact writers.
- Create `scripts/evaluation/evaluate_vae_latent_reconstruction.py`: CLI and evaluation orchestration.
- Create `tests/test_vae_latent_reconstruction_evaluation.py`: behavior and dependency-boundary tests.
- Modify `README_RUN.md`: server command and evidence boundary.
- Modify `docs/agent-memory/PROJECT.md` after verification: stable tool availability.
- Modify `docs/agent-memory/EXPERIMENTS.md` only after the server run: measured results.

### Task 1: Validate statistics and construct one full-sequence manifest

**Files:**
- Create: `core-code/skimba-main-7-1/scripts/evaluation/vae_latent_reconstruction.py`
- Create: `core-code/skimba-main-7-1/tests/test_vae_latent_reconstruction_evaluation.py`

**Interfaces:**
- Consumes: `<root>/sequences/<sequence>/voxels/<frame>.<suffix>` and statistics JSON from `compute_latent_channel_stats.py`.
- Produces: `load_channel_stats(path, latent_shape) -> dict`, `build_full_manifest(gt_root, latent_root, invalid_root, sequence="08") -> list[dict]`, `load_raw_latent(path, latent_shape) -> np.ndarray`, and `load_masked_target(...) -> np.ndarray`.

- [ ] **Step 1: Write failing statistics tests**

~~~python
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "scripts" / "evaluation" / "vae_latent_reconstruction.py"


def load_core_module():
    spec = importlib.util.spec_from_file_location("vae_latent_reconstruction", CORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stats_require_train_mean_shape_and_positive_std(tmp_path):
    module = load_core_module()
    path = tmp_path / "stats.json"
    payload = {
        "split": "train",
        "export_mode": "mean",
        "latent_shape": [2, 1, 1, 2],
        "mean": [1.0, -2.0],
        "std": [0.5, 4.0],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert module.load_channel_stats(path, (2, 1, 1, 2))["std"] == [0.5, 4.0]

    for change, message in (
        ({"split": "valid"}, "split=train"),
        ({"export_mode": "sample"}, "export_mode=mean"),
        ({"latent_shape": [8, 64, 64, 8]}, "latent_shape"),
        ({"mean": [1.0]}, "mean/std"),
        ({"std": [0.5, 0.0]}, "strictly positive"),
    ):
        broken = {**payload, **change}
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            module.load_channel_stats(path, (2, 1, 1, 2))
~~~

- [ ] **Step 2: Verify RED**

Run:

~~~bash
cd core-code/skimba-main-7-1
python -m pytest tests/test_vae_latent_reconstruction_evaluation.py::test_stats_require_train_mean_shape_and_positive_std -v
~~~

Expected: FAIL because the core module/function does not exist.

- [ ] **Step 3: Implement strict statistics validation**

~~~python
"""Pure helpers for raw versus normalization-roundtrip VAE evaluation."""

import csv
import json
import math
from pathlib import Path

import numpy as np

from scripts.data.export_z0_semantic_latents import read_dense_label, read_invalid_mask


DEFAULT_LATENT_SHAPE = (8, 64, 64, 8)
SPATIAL_SHAPE = (256, 256, 32)


def load_channel_stats(path, latent_shape=DEFAULT_LATENT_SHAPE):
    latent_shape = tuple(int(value) for value in latent_shape)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("split") != "train":
        raise ValueError("Latent statistics must declare split=train")
    if payload.get("export_mode") != "mean":
        raise ValueError("Latent statistics must declare export_mode=mean")
    if tuple(payload.get("latent_shape", ())) != latent_shape:
        raise ValueError("Latent statistics latent_shape does not match evaluation")
    channels = latent_shape[0]
    mean = [float(value) for value in payload.get("mean", [])]
    std = [float(value) for value in payload.get("std", [])]
    if len(mean) != channels or len(std) != channels:
        raise ValueError(f"Latent statistics mean/std must each contain {channels} values")
    if not all(math.isfinite(value) for value in mean + std):
        raise ValueError("Latent statistics mean/std must be finite")
    if not all(value > 0.0 for value in std):
        raise ValueError("Latent statistics std must be strictly positive")
    return {**payload, "mean": mean, "std": std, "path": str(Path(path).resolve())}
~~~

- [ ] **Step 4: Verify GREEN**

Repeat Step 2. Expected: PASS.

- [ ] **Step 5: Write failing manifest, size, and invalid-mask tests**

~~~python
def create_frame(root, frame, suffix, data=b"x"):
    path = root / "sequences" / "08" / "voxels" / f"{frame}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_manifest_is_sorted_and_requires_all_counterparts(tmp_path):
    module = load_core_module()
    for frame in ("000002", "000001"):
        create_frame(tmp_path / "gt", frame, "label")
        create_frame(tmp_path / "latent", frame, "bin")
        create_frame(tmp_path / "invalid", frame, "invalid")
    manifest = module.build_full_manifest(
        tmp_path / "gt", tmp_path / "latent", tmp_path / "invalid"
    )
    assert [row["frame"] for row in manifest] == ["000001", "000002"]
    Path(manifest[0]["latent_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="Missing latent"):
        module.build_full_manifest(
            tmp_path / "gt", tmp_path / "latent", tmp_path / "invalid"
        )


def test_latent_loader_rejects_wrong_size_and_nonfinite(tmp_path):
    module = load_core_module()
    path = tmp_path / "latent.bin"
    np.array([1, 2, 3], dtype=np.float32).tofile(path)
    with pytest.raises(ValueError, match="expected 4"):
        module.load_raw_latent(path, (2, 1, 1, 2))
    np.array([1, np.nan, 3, 4], dtype=np.float32).tofile(path)
    with pytest.raises(ValueError, match="non-finite"):
        module.load_raw_latent(path, (2, 1, 1, 2))


def test_target_masks_invalid_voxels(monkeypatch):
    module = load_core_module()
    monkeypatch.setattr(
        module, "read_dense_label",
        lambda path, spatial_shape: np.array([[[0, 1, 2, 1]]], dtype=np.uint16),
    )
    monkeypatch.setattr(
        module, "read_invalid_mask",
        lambda path, spatial_shape: np.array([[[0, 1, 0, 0]]], dtype=np.uint8),
    )
    target = module.load_masked_target(
        "a.label", "a.invalid", np.array([0, 1, 2], dtype=np.uint8),
        spatial_shape=(1, 1, 4),
    )
    assert target.tolist() == [[[0, 255, 2, 1]]]
~~~

- [ ] **Step 6: Verify RED**

~~~bash
cd core-code/skimba-main-7-1
python -m pytest tests/test_vae_latent_reconstruction_evaluation.py -k "manifest or latent_loader or target_masks" -v
~~~

Expected: FAIL because the functions are absent.

- [ ] **Step 7: Implement manifest, latent, and target loading**

~~~python
def _voxel_dir(root, sequence):
    return Path(root) / "sequences" / str(sequence).zfill(2) / "voxels"


def build_full_manifest(gt_root, latent_root, invalid_root, sequence="08"):
    sequence = str(sequence).zfill(2)
    labels = sorted(_voxel_dir(gt_root, sequence).glob("*.label"))
    if not labels:
        raise FileNotFoundError(f"No labels under {_voxel_dir(gt_root, sequence)}")
    rows = []
    for label_path in labels:
        frame = label_path.stem
        latent_path = _voxel_dir(latent_root, sequence) / f"{frame}.bin"
        invalid_path = _voxel_dir(invalid_root, sequence) / f"{frame}.invalid"
        if not latent_path.is_file():
            raise FileNotFoundError(f"Missing latent for {sequence}/{frame}: {latent_path}")
        if not invalid_path.is_file():
            raise FileNotFoundError(
                f"Missing invalid mask for {sequence}/{frame}: {invalid_path}"
            )
        rows.append({
            "sequence": sequence,
            "frame": frame,
            "label_path": str(label_path),
            "latent_path": str(latent_path),
            "invalid_path": str(invalid_path),
        })
    return rows


def load_raw_latent(path, latent_shape=DEFAULT_LATENT_SHAPE):
    latent_shape = tuple(int(value) for value in latent_shape)
    raw = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(latent_shape))
    if raw.size != expected:
        raise ValueError(f"{path} has {raw.size} float32 values; expected {expected}")
    latent = raw.reshape(latent_shape)
    if not np.isfinite(latent).all():
        raise ValueError(f"{path} contains non-finite latent values")
    return latent


def load_masked_target(
    label_path, invalid_path, learning_map_lut, ignore_label=255,
    spatial_shape=SPATIAL_SHAPE,
):
    raw = read_dense_label(label_path, spatial_shape=spatial_shape)
    lower = (raw.astype(np.uint32) & 0xFFFF).reshape(-1)
    if lower.size and int(lower.max()) >= len(learning_map_lut):
        raise ValueError(f"{label_path} contains a raw label outside the LUT")
    target = learning_map_lut[lower].reshape(spatial_shape).astype(np.uint8)
    invalid = read_invalid_mask(invalid_path, spatial_shape=spatial_shape)
    if invalid is None:
        raise FileNotFoundError(f"Missing invalid mask: {invalid_path}")
    target = target.copy()
    target[invalid == 1] = ignore_label
    return target
~~~

- [ ] **Step 8: Verify Task 1 and commit**

~~~bash
cd core-code/skimba-main-7-1
python -m pytest tests/test_vae_latent_reconstruction_evaluation.py -v
git diff --check -- scripts/evaluation/vae_latent_reconstruction.py tests/test_vae_latent_reconstruction_evaluation.py
git add scripts/evaluation/vae_latent_reconstruction.py tests/test_vae_latent_reconstruction_evaluation.py
git commit -m "feat: validate VAE latent evaluation inputs"
~~~

Expected: tests PASS, whitespace check exits 0, and the commit contains only the core/test files.

### Task 2: Implement raw and normalization-round-trip evidence

**Files:**
- Modify: `core-code/skimba-main-7-1/scripts/evaluation/vae_latent_reconstruction.py`
- Modify: `core-code/skimba-main-7-1/tests/test_vae_latent_reconstruction_evaluation.py`

**Interfaces:**
- Consumes: one decoder with `decode(tensor)`, one raw latent, validated stats, and masked target.
- Produces: `decode_raw_and_roundtrip`, `frame_reconstruction_evidence`, `compare_branch_outputs`, and `aggregate_evaluation`.

- [ ] **Step 1: Write a failing decoder-input test**

~~~python
import torch


class RecordingDecoder:
    def __init__(self):
        self.inputs = []

    def decode(self, latent):
        self.inputs.append(latent.detach().clone())
        return torch.cat((latent[:, :1], -latent[:, :1]), dim=1)


def test_decoder_gets_raw_then_exact_roundtrip():
    module = load_core_module()
    decoder = RecordingDecoder()
    latent = np.array([[[[1.25, 2.75]]], [[[6.0, -2.0]]]], dtype=np.float32)
    outputs = module.decode_raw_and_roundtrip(
        decoder, latent, {"mean": [1.0, -2.0], "std": [0.5, 4.0]},
        torch.device("cpu"),
    )
    raw = torch.from_numpy(latent).unsqueeze(0)
    mean = torch.tensor([1.0, -2.0]).reshape(1, 2, 1, 1, 1)
    std = torch.tensor([0.5, 4.0]).reshape(1, 2, 1, 1, 1)
    expected = ((raw - mean) / std) * std + mean
    assert torch.equal(decoder.inputs[0], raw)
    assert torch.equal(decoder.inputs[1], expected)
    assert torch.equal(outputs["roundtrip_latent"], expected)
~~~

- [ ] **Step 2: Verify RED**

~~~bash
cd core-code/skimba-main-7-1
python -m pytest tests/test_vae_latent_reconstruction_evaluation.py::test_decoder_gets_raw_then_exact_roundtrip -v
~~~

Expected: FAIL because `decode_raw_and_roundtrip` is absent.

- [ ] **Step 3: Implement the branch transform**

~~~python
def decode_raw_and_roundtrip(autoencoder, latent, stats, device):
    import torch

    raw = torch.from_numpy(np.asarray(latent, dtype=np.float32)).unsqueeze(0).to(device)
    channels = raw.shape[1]
    mean = torch.tensor(stats["mean"], dtype=raw.dtype, device=device).reshape(
        1, channels, 1, 1, 1
    )
    std = torch.tensor(stats["std"], dtype=raw.dtype, device=device).reshape(
        1, channels, 1, 1, 1
    )
    normalized = (raw - mean) / std
    roundtrip = normalized * std + mean
    with torch.no_grad():
        raw_logits = autoencoder.decode(raw)
        roundtrip_logits = autoencoder.decode(roundtrip)
    if not torch.isfinite(raw_logits).all() or not torch.isfinite(roundtrip_logits).all():
        raise ValueError("VAE decoder produced non-finite logits")
    return {
        "raw_latent": raw,
        "roundtrip_latent": roundtrip,
        "raw_logits": raw_logits,
        "roundtrip_logits": roundtrip_logits,
    }
~~~

- [ ] **Step 4: Verify GREEN**

Repeat Step 2. Expected: PASS.

- [ ] **Step 5: Write failing masked-metric and weighted-aggregation tests**

~~~python
def test_aggregate_ce_is_voxel_weighted_and_semantic_miou_excludes_empty():
    module = load_core_module()
    first = {"confusion": [[1, 0], [0, 0]], "ce_sum": 1.0, "valid_voxels": 1}
    second = {"confusion": [[0, 0], [0, 9]], "ce_sum": 0.9, "valid_voxels": 9}
    comparison_one = {
        "latent": {"element_count": 1, "differing_elements": 0, "max_abs_error": 0.0, "abs_error_sum": 0.0},
        "logits": {"element_count": 1, "differing_elements": 0, "max_abs_error": 0.0, "abs_error_sum": 0.0},
        "prediction": {"valid_voxels": 1, "differing_voxels": 0},
    }
    comparison_nine = {
        "latent": {"element_count": 1, "differing_elements": 0, "max_abs_error": 0.0, "abs_error_sum": 0.0},
        "logits": {"element_count": 1, "differing_elements": 0, "max_abs_error": 0.0, "abs_error_sum": 0.0},
        "prediction": {"valid_voxels": 9, "differing_voxels": 0},
    }
    rows = [
        {"raw": first, "roundtrip": first, "comparison": comparison_one},
        {"raw": second, "roundtrip": second, "comparison": comparison_nine},
    ]
    result = module.aggregate_evaluation(rows, num_classes=2)
    assert result["raw"]["mean_ce"] == pytest.approx(0.19)
    assert result["raw"]["semantic_miou_percent"] == pytest.approx(100.0)
    assert result["raw"]["voxel_accuracy_percent"] == pytest.approx(100.0)


def test_prediction_disagreement_ignores_target_255():
    module = load_core_module()
    target = np.array([[[0, 1, 255, 2]]], dtype=np.uint8)
    raw_logits = torch.tensor(
        [[[[[8.0, 0.0, 0.0, 0.0]]], [[[0.0, 8.0, 8.0, 0.0]]], [[[0.0, 0.0, 0.0, 8.0]]]]]
    )
    roundtrip_logits = raw_logits.clone()
    roundtrip_logits[:, 2, 0, 0, 2] = 100.0
    comparison = module.compare_branch_outputs(
        torch.zeros(1), torch.zeros(1), raw_logits, roundtrip_logits, target
    )
    assert comparison["prediction"]["valid_voxels"] == 3
    assert comparison["prediction"]["differing_voxels"] == 0


def test_frame_evidence_rejects_prediction_target_shape_mismatch():
    module = load_core_module()
    logits = torch.zeros((1, 3, 1, 1, 3), dtype=torch.float32)
    target = np.zeros((1, 1, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="Prediction shape"):
        module.frame_reconstruction_evidence(logits, target, num_classes=3)
~~~

- [ ] **Step 6: Verify RED**

~~~bash
cd core-code/skimba-main-7-1
python -m pytest tests/test_vae_latent_reconstruction_evaluation.py -k "aggregate_ce or prediction_disagreement" -v
~~~

Expected: FAIL because comparison/aggregation functions are absent.

- [ ] **Step 7: Implement per-frame and aggregate evidence**

Implement `frame_reconstruction_evidence` with PyTorch `cross_entropy(..., reduction="sum", ignore_index=255)` and `confusion_from_labels`. Implement `compare_branch_outputs` with max/mean absolute tensor errors and valid-target prediction disagreement. Implement `aggregate_evaluation` by:

~~~python
confusion = sum(np.asarray(row[branch]["confusion"], dtype=np.int64) for row in rows)
class_iou = per_class_iou_percent(confusion)
semantic = class_iou[1:]
mean_ce = sum(row[branch]["ce_sum"] for row in rows) / sum(
    row[branch]["valid_voxels"] for row in rows
)
occupancy_intersection = int(confusion[1:, 1:].sum())
occupancy_union = int(confusion.sum() - confusion[0, 0])
voxel_accuracy = 100.0 * np.diag(confusion).sum() / confusion.sum()
~~~

For every frame, derive `semantic_miou_percent`, `occupancy_iou_percent`,
`voxel_accuracy_percent`, `mean_ce`, `valid_voxels`, and
`per_class_iou_percent` from that frame's confusion/CE evidence. Return `raw`,
`roundtrip`, and `comparison` dictionaries containing the same named metrics at
dataset level, with aggregate values recalculated from sums rather than averaged
from the frame metrics.

- [ ] **Step 8: Verify Task 2 and commit**

~~~bash
cd core-code/skimba-main-7-1
python -m pytest tests/test_vae_latent_reconstruction_evaluation.py -v
git diff --check -- scripts/evaluation/vae_latent_reconstruction.py tests/test_vae_latent_reconstruction_evaluation.py
git add scripts/evaluation/vae_latent_reconstruction.py tests/test_vae_latent_reconstruction_evaluation.py
git commit -m "feat: compare raw and normalized VAE latent decoding"
~~~

Expected: tests PASS and only task-owned paths are committed.

### Task 3: Add the decoder-only CLI and reports

**Files:**
- Create: `core-code/skimba-main-7-1/scripts/evaluation/evaluate_vae_latent_reconstruction.py`
- Modify: `core-code/skimba-main-7-1/scripts/evaluation/vae_latent_reconstruction.py`
- Modify: `core-code/skimba-main-7-1/tests/test_vae_latent_reconstruction_evaluation.py`

**Interfaces:**
- Consumes: existing `load_training_config`, `resolve_vae_root`, `resolve_gt_root`, `resolve_dataset_root`, `load_checkpoint`, `load_autoencoder_from_checkpoint`, and `build_learning_map_lut`.
- Produces: `build_argparser`, `resolve_runtime`, `run`, `main`, `write_artifacts`, `summary.json`, `frames.csv`, and `report.md`.

- [ ] **Step 1: Write failing CLI-boundary tests**

~~~python
SCRIPT = ROOT / "scripts" / "evaluation" / "evaluate_vae_latent_reconstruction.py"


def load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_vae_latent_reconstruction", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_and_forbidden_import_boundary():
    module = load_cli_module()
    args = module.build_argparser().parse_args([])
    assert args.sequence == "08"
    assert args.device == "auto"
    assert args.latent_shape == "8,64,64,8"

    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        "train_diffusion_network_2",
        "model_builder_3D_Voxel_unet_diffusion",
        "LatentDiffusion",
        "SegMamba",
        "diffusers",
        "torch.optim",
        "ConditionFusionCompressor",
    )
    assert not [name for name in forbidden if name in source]
~~~

- [ ] **Step 2: Verify RED**

~~~bash
cd core-code/skimba-main-7-1
python -m pytest tests/test_vae_latent_reconstruction_evaluation.py -k cli -v
~~~

Expected: FAIL because the CLI file is absent.

- [ ] **Step 3: Implement parser and runtime resolution**

The parser exposes:

~~~python
parser.add_argument("--config-path", default="config/semantickitti_autoencoder.yaml")
parser.add_argument("--vae-checkpoint", default="")
parser.add_argument("--latent-root", default="")
parser.add_argument("--gt-root", default="")
parser.add_argument("--invalid-root", default="")
parser.add_argument("--label-mapping", default="")
parser.add_argument("--stats-path", default="")
parser.add_argument("--output-dir", default="")
parser.add_argument("--sequence", default="08")
parser.add_argument("--latent-shape", default="8,64,64,8")
parser.add_argument("--device", default="auto")
~~~

`resolve_runtime` loads the YAML, resolves relative label-mapping paths against `PROJECT_ROOT`, uses `train_params.vae_checkpoint`, `resolve_vae_root`, `resolve_gt_root`, `dataset_params.invalid_root`, and `model_params.latent_normalization.stats_path` as defaults, and fails before decoder construction if any required path is missing.

Create decoder arguments with `SimpleNamespace` using the same fields as `verify_exported_vae_latents._model_args`, but do not load semantic embeddings because this evaluator only decodes saved latents.

- [ ] **Step 4: Verify CLI defaults GREEN**

Repeat Step 2. Expected: PASS.

- [ ] **Step 5: Write a failing artifact-consistency test**

~~~python
def test_artifacts_agree_on_frame_count_and_metrics(tmp_path):
    module = load_core_module()
    result = {
        "status": "pass",
        "num_frames": 1,
        "raw": {"semantic_miou_percent": 99.0, "occupancy_iou_percent": 98.0, "voxel_accuracy_percent": 97.0, "mean_ce": 0.01},
        "roundtrip": {"semantic_miou_percent": 99.0, "occupancy_iou_percent": 98.0, "voxel_accuracy_percent": 97.0, "mean_ce": 0.01},
        "comparison": {
            "latent": {"max_abs_error": 0.0, "mean_abs_error": 0.0},
            "logits": {"max_abs_error": 0.0, "mean_abs_error": 0.0},
            "prediction": {"differing_voxels": 0, "valid_voxels": 4, "disagreement_rate_percent": 0.0},
        },
    }
    frames = [{
        "sequence": "08",
        "frame": "000000",
        "raw": {"semantic_miou_percent": 99.0, "occupancy_iou_percent": 98.0, "voxel_accuracy_percent": 97.0, "mean_ce": 0.01, "valid_voxels": 4},
        "roundtrip": {"semantic_miou_percent": 99.0, "occupancy_iou_percent": 98.0, "voxel_accuracy_percent": 97.0, "mean_ce": 0.01, "valid_voxels": 4},
        "comparison": {"prediction": {"differing_voxels": 0}},
    }]
    paths = module.write_artifacts(result, frames, tmp_path)
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    csv_text = (tmp_path / "frames.csv").read_text(encoding="utf-8")
    assert saved["num_frames"] == 1
    assert "99.000" in report
    assert "000000" in csv_text
    assert len(csv_text.splitlines()) == 2
    assert set(paths) == {"summary", "frames", "report"}
~~~

- [ ] **Step 6: Verify RED, implement reports, verify GREEN**

Run the single test; expected FAIL because `write_artifacts` is absent. Implement
the writer with this structure:

~~~python
def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_artifacts(result, frame_rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    frames_path = output_dir / "frames.csv"
    report_path = output_dir / "report.md"
    summary_path.write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fields = (
        "sequence", "frame", "valid_voxels", "raw_mean_ce",
        "roundtrip_mean_ce", "raw_semantic_miou_percent",
        "roundtrip_semantic_miou_percent", "raw_occupancy_iou_percent",
        "roundtrip_occupancy_iou_percent", "raw_voxel_accuracy_percent",
        "roundtrip_voxel_accuracy_percent", "prediction_differing_voxels",
    )
    with frames_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in frame_rows:
            writer.writerow({
                "sequence": row["sequence"],
                "frame": row["frame"],
                "valid_voxels": row["raw"]["valid_voxels"],
                "raw_mean_ce": row["raw"]["mean_ce"],
                "roundtrip_mean_ce": row["roundtrip"]["mean_ce"],
                "raw_semantic_miou_percent": row["raw"]["semantic_miou_percent"],
                "roundtrip_semantic_miou_percent": row["roundtrip"]["semantic_miou_percent"],
                "raw_occupancy_iou_percent": row["raw"]["occupancy_iou_percent"],
                "roundtrip_occupancy_iou_percent": row["roundtrip"]["occupancy_iou_percent"],
                "raw_voxel_accuracy_percent": row["raw"]["voxel_accuracy_percent"],
                "roundtrip_voxel_accuracy_percent": row["roundtrip"]["voxel_accuracy_percent"],
                "prediction_differing_voxels": row["comparison"]["prediction"]["differing_voxels"],
            })
    raw = result["raw"]
    roundtrip = result["roundtrip"]
    prediction = result["comparison"]["prediction"]
    report_path.write_text("\n".join((
        "# VAE Latent Reconstruction Evaluation",
        "",
        f"- Frames: {result['num_frames']}",
        f"- Raw semantic mIoU: {raw['semantic_miou_percent']:.3f}%",
        f"- Round-trip semantic mIoU: {roundtrip['semantic_miou_percent']:.3f}%",
        f"- Raw occupancy IoU: {raw['occupancy_iou_percent']:.3f}%",
        f"- Round-trip occupancy IoU: {roundtrip['occupancy_iou_percent']:.3f}%",
        f"- Raw voxel accuracy: {raw['voxel_accuracy_percent']:.3f}%",
        f"- Round-trip voxel accuracy: {roundtrip['voxel_accuracy_percent']:.3f}%",
        f"- Raw CE: {raw['mean_ce']:.9f}",
        f"- Round-trip CE: {roundtrip['mean_ce']:.9f}",
        f"- Latent max/mean error: {result['comparison']['latent']['max_abs_error']:.9g}/{result['comparison']['latent']['mean_abs_error']:.9g}",
        f"- Logit max/mean error: {result['comparison']['logits']['max_abs_error']:.9g}/{result['comparison']['logits']['mean_abs_error']:.9g}",
        f"- Prediction disagreement: {prediction['differing_voxels']}/{prediction['valid_voxels']} ({prediction['disagreement_rate_percent']:.9f}%)",
        f"- Device: {result.get('device', 'unavailable')}",
        f"- VAE checkpoint: `{result.get('vae_checkpoint', 'unavailable')}`",
        f"- Latent root: `{result.get('latent_root', 'unavailable')}`",
        "",
        "This is full-target VAE information-retention evidence, not scene-completion or generation evidence.",
        "",
    )), encoding="utf-8")
    return {"summary": summary_path, "frames": frames_path, "report": report_path}
~~~

Repeat the test. Expected: PASS.

- [ ] **Step 7: Implement evaluation orchestration**

`run(args)` must execute in this order:

1. Resolve config/paths and choose `cuda:0` when `--device auto` and CUDA is available, else CPU.
2. Validate stats and build one full manifest.
3. Load the checkpoint and one `AutoEncoderKL` through `load_autoencoder_from_checkpoint`.
4. Build the learning-map LUT.
5. For every manifest row: load raw latent and masked target, decode both branches, compute per-frame evidence, and retain no prediction tensor in public output.
6. Aggregate from the complete frame list.
7. Require finite aggregate mIoU, occupancy IoU, voxel accuracy, CE, tensor
   error, and disagreement-rate values for both branches/comparison; absent
   per-class IoUs may remain `None` in JSON.
8. Write artifacts and print the report path.
9. Return the result; `main` returns 0 on success and 2 with a concise stderr message on failure.

Use this result skeleton:

~~~python
result = {
    "status": "pass",
    "config_path": str(runtime.config_path.resolve()),
    "vae_checkpoint": str(runtime.vae_checkpoint.resolve()),
    "latent_root": str(runtime.latent_root.resolve()),
    "gt_root": str(runtime.gt_root.resolve()),
    "invalid_root": str(runtime.invalid_root.resolve()),
    "stats": stats,
    "sequence": runtime.sequence,
    "device": str(device),
    "latent_shape": list(runtime.latent_shape),
    "num_frames": len(frame_rows),
    **aggregate_evaluation(frame_rows, num_classes),
}
~~~

- [ ] **Step 8: Verify CLI, syntax, and whitespace**

~~~bash
cd core-code/skimba-main-7-1
python -m pytest tests/test_vae_latent_reconstruction_evaluation.py -v
PYTHONPYCACHEPREFIX=/private/tmp/skimba-pyc python -m py_compile scripts/evaluation/vae_latent_reconstruction.py scripts/evaluation/evaluate_vae_latent_reconstruction.py
python scripts/evaluation/evaluate_vae_latent_reconstruction.py --help
git diff --check -- scripts/evaluation/vae_latent_reconstruction.py scripts/evaluation/evaluate_vae_latent_reconstruction.py tests/test_vae_latent_reconstruction_evaluation.py
~~~

Expected: tests PASS, compilation/help exit 0, and whitespace check exits 0.

- [ ] **Step 9: Commit**

~~~bash
cd core-code/skimba-main-7-1
git add scripts/evaluation/vae_latent_reconstruction.py scripts/evaluation/evaluate_vae_latent_reconstruction.py tests/test_vae_latent_reconstruction_evaluation.py
git commit -m "feat: evaluate VAE latent reconstruction without training"
~~~

Expected: only evaluator code/tests are committed.

### Task 4: Document and fully verify the implementation

**Files:**
- Modify: `core-code/skimba-main-7-1/README_RUN.md`
- Modify: `docs/agent-memory/PROJECT.md`
- Modify: `core-code/skimba-main-7-1/tests/test_vae_latent_reconstruction_evaluation.py`

**Interfaces:**
- Consumes: completed evaluator.
- Produces: server command, evidence warning, and durable verified project context.

- [ ] **Step 1: Write a failing runbook-contract test**

~~~python
def test_runbook_documents_training_free_full_sequence_evaluation():
    text = (ROOT / "README_RUN.md").read_text(encoding="utf-8")
    assert "evaluate_vae_latent_reconstruction.py" in text
    assert "不训练" in text
    assert "sequence 08" in text
    assert "normalize" in text
    assert "denormalize" in text
~~~

- [ ] **Step 2: Verify RED**

~~~bash
cd core-code/skimba-main-7-1
python -m pytest tests/test_vae_latent_reconstruction_evaluation.py::test_runbook_documents_training_free_full_sequence_evaluation -v
~~~

Expected: FAIL because the command is not documented.

- [ ] **Step 3: Add the runbook command and interpretation**

~~~~markdown
## 纯 VAE/latent 重建评估（不训练）

该入口不读取 condition，不构建 diffusion，也不执行 optimizer。默认遍历完整
SemanticKITTI validation sequence 08，对比 raw latent decode 与
`normalize -> denormalize -> decode`。

~~~bash
python scripts/evaluation/evaluate_vae_latent_reconstruction.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --device cuda:0 \
  --output-dir /path/to/project-master-results/skimba-main-7-1/vae_latent_reconstruction
~~~

输出为 `summary.json`、`frames.csv` 和 `report.md`。结果验证完整语义目标经
VAE latent 后的信息保真度及标准化边界的数值可逆性；它不是无条件生成、
scene completion、condition 或 diffusion 有效性的证据。
~~~~

- [ ] **Step 4: Run focused and full local verification**

~~~bash
cd core-code/skimba-main-7-1
python -m pytest tests/test_vae_latent_reconstruction_evaluation.py tests/test_verify_exported_vae_latents_script.py tests/test_vae_audit.py -v
MPLBACKEND=Agg PYTHONPYCACHEPREFIX=/private/tmp/skimba-pyc python -m pytest tests -q
~~~

Expected: both commands exit 0. If not, report exact failures and fix only regressions introduced by this work.

- [ ] **Step 5: Prove the training/dependency boundary**

Run separately:

~~~bash
cd /Users/zhao/project-master
git log --format= --name-only 18d66b7..HEAD -- core-code/skimba-main-7-1/train_diffusion_network_2.py
rg -n "train_diffusion_network_2|model_builder_3D_Voxel_unet_diffusion|LatentDiffusion|SegMamba|diffusers|torch\.optim|ConditionFusionCompressor" core-code/skimba-main-7-1/scripts/evaluation/evaluate_vae_latent_reconstruction.py core-code/skimba-main-7-1/scripts/evaluation/vae_latent_reconstruction.py
git diff --check
~~~

Expected: the first command prints nothing and exits 0, the third command exits
0, and `rg` prints no matches and exits 1. The worktree already had user changes
to the training script before this feature; proving that the new commits contain
no training-script path is the correct boundary check. Do not revert those user
changes.

- [ ] **Step 6: Update durable memory**

Merge this verified statement into `docs/agent-memory/PROJECT.md`:

~~~markdown
- **Verified — `skimba-main-7-1` provides a training-free full-sequence VAE latent reconstruction evaluator.** It decodes saved raw posterior-mean latents and a train-statistics normalize/de-normalize round trip through the same frozen semantic VAE, applies SemanticKITTI invalid masking, and reports aggregate confusion-matrix mIoU, occupancy IoU, voxel accuracy, valid-voxel-weighted CE, round-trip tensor error, and prediction disagreement. It imports no condition, diffusion, scheduler, optimizer, or training path. Full sequence-08 Linux/CUDA values remain **Unavailable** until the server command runs. Evidence: the evaluation CLI, pure helpers, behavioral tests, and runbook.
~~~

Use repository-relative Markdown links to the four evidence files and merge with any overlapping entry instead of duplicating it.

- [ ] **Step 7: Final diff review and documentation commit**

~~~bash
cd /Users/zhao/project-master
git status --short
git diff --check -- core-code/skimba-main-7-1/README_RUN.md core-code/skimba-main-7-1/tests/test_vae_latent_reconstruction_evaluation.py docs/agent-memory/PROJECT.md
git add core-code/skimba-main-7-1/README_RUN.md core-code/skimba-main-7-1/tests/test_vae_latent_reconstruction_evaluation.py docs/agent-memory/PROJECT.md
git commit -m "docs: document VAE latent reconstruction evidence"
~~~

Expected: the commit contains only the runbook, its test, and the merged memory entry.

### Task 5: Run and record the server experiment

**Files:**
- Generated outside repository: `<output-dir>/summary.json`, `frames.csv`, `report.md`
- Modify only after successful server execution: `docs/agent-memory/EXPERIMENTS.md`

**Interfaces:**
- Consumes: server VAE checkpoint, raw latent files, GT/invalid files, and train-only stats.
- Produces: complete measured sequence-08 evidence.

- [ ] **Step 1: Verify server artifacts**

~~~bash
cd core-code/skimba-main-7-1
python scripts/evaluation/evaluate_vae_latent_reconstruction.py --help
test -f /mnt/data/datasets/kitti/odometry/skimba_data/model_save_path/vae_v2/best_210_99.12439993681576.pth
test -f /mnt/data/projects/skimba-main-7-1/latent_channel_stats.json
test -d /mnt/data/datasets/kitti/odometry/skimba_data/VAE_Encoder_Features_Semantic20_epoch210/sequences/08/voxels
test -d /mnt/data/datasets/kitti/odometry/skimba_data/data_odometry_voxels_all/sequences/08/voxels
~~~

Expected: all commands exit 0. If paths differ, use CLI overrides rather than editing the shared training YAML.

- [ ] **Step 2: Run every sequence-08 frame**

~~~bash
python scripts/evaluation/evaluate_vae_latent_reconstruction.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --device cuda:0 \
  --output-dir /mnt/data/projects/project-master-results/skimba-main-7-1/vae_latent_reconstruction
~~~

Expected: exit 0 after every discovered frame and print the absolute report path.

- [ ] **Step 3: Validate artifacts**

~~~bash
python -m json.tool /mnt/data/projects/project-master-results/skimba-main-7-1/vae_latent_reconstruction/summary.json
wc -l /mnt/data/projects/project-master-results/skimba-main-7-1/vae_latent_reconstruction/frames.csv
sed -n '1,120p' /mnt/data/projects/project-master-results/skimba-main-7-1/vae_latent_reconstruction/report.md
~~~

Expected: JSON parses; CSV line count equals `num_frames + 1`; report contains both branches and the evidence warning.

- [ ] **Step 4: Record only measured evidence**

Add an `EXPERIMENTS.md` entry with the exact command, code commit, resolved paths, device, frame count, raw metrics, round-trip metrics, tensor errors, disagreement, and artifact links. Mark observed measurements **Verified** and state that this tests VAE information retention and standardization reversibility—not completion, generation, condition, or diffusion.

- [ ] **Step 5: Commit the experiment entry only**

~~~bash
cd /Users/zhao/project-master
git add docs/agent-memory/EXPERIMENTS.md
git commit -m "docs: record full VAE latent reconstruction evaluation"
~~~

Expected: commit contains only the measured experiment entry and is not created before server artifacts are inspected.
