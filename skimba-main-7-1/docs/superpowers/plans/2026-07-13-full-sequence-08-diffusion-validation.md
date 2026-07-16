# Full Sequence 08 Diffusion Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone, reproducible command that selects the best diffusion checkpoint and evaluates every validation frame in SemanticKITTI sequence 08 without resuming training.

**Architecture:** Put deterministic, CPU-testable checkpoint/configuration/metric/report helpers in `utils/full_diffusion_validation.py`. Keep CUDA model construction and the full data loop in `scripts/evaluation/evaluate_diffusion_checkpoint.py`, reusing the current builders and checkpoint-loading functions while overriding only the copied validation-loader configuration.

**Tech Stack:** Python 3.9+, PyTorch, NumPy, PyYAML, existing Skimba builders and losses, pytest.

## Global Constraints

- Default config is `config/semantickitti_autoencoder.yaml`.
- Default seed is exactly `20260713`; it controls Python, NumPy, PyTorch, and CUDA RNG inputs without claiming bitwise deterministic CUDA kernels.
- Automatic selection accepts only `best_<epoch>_<semantic-mIoU>.pth`, ranks by encoded mIoU descending, and breaks ties by epoch descending.
- Full validation enforces `imageset: val`, `batch_size: 1`, `shuffle: false`, and no `frame_divisor`.
- The label mapping must define the validation split as exactly sequence 08.
- Evaluation uses the current CFG scale, DDIM inference steps, invalid/255 masks, strict diffusion loading, configured VAE loading, and latent-normalization checks.
- No per-frame `.label` files are saved by default.
- Do not modify or stage unrelated dirty-worktree files.

---

### Task 1: Pure validation helpers

**Files:**
- Create: `core-code/skimba-main-7-1/utils/full_diffusion_validation.py`
- Create: `core-code/skimba-main-7-1/tests/test_full_diffusion_validation.py`

**Interfaces:**
- Produces: `resolve_best_checkpoint(explicit_checkpoint: str, model_save_path: str) -> pathlib.Path`
- Produces: `prepare_full_validation_loader_config(config: dict) -> dict`
- Produces: `require_sequence_08_validation_split(label_mapping_path: str) -> None`
- Produces: `seed_random_generators(seed: int) -> None`
- Produces: `metrics_from_confusion(confusion: numpy.ndarray) -> dict`
- Produces: `write_validation_reports(result: dict, output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]`

- [ ] **Step 1: Write failing checkpoint-selection and validation-config tests**

```python
def test_resolve_best_checkpoint_uses_miou_then_epoch(tmp_path):
    for name in ("best_10_4.5.pth", "best_20_4.5.pth", "best_99_bad.pth"):
        (tmp_path / name).touch()
    assert resolve_best_checkpoint("", str(tmp_path)).name == "best_20_4.5.pth"


def test_prepare_full_validation_loader_config_removes_sampling():
    original = {"imageset": "test", "batch_size": 4, "shuffle": True, "frame_divisor": 10}
    prepared = prepare_full_validation_loader_config(original)
    assert prepared == {"imageset": "val", "batch_size": 1, "shuffle": False}
    assert original["frame_divisor"] == 10
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_full_diffusion_validation.py -q`

Expected: collection fails because `utils.full_diffusion_validation` does not exist.

- [ ] **Step 3: Implement checkpoint and loader-config helpers**

Implement a strict anchored filename regex, reject missing explicit checkpoints,
ignore malformed automatic candidates, raise when no valid candidate exists, and
return a copied loader dictionary with the enforced full-validation values.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_full_diffusion_validation.py -q`

Expected: checkpoint and loader-config tests pass.

- [ ] **Step 5: Add failing split, seed, metric, and report tests**

Tests must establish that only `split.valid: [8]` is accepted; repeated calls to
`seed_random_generators(123)` reproduce Python, NumPy, and torch samples; a known
2-class confusion matrix produces correct per-class and occupancy IoU; report
serialization writes finite JSON metrics and readable checkpoint/frame/seed
fields only after receiving a complete result.

- [ ] **Step 6: Run the new tests and verify RED**

Run: `python -m pytest tests/test_full_diffusion_validation.py -q`

Expected: failures identify the missing split, seed, metric, and report helpers.

- [ ] **Step 7: Implement the remaining pure helpers**

Use `yaml.safe_load`, `random.seed`, `numpy.random.seed`, `torch.manual_seed`, and
`torch.cuda.manual_seed_all` when CUDA is available. Compute confusion metrics
with rows as predictions and columns as ground truth, exclude class 0 from
semantic mIoU, and reject non-finite aggregate values before writing indented
JSON plus a text report.

- [ ] **Step 8: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_full_diffusion_validation.py -q`

Expected: all helper tests pass with zero failures.

- [ ] **Step 9: Review the Task 1 diff**

Run: `git diff --check -- core-code/skimba-main-7-1/utils/full_diffusion_validation.py core-code/skimba-main-7-1/tests/test_full_diffusion_validation.py`

Expected: exit status 0 and no output. Do not commit without explicit user authorization.

---

### Task 2: Standalone CUDA evaluation entry point

**Files:**
- Create: `core-code/skimba-main-7-1/scripts/evaluation/__init__.py`
- Create: `core-code/skimba-main-7-1/scripts/evaluation/evaluate_diffusion_checkpoint.py`
- Modify: `core-code/skimba-main-7-1/tests/test_full_diffusion_validation.py`

**Interfaces:**
- Consumes: all Task 1 helper functions.
- Produces: `build_parser() -> argparse.ArgumentParser`
- Produces: `evaluate(args: argparse.Namespace) -> dict`
- Produces: `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write failing source-contract and parser tests**

Tests import the lightweight script module and assert defaults for config, seed,
and device. Source-contract assertions require calls to
`prepare_full_validation_loader_config`,
`require_sequence_08_validation_split`, `load_model_state`,
`load_autoencoder_state_from_checkpoint`, `invalid_mask_path`,
`torch.no_grad`, and `write_validation_reports`, and reject any optimizer,
scheduler, backward, or training-resume call.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/test_full_diffusion_validation.py -q`

Expected: failure because the evaluation entry point does not exist.

- [ ] **Step 3: Implement CLI and lazy runtime imports**

The parser exposes `--config-path`, `--checkpoint`, `--seed`, `--device`, and
`--output-dir`. Keep project model/builder imports inside `evaluate` so `--help`
and unit tests can load the module without CUDA extensions.

- [ ] **Step 4: Implement model and full-loader setup**

Load the config, validate sequence 08, resolve the checkpoint, seed before model
construction, reject unavailable CUDA requests, build the model, strictly load
the diffusion state, load the configured VAE with configured strictness, move to
the requested device, and build the loader from a copied validation config with
no divisor. Ignore the returned training loader and never iterate it.

- [ ] **Step 5: Implement the evaluation loop**

Under `model.eval()` and `torch.no_grad()`, move latent/condition/GT tensors to the
device, call the existing model with `train=False`, compute cross entropy, take
`argmax(dim=1)`, load each frame's invalid mask, apply `get_eval_mask`, update a
global confusion matrix, and accumulate per-frame epsilon MSE and cross entropy.
Fail when the loader yields zero frames.

- [ ] **Step 6: Implement final result and output paths**

Combine `metrics_from_confusion` with checkpoint/config/seed/frame count and mean
losses. Default the output directory to
`<model_save_path>/full_validation/<checkpoint-stem>/seed_<seed>`, write both
reports, print their paths and summary metrics, and return zero only after the
reports exist.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_full_diffusion_validation.py -q`

Expected: all tests pass with zero failures.

- [ ] **Step 8: Verify CLI help without constructing a model**

Run: `python scripts/evaluation/evaluate_diffusion_checkpoint.py --help`

Expected: exit status 0 and help listing all five options.

- [ ] **Step 9: Review the Task 2 diff**

Run: `git diff --check -- core-code/skimba-main-7-1/scripts/evaluation core-code/skimba-main-7-1/tests/test_full_diffusion_validation.py`

Expected: exit status 0 and no output. Do not commit without explicit user authorization.

---

### Task 3: Runbook and repository verification

**Files:**
- Modify: `core-code/skimba-main-7-1/README_RUN.md`
- Modify only if durable knowledge is new and non-duplicative: `docs/agent-memory/PROJECT.md`

**Interfaces:**
- Consumes: the Task 2 command contract.
- Produces: a copy-paste server command and explicit interpretation of full-sequence metrics.

- [ ] **Step 1: Add a failing documentation contract test**

Add a test requiring `README_RUN.md` to name the new script, show an automatic
checkpoint-selection command, explain `--checkpoint`, state that all sequence 08
frames are evaluated, and identify the JSON/text output directory.

- [ ] **Step 2: Run the documentation test and verify RED**

Run: `python -m pytest tests/test_full_diffusion_validation.py -q`

Expected: README contract assertion fails.

- [ ] **Step 3: Add the runbook section**

Document this command:

```bash
python scripts/evaluation/evaluate_diffusion_checkpoint.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --seed 20260713 \
  --device cuda:0
```

Also document explicit `--checkpoint`, full sequence 08 coverage, no training
resume, no default `.label` export, and report locations.

- [ ] **Step 4: Run focused and full project tests**

Run: `python -m pytest tests/test_full_diffusion_validation.py -q`

Expected: all focused tests pass.

Run: `python -m pytest tests -q`

Expected: all project tests pass; if unrelated pre-existing failures occur,
record exact failing test names and compare against a clean baseline before
making any completion claim.

- [ ] **Step 5: Run syntax and whitespace verification**

Run: `PYTHONPYCACHEPREFIX=/tmp/skimba-pycache python -m py_compile utils/full_diffusion_validation.py scripts/evaluation/evaluate_diffusion_checkpoint.py`

Expected: exit status 0 and no output.

Run: `git diff --check`

Expected: exit status 0 and no whitespace errors in the new work.

- [ ] **Step 6: Update durable memory only if not already recorded**

Merge one evidence-backed `PROJECT.md` entry describing the new standalone full
sequence 08 evaluation contract, local test evidence, and the unavailable
Linux/CUDA result. Preserve all unrelated memory edits and do not duplicate the
existing sampled-validation audit entry.

- [ ] **Step 7: Inspect final scope**

Run: `git status --short` and targeted `git diff --stat`/`git diff` commands for
the files in this plan. Confirm no unrelated file was modified or staged by this
implementation. Do not commit without explicit user authorization.
