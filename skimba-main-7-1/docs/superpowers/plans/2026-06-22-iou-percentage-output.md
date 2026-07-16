# IoU Percentage Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all IoU and mIoU results from `train_diffusion_network_2.py` display and log in percentage units without changing the underlying evaluation formulas.

**Architecture:** Keep evaluator outputs and confusion-matrix calculations as raw ratios. Derive explicitly named percentage values at the output boundary and use those values for terminal output, text results, TensorBoard, and the existing percentage-scale best-mIoU checkpoint path.

**Tech Stack:** Python 3, NumPy, PyTorch/TensorBoard, pytest source-level regression tests.

---

### Task 1: Lock the percentage output contract with a failing test

**Files:**
- Create: `core-code/skimba-main-7-1/tests/test_training_metric_units.py`
- Read: `core-code/skimba-main-7-1/train_diffusion_network_2.py`

- [ ] **Step 1: Write the failing source-level regression test**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = ROOT / "train_diffusion_network_2.py"


def test_training_reports_all_iou_metrics_as_percentages():
    source = TRAIN_SOURCE.read_text(encoding="utf-8")

    assert "class_iou_percent = class_jaccard * 100.0" in source
    assert "val_miou_percent = m_jaccard * 100.0" in source
    assert "completion_iou_percent = acc_completion * 100.0" in source
    assert "{class_iou_percent[i]:.3f}%" in source
    assert "{completion_iou_percent:.3f}%" in source
    assert "{val_miou_percent:.3f}%" in source
    assert 'writer.add_scalar("mIoU_val_mean", val_miou_percent, start_epoch)' in source
    assert 'writer.add_scalar("iou_val_mean", completion_iou_percent, start_epoch)' in source
```

- [ ] **Step 2: Run the focused test and verify the expected failure**

Run:

```bash
python -m pytest core-code/skimba-main-7-1/tests/test_training_metric_units.py -q
```

Expected: `1 failed`; the first missing assertion is `class_iou_percent = class_jaccard * 100.0`, proving the mixed-unit implementation does not satisfy the new contract.

### Task 2: Convert display and logging boundaries to percentages

**Files:**
- Modify: `core-code/skimba-main-7-1/train_diffusion_network_2.py:337-382`
- Test: `core-code/skimba-main-7-1/tests/test_training_metric_units.py`

- [ ] **Step 1: Derive explicit percentage values without changing raw metric formulas**

Replace the validation metric output block with the following unit flow:

```python
_, class_jaccard = evaluator.getIoU()
m_jaccard = class_jaccard[1:].mean()
class_iou_percent = class_jaccard * 100.0
val_miou_percent = m_jaccard * 100.0

# Existing confusion-matrix formula remains unchanged.
acc_completion = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0])
completion_iou_percent = acc_completion * 100.0
```

- [ ] **Step 2: Use percentage values for every human-readable result**

Build each human-readable result once, then send the same string to the terminal and text file:

```python
class_iou_line = (
    f"IoU class {i} [{class_strings[class_inv_remap[i]]}] = "
    f"{class_iou_percent[i]:.3f}%"
)
print(class_iou_line)
with open(outpath, 'a', encoding='utf-8') as file:
    file.write(class_iou_line + "\n")

validation_iou_line = (
    f"Current val completion iou is {completion_iou_percent:.3f}% "
    f"and Current val miou is {val_miou_percent:.3f}%"
)
print(validation_iou_line)
with open(outpath, 'a', encoding='utf-8') as file:
    file.write(validation_iou_line + "\n")

print(
    f"Current val miou is {val_miou_percent:.3f}% while the best val miou is "
    f"{best_val_miou:.3f}%"
)
```

- [ ] **Step 3: Keep checkpoint selection and TensorBoard in consistent percentage units**

```python
if best_val_miou < val_miou_percent:
    best_val_miou = val_miou_percent

writer.add_scalar("mIoU_val_mean", val_miou_percent, start_epoch)
writer.add_scalar("iou_val_mean", completion_iou_percent, start_epoch)
```

This preserves the current percentage scale of `best_val_miou`, checkpoint filenames, and the checkpoint `Loss` field.

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
python -m pytest core-code/skimba-main-7-1/tests/test_training_metric_units.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Run the containing training-source regression suite**

Run:

```bash
python -m pytest core-code/skimba-main-7-1/tests/test_training_metric_units.py core-code/skimba-main-7-1/tests/test_joint_condition_training_source.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Compile the changed Python files**

Run:

```bash
python -m py_compile core-code/skimba-main-7-1/train_diffusion_network_2.py core-code/skimba-main-7-1/tests/test_training_metric_units.py
```

Expected: exit code `0` with no output.

### Task 3: Review repository memory and hand off evidence

**Files:**
- Read: `docs/agent-memory/README.md`
- Read: `docs/agent-memory/PROJECT.md`
- Optionally modify: `docs/agent-memory/LESSONS.md` only if verification reveals durable, non-duplicate knowledge.

- [ ] **Step 1: Inspect the final diff**

Run:

```bash
git diff -- core-code/skimba-main-7-1/train_diffusion_network_2.py core-code/skimba-main-7-1/tests/test_training_metric_units.py
```

Expected: only the percentage output changes plus the focused regression test; the pre-existing checkpoint-normalization edit remains untouched.

- [ ] **Step 2: Classify memory impact**

This formatting-only change is routine unless it uncovers a reusable unit-consistency pitfall not already recorded. If no durable knowledge results, leave experience memory unchanged as required by `docs/agent-memory/README.md`.

- [ ] **Step 3: Report verification limits**

Report exact test counts and compilation status. State that local source/test verification does not establish Linux/CUDA runtime correctness or model-performance impact.

No implementation commit is planned because `train_diffusion_network_2.py` already contains an unrelated uncommitted user change; committing the whole file could incorrectly capture that work.
