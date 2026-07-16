# CFG Per-Sample Element Dropout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CFG training independently mask approximately 10% of the condition elements inside every sample instead of masking approximately 10% of whole samples.

**Architecture:** Keep condition merging and the training call path unchanged. Replace the batch-shaped random mask in `LatentDiffusion._sample_condition_keep_mask` with a tensor-shaped Bernoulli mask, while retaining the existing probability boundary behavior and leaving sampling-time CFG untouched.

**Tech Stack:** Python, PyTorch, pytest

---

### Task 1: Specify element-level condition masking with failing tests

**Files:**
- Create: `core-code/skimba-main-7/tests/test_latent_diffusion_dropout.py`
- Modify: `core-code/skimba-main-7/tests/test_latent_diffusion_source.py`

- [ ] **Step 1: Add behavioral tests for mixed masks and probability boundaries**

Create `core-code/skimba-main-7/tests/test_latent_diffusion_dropout.py`:

```python
import torch

from stable_diffusion.models.latent_diffusion import LatentDiffusion


def test_condition_dropout_masks_elements_independently_within_each_sample(monkeypatch):
    context = torch.ones((2, 1, 1, 1, 4))
    random_values = torch.tensor(
        [0.05, 0.20, 0.09, 0.90, 0.01, 0.30, 0.08, 0.80]
    ).reshape_as(context)

    def fake_rand_like(tensor):
        assert tensor is context
        return random_values

    monkeypatch.setattr(torch, "rand_like", fake_rand_like)

    mask = LatentDiffusion._sample_condition_keep_mask(None, context, 0.1)

    assert mask.dtype == torch.bool
    assert mask.shape == context.shape
    assert mask[0].flatten().tolist() == [False, True, False, True]
    assert mask[1].flatten().tolist() == [False, True, False, True]


def test_condition_dropout_probability_boundaries():
    context = torch.ones((2, 3, 2, 2, 2))

    keep_all = LatentDiffusion._sample_condition_keep_mask(None, context, 0.0)
    drop_all = LatentDiffusion._sample_condition_keep_mask(None, context, 1.0)

    assert torch.all(keep_all)
    assert not torch.any(drop_all)
```

- [ ] **Step 2: Update the source regression assertions**

In `test_cfg_dropout_logic_is_in_main_latent_diffusion_file`, replace the old batch-mask assertions with:

```python
    assert "element_dropout = torch.rand_like(context_emb_all)" in source
    assert "condition_keep_mask = element_dropout > condition_dropout_prob" in source
    assert "(batch_size, 1, 1, 1, 1)" not in source
    assert "return condition_keep_mask.expand_as(context_emb_all)" not in source
```

Keep the existing assertions for the training UNet call, single-pass training, zero unconditional condition, and device portability.

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
cd core-code/skimba-main-7
python -m pytest tests/test_latent_diffusion_dropout.py tests/test_latent_diffusion_source.py -v
```

Expected: the mixed-mask test fails because the current implementation does not call `torch.rand_like`, and the source regression test fails because `element_dropout` is absent.

### Task 2: Implement per-element CFG condition masking

**Files:**
- Modify: `core-code/skimba-main-7/stable_diffusion/models/latent_diffusion.py`
- Test: `core-code/skimba-main-7/tests/test_latent_diffusion_dropout.py`
- Test: `core-code/skimba-main-7/tests/test_latent_diffusion_source.py`

- [ ] **Step 1: Replace the batch-level random mask with a tensor-level mask**

Replace the non-boundary body of `_sample_condition_keep_mask` with:

```python
        element_dropout = torch.rand_like(context_emb_all)
        condition_keep_mask = element_dropout > condition_dropout_prob
        return condition_keep_mask
```

- [ ] **Step 2: Correct the training documentation**

Change the training section of `pred_noise` to:

```python
        Training:
        - independently drop condition elements within every sample in one
          denoiser pass; dropped elements use zero condition.
```

- [ ] **Step 3: Run the focused tests and verify GREEN**

Run:

```bash
cd core-code/skimba-main-7
python -m pytest tests/test_latent_diffusion_dropout.py tests/test_latent_diffusion_source.py -v
```

Expected: all focused tests pass.

- [ ] **Step 4: Run the complete available test suite**

Run:

```bash
cd core-code/skimba-main-7
python -m pytest -v
```

Expected: all collected tests pass. If an optional project dependency prevents collection, record the exact missing dependency and retain the successful focused-test result without claiming the full suite passes.

- [ ] **Step 5: Review the final diff**

Run:

```bash
git diff --check
git diff -- core-code/skimba-main-7/stable_diffusion/models/latent_diffusion.py core-code/skimba-main-7/tests/test_latent_diffusion_dropout.py core-code/skimba-main-7/tests/test_latent_diffusion_source.py
```

Expected: no whitespace errors; the diff contains only the mask implementation, its documentation, and its regression tests.

- [ ] **Step 6: Commit the implementation**

```bash
git add core-code/skimba-main-7/stable_diffusion/models/latent_diffusion.py core-code/skimba-main-7/tests/test_latent_diffusion_dropout.py core-code/skimba-main-7/tests/test_latent_diffusion_source.py
git commit -m "fix: apply CFG dropout within each sample"
```
