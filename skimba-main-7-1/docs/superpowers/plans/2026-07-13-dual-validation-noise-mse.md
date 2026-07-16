# Dual Validation Noise MSE Implementation Plan

> **For agent:** Execute this plan with `superpowers:executing-plans` for inline work, or `superpowers:subagent-driven-development` if the user explicitly chooses delegated execution. Follow test-driven development and verification-before-completion.

**Goal:** Log and plot train conditional noise MSE, validation conditional noise MSE, and validation CFG-guided noise MSE while preserving validation DDIM sampling at the configured guidance scale.

**Architecture:** Extend `LatentDiffusion.pred_noise` with an opt-in inference return that exposes guided, unconditional, and conditional predictions from the existing batched CFG UNet call. The validation wrapper computes both validation MSE values from those predictions, the training entrypoint records both TensorBoard scalars while preserving legacy `val_loss`, and the plotting script renders all available series with backward-compatible log parsing.

**Tech Stack:** Python, PyTorch, TensorBoard event accumulator, Matplotlib, pytest.

---

## Constraints and invariants

- Modify only `/Users/zhao/project-master/core-code/skimba-main-7-1`.
- Do not connect to or modify `campus-server`.
- Do not change optimizer steps, gradients, model parameters, training loss, or checkpoint format.
- Do not add another UNet forward pass during validation.
- Keep DDIM sampling on `self.guidance_scale` (currently configured as `3.0`).
- Preserve the historical `val_loss` TensorBoard tag as CFG-guided validation MSE.
- Do not invent conditional validation values for old event files that never logged them.

## Task 1: Add regression tests for CFG component reuse

**Files:**

- Modify: `tests/test_latent_diffusion_dropout.py`
- Test: `tests/test_latent_diffusion_dropout.py`

- [ ] **Step 1: Write a failing test for returning all CFG components**

Add a test using the existing `CFGFormulaDenoiser` fixture:

```python
def test_inference_can_return_cfg_components_without_extra_denoiser_call():
    diffusion, denoiser = make_diffusion(CFGFormulaDenoiser())
    noised_sample = torch.randn(2, 1, 2, 2, 2)
    context = torch.randn(2, 1, 2, 2, 2)
    timestep = torch.tensor([1, 2])

    guided, unconditional, conditional = diffusion.pred_noise(
        noised_sample=noised_sample,
        context_emb_all=context,
        time_step=timestep,
        guidance_scale=3.0,
        train=False,
        return_cfg_components=True,
    )

    assert len(denoiser.calls) == 1
    torch.testing.assert_close(unconditional, torch.full_like(unconditional, 2.0))
    torch.testing.assert_close(conditional, torch.full_like(conditional, 5.0))
    torch.testing.assert_close(guided, torch.full_like(guided, 11.0))
```

- [ ] **Step 2: Run the focused test and confirm it fails for the missing API**

Run:

```bash
cd /Users/zhao/project-master/core-code/skimba-main-7-1
python3 -m pytest tests/test_latent_diffusion_dropout.py -q
```

Expected: the new test fails because `pred_noise` does not accept `return_cfg_components`.

- [ ] **Step 3: Implement the opt-in inference return**

Modify `stable_diffusion/models/latent_diffusion.py`:

```python
def pred_noise(
    self,
    noised_sample,
    context_emb_all,
    time_step,
    guidance_scale=1.0,
    train=True,
    condition_dropout_prob=0.1,
    return_cfg_components=False,
):
```

In training mode, reject the inference-only option with a clear `ValueError`. In inference mode, keep the existing single batched UNet call, calculate:

```python
pred_noise_guided = pred_noise_uncond + guidance_scale * (
    pred_noise_cond - pred_noise_uncond
)
```

Return `(pred_noise_guided, pred_noise_uncond, pred_noise_cond)` only when `return_cfg_components=True`; otherwise return the guided tensor exactly as before.

- [ ] **Step 4: Run the focused test and confirm it passes**

Run:

```bash
cd /Users/zhao/project-master/core-code/skimba-main-7-1
python3 -m pytest tests/test_latent_diffusion_dropout.py -q
```

Expected: all tests pass, including the existing default-return and CFG formula tests.

- [ ] **Step 5: Commit the focused API change**

```bash
git add core-code/skimba-main-7-1/stable_diffusion/models/latent_diffusion.py core-code/skimba-main-7-1/tests/test_latent_diffusion_dropout.py
git commit -m "feat: expose validation CFG noise components"
```

## Task 2: Compute and log both validation noise MSE metrics

**Files:**

- Create: `tests/test_dual_validation_noise_mse.py`
- Modify: `network/cylinder_3D_Unet_mamba_diffusion.py`
- Modify: `train_diffusion_network_2.py`
- Test: `tests/test_dual_validation_noise_mse.py`
- Test: `tests/test_joint_condition_training_source.py`

- [ ] **Step 1: Write a failing wrapper test with deterministic predictions**

Create a small fake diffusion model that records calls and returns, from one `pred_noise` call, a conditional prediction filled with `5`, an unconditional prediction filled with `2`, and a guided prediction filled with `11`. Patch the validation noise to ones and assert:

```python
conditional_mse, cfg_mse, reconstruction = model(
    img_feature=img_feature,
    partial_feature=partial_feature,
    complete_voxel=complete_voxel,
    train=False,
)

torch.testing.assert_close(conditional_mse, torch.tensor(16.0))
torch.testing.assert_close(cfg_mse, torch.tensor(100.0))
assert fake_diffusion.pred_noise_call_count == 1
assert fake_diffusion.return_cfg_components is True
assert fake_diffusion.sample_guidance_scale == model.guidance_scale
```

The fake decoder should return a correctly shaped zero voxel tensor so the reconstruction path remains exercised without loading real weights.

- [ ] **Step 2: Add failing source assertions for TensorBoard tags**

Extend `tests/test_joint_condition_training_source.py` to assert that `train_diffusion_network_2.py` contains both:

```python
writer.add_scalar("val_loss_conditional", val_loss_conditional_mean, epoch)
writer.add_scalar("val_loss", val_loss_cfg_mean, epoch)
```

Also assert the validation call unpacks three values in the order conditional MSE, CFG MSE, reconstruction.

- [ ] **Step 3: Run focused tests and confirm the expected failures**

Run:

```bash
cd /Users/zhao/project-master/core-code/skimba-main-7-1
python3 -m pytest tests/test_dual_validation_noise_mse.py tests/test_joint_condition_training_source.py -q
```

Expected: failures show that the wrapper returns one validation MSE and the training entrypoint lacks the new scalar.

- [ ] **Step 4: Implement dual validation MSE in the wrapper**

In `network/cylinder_3D_Unet_mamba_diffusion.py`, replace the validation `pred_noise` use with:

```python
pred_noise_cfg, _, pred_noise_conditional = self.model_part.pred_noise(
    noised_sample=noised_sample,
    context_emb_all=condition,
    time_step=random_time_step,
    guidance_scale=self.guidance_scale,
    train=False,
    return_cfg_components=True,
)
loss_val_conditional = F.mse_loss(
    pred_noise_conditional, noise_complete, reduction="mean"
)
loss_val_cfg = F.mse_loss(
    pred_noise_cfg, noise_complete, reduction="mean"
)
```

Keep `self.model_part.sample(... guidance_scale=self.guidance_scale, ...)` unchanged and return:

```python
return loss_val_conditional, loss_val_cfg, recon_voxel
```

- [ ] **Step 5: Log both values without changing legacy semantics**

In `train_diffusion_network_2.py`:

- Maintain separate `val_loss_conditional_list` and `val_loss_cfg_list` accumulators.
- Unpack `loss_val_conditional_mse, loss_val_cfg_mse, recon_voxel`.
- Print both named values in the validation status line.
- Average both lists independently.
- Write `val_loss_conditional` from the conditional mean.
- Continue writing `val_loss` from the CFG-guided mean.

- [ ] **Step 6: Run focused tests and confirm they pass**

Run:

```bash
cd /Users/zhao/project-master/core-code/skimba-main-7-1
python3 -m pytest tests/test_dual_validation_noise_mse.py tests/test_joint_condition_training_source.py -q
```

Expected: all focused tests pass.

- [ ] **Step 7: Commit the validation and logging change**

```bash
git add core-code/skimba-main-7-1/network/cylinder_3D_Unet_mamba_diffusion.py core-code/skimba-main-7-1/train_diffusion_network_2.py core-code/skimba-main-7-1/tests/test_dual_validation_noise_mse.py core-code/skimba-main-7-1/tests/test_joint_condition_training_source.py
git commit -m "feat: log conditional and CFG validation MSE"
```

## Task 3: Plot all available noise MSE curves

**Files:**

- Modify: `plot_training_curves.py`
- Create: `tests/test_plot_training_curves.py`
- Test: `tests/test_plot_training_curves.py`

- [ ] **Step 1: Write failing tests for new and legacy console formats**

Create tests that write temporary log snippets. For the new format, assert the parser returns all three relevant tags when the log contains:

```text
epoch 50 validation loss_val_conditional_mse 0.15 loss_val_cfg_mse 0.18 reconstruction_loss 1.20
```

For the legacy format, assert:

```text
epoch 40 validation loss_val_mse 0.17 reconstruction_loss 1.10
```

still maps only to `val_loss`, never fabricating `val_loss_conditional`.

- [ ] **Step 2: Write a failing plot-label test**

Call `draw_noise_loss_series` on an axis and assert it renders these three labels when all series exist:

```python
{
    "Train conditional noise MSE",
    "Validation conditional noise MSE",
    "Validation CFG-guided noise MSE",
}
```

- [ ] **Step 3: Run plot tests and confirm they fail**

Run:

```bash
cd /Users/zhao/project-master/core-code/skimba-main-7-1
MPLBACKEND=Agg python3 -m pytest tests/test_plot_training_curves.py -q
```

Expected: failures show the missing tag, parser pattern, and third plotted series.

- [ ] **Step 4: Extend TensorBoard tag extraction and labels**

In `plot_training_curves.py`, set:

```python
NOISE_LOSS_TAGS = (
    "train_loss_mse_mean",
    "val_loss_conditional",
    "val_loss",
)
```

Update `draw_noise_loss_series` to draw:

- `train_loss_mse_mean`: blue solid, dense training curve.
- `val_loss_conditional`: orange dashed markers, sparse validation checkpoints.
- `val_loss`: red dotted markers, sparse CFG-guided validation checkpoints.

Use scale-neutral labels because historical event files do not record the configured guidance scale in the scalar tag.

- [ ] **Step 5: Add backward-compatible console parsing**

Add a new regex for the explicit two-MSE validation status line while retaining the legacy `loss_val_mse` regex. Parse new logs into both `val_loss_conditional` and `val_loss`; parse old logs only into `val_loss`.

The existing full-range and late-stage `noise_loss_comparison.png` output should automatically show three curves when the new scalar exists and two curves for historical logs.

- [ ] **Step 6: Run plot tests and confirm they pass**

Run:

```bash
cd /Users/zhao/project-master/core-code/skimba-main-7-1
MPLBACKEND=Agg python3 -m pytest tests/test_plot_training_curves.py -q
```

Expected: all plot parser and label tests pass.

- [ ] **Step 7: Commit the plotting change**

```bash
git add core-code/skimba-main-7-1/plot_training_curves.py core-code/skimba-main-7-1/tests/test_plot_training_curves.py
git commit -m "feat: plot dual validation noise MSE curves"
```

## Task 4: Full verification and durable project memory

**Files:**

- Verify: all files changed above
- Modify if durable and non-conflicting: `docs/agent-memory/DECISIONS.md`
- Modify if durable and non-conflicting: `docs/agent-memory/LESSONS.md`

- [ ] **Step 1: Run the relevant regression suite**

Run:

```bash
cd /Users/zhao/project-master/core-code/skimba-main-7-1
MPLBACKEND=Agg python3 -m pytest tests/test_latent_diffusion_dropout.py tests/test_dual_validation_noise_mse.py tests/test_joint_condition_training_source.py tests/test_plot_training_curves.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Compile changed Python files without writing caches into the repository**

Run:

```bash
cd /Users/zhao/project-master/core-code/skimba-main-7-1
PYTHONPYCACHEPREFIX=/tmp/codex-pycache python3 -m py_compile stable_diffusion/models/latent_diffusion.py network/cylinder_3D_Unet_mamba_diffusion.py train_diffusion_network_2.py plot_training_curves.py
```

Expected: exit code `0` and no syntax errors.

- [ ] **Step 3: Inspect the final diff for scope and invariants**

Run:

```bash
cd /Users/zhao/project-master
git diff --check
git diff -- core-code/skimba-main-7-1/stable_diffusion/models/latent_diffusion.py core-code/skimba-main-7-1/network/cylinder_3D_Unet_mamba_diffusion.py core-code/skimba-main-7-1/train_diffusion_network_2.py core-code/skimba-main-7-1/plot_training_curves.py core-code/skimba-main-7-1/tests
```

Confirm:

- Only one validation `pred_noise`/UNet call supplies both MSEs.
- The training branch and `train_loss_mse_mean` calculation are unchanged.
- DDIM sampling still uses `self.guidance_scale`.
- `val_loss` still means CFG-guided validation MSE.
- Old logs do not acquire fabricated conditional MSE points.
- Unrelated dirty files and user changes remain untouched.

- [ ] **Step 4: Update agent memory only if the knowledge is durable and the dirty files can be merged safely**

Merge a concise decision recording the scalar semantics:

```text
train_loss_mse_mean = conditional training noise MSE (scale 1 behavior)
val_loss_conditional = conditional validation noise MSE (scale 1 behavior)
val_loss = CFG-guided validation noise MSE (legacy-compatible tag)
```

Record the reusable lesson that conditional and guided validation predictions can be extracted from one batched CFG forward. If existing uncommitted user edits overlap these entries, do not modify the memory files; report the conflict instead.

- [ ] **Step 5: Commit only verified, non-conflicting memory updates**

```bash
git add docs/agent-memory/DECISIONS.md docs/agent-memory/LESSONS.md
git commit -m "docs: record validation MSE metric semantics"
```

Skip this commit when no memory update is needed or when the files contain overlapping in-progress user changes.

## Self-review checklist

- [ ] Every requested curve has an explicit scalar source and unambiguous label.
- [ ] Conditional and guided validation metrics use the exact same sampled timestep and target noise.
- [ ] The conditional branch is taken before applying CFG guidance.
- [ ] Validation does not run an extra UNet forward.
- [ ] DDIM generation remains CFG-guided.
- [ ] Training outputs, gradients, and optimization are unchanged.
- [ ] Legacy event files and console logs remain readable without invented data.
- [ ] Tests demonstrate numerical branch correctness, call count, logging semantics, and plot labels.
- [ ] Verification commands are rerun immediately before reporting completion.
