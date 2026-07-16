# True Classifier-Free Guidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace element-wise condition corruption with standard sample-level CFG and make its active train/validation parameters YAML-configurable.

**Architecture:** Preserve the existing independent condition encoders and channel-wise fusion. Apply a broadcastable sample-level mask to the fused condition before `SegMamba` concatenates it with the unchanged noisy latent; at inference, evaluate zero/full conditions on duplicated copies of the same latent and combine only the predicted epsilon tensors.

**Tech Stack:** Python, PyTorch, StrictYAML, pytest

---

### Task 1: Specify true CFG behavior

**Files:**
- Modify: `tests/test_latent_diffusion_dropout.py`
- Modify: `tests/test_latent_diffusion_source.py`
- Modify: `tests/test_joint_condition_training_source.py`

- [x] Replace the element-wise test with sample-level mask shape, consistency, frequency, and boundary tests.
- [x] Add behavioral denoiser-spy tests for one-pass training, joint condition dropping, unchanged latent reuse, zero unconditional input, and the CFG equation.
- [x] Add source/config regression assertions for active YAML parameter wiring.
- [x] Run the focused tests and confirm failures specifically identify the old element-wise mask and hard-coded wrapper values.

### Task 2: Implement sample-level CFG and configuration wiring

**Files:**
- Modify: `stable_diffusion/models/latent_diffusion.py`
- Modify: `network/cylinder_3D_Unet_mamba_diffusion.py`
- Modify: `builder/model_builder_3D_Voxel_unet_diffusion.py`
- Modify: `config/config.py`
- Modify: `config/semantickitti_autoencoder.yaml`

- [x] Generate a Bernoulli mask shaped `[B,1,1,1,1]` and broadcast it over the fused condition without rescaling retained values.
- [x] Store CFG values on `cylinder_asym`; pass dropout only to the single training denoiser call and use configured inference values in validation/sampling.
- [x] Read defaults from `model_params`, pass them through the builder, and declare concrete values in the active YAML and StrictYAML schema.
- [x] Run the focused tests until green without altering unrelated model, VAE, scheduler, optimizer, or dataset settings.

### Task 3: Verify and preserve durable evidence

**Files:**
- Modify as needed: `docs/agent-memory/PROJECT.md`
- Modify as needed: `docs/agent-memory/DECISIONS.md`
- Modify as needed: `docs/agent-memory/LESSONS.md`

- [x] Run CFG-focused and directly related tests, Python syntax compilation, and `git diff --check`.
- [x] Review the scoped diff against every requested CFG invariant and confirm pre-existing unrelated edits remain intact.
- [x] Merge non-duplicative Verified/Unavailable knowledge into agent memory; record GPU training and performance evidence as unavailable.
- [x] Do not commit, stage, push, or modify unrelated files.
