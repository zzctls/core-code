# Two-GPU DDP Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one Skimba diffusion model across two RTX 5090 GPUs with a per-GPU batch of 8, correct data sharding, and rank-0-only validation/output.

**Architecture:** A small `utils/distributed_training.py` module owns process-group initialization, rank metadata, sampler construction, barriers, model unwrapping, and cleanup. The existing entry point uses that context to select the local CUDA device, wrap training in DDP, and isolate validation and filesystem side effects to rank 0; the data builder accepts the sampler explicitly.

**Tech Stack:** Python, PyTorch `torch.distributed`, `DistributedDataParallel`, `DistributedSampler`, pytest, StrictYAML.

---

### Task 1: Add testable distributed runtime helpers

**Files:**
- Create: `core-code/skimba-main-7-1/utils/distributed_training.py`
- Create: `core-code/skimba-main-7-1/tests/test_distributed_training.py`

- [ ] **Step 1: Write failing tests for context resolution, sampler selection, and unwrapping**

Add tests that assert:

```python
context = resolve_distributed_context(
    {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"},
    cuda_available=True,
    cuda_device_count=2,
)
assert context.world_size == 2
assert context.rank == 1
assert context.local_rank == 1
assert context.distributed is True
assert context.is_main is False
```

Also cover the direct-Python default (`world_size=1`, rank/local rank zero), rejection of missing/invalid ranks or insufficient GPUs, `DistributedSampler` creation only for distributed mode, and `unwrap_model(DDP-like-wrapper)` returning `.module` while an ordinary module is returned unchanged.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
cd core-code/skimba-main-7-1
python -m pytest tests/test_distributed_training.py -q
```

Expected: collection/import failure because `utils.distributed_training` does not exist.

- [ ] **Step 3: Implement the minimal helper module**

Implement an immutable `DistributedContext` with `rank`, `local_rank`, `world_size`, `distributed`, `is_main`, and `device`; a pure `resolve_distributed_context(...)`; and runtime helpers:

```python
def initialize_distributed(environ=os.environ):
    context = resolve_distributed_context(
        environ,
        cuda_available=torch.cuda.is_available(),
        cuda_device_count=torch.cuda.device_count(),
    )
    torch.cuda.set_device(context.local_rank)
    if context.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
    return replace(context, device=torch.device("cuda", context.local_rank))

def build_train_sampler(dataset, context, shuffle):
    if not context.distributed:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=shuffle,
    )
```

Add `barrier(context)`, `cleanup_distributed(context)`, and `unwrap_model(model)` without introducing process-group side effects at import time.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run the same pytest command. Expected: all tests pass.

### Task 2: Shard the training DataLoader

**Files:**
- Modify: `core-code/skimba-main-7-1/builder/data_builder_autoencoder.py`
- Modify: `core-code/skimba-main-7-1/tests/test_distributed_training.py`

- [ ] **Step 1: Write a failing source/behavior test for builder integration**

Assert that `build(...)` accepts `distributed_context`, obtains the training sampler through `build_train_sampler`, passes it as `sampler=train_sampler`, and uses `shuffle=False` whenever the sampler is present. Assert the validation loader remains unsharded.

- [ ] **Step 2: Run the focused test and confirm RED**

Expected: failure because the builder has no distributed context or sampler.

- [ ] **Step 3: Implement sampler wiring**

Extend `build(..., distributed_context=None)`, construct `train_sampler`, and configure:

```python
sampler=train_sampler,
shuffle=train_dataloader_config["shuffle"] if train_sampler is None else False,
```

Keep validation unchanged so rank 0 evaluates the complete validation set.

- [ ] **Step 4: Run the focused test and confirm GREEN**

Run `python -m pytest tests/test_distributed_training.py -q`. Expected: pass.

### Task 3: Make model-created tensors local-device safe

**Files:**
- Modify: `core-code/skimba-main-7-1/network/cylinder_3D_Unet_mamba_diffusion.py`
- Modify: `core-code/skimba-main-7-1/tests/test_distributed_training.py`

- [ ] **Step 1: Add a failing source test**

Assert that the wrapper contains no module-level `torch.device('cuda:0')`, uses `torch.randn_like(val_VAE_features_change)`, and creates timesteps with `device=val_VAE_features_change.device` in both training and validation paths.

- [ ] **Step 2: Run the focused test and confirm RED**

Expected: failure on the current global `cuda:0` allocation.

- [ ] **Step 3: Replace global-device allocations**

Use:

```python
noise_complete = torch.randn_like(val_VAE_features_change)
timesteps_complete = torch.randint(
    noise_steps,
    (val_VAE_features_change.shape[0],),
    device=val_VAE_features_change.device,
    dtype=torch.long,
)
```

Do not change noise distribution, timestep range, loss, CFG, or sampling semantics.

- [ ] **Step 4: Run focused CFG and distributed tests**

Run:

```bash
python -m pytest tests/test_distributed_training.py tests/test_latent_diffusion_dropout.py -q
```

Expected: pass.

### Task 4: Convert the training entry point to DDP

**Files:**
- Modify: `core-code/skimba-main-7-1/train_diffusion_network_2.py`
- Modify: `core-code/skimba-main-7-1/tests/test_distributed_training.py`

- [ ] **Step 1: Add failing tests for training-loop invariants**

Tests must establish that the source:

- removes the hard-coded `CUDA_VISIBLE_DEVICES` assignment;
- calls `initialize_distributed()` before model/device setup;
- wraps the model with `DistributedDataParallel(..., device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)` because condition-encoder branches can leave trainable parameters unused for a given feature representation;
- passes distributed context to the data builder and calls `train_loader.sampler.set_epoch(start_epoch)`;
- creates writer/progress/output/checkpoints only on `context.is_main`;
- performs a barrier before and after rank-0 validation;
- validates and saves via `unwrap_model(my_model)` to avoid rank-0-only DDP forward collectives and `module.` checkpoint prefixes;
- always invokes distributed cleanup from a `finally` block.

- [ ] **Step 2: Run tests and confirm RED**

Expected: failures for each missing DDP invariant.

- [ ] **Step 3: Implement initialization and DDP wrapping**

Initialize context at the beginning of `main`, use `context.device`, load checkpoints with `map_location=context.device`, move the model to that device, then wrap only when distributed. Keep the optimizer bound to wrapped model parameters and retain the raw model through `unwrap_model` for VAE freezing, validation, and saving.

- [ ] **Step 4: Restrict side effects to rank 0**

Create `SummaryWriter`, tqdm, directories, routine prints, metric files, and checkpoints only on the main rank. Use a disabled/no-op progress path and guard writer calls on other ranks.

- [ ] **Step 5: Implement synchronized rank-0 validation**

At validation epochs, all ranks enter a pre-validation barrier; only rank 0 runs the existing full validation loop through the unwrapped model; all ranks enter a post-validation barrier. Restore training mode on all ranks afterward.

- [ ] **Step 6: Preserve checkpoint compatibility and cleanup**

Save `unwrap_model(my_model).state_dict()`. Load plain or protect checkpoint states into the unwrapped model before DDP wrapping. Close the writer when present and destroy the process group in `finally`.

- [ ] **Step 7: Run focused tests and syntax compilation**

Run:

```bash
python -m pytest tests/test_distributed_training.py tests/test_training_metric_units.py tests/test_latent_diffusion_dropout.py -q
python -m py_compile train_diffusion_network_2.py builder/data_builder_autoencoder.py network/cylinder_3D_Unet_mamba_diffusion.py utils/distributed_training.py
```

Expected: all tests pass and compilation exits zero.

### Task 5: Configure and document the two-GPU run

**Files:**
- Modify: `core-code/skimba-main-7-1/config/semantickitti_autoencoder.yaml`
- Modify: `core-code/skimba-main-7-1/README_RUN.md`
- Modify: `core-code/skimba-main-7-1/tests/test_distributed_training.py`

- [ ] **Step 1: Add failing configuration/documentation tests**

Load the YAML and assert training batch size `8`, training workers `8`, validation batch size `1`, validation workers `4`, and learning rate `1e-3`. Assert the run guide contains `torchrun --standalone --nproc_per_node=2`, explains that batch size/workers are per process, and includes the global batch formula.

- [ ] **Step 2: Run tests and confirm RED**

Expected: failures because the current YAML is `4/0/1/0` and the guide documents only single-process Python.

- [ ] **Step 3: Update YAML and run guide**

Set per-GPU batch/workers to `8/8`, validation to `1/4`, preserve learning rate `1e-3`, and document both the recommended two-GPU command and supported single-GPU direct-Python command. Include the server smoke-test checklist from the approved design.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run `python -m pytest tests/test_distributed_training.py -q`. Expected: pass.

### Task 6: Full verification and durable memory

**Files:**
- Modify if warranted: `docs/agent-memory/PROJECT.md`
- Modify if warranted: `docs/agent-memory/DECISIONS.md`
- Modify if warranted: `docs/agent-memory/LESSONS.md`

- [ ] **Step 1: Run the relevant project suite**

Run:

```bash
cd core-code/skimba-main-7-1
python -m pytest tests -q
python -m py_compile train_diffusion_network_2.py builder/data_builder_autoencoder.py network/cylinder_3D_Unet_mamba_diffusion.py utils/distributed_training.py
git diff --check
```

Expected: relevant tests pass, compilation succeeds, and the diff has no whitespace errors. Record any environment-driven skips or unavailable dependencies exactly.

- [ ] **Step 2: Review the final diff against the approved design**

Verify data sharding, global batch semantics, rank-0 side effects, device locality, validation barriers, checkpoint format, single-process compatibility, and documentation. Do not claim Linux/NCCL runtime or throughput evidence from local CPU tests.

- [ ] **Step 3: Update agent memory only with durable verified outcomes**

Record the accepted DDP decision and locally verified behavior, clearly marking two-GPU CUDA runtime, memory, throughput, and convergence as unavailable until the server run. Preserve the unrelated untracked `core-code/SemCity/` tree.

- [ ] **Step 4: Commit implementation changes intentionally**

Stage only the files named by this plan, inspect the staged diff, and commit with a focused message such as:

```bash
git commit -m "feat: support two-gpu DDP diffusion training"
```
