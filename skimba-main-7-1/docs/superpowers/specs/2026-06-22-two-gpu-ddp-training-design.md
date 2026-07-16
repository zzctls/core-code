# Two-GPU DDP Training Design

## Objective

Enable `train_diffusion_network_2.py` to train one model across two RTX 5090 GPUs with PyTorch `DistributedDataParallel` (DDP), prioritizing throughput. The configured training batch size is per process/per GPU: `8` per GPU produces a global batch size of `16` with two processes.

## Verified Current State

- The entry point hides all but physical GPU 1 with `CUDA_VISIBLE_DEVICES='1'` and always selects `cuda:0`.
- The training loader uses ordinary shuffle and has no `DistributedSampler`.
- Every process would currently create TensorBoard output, progress bars, validation artifacts, and checkpoints.
- The model wrapper creates noise and timestep tensors on a module-level `cuda:0`, which would place rank 1 tensors on the wrong device.
- The live YAML currently uses training batch size `4` and zero loader workers.

Evidence: `train_diffusion_network_2.py`, `builder/data_builder_autoencoder.py`, `network/cylinder_3D_Unet_mamba_diffusion.py`, and `config/semantickitti_autoencoder.yaml` as inspected on 2026-06-22.

## Chosen Architecture

Launch one process per GPU with:

```bash
torchrun --standalone --nproc_per_node=2 train_diffusion_network_2.py \
  -y config/semantickitti_autoencoder.yaml
```

Each process reads `LOCAL_RANK`, selects that CUDA device, initializes an NCCL process group, builds the same model, and wraps it with DDP. The code will retain a reference to the unwrapped model for checkpoint compatibility and rank-0-only validation.

The implementation will fail clearly if a multi-process launch lacks CUDA/NCCL prerequisites. A normal direct Python launch remains supported as a single-process path so existing inspection and debugging workflows do not require `torchrun`.

## Data Flow

The data builder will accept explicit distributed context. During distributed training it will attach a `DistributedSampler` to the training dataset and disable DataLoader-level shuffle. The training loop will call `sampler.set_epoch(epoch)` before every epoch so all ranks receive deterministic but newly shuffled, non-overlapping shards.

Validation will remain unsharded and run only on rank 0 to preserve the current full-dataset metric calculation and artifact layout. Rank 0 will validate through the unwrapped model while the other rank waits at a barrier; this avoids DDP forward-time buffer synchronization deadlocks. All ranks resume training only after rank 0 completes validation.

## Device Safety

The model wrapper will derive newly created noise and timestep tensors from the input tensor's device rather than a global `cuda:0`. Host-to-device transfers in the training entry point will target the process-local device and use direct dtype/device conversion.

## Side Effects and Checkpoints

Only rank 0 will:

- create the TensorBoard writer;
- display progress and routine training output;
- create validation directories and metric files;
- save best/protect checkpoints.

Checkpoints will store the unwrapped model state dict, avoiding a new `module.` prefix and preserving the existing loader format. Optimizer and scheduler state remain included in protect checkpoints. Process groups will be destroyed during orderly shutdown.

## Configuration and Optimization Semantics

The YAML training settings will become:

```yaml
train_data_loader:
  batch_size: 8  # per GPU; global batch size is 16 with two ranks
  num_workers: 8 # per rank; 16 workers total with two ranks
```

Validation remains batch size `1`; its worker count becomes `4` because only rank 0 consumes it. The learning rate remains `1e-3` initially. Scaling it automatically would confound the DDP throughput change with an unverified optimization change. Linux/CUDA throughput, peak memory, and convergence remain unavailable until measured on the server.

Because global batch increases from `4` to `16`, the number of optimizer steps per epoch falls by approximately four times. Epoch-based validation stays unchanged; any step-count-based reporting must be interpreted using the new global batch.

## Testing

Focused CPU tests will verify:

- distributed environment parsing and rank-0 detection;
- distributed sampler selection and shuffle behavior;
- model unwrapping for prefix-free checkpoints;
- device-local noise/timestep allocation by source inspection or a device-agnostic forward seam;
- YAML per-GPU batch and worker values;
- documented `torchrun` command.

The existing focused test suite and Python syntax compilation will run after implementation. These checks cannot establish NCCL correctness, two-GPU memory use, throughput, or convergence; those require a server smoke test followed by a controlled training run.

## Server Acceptance Check

Run a short two-GPU job and verify:

1. both GPUs hold a model and show sustained utilization;
2. rank 0 is the only process writing logs and checkpoints;
3. no device mismatch or collective hang occurs at the first validation boundary;
4. peak memory stays below approximately 28 GiB per GPU;
5. samples per second improve over the recorded single-GPU `batch_size=4` run.

