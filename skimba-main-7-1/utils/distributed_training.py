import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Mapping, Optional

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


VALIDATION_BARRIER_TIMEOUT = timedelta(hours=24)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    control_group: Optional[object] = None

    @property
    def distributed(self):
        return self.world_size > 1

    @property
    def is_main(self):
        return self.rank == 0


def _parse_environment_integer(environ, name, default=None):
    value = environ.get(name)
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"{name} is required when WORLD_SIZE is greater than 1")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def resolve_distributed_context(
    environ: Mapping[str, str],
    cuda_available: bool,
    cuda_device_count: int,
) -> DistributedContext:
    world_size = _parse_environment_integer(environ, "WORLD_SIZE", default=1)
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be at least 1")

    if world_size > 1:
        rank = _parse_environment_integer(environ, "RANK")
        local_rank = _parse_environment_integer(environ, "LOCAL_RANK")
    else:
        rank = _parse_environment_integer(environ, "RANK", default=0)
        local_rank = _parse_environment_integer(environ, "LOCAL_RANK", default=0)

    if not 0 <= rank < world_size:
        raise ValueError(f"RANK must be in [0, {world_size}), got {rank}")
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be non-negative, got {local_rank}")
    if not cuda_available:
        raise RuntimeError("CUDA is required for diffusion training")
    if local_rank >= cuda_device_count:
        raise RuntimeError(
            f"LOCAL_RANK {local_rank} requires more visible CUDA devices; "
            f"only {cuda_device_count} detected"
        )

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def initialize_distributed(environ: Optional[Mapping[str, str]] = None):
    if environ is None:
        environ = os.environ
    context = resolve_distributed_context(
        environ,
        cuda_available=torch.cuda.is_available(),
        cuda_device_count=torch.cuda.device_count(),
    )
    torch.cuda.set_device(context.local_rank)
    if context.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        control_group = dist.new_group(
            backend="gloo",
            timeout=VALIDATION_BARRIER_TIMEOUT,
        )
        context = DistributedContext(
            rank=context.rank,
            local_rank=context.local_rank,
            world_size=context.world_size,
            device=context.device,
            control_group=control_group,
        )
    return context


def build_train_sampler(dataset, context, shuffle):
    if context is None or not context.distributed:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=shuffle,
    )


def barrier(context):
    if context.distributed:
        dist.barrier(group=context.control_group)


def cleanup_distributed(context):
    if context is not None and context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return getattr(model, "module", model)
