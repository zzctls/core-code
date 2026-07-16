from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import re

import pytest
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler

from utils.distributed_training import (
    DistributedContext,
    barrier,
    build_train_sampler,
    initialize_distributed,
    resolve_distributed_context,
    unwrap_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_resolves_two_process_context_from_torchrun_environment():
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


def test_direct_python_defaults_to_main_process_on_gpu_zero():
    context = resolve_distributed_context(
        {},
        cuda_available=True,
        cuda_device_count=2,
    )

    assert context == DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cuda", 0),
    )
    assert context.distributed is False
    assert context.is_main is True


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"WORLD_SIZE": "2", "RANK": "0"}, "LOCAL_RANK"),
        ({"WORLD_SIZE": "2", "RANK": "2", "LOCAL_RANK": "0"}, "RANK"),
        ({"WORLD_SIZE": "0", "RANK": "0", "LOCAL_RANK": "0"}, "WORLD_SIZE"),
    ],
)
def test_rejects_invalid_torchrun_environment(environment, message):
    with pytest.raises(ValueError, match=message):
        resolve_distributed_context(
            environment,
            cuda_available=True,
            cuda_device_count=2,
        )


def test_requires_cuda_and_enough_visible_devices():
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_distributed_context({}, cuda_available=False, cuda_device_count=0)

    with pytest.raises(RuntimeError, match="visible CUDA devices"):
        resolve_distributed_context(
            {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"},
            cuda_available=True,
            cuda_device_count=1,
        )


def test_builds_distributed_sampler_only_for_distributed_training():
    dataset = TensorDataset(torch.arange(8))
    single_context = DistributedContext(0, 0, 1, torch.device("cuda", 0))
    distributed_context = replace(single_context, world_size=2)

    assert build_train_sampler(dataset, single_context, shuffle=True) is None

    sampler = build_train_sampler(dataset, distributed_context, shuffle=True)
    assert isinstance(sampler, DistributedSampler)
    assert sampler.num_replicas == 2
    assert sampler.rank == 0
    assert sampler.shuffle is True


def test_unwrap_model_handles_wrapped_and_plain_models():
    model = torch.nn.Linear(2, 1)

    class Wrapper:
        def __init__(self, module):
            self.module = module

    assert unwrap_model(model) is model
    assert unwrap_model(Wrapper(model)) is model


def test_validation_barrier_uses_long_timeout_gloo_control_group(monkeypatch):
    control_group = object()
    calls = {}

    def record_new_group(**kwargs):
        calls["new_group"] = kwargs
        return control_group

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda device: calls.setdefault("device", device),
    )
    monkeypatch.setattr(
        "utils.distributed_training.dist.init_process_group",
        lambda **kwargs: calls.setdefault("init", kwargs),
    )
    monkeypatch.setattr(
        "utils.distributed_training.dist.new_group",
        record_new_group,
    )
    monkeypatch.setattr(
        "utils.distributed_training.dist.barrier",
        lambda **kwargs: calls.setdefault("barrier", kwargs),
    )

    context = initialize_distributed(
        {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"}
    )
    barrier(context)

    assert calls["init"] == {"backend": "nccl", "init_method": "env://"}
    assert calls["new_group"]["backend"] == "gloo"
    assert calls["new_group"]["timeout"] >= timedelta(hours=12)
    assert context.control_group is control_group
    assert calls["barrier"] == {"group": control_group}


def test_data_builder_wires_only_the_training_loader_to_distributed_sampler():
    source = (PROJECT_ROOT / "builder" / "data_builder_autoencoder.py").read_text(
        encoding="utf-8"
    )

    assert "distributed_context=None" in source
    assert "build_train_sampler" in source
    assert "train_sampler = build_train_sampler(" in source
    assert "sampler=train_sampler" in source
    assert 'shuffle=train_dataloader_config["shuffle"] if train_sampler is None else False' in source
    assert "sampler=val_sampler" not in source


def test_diffusion_wrapper_creates_noise_and_timesteps_on_input_device():
    source = (
        PROJECT_ROOT / "network" / "cylinder_3D_Unet_mamba_diffusion.py"
    ).read_text(encoding="utf-8")

    assert "torch.device('cuda:0')" not in source
    assert source.count("torch.randn_like(val_VAE_features_change)") == 2
    assert source.count("device=val_VAE_features_change.device") == 2


def test_training_entrypoint_uses_ddp_and_rank_safe_side_effects():
    source = (PROJECT_ROOT / "train_diffusion_network_2.py").read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES" not in source
    assert "context = initialize_distributed()" in source
    assert "pytorch_device = context.device" in source
    assert "DistributedDataParallel(" in source
    assert "device_ids=[context.local_rank]" in source
    assert "output_device=context.local_rank" in source
    assert "find_unused_parameters=True" in source
    assert "distributed_context=context" in source
    assert "train_dataset_loader.sampler.set_epoch(start_epoch)" in source
    assert "tensorboard_log_dir = os.path.join(model_save_path, 'logs')" in source
    assert "writer = SummaryWriter(tensorboard_log_dir)" in source
    assert "disable=not context.is_main" in source
    assert "if should_validate:" in source
    assert source.count("barrier(context)") >= 2
    assert "if should_validate and context.is_main:" in source
    assert "raw_model = unwrap_model(my_model)" in source
    assert "raw_model.state_dict()" in source
    assert "finally:" in source
    assert "cleanup_distributed(context)" in source


def test_default_config_targets_two_gpu_throughput_without_scaling_learning_rate():
    source = (PROJECT_ROOT / "config" / "semantickitti_autoencoder.yaml").read_text(
        encoding="utf-8"
    )

    train_section = re.search(
        r"train_data_loader:\n(?P<body>.*?)(?=\nval_data_loader:)", source, re.DOTALL
    ).group("body")
    val_section = re.search(
        r"val_data_loader:\n(?P<body>.*?)(?=\n\n#+)", source, re.DOTALL
    ).group("body")
    assert "batch_size: 12" in train_section
    assert "num_workers: 8" in train_section
    assert "batch_size: 1" in val_section
    assert "num_workers: 4" in val_section
    assert "max_num_epochs: 1500" in source
    assert "eval_every_n_epochs: 10" in source
    assert "eval_first_phase_epochs: 450" in source
    assert "eval_first_phase_every_n_epochs: 30" in source
    assert "eval_after_phase_every_n_epochs: 10" in source
    assert "learning_rate: 1e-4" in source
    assert "warmup_start_lr: 1e-6" in source
    assert "warmup_epochs: 15" in source


def test_run_guide_documents_torchrun_and_per_process_batch_semantics():
    guide = (PROJECT_ROOT / "README_RUN.md").read_text(encoding="utf-8")

    assert "torchrun --standalone --nproc_per_node=2" in guide
    assert "每个进程" in guide
    assert "全局 batch size = 每卡 batch size × GPU 数量" in guide
    assert "28 GiB" in guide
