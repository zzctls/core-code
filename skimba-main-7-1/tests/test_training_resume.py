import sys
import types
import importlib.util
from pathlib import Path

import torch

from utils.warmup_lr import ConstantLearningRate


ROOT = Path(__file__).resolve().parents[1]


def load_training_module():
    stubbed_modules = ("builder", "dataloader.pc_dataset_gai")
    original_modules = {
        name: sys.modules.get(name)
        for name in stubbed_modules
    }

    pc_dataset_stub = types.ModuleType("dataloader.pc_dataset_gai")
    pc_dataset_stub.get_eval_mask = None
    pc_dataset_stub.unpack = None
    sys.modules["dataloader.pc_dataset_gai"] = pc_dataset_stub

    builder_stub = types.ModuleType("builder")
    builder_stub.data_builder_autoencoder = None
    builder_stub.loss_builder = None
    builder_stub.model_builder_3D_Voxel_unet_diffusion = None
    sys.modules["builder"] = builder_stub

    try:
        spec = importlib.util.spec_from_file_location(
            "_training_resume_under_test",
            ROOT / "train_diffusion_network_2.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_resolve_model_checkpoint_path_prefers_cli_then_yaml_then_default():
    training = load_training_module()

    assert training.resolve_model_checkpoint_path("/cli/protect.pth", "/models", "/yaml/protect.pth") == "/cli/protect.pth"
    assert training.resolve_model_checkpoint_path("", "/models", "/yaml/protect.pth") == "/yaml/protect.pth"
    assert training.resolve_model_checkpoint_path("", "/models", "") == "/models/0.pth"


def test_restore_training_state_resumes_epoch_best_metric_and_global_iter():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = ConstantLearningRate(optimizer, lr=0.1)
    checkpoint = {
        "epoch": 17,
        "Loss": 4.2,
        "global_iter": 1234,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    training = load_training_module()

    start_epoch, best_val_miou, global_iter = training.restore_training_state(
        optimizer,
        scheduler,
        checkpoint,
        steps_per_epoch=99,
    )

    assert start_epoch == 18
    assert best_val_miou == 4.2
    assert global_iter == 1234


def test_restore_training_state_reconstructs_global_iter_for_legacy_checkpoints():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = ConstantLearningRate(optimizer, lr=0.1)
    checkpoint = {
        "epoch": 17,
        "Loss": 4.2,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    training = load_training_module()

    _, _, global_iter = training.restore_training_state(
        optimizer,
        scheduler,
        checkpoint,
        steps_per_epoch=99,
    )

    assert global_iter == 17 * 99
