from pathlib import Path

from config.config import load_config_data
from utils.validation_schedule import should_validate_epoch


ROOT = Path(__file__).resolve().parents[1]


def test_default_autoencoder_config_loads_with_strict_schema():
    config = load_config_data(ROOT / "config" / "semantickitti_autoencoder.yaml")

    assert config["train_params"]["resume_checkpoint"] == ""


def test_default_diffusion_training_sampling_and_validation_schedule():
    config = load_config_data(ROOT / "config" / "semantickitti_autoencoder.yaml")
    train_params = config["train_params"]

    assert "frame_divisor" not in config["train_data_loader"]
    assert config["val_data_loader"]["frame_divisor"] == 10
    assert should_validate_epoch(30, 600, train_params)
    assert should_validate_epoch(450, 600, train_params)
    assert should_validate_epoch(460, 600, train_params)
    assert should_validate_epoch(600, 600, train_params)
    assert not should_validate_epoch(10, 600, train_params)
    assert not should_validate_epoch(455, 600, train_params)
