import importlib.util
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data" / "create_zero_partial_condition_features.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("create_zero_partial_condition_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_infers_partial_condition_path_from_semantic_vae_root():
    module = load_script_module()

    vae_path = Path("/data/VAE_Encoder_Features_Semantic20/sequences/08/voxels/000123.bin")

    assert module.infer_partial_condition_path(vae_path) == (
        Path("/data/Condition_Features_2/sequences/08/voxels/000123.bin")
    )


def test_script_writes_float32_zero_partial_feature():
    module = load_script_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "sequences" / "08" / "voxels" / "000123.bin"
        result = module.write_zero_partial_file(
            out_path,
            channels=8,
            spatial_shape=(2, 2, 1),
            overwrite=True,
            dry_run=False,
        )

        values = np.fromfile(out_path, dtype=np.float32)

        assert result["status"] == "written"
        assert result["channels"] == 8
        assert result["num_values"] == 32
        assert values.dtype == np.float32
        assert values.shape == (32,)
        assert np.count_nonzero(values) == 0


def test_script_skips_existing_file_without_overwrite():
    module = load_script_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "000123.bin"
        np.ones(32, dtype=np.float32).tofile(out_path)

        result = module.write_zero_partial_file(
            out_path,
            channels=8,
            spatial_shape=(2, 2, 1),
            overwrite=False,
            dry_run=False,
        )
        values = np.fromfile(out_path, dtype=np.float32)

        assert result["status"] == "skipped_exists"
        assert np.count_nonzero(values) == 32


def test_script_generates_paths_from_vae_files():
    module = load_script_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        vae_dir = root / "VAE_Encoder_Features_Semantic20" / "sequences" / "08" / "voxels"
        vae_dir.mkdir(parents=True)
        np.zeros(8 * 2 * 2 * 1, dtype=np.float32).tofile(vae_dir / "000001.bin")

        results = module.create_zero_partial_features(
            vae_root=root / "VAE_Encoder_Features_Semantic20",
            output_root=None,
            sequences=["08"],
            frames=[],
            num_samples=0,
            latent_folder="voxels",
            partial_folder="voxels",
            ext=".bin",
            channels=8,
            spatial_shape=(2, 2, 1),
            overwrite=True,
            dry_run=False,
        )

        out_path = root / "Condition_Features_2" / "sequences" / "08" / "voxels" / "000001.bin"

        assert len(results) == 1
        assert results[0]["path"] == str(out_path)
        assert out_path.exists()
