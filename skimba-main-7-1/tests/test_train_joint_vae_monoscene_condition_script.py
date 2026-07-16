import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "train_joint_vae_monoscene_condition.py"


def load_script_module():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("train_joint_vae_monoscene_condition", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_maps_semantic_vae_root_to_current_image_condition_root():
    module = load_script_module()
    vae_root = Path("/data/VAE_Encoder_Features_Semantic20")
    vae_path = vae_root / "sequences" / "08" / "voxels" / "000123.bin"

    assert module.infer_image_condition_root(vae_root) == Path(
        "/data/Image_transform_Voxel_Condition_Features"
    )
    assert module.image_condition_path_for(vae_path, module.infer_image_condition_root(vae_root)) == (
        Path("/data/Image_transform_Voxel_Condition_Features/sequences/08/voxels/000123.bin")
    )


def test_script_uses_vae_latent_files_as_frame_source():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "--config_path" in source
    assert "--vae_root" in source
    assert "resolve_vae_root(configs)" in source
    assert "resolve_kitti_root(configs)" in source
    assert "infer_image_condition_root(vae_root)" in source
    assert "def list_vae_files" in source
    assert "def sample_from_vae_path" in source
    assert "image_2" in source
    assert "calib.txt" in source


def test_script_exports_raw_monoscene_64_channel_features_as_legacy_helper():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "Image_transform_Voxel_Condition_Features" in source
    assert "expected_shape = (args.project_feature_dim, *LATENT_SHAPE)" in source
    assert "np.ascontiguousarray(output).tofile" in source
    assert "parser.add_argument(\"--project_feature_dim\", \"--project-feature-dim\", type=int, default=64)" in source
    assert "default diffusion config expects separate 8-channel condition files" in source
    assert "from stable_diffusion.models.condition_encoder" not in source
    assert "ImageConditionEncoder(" not in source
    assert "AutoEncoderKL" not in source
    assert "SegMamba" not in source
    assert "diffusion_loss" not in source
