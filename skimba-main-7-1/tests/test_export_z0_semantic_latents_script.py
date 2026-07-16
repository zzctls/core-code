import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data" / "export_z0_semantic_latents.py"
REMOVED_SCRIPT = ROOT / "scripts" / "data" / "export_skimba_semantic_vae_latents.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("export_z0_semantic_latents", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_z0_exporter_owns_skimba_path_and_checkpoint_key_contract():
    module = load_script_module()

    assert module.output_latent_path("out", "8", "123") == (
        Path("out") / "sequences" / "08" / "voxels" / "000123.bin"
    )
    assert module.normalize_autoencoder_key("model_part.encoder.conv_input.0.weight") == (
        "encoder.conv_input.0.weight"
    )
    assert module.normalize_autoencoder_key(
        "module.model_part.decoder.out_conv.conv_1x1.weight"
    ) == "decoder.out_conv.conv_1x1.weight"
    assert module.normalize_embedding_key("module.semantic_embedding.weight") == (
        "semantic_embedding.weight"
    )


def test_z0_exporter_uses_embedding_encode_and_raw_float32_bin_output():
    source = SCRIPT.read_text(encoding="utf-8")

    parser = load_script_module().build_argparser()
    args = parser.parse_args(
        ["--config_path", "config/semantickitti_autoencoder.yaml", "--device", "cpu"]
    )
    assert args.config_path == "config/semantickitti_autoencoder.yaml"
    assert args.device == "cpu"
    assert "semantic_embedding.weight" in source
    assert "autoencoder.encode" in source
    assert "dist.sample()" in source
    assert "dist.mean" in source
    assert "astype(np.float32).tofile" in source
    assert not REMOVED_SCRIPT.exists()
