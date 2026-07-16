import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data" / "validate_semantic_vae_latents.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("validate_semantic_vae_latents", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_exposes_pair_and_dataset_latent_path_helpers():
    module = load_script_module()

    assert module.parse_csv_items("00, 01,,08") == ["00", "01", "08"]
    assert module.parse_shape("8,64,64,8") == (8, 64, 64, 8)
    assert module.latent_path_for("root", "08", "123", "latents", ".pt") == (
        Path("root") / "sequences" / "08" / "latents" / "000123.pt"
    )
    assert module.label_path_for("dataset", "8", "123") == (
        Path("dataset") / "sequences" / "08" / "voxels" / "000123.label"
    )


def test_script_has_skimba_bin_defaults_for_new_exporter():
    module = load_script_module()

    args = module.build_argparser().parse_args(
        [
            "--vae_checkpoint",
            "model.pth",
            "--latent_root",
            "latents",
            "--dataset_root",
            "dataset",
            "--skimba_bin",
        ]
    )

    module.apply_latent_format_defaults(args)

    assert args.latent_folder == "voxels"
    assert args.latent_ext == ".bin"


def test_script_contains_content_metrics_and_threshold_checks():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "--config_path" in source
    assert "resolve_vae_root(configs)" in source
    assert "resolve_gt_root(configs)" in source
    assert "train_params" in source
    assert "F.cross_entropy" in source
    assert "semantic_miou" in source
    assert "occupancy_iou" in source
    assert "--skimba_bin" in source
    assert "--min_miou" in source
    assert "--min_occ_iou" in source
    assert "--max_ce" in source
    assert "raise SystemExit" in source
