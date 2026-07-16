import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data" / "export_mask2former_semantic_probabilities.py"


def load_module():
    spec = importlib.util.spec_from_file_location("export_mask2former_semantic_probabilities", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semantic_probability_path_follows_kitti_sequence_layout():
    module = load_module()
    image_path = Path("/data/SemanticKITTI/sequences/08/image_2/000123.png")

    out_path = module.semantic_probability_output_path(
        image_path,
        "/data/Mask2Former_Semantic_Probabilities",
    )

    assert out_path == Path(
        "/data/Mask2Former_Semantic_Probabilities/sequences/08/image_2/000123.npz"
    )


def test_semantic_scores_are_normalized_to_probabilities():
    module = load_module()
    scores = np.array(
        [
            [[2.0, 0.0], [0.0, 1.0]],
            [[1.0, 3.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 2.0]],
        ],
        dtype=np.float32,
    )

    probabilities = module.semantic_scores_to_probabilities(scores)

    assert probabilities.shape == scores.shape
    assert probabilities.dtype == np.float32
    assert np.allclose(probabilities.sum(axis=0), 1.0)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_exporter_uses_mask2former_default_predictor_and_saves_npz_probabilities():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "from detectron2.engine.defaults import DefaultPredictor" in source
    assert "from mask2former import add_maskformer2_config" in source
    assert "predictions[\"sem_seg\"]" in source
    assert "np.savez_compressed" in source
    assert "probabilities=probabilities.astype(np.float32)" in source
    assert "Mask2Former semantic probabilities output_root" in source


def test_exporter_exposes_cityscapes_cli_defaults_and_required_arguments():
    source = SCRIPT.read_text(encoding="utf-8")

    for token in [
        "configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml",
        "weights/cityscapes_semantic_R50.pkl",
        "--kitti-root",
        "--output-root",
        "--mask2former-root",
        "--config-file",
        "--checkpoint",
        "--sequences",
        "--num-samples",
        "--dry-run",
        "--overwrite",
        "--out-json",
        "--out-csv",
    ]:
        assert token in source
