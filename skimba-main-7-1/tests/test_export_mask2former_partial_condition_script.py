from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data" / "export_mask2former_partial_condition_features.py"


def test_exporter_uses_monoscene_projection_and_writes_partial_condition():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "from monoscene.data.utils.helpers import vox2pix" in source
    assert "from monoscene.models.flosp import FLoSP" in source
    assert "build_semantic_condition_2d" in source
    assert "compute_depth_surface_weights" in source
    assert "condition_path_for" in source
    assert "resolve_depth_path" in source
    assert "project_semantic_condition" in source
    assert "apply_depth_surface_weights" in source
    assert "voxel_depth" in source
    assert "Mask2Former partial_condition output_root" in source


def test_exporter_exposes_required_cli_arguments():
    source = SCRIPT.read_text(encoding="utf-8")

    for flag in [
        "--semantic-root",
        "--mapping-path",
        "--output-root",
        "--monoscene-root",
        "--sequences",
        "--num-samples",
        "--projection-mode",
        "--depth-root",
        "--dry-run",
        "--overwrite",
        "--out-json",
        "--out-csv",
    ]:
        assert flag in source
