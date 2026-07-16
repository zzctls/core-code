from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = ROOT / "train_diffusion_network_2.py"


def test_training_reports_all_iou_metrics_as_percentages():
    source = TRAIN_SOURCE.read_text(encoding="utf-8")

    assert "class_iou_percent = class_jaccard * 100.0" in source
    assert "val_miou_percent = m_jaccard * 100.0" in source
    assert "completion_iou_percent = acc_completion * 100.0" in source
    assert "{class_iou_percent[i]:.3f}%" in source
    assert "{completion_iou_percent:.3f}%" in source
    assert "{val_miou_percent:.3f}%" in source
    assert 'writer.add_scalar("mIoU_val_mean", val_miou_percent, start_epoch)' in source
    assert 'writer.add_scalar("iou_val_mean", completion_iou_percent, start_epoch)' in source
