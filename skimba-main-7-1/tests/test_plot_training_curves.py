import matplotlib.pyplot as plt

import plot_training_curves as curves


def test_parse_console_log_reads_both_validation_noise_mse_values(tmp_path):
    log_path = tmp_path / "training.log"
    log_path.write_text(
        "epoch 50 iter 7 total train loss 0.14000000\n"
        "epoch 50 iter 3 loss_val_conditional_mse 0.15000000 "
        "loss_val_cfg_mse 0.18000000 reconstruction_loss 1.20000000\n",
        encoding="utf-8",
    )

    scalars = curves.parse_console_log(log_path)

    assert scalars["train_loss_mse_mean"] == [(50, 0.14)]
    assert scalars["val_loss_conditional"] == [(50, 0.15)]
    assert scalars["val_loss"] == [(50, 0.18)]
    assert scalars["val_reconstruction_loss_mean"] == [(50, 1.2)]


def test_parse_console_log_keeps_legacy_validation_mse_without_fabricating_conditional(
    tmp_path,
):
    log_path = tmp_path / "legacy.log"
    log_path.write_text(
        "epoch 40 iter 2 loss_val_mse 0.17000000 "
        "reconstruction_loss 1.10000000\n",
        encoding="utf-8",
    )

    scalars = curves.parse_console_log(log_path)

    assert scalars["val_loss"] == [(40, 0.17)]
    assert "val_loss_conditional" not in scalars
    assert scalars["val_reconstruction_loss_mean"] == [(40, 1.1)]


def test_draw_noise_loss_series_uses_unambiguous_three_curve_labels():
    scalars = {
        "train_loss_mse_mean": [(1, 0.2)],
        "val_loss_conditional": [(1, 0.21)],
        "val_loss": [(1, 0.25)],
    }
    figure, axis = plt.subplots()

    curves.draw_noise_loss_series(axis, scalars)

    assert {line.get_label() for line in axis.lines} == {
        "Train conditional noise MSE",
        "Validation conditional noise MSE",
        "Validation CFG-guided noise MSE",
    }
    plt.close(figure)
