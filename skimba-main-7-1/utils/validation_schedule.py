def _positive_epoch_interval(value, name):
    if value <= 0:
        raise ValueError("{} must be a positive epoch interval, got {}".format(name, value))
    return value


def validation_interval_for_epoch(epoch, train_hypers):
    first_phase_epochs = train_hypers.get("eval_first_phase_epochs")
    first_phase_interval = train_hypers.get("eval_first_phase_every_n_epochs")
    after_phase_interval = train_hypers.get(
        "eval_after_phase_every_n_epochs",
        train_hypers.get("eval_every_n_epochs", 5),
    )

    if (
        first_phase_epochs is not None
        and first_phase_interval is not None
        and epoch <= first_phase_epochs
    ):
        return _positive_epoch_interval(
            first_phase_interval,
            "eval_first_phase_every_n_epochs",
        )

    return _positive_epoch_interval(
        after_phase_interval,
        "eval_after_phase_every_n_epochs",
    )


def should_validate_epoch(epoch, max_epoch, train_hypers):
    if epoch == max_epoch:
        return True
    interval = validation_interval_for_epoch(epoch, train_hypers)
    return epoch % interval == 0
