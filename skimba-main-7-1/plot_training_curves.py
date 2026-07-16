import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


OUTPUT_RE = re.compile(r"output_(\d+)\.txt$")
VAL_RE = re.compile(
    r"Current val completion iou is\s+([0-9.+-eE]+)%?"
    r"\s+and Current val miou is\s+([0-9.+-eE]+)%?"
)
CLASS_IOU_RE = re.compile(
    r"IoU class\s+(\d+)\s+\[([^\]]+)\]\s*=\s*([0-9.+-eE]+)"
)
BEST_RE = re.compile(r"best_(\d+)_([0-9.+-eE]+)\.pth$")
TRAIN_LOSS_RE = re.compile(
    r"epoch\s+(\d+)\s+iter\s+\d+\s+total train loss\s+([0-9.+-eE]+)"
)
VAL_LOSS_RE = re.compile(
    r"epoch\s+(\d+)\s+iter\s+\d+\s+loss_val_mse\s+([0-9.+-eE]+)"
    r"\s+reconstruction_loss\s+([0-9.+-eE]+)"
)
VAL_LOSS_DUAL_RE = re.compile(
    r"epoch\s+(\d+)\s+iter\s+\d+\s+loss_val_conditional_mse\s+([0-9.+-eE]+)"
    r"\s+loss_val_cfg_mse\s+([0-9.+-eE]+)"
    r"\s+reconstruction_loss\s+([0-9.+-eE]+)"
)

NOISE_LOSS_TAGS = (
    "train_loss_mse_mean",
    "val_loss_conditional",
    "val_loss",
)
RECONSTRUCTION_LOSS_TAG = "val_reconstruction_loss_mean"
LOSS_TAGS = (*NOISE_LOSS_TAGS, RECONSTRUCTION_LOSS_TAG)
VALIDATION_TAGS = ("mIoU_val_mean", "iou_val_mean")


def mean(values):
    return sum(values) / len(values) if values else None


def parse_output_files(model_dir):
    validation_rows = []
    class_rows = []

    for path in sorted(Path(model_dir).glob("output_*.txt")):
        match = OUTPUT_RE.match(path.name)
        if not match:
            continue

        epoch = int(match.group(1))
        completion_iou = None
        semantic_miou = None

        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            validation_match = VAL_RE.search(line)
            if validation_match:
                completion_iou = float(validation_match.group(1))
                semantic_miou = float(validation_match.group(2))

            class_match = CLASS_IOU_RE.search(line)
            if class_match:
                class_rows.append(
                    {
                        "epoch": epoch,
                        "class_id": int(class_match.group(1)),
                        "class_name": class_match.group(2),
                        "iou": float(class_match.group(3)),
                    }
                )

        if completion_iou is not None and semantic_miou is not None:
            validation_rows.append(
                {
                    "epoch": epoch,
                    "completion_iou": completion_iou,
                    "semantic_miou": semantic_miou,
                }
            )

    validation_rows.sort(key=lambda row: row["epoch"])
    class_rows.sort(key=lambda row: (row["epoch"], row["class_id"]))
    return validation_rows, class_rows


def parse_best_checkpoints(model_dir):
    rows = []
    for path in Path(model_dir).glob("best_*.pth"):
        match = BEST_RE.match(path.name)
        if match:
            rows.append(
                {
                    "epoch": int(match.group(1)),
                    "best_semantic_miou": float(match.group(2)),
                }
            )
    return sorted(rows, key=lambda row: row["epoch"])


def parse_console_log(path):
    if not path:
        return {}

    train_losses = defaultdict(list)
    conditional_val_losses = defaultdict(list)
    val_losses = defaultdict(list)
    reconstruction_losses = defaultdict(list)

    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        train_match = TRAIN_LOSS_RE.search(line)
        if train_match:
            train_losses[int(train_match.group(1))].append(float(train_match.group(2)))

        dual_val_match = VAL_LOSS_DUAL_RE.search(line)
        if dual_val_match:
            epoch = int(dual_val_match.group(1))
            conditional_val_losses[epoch].append(float(dual_val_match.group(2)))
            val_losses[epoch].append(float(dual_val_match.group(3)))
            reconstruction_losses[epoch].append(float(dual_val_match.group(4)))
        else:
            val_match = VAL_LOSS_RE.search(line)
            if val_match:
                epoch = int(val_match.group(1))
                val_losses[epoch].append(float(val_match.group(2)))
                reconstruction_losses[epoch].append(float(val_match.group(3)))

    scalars = {}
    if train_losses:
        scalars["train_loss_mse_mean"] = [
            (epoch, mean(values)) for epoch, values in sorted(train_losses.items())
        ]
    if conditional_val_losses:
        scalars["val_loss_conditional"] = [
            (epoch, mean(values))
            for epoch, values in sorted(conditional_val_losses.items())
        ]
    if val_losses:
        scalars["val_loss"] = [
            (epoch, mean(values)) for epoch, values in sorted(val_losses.items())
        ]
    if reconstruction_losses:
        scalars["val_reconstruction_loss_mean"] = [
            (epoch, mean(values))
            for epoch, values in sorted(reconstruction_losses.items())
        ]
    return scalars


def read_tensorboard_scalars(log_dir):
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return {}, "tensorboard is not installed; TensorBoard logs were skipped"

    log_dir = Path(log_dir)
    event_files = sorted(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        return {}, f"no TensorBoard event files found under {log_dir}"

    scalars = defaultdict(list)
    for event_file in event_files:
        accumulator = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        try:
            accumulator.Reload()
        except Exception as exc:
            print(f"Warning: skipped unreadable event file {event_file}: {exc}")
            continue

        for tag in accumulator.Tags().get("scalars", []):
            scalars[tag].extend(
                (event.step, event.value) for event in accumulator.Scalars(tag)
            )

    return deduplicate_scalars(scalars), None


def deduplicate_scalars(scalars):
    result = {}
    for tag, values in scalars.items():
        by_step = {}
        for step, value in values:
            by_step[int(step)] = float(value)
        result[tag] = sorted(by_step.items())
    return result


def merge_scalars(primary, fallback):
    merged = dict(fallback)
    for tag, values in primary.items():
        if values:
            merged[tag] = values
    return merged


def validation_values_are_percent(rows):
    if not rows:
        return True
    return any(
        abs(row["completion_iou"]) > 1.0 or abs(row["semantic_miou"]) > 1.0
        for row in rows
    )


def normalize_validation_rows(rows):
    if validation_values_are_percent(rows):
        return [dict(row) for row in rows]

    normalized = []
    for row in rows:
        normalized.append(
            {
                **row,
                "completion_iou": row["completion_iou"] * 100.0,
                "semantic_miou": row["semantic_miou"] * 100.0,
            }
        )
    return normalized


def normalize_percent_series(values):
    if not values:
        return []
    already_percent = any(abs(value) > 1.0 for _, value in values)
    if already_percent:
        return list(values)
    return [(step, value * 100.0) for step, value in values]


def write_csv(path, rows, fieldnames):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_tag_name(tag):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


def next_run_dir(base_dir):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_numbers = []
    for path in base_dir.iterdir():
        match = re.fullmatch(r"run_(\d+)", path.name)
        if path.is_dir() and match:
            run_numbers.append(int(match.group(1)))

    run_dir = base_dir / f"run_{max(run_numbers, default=0) + 1:03d}"
    run_dir.mkdir()
    return run_dir


def resolve_log_dir(model_dir, log_dir):
    if log_dir:
        return Path(log_dir)

    candidates = (
        Path(model_dir) / "logs",
        Path(model_dir) / "logs_ddp",
        Path("logs"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(model_dir) / "logs"


def mark_empty(ax, message):
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        alpha=0.65,
    )


def plot_validation(ax, validation_rows, best_rows, scalars):
    plotted = False

    if validation_rows:
        rows = normalize_validation_rows(validation_rows)
        epochs = [row["epoch"] for row in rows]
        ax.plot(
            epochs,
            [row["semantic_miou"] for row in rows],
            marker="o",
            linewidth=2,
            label="Semantic mIoU",
        )
        ax.plot(
            epochs,
            [row["completion_iou"] for row in rows],
            marker="s",
            linewidth=2,
            label="Completion IoU",
        )
        plotted = True
    elif scalars.get("mIoU_val_mean"):
        values = normalize_percent_series(scalars["mIoU_val_mean"])
        ax.plot(
            [step for step, _ in values],
            [value for _, value in values],
            marker="o",
            linewidth=2,
            label="Semantic mIoU",
        )
        plotted = True

    if not validation_rows and scalars.get("iou_val_mean"):
        values = normalize_percent_series(scalars["iou_val_mean"])
        ax.plot(
            [step for step, _ in values],
            [value for _, value in values],
            marker="s",
            linewidth=2,
            label="Completion IoU",
        )
        plotted = True

    if best_rows:
        values = normalize_percent_series(
            [(row["epoch"], row["best_semantic_miou"]) for row in best_rows]
        )
        ax.step(
            [step for step, _ in values],
            [value for _, value in values],
            where="post",
            linestyle=":",
            linewidth=1.8,
            label="Best checkpoint mIoU",
        )
        plotted = True

    ax.set_title("Validation Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Percent (%)")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    else:
        mark_empty(ax, "No validation metrics found")


def epoch_limits(validation_rows, best_rows, scalars):
    epochs = []
    epochs.extend(row["epoch"] for row in validation_rows)
    epochs.extend(row["epoch"] for row in best_rows)
    for tag in (*LOSS_TAGS, "lr_rate"):
        epochs.extend(step for step, _ in scalars.get(tag, []))
    if not epochs:
        return None

    lower = min(epochs)
    upper = max(epochs)
    if lower == upper:
        return (lower - 1, upper + 1)
    return (lower, upper)


def plot_noise_losses(ax, scalars, x_limits=None):
    plotted = False
    labels = {
        "train_loss_mse_mean": "Train conditional noise MSE",
        "val_loss_conditional": "Validation conditional noise MSE",
        "val_loss": "Validation CFG-guided noise MSE",
    }
    for tag in NOISE_LOSS_TAGS:
        if not scalars.get(tag):
            continue
        ax.plot(
            [step for step, _ in scalars[tag]],
            [value for _, value in scalars[tag]],
            marker="o",
            linewidth=1.8,
            label=labels[tag],
        )
        plotted = True

    ax.set_title("Noise Prediction Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    else:
        if x_limits:
            ax.set_xlim(*x_limits)
        mark_empty(ax, "No noise-loss scalars found")


def noise_loss_values_after(scalars, start_epoch):
    return [
        value
        for tag in NOISE_LOSS_TAGS
        for step, value in scalars.get(tag, [])
        if step >= start_epoch
    ]


def padded_limits(values, padding_ratio=0.08):
    if not values:
        return None
    lower = min(values)
    upper = max(values)
    span = upper - lower
    padding = span * padding_ratio if span > 0 else max(abs(lower) * padding_ratio, 1e-3)
    return lower - padding, upper + padding


def draw_noise_loss_series(ax, scalars, start_epoch=None):
    styles = {
        "train_loss_mse_mean": {
            "label": "Train conditional noise MSE",
            "color": "tab:blue",
            "linestyle": "-",
            "linewidth": 1.6,
        },
        "val_loss_conditional": {
            "label": "Validation conditional noise MSE",
            "color": "tab:orange",
            "linestyle": "--",
            "linewidth": 2.0,
            "marker": "s",
            "markersize": 5,
        },
        "val_loss": {
            "label": "Validation CFG-guided noise MSE",
            "color": "tab:red",
            "linestyle": ":",
            "linewidth": 2.0,
            "marker": "o",
            "markersize": 5,
        },
    }
    plotted = False
    for tag in NOISE_LOSS_TAGS:
        values = scalars.get(tag, [])
        if start_epoch is not None:
            values = [(step, value) for step, value in values if step >= start_epoch]
        if not values:
            continue
        ax.plot(
            [step for step, _ in values],
            [value for _, value in values],
            **styles[tag],
        )
        plotted = True
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    else:
        mark_empty(ax, "No noise-loss scalars found")
    return plotted


def plot_noise_loss_comparison(scalars, output_path, zoom_start_epoch=50):
    import matplotlib.pyplot as plt

    fig, (ax_full, ax_zoom) = plt.subplots(
        1,
        2,
        figsize=(13, 4.8),
        constrained_layout=True,
    )

    draw_noise_loss_series(ax_full, scalars)
    ax_full.set_title("Full Training")
    ax_full.set_xlabel("Epoch")
    ax_full.set_ylabel("MSE loss")

    draw_noise_loss_series(ax_zoom, scalars, start_epoch=zoom_start_epoch)
    last_epoch = max(
        (step for tag in NOISE_LOSS_TAGS for step, _ in scalars.get(tag, [])),
        default=zoom_start_epoch,
    )
    ax_zoom.set_title(f"Late-stage Detail (Epoch {zoom_start_epoch}-{last_epoch})")
    ax_zoom.set_xlabel("Epoch")
    ax_zoom.set_ylabel("MSE loss")
    ax_zoom.set_xlim(left=zoom_start_epoch)
    zoom_limits = padded_limits(noise_loss_values_after(scalars, zoom_start_epoch))
    if zoom_limits:
        ax_zoom.set_ylim(*zoom_limits)

    fig.suptitle("Noise Prediction Loss: Full View and Late-stage Detail", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_reconstruction_loss(ax, scalars, x_limits=None):
    values = scalars.get(RECONSTRUCTION_LOSS_TAG, [])
    if values:
        ax.plot(
            [step for step, _ in values],
            [value for _, value in values],
            marker="o",
            linewidth=1.8,
            color="tab:green",
            label="Validation sampled semantic CE",
        )
        ax.legend()
    else:
        if x_limits:
            ax.set_xlim(*x_limits)
        mark_empty(ax, "No reconstruction-loss scalar found")
    ax.set_title("Validation Sampled Semantic Cross-Entropy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(True, alpha=0.3)


def class_iou_matrix(class_rows):
    epochs = sorted({row["epoch"] for row in class_rows})
    classes = sorted(
        {(row["class_id"], row["class_name"]) for row in class_rows},
        key=lambda item: item[0],
    )
    lookup = {
        (row["epoch"], row["class_id"]): row["iou"]
        for row in class_rows
    }
    matrix = [
        [lookup.get((epoch, class_id), float("nan")) for epoch in epochs]
        for class_id, _ in classes
    ]
    return epochs, classes, matrix


def plot_class_iou(ax, class_rows):
    if not class_rows:
        ax.set_title("Per-class IoU")
        mark_empty(ax, "No per-class IoU found")
        return

    epochs, classes, matrix = class_iou_matrix(class_rows)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("Per-class IoU")
    ax.set_xlabel("Validation epoch")
    ax.set_ylabel("Class")
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels(epochs, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([f"{class_id}: {name}" for class_id, name in classes], fontsize=7)
    ax.figure.colorbar(image, ax=ax, label="IoU")


def plot_learning_rate(ax, scalars, x_limits=None):
    if scalars.get("lr_rate"):
        ax.plot(
            [step for step, _ in scalars["lr_rate"]],
            [value for _, value in scalars["lr_rate"]],
            marker="o",
            linewidth=1.8,
            label="Learning rate",
        )
        ax.legend()
    else:
        if x_limits:
            ax.set_xlim(*x_limits)
        mark_empty(ax, "No learning-rate scalar found")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=False)
    ax.grid(True, alpha=0.3)


def plot_curves(validation_rows, class_rows, best_rows, scalars, output_path):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_validation = fig.add_subplot(grid[0, 0])
    ax_noise_loss = fig.add_subplot(grid[0, 1])
    ax_reconstruction_loss = fig.add_subplot(grid[1, 0])
    ax_lr = fig.add_subplot(grid[1, 1])
    x_limits = epoch_limits(validation_rows, best_rows, scalars)

    plot_validation(ax_validation, validation_rows, best_rows, scalars)
    plot_noise_losses(ax_noise_loss, scalars, x_limits=x_limits)
    plot_reconstruction_loss(ax_reconstruction_loss, scalars, x_limits=x_limits)
    plot_learning_rate(ax_lr, scalars, x_limits=x_limits)
    fig.suptitle("SKIMBA Training Curves", fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_outputs(out_dir, validation_rows, class_rows, best_rows, scalars):
    write_csv(
        out_dir / "validation_metrics.csv",
        normalize_validation_rows(validation_rows),
        ["epoch", "completion_iou", "semantic_miou"],
    )
    write_csv(
        out_dir / "per_class_iou.csv",
        class_rows,
        ["epoch", "class_id", "class_name", "iou"],
    )
    write_csv(
        out_dir / "best_checkpoints.csv",
        best_rows,
        ["epoch", "best_semantic_miou"],
    )
    for tag, values in scalars.items():
        write_csv(
            out_dir / f"{safe_tag_name(tag)}.csv",
            [{"epoch": step, "value": value} for step, value in values],
            ["epoch", "value"],
        )


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Plot SKIMBA diffusion training and validation curves."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing output_*.txt and best_*.pth.",
    )
    parser.add_argument(
        "--log-dir",
        default="",
        help="TensorBoard log directory. Defaults to model-dir/logs or ./logs.",
    )
    parser.add_argument(
        "--console-log",
        default="",
        help="Optional saved terminal log used when TensorBoard events are unavailable.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Base output directory. Defaults to model-dir/curves.",
    )
    parser.add_argument(
        "--flat-out-dir",
        action="store_true",
        help="Write directly into out-dir instead of creating run_###.",
    )
    parser.add_argument(
        "--noise-zoom-start",
        type=int,
        default=50,
        help="First epoch shown in the late-stage noise-loss detail panel.",
    )
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    model_dir = Path(args.model_dir).expanduser()
    log_dir = resolve_log_dir(model_dir, args.log_dir)
    base_out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir
        else model_dir / "curves"
    )
    if args.flat_out_dir:
        out_dir = base_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = next_run_dir(base_out_dir)

    validation_rows, class_rows = parse_output_files(model_dir)
    best_rows = parse_best_checkpoints(model_dir)
    tensorboard_scalars, warning = read_tensorboard_scalars(log_dir)
    console_scalars = parse_console_log(args.console_log)
    scalars = merge_scalars(tensorboard_scalars, console_scalars)

    write_outputs(out_dir, validation_rows, class_rows, best_rows, scalars)

    output_png = out_dir / "training_curves.png"
    noise_output_png = out_dir / "noise_loss_comparison.png"
    try:
        plot_curves(validation_rows, class_rows, best_rows, scalars, output_png)
        print(f"Wrote plot: {output_png}")
        plot_noise_loss_comparison(
            scalars,
            noise_output_png,
            zoom_start_epoch=args.noise_zoom_start,
        )
        print(f"Wrote plot: {noise_output_png}")
    except ImportError:
        print("Warning: matplotlib is not installed; CSV files were still written.")
        print("Install it with: pip install matplotlib")

    print(f"Output directory: {out_dir}")
    print(f"Validation epochs: {len(validation_rows)}")
    print(f"Per-class IoU rows: {len(class_rows)}")
    print(f"TensorBoard/console scalar tags: {len(scalars)}")
    if warning:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
