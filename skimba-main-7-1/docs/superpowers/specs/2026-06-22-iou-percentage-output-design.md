# IoU Percentage Output Design

## Objective

Make every user-facing IoU and mIoU value in `train_diffusion_network_2.py` use percentage units, while preserving the correctness and ordering of the underlying evaluation calculations.

## Scope

- Convert per-class IoU, completion IoU, current mIoU, and best mIoU displays to percentages.
- Apply the same units to terminal output, `output_*.txt`, and TensorBoard scalars.
- Append `%` to human-readable terminal and text-file values.
- Keep the existing three-decimal display precision.
- Do not change the confusion-matrix calculation, class averaging, checkpoint selection, model behavior, or loss reporting.

## Design

The evaluator and confusion-matrix formulas continue to produce ratios in `[0, 1]`. Immediately after each raw metric is calculated, the training script derives an explicitly named percentage value by multiplying by `100.0`. Only percentage values cross user-facing output boundaries.

Checkpoint ranking continues to compare mIoU values in one consistent unit. Since multiplication by a positive constant preserves ordering, selecting the best checkpoint remains equivalent to comparing raw ratios. The checkpoint filename and stored `Loss` field retain their existing percentage-scale behavior.

## Output Contract

- Per-class line: `IoU class ... = 12.345%`
- Validation summary: `Current val completion iou is 12.345% and Current val miou is 23.456%`
- Best-metric summary: `Current val miou is 23.456% while the best val miou is 24.567%`
- TensorBoard tags `mIoU_val_mean` and `iou_val_mean`: percentage-scale numeric values.

## Verification

Add a focused source-level regression test following the subproject's existing training-entry-point test style. The test will first fail against the mixed-unit implementation, then verify that all named output paths use explicit percentage values and that the human-readable output includes `%`. Run the focused test, the containing test module, and Python syntax compilation for the modified files.

Local verification establishes source and unit-test correctness only. Linux/CUDA runtime behavior and performance remain unavailable on the Apple Silicon development machine.
