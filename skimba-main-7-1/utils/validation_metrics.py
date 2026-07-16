import torch


def mask_invalid_voxels(target, invalid_voxels, ignore_label):
    """Return a target copy whose invalid voxels use the loss ignore label."""
    invalid_mask = torch.as_tensor(
        invalid_voxels,
        dtype=torch.bool,
        device=target.device,
    )
    if invalid_mask.ndim == target.ndim - 1:
        if target.shape[0] != 1:
            raise ValueError(
                "A single invalid mask can only be applied to a batch of size 1"
            )
        invalid_mask = invalid_mask.unsqueeze(0)
    if invalid_mask.shape != target.shape:
        raise ValueError(
            f"Invalid mask shape {tuple(invalid_mask.shape)} does not match "
            f"target shape {tuple(target.shape)}"
        )
    return target.masked_fill(invalid_mask, ignore_label)
