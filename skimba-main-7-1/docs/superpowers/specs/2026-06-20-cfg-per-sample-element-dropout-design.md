# CFG Per-Sample Element Dropout Design

## Goal

Change CFG condition dropout during training from dropping whole batch samples to independently dropping condition elements within every sample.

## Behavior

- The merged condition tensor keeps its existing `[B, C, W, L, H]` shape.
- For `0 < condition_dropout_prob < 1`, generate one random value for every tensor element.
- Keep an element when its random value is greater than `condition_dropout_prob`; otherwise replace it with zero.
- With the default probability `0.1`, every sample therefore has approximately 10% of its condition elements removed independently.
- Preserve the existing boundary behavior: probability `0` keeps every element and probability `1` removes every element.
- Do not rescale retained values. This is condition masking for CFG training, not activation dropout.
- CFG sampling behavior remains unchanged.

## Implementation

Update `LatentDiffusion._sample_condition_keep_mask` to create a random tensor with the same shape and device as `context_emb_all`. Remove the batch-sized random tensor and mask expansion, which currently cause entire samples to be kept or removed together.

Update the method and training docstrings to describe element-level masking within each sample.

## Tests

- Add a behavioral test proving a nontrivial probability produces mixed kept and dropped elements inside each sample.
- Verify the generated mask has the same shape and boolean dtype as the condition tensor.
- Verify probabilities `0` and `1` return all-keep and all-drop masks.
- Update the existing source regression test so it rejects the previous batch-sized mask implementation.
- Run the focused CFG tests, followed by the available project test suite if its dependencies permit.

## Scope

Only training-time condition masking changes. Condition merging, UNet calls, guidance calculation, sampling, configuration defaults, and tensor shapes remain unchanged.
