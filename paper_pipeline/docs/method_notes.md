# Method notes

## Main classification flow

1. Cosine similarity uses the element list of each 6-topic label.
2. Best-only top-1 assignment is the main operational path.
3. Gemma validates the assigned topic with the stricter `mention_elements` prompt.
4. Supervised BERTopic is fit only on full best-only `Gemma yes`.

## Why keep Gemma in the flow

The current plan preserves Gemma in the pipeline because the operational sequence matches the earlier line, but the stricter prompt and revised taxonomy should make it easier to audit and recalibrate.

## Why a single unified root

The previous line became difficult to manage because related artifacts were spread across:

- `outputs/`
- `notebooks/`
- `scripts/`

This root is designed to be future GitHub-ready and paper-release-ready.
