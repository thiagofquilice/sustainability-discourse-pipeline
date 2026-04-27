# Colab runtime for the unified 6-topic paper pipeline

This Colab setup is intentionally based on the exact runtime bootstrap that actually worked with Gemma 4 in Colab. It is not optional cleanup.

## Official assets

- Notebook: `colab/colab_gemma_validation_6topic.ipynb`
- Runner: `colab/run_hf_gemma_binary_validation_6topic_local.py`
- Prompt: `colab/gemma_prompt_template.txt`

## Default runtime settings

- `MODEL_NAME = google/gemma-4-E4B-it`
- `BATCH_SIZE = 256`
- `SAVE_EVERY = 2048`

Fallback for weaker runtimes:

- `BATCH_SIZE = 128`
- `SAVE_EVERY = 1024`

## Why the notebook is verbose

The notebook keeps:

1. package uninstall
2. clean reinstall with the CUDA 12.8 PyTorch index
3. version checks
4. Hugging Face environment variables
5. a Gemma smoke test

That sequence came from the known-good Colab session that finally worked. Keeping it reduces avoidable runtime failures.

## What to upload to Drive

From this pipeline root:

- `outputs/pilot_1000/input/six_topic_pilot_1000.parquet`
- `outputs/full_run/input/six_topic_best_only_remaining.parquet`
- `colab/run_hf_gemma_binary_validation_6topic_local.py`

Optional reference upload:

- `colab/gemma_prompt_template.txt`

## Recommended output folders in Drive

- `six_topic_pilot_output/`
- `six_topic_best_only_full_output/`
- `six_topic_extra_candidates_output/`

## After the pilot comes back

Move returned pilot files into:

- `outputs/pilot_1000/output/`

Then run:

- `scripts/build_6topic_post_gemma_manual_review.py`
- `scripts/build_6topic_paper_audit_sample.py`

Only after that should you launch the best-only full run.
