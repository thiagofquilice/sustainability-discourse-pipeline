# Colab Full Run: Micro-Topic Evolution

This package is dedicated only to the Colab execution of the full micro-topic evolution workflow:

1. Phase 01: annual summaries
2. Phase 02: phase-based evolution narratives

It is intentionally separated from:

- the earlier 6-topic Gemma validation work
- the isolated single-micro-topic Colab test
- the local Ollama runs

## Folder structure

- `inputs/`
  - upload these files to Colab
  - a ready-made zip bundle is included to avoid many separate uploads
- `outputs/`
  - this documents what to download after the Colab run
- `run_hf_gemma_micro_topic_year_summaries_colab.py`
  - Phase 01 runner
- `run_hf_gemma_micro_topic_evolution_synthesis_colab.py`
  - Phase 02 runner
- `colab_micro_topic_evolution_full_run.ipynb`
  - notebook that runs the full process

## Recommended Colab flow

1. Open `colab_micro_topic_evolution_full_run.ipynb`
2. Connect to a GPU runtime
3. Upload `inputs/micro_topic_evolution_full_inputs.zip` to `/content`
4. Unzip it in `/content`
5. Run the notebook top to bottom
6. Download the final zip produced by the notebook

## Method wording to preserve

- BERTopic `topics_over_time` is used as temporal topic tracking / temporal topic representation.
- The BERTopic micro-topic identity stays fixed after fitting.
- Year-specific words reflect temporal variation in wording and prevalence.
- Do not describe this workflow as a strict dynamic topic model with topics re-fitted each year.
