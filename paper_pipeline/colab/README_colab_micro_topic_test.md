# Colab Micro-Topic Test

This package runs the annual-summary stage for one isolated BERTopic micro-topic in Colab with Hugging Face Gemma.

## Test target

- subgroup: `academic_T4`
- micro topic id: `4`
- topic name: `4_water_groundwater_basin_river`

## Files to upload to Drive or Colab working directory

- `micro_topic_year_evidence.csv`
- `selected_micro_topics.csv`
- `run_hf_gemma_micro_topic_year_summaries_colab.py`

## Recommended Colab command

```bash
python run_hf_gemma_micro_topic_year_summaries_colab.py \
  --input-csv micro_topic_year_evidence.csv \
  --output-dir year_summaries \
  --model-name google/gemma-4-E4B-it \
  --batch-size 8 \
  --save-every 25
```

## Expected output files

- `year_summaries/micro_topic_year_summaries.csv`
- `year_summaries/micro_topic_year_summaries.jsonl`
- `year_summaries/raw_responses.jsonl`
- `year_summaries/error_log.csv`
- `year_summaries/run_manifest.json`

## Method wording to preserve

- BERTopic `topics_over_time` is used as temporal topic tracking / temporal topic representation.
- The BERTopic micro-topic stays fixed.
- The year-specific words show how topic wording and prevalence vary over time.
- Do not describe this as a strict dynamic topic model with topics re-estimated each year.
