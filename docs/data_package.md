# Data Package Expected By The Reproduction Scripts

The repository does not include raw data or large intermediate artifacts. To
run the paper-facing scripts, place the final derived CSV files below in
`data/final_artifacts/`.

| File | Purpose |
|---|---|
| `corporate_focus_master_review.csv` | Master corporate-centered review table linking non-corporate candidate topics to corporate anchors. |
| `corporate_focus_commented_decision_rows.csv` | Final review decisions used to classify external topics as aligned, relevant but unpaired, or excluded. |
| `included_corporate_groups.csv` | Final retained corporate anchor inventory. |
| `macro_topic_document_counts_final_by_source.csv` | Final document counts by source and macro-topic. |
| `corporate_topic_external_relative_longitudinal.csv` | Annual prevalence series for corporate anchors and aligned academic/media aggregates. |
| `aggregate_spearman_peak_summary_table.csv` | Paper-facing aggregate source-relation diagnostics: same-year Spearman, first active-year gap, and peak-year gap. |
| `individual_spearman_peak_summary_table.csv` | Paper-facing individual topic-pair diagnostics using the same simplified metrics. |

The Streamlit companion app uses a richer derived package built from the final
pipeline workspace:

```bash
python scripts/build_streamlit_app_data.py \
  --pipeline-root /path/to/paper_6topic_discourse_pipeline \
  --output-dir data/app_data \
  --public-safe
```

The `--public-safe` mode exports short snippets and metadata only. It does not
export full representative-document text, raw corpora, embeddings, or model
folders.
