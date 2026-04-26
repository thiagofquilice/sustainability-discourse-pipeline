# Sustainability Discourse Pipeline

Small reproduction scripts for the paper's corporate-centered sustainability
discourse analysis.

This repository intentionally contains only the paper-facing scripts. It does
not include raw corpora, embeddings, LLM outputs, BERTopic model folders,
notebooks, or large intermediate files.

## Expected inputs

Place the final derived CSV files in `data/final_artifacts/`:

```text
corporate_focus_master_review.csv
corporate_focus_commented_decision_rows.csv
included_corporate_groups.csv
macro_topic_document_counts_final_by_source.csv
corporate_topic_external_relative_longitudinal.csv
aggregate_spearman_peak_summary_table.csv
individual_spearman_peak_summary_table.csv
```

These files are ignored by Git. They are expected to come from the final
analysis workspace used for the paper. See `docs/data_package.md` for a
file-by-file description.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_all.py \
  --input-dir data/final_artifacts \
  --output-dir outputs/paper_results
```

The command writes summary CSVs and figures to `outputs/paper_results/`.

## Streamlit reader companion

The repository also includes a Streamlit companion app for readers of the
paper. It is organized as a guided reading path rather than a technical
dashboard: overview of the study, domain-level patterns, source-specific
topics, corporate-external relations, and a secondary timing-diagnostics view.
The app uses a separate derived data package in `data/app_data/`, which is
ignored by Git.

Build the app data from the final paper pipeline workspace:

```bash
python scripts/build_streamlit_app_data.py \
  --pipeline-root /path/to/paper_6topic_discourse_pipeline \
  --output-dir data/app_data \
  --public-safe
```

Then run the companion app:

```bash
streamlit run streamlit_app.py
```

The default `--public-safe` mode exports only short representative snippets,
metadata, year, source, chunk IDs, document IDs, and source links when available.
It does not export full representative-document text to `data/app_data/`.
Temporal charts use annual document prevalence: topic documents in a given year
divided by retained documents from the same source, macro-topic, and year.
Merged-topic composition tables list only original microtopic IDs, labels,
counts, and top-word representations. The advanced timing-diagnostics view
reports the simplified diagnostics used in the paper: same-year Spearman
correlations, first active-year gaps, and peak-year gaps.

## Method supplement

Paper-facing methodological notes are provided in `docs/method/`. These files
replace the planned appendix material and document the final workflow in stages:

- corpus construction, taxonomy, embedding assignment, and validation;
- BERTopic modeling, review, and merge procedures;
- corporate-centered candidate selection, thresholds, manual recovery, and
  final review outcomes;
- longitudinal prevalence and source-relation diagnostics.

Prompt templates used for validation and temporal synthesis are included in
`docs/method/prompts/`.

## Scripts

- `scripts/01_build_macro_results.py`: builds the sample construction,
  macro-topic document counts, macro-topic coverage, macro-topic summary, and
  simplified source-relation timing CSVs.
- `scripts/02_render_macro_figures.py`: renders the sample, document-count,
  coverage, macro-summary, and source-relation timing figures.
- `scripts/03_render_longitudinal_panels.py`: renders longitudinal panels by
  macro-topic from the final annual-prevalence series.
- `scripts/build_streamlit_app_data.py`: builds the public-safe data package for
  the Streamlit companion app, including annual-prevalence series and temporal
  relation tables, and merged-topic composition tables.
- `scripts/run_all.py`: runs the three steps above.

## Notes

The scripts assume the final artifact schemas used in the paper. They are kept
simple on purpose and do not try to support earlier experimental formats.
