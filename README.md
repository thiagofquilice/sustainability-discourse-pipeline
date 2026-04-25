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
aggregate_pair_precedence_classification.csv
macro_topic_document_counts_final_by_source.csv
corporate_topic_external_relative_longitudinal.csv
```

These files are ignored by Git. They are expected to come from the final
analysis workspace used for the paper.

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

## Streamlit explorer

The repository also includes a Streamlit app for browsing the paper results by
macro-topic, source, corporate anchor, external counterpart, and lagged
corporate-external temporal relation. The app uses a separate derived data
package in `data/app_data/`, which is ignored by Git.

Build the app data from the final paper pipeline workspace:

```bash
python scripts/build_streamlit_app_data.py \
  --pipeline-root /path/to/paper_6topic_discourse_pipeline \
  --output-dir data/app_data \
  --public-safe
```

Then run the explorer:

```bash
streamlit run streamlit_app.py
```

The default `--public-safe` mode exports only short representative snippets,
metadata, year, source, chunk IDs, document IDs, and source links when available.
It does not export full representative-document text to `data/app_data/`.
Temporal charts use annual document prevalence: topic documents in a given year
divided by all source-domain documents in that same year.

## Scripts

- `scripts/01_build_macro_results.py`: builds the sample construction,
  macro-topic coverage, macro-topic summary, and temporal-summary CSVs.
- `scripts/02_render_macro_figures.py`: renders the sample, coverage, macro
  summary, and temporal-summary figures.
- `scripts/03_render_longitudinal_panels.py`: renders longitudinal panels by
  macro-topic from the final relative-salience series.
- `scripts/build_streamlit_app_data.py`: builds the public-safe data package for
  the Streamlit explorer, including annual-prevalence series and temporal
  relation tables.
- `scripts/run_all.py`: runs the three steps above.

## Notes

The scripts assume the final artifact schemas used in the paper. They are kept
simple on purpose and do not try to support earlier experimental formats.
