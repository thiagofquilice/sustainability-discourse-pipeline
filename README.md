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
  --pipeline-root /path/to/final_pipeline_workspace \
  --output-dir data/app_data \
  --public-safe
```

Then run the companion app:

```bash
streamlit run streamlit_app.py
```

### Public deployment

The repository includes a public-facing `data/app_data/` package for Streamlit
deployment. It contains derived topic metadata, annual prevalence series,
representative research units or source-limited excerpts, document IDs, source
metadata, dates when available, and links when available. It does not contain
raw corpora, embeddings, or BERTopic model folders.

To publish on Streamlit Community Cloud:

1. Open `https://share.streamlit.io/`.
2. Choose this GitHub repository.
3. Set the main file path to `streamlit_app.py`.
4. Deploy from the `main` branch.

The default `--public-safe` mode uses a source-aware display policy. Academic
and corporate examples are exported as the full abstract/chunk units used in the
research, with links or source references when available. Guardian media
examples are exported as excerpts capped at 300 words, with article date and a
link to the original Guardian article. The Guardian cap is intentional because
Guardian terms govern reuse of Guardian content. Use `--local-full-text` only for
private/local review.
Temporal charts use annual document prevalence: topic documents in a given year
divided by retained documents from the same source, macro-topic, and year.
Merged-topic composition tables list only original microtopic IDs, labels,
counts, and top-word representations. The advanced timing-diagnostics view
reports the simplified diagnostics used in the paper: same-year Spearman
correlations, first active-year gaps, and peak-year gaps.

## Full paper pipeline scripts

The repository also includes the fuller method code used during the paper
pipeline in `paper_pipeline/`. This is an archival, code-only layer for readers
who want to inspect or rerun the method from earlier stages if the external
input data and credentials are available. It includes scripts for:

- upstream corpus chunking, source/industry subset construction, and BGE
  document-embedding generation;
- descriptor embedding and cosine assignment;
- LLM/Gemma validation and temporal synthesis;
- BERTopic fitting for the source-by-domain microtopic models;
- topic review, merge, corporate-centered selection, and final paper tables.

The full pipeline layer does not include raw corpora, embeddings, model folders,
representative-document exports, notebooks, or generated outputs. Its example
configuration is `paper_pipeline/config/paper_6topic_pipeline_config.example.json`.
Use `requirements-full.txt` for the heavier optional dependencies needed by
the full pipeline.

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
- `scripts/build_streamlit_app_data.py`: builds the public Streamlit data package,
  including source-aware representative evidence, annual-prevalence series,
  temporal relation tables, and merged-topic composition tables.
- `scripts/run_all.py`: runs the three steps above.

## Notes

The scripts assume the final artifact schemas used in the paper. They are kept
simple on purpose and do not try to support earlier experimental formats.
