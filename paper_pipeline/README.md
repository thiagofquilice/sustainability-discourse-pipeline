# Full Paper Pipeline Code

This folder contains the fuller method code used to build the paper pipeline.
It is included for transparency and inspection. It is not a data release.

The top-level `scripts/` directory in the repository is the lightweight
paper-facing reproduction layer: it rebuilds final tables and figures from final
derived artifacts. This `paper_pipeline/` directory goes further upstream and
contains the code used for embedding assignment, LLM validation, BERTopic
modeling, topic review/merge, corporate-centered selection, temporal synthesis,
and final paper exports.

## What Is Included

- `catalog/`: the six macro-topic descriptor catalog used for environmental
  issue assignment.
- `config/`: an example configuration with portable placeholder paths.
- `upstream_preparation/`: optional corpus construction, BGE embedding
  generation, and source/industry subset scripts used before the six-topic
  paper workflow.
- `utils/`: helper modules required by the upstream corpus construction script.
- `scripts/`: the main local Python scripts used across the final workflow.
- `colab/`: Gemma/Hugging Face scripts and prompt material used for LLM
  validation and temporal synthesis in Colab/local GPU settings.
- `docs/`: working method notes from the project.

## What Is Not Included

The folder does not include raw corpora, SEC filings, Guardian text, Semantic
Scholar records, embeddings, BERTopic model folders, generated outputs,
workbooks, notebooks, zips, or representative-document exports. Generated
folders such as `paper_pipeline/outputs/`, `paper_pipeline/data/`, and Colab
`inputs/` or `outputs/` folders are ignored by Git.

## Main Workflow Map

The exact execution order depended on the available generated artifacts, but
the final paper-facing workflow is represented by these stages:

0. Corpus preparation and document embeddings:
   `upstream_preparation/prepare_corpus.py`,
   `upstream_preparation/run_topic_model.py`, and
   `upstream_preparation/build_three_source_subset.py`.
1. Macro-topic descriptor catalog:
   `scripts/build_6topic_catalog.py` and `catalog/six_topic_discourse_catalog.*`.
2. Embedding/cosine assignment:
   `scripts/run_6topic_discourse_cosine.py`.
3. LLM binary validation:
   `colab/run_hf_gemma_binary_validation_6topic_local.py` and related Colab
   package builders.
4. BERTopic source-by-domain microtopic modeling:
   `scripts/run_6topic_micro_unsupervised.py` and
   `scripts/run_6topic_bertopic_phase.py`.
5. Topic review and merge:
   `scripts/build_microtopic_merge_first_review.py`,
   `scripts/build_microtopic_merge_first_group_review.py`,
   `scripts/materialize_microtopic_group_review_mapping.py`, and
   `scripts/build_merged_microtopic_outputs.py`.
6. Temporal summaries and evolution narratives:
   `scripts/build_micro_topic_evolution_inputs.py`,
   `scripts/run_micro_topic_year_summaries.py`,
   `scripts/run_micro_topic_evolution_synthesis.py`, and
   `colab/micro_topic_evolution_full/run_hf_gemma_micro_topic_evolution_synthesis_colab.py`.
7. Corporate-centered review and sample construction:
   `scripts/build_corporate_focus_review.py`,
   `scripts/build_corporate_focus_hybrid_review_workbook.py`,
   `scripts/build_corporate_focus_override_packages.py`, and
   `scripts/build_corporate_focus_relative_longitudinal_series.py`.
8. Source-relation diagnostics and paper outputs:
   `scripts/analyze_corporate_focus_source_topic_relations.py`,
   `scripts/build_spearman_peak_summary_table.py`,
   `scripts/render_corporate_focus_relative_longitudinal_figures.py`, and
   `scripts/build_paper_appendix_method_tables.py`.

## Running The Full Pipeline

The code assumes external inputs with schemas matching the project workspace.
Start by copying and adapting:

```bash
cp paper_pipeline/config/paper_6topic_pipeline_config.example.json \
  paper_pipeline/config/paper_6topic_pipeline_config.json
```

Then update the paths to point to your local corpus, embedding files, and output
workspace. Most scripts should be run from the repository root so that the
portable `paper_pipeline/...` placeholders resolve consistently.

For dependencies, use:

```bash
pip install -r requirements-full.txt
```

Gemma/Hugging Face scripts require access to the selected model and may require
a local or Colab GPU environment.

If starting from raw JSONL inputs rather than from an already prepared corpus,
see `upstream_preparation/README.md` first.
