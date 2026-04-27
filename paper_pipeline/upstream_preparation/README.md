# Upstream Preparation Scripts

This folder contains the optional upstream scripts needed to reproduce the
initial corpus and embedding inputs used by the paper pipeline, assuming the
underlying raw/source files are available outside this repository.

These scripts are code-only. They do not include raw Guardian articles,
Semantic Scholar records, SEC filing text, embeddings, BERTopic model folders,
or generated outputs.

## Scripts

- `prepare_corpus.py`: standardizes and chunks the three-source corpus into
  `data/processed/full_corpus_chunked.parquet`.
- `run_topic_model.py`: computes BGE document embeddings and can fit a general
  BERTopic model over the prepared corpus. Use `--stop-after-embeddings` when
  only the embedding matrix is needed for the six-topic paper pipeline.
- `prepare_source_subset.py`: creates source-specific corpus and embedding
  subsets while preserving row alignment.
- `prepare_industry_subset.py`: creates industry-specific corpus and embedding
  subsets while preserving row alignment.
- `build_three_source_subset.py`: keeps full academic and media sources and
  selected corporate industries for the three-source paper corpus.

## Expected External Inputs

Place raw JSONL inputs under `paper_pipeline/data/raw/` or pass explicit paths.
The corpus preparation script expects:

- `guardian_dataset.jsonl`
- `papers_dataset.jsonl`
- one or more `10K*.jsonl` files

The generated parquet and embedding files remain under `paper_pipeline/data/`
and are ignored by Git.

## Example

```bash
python paper_pipeline/upstream_preparation/prepare_corpus.py \
  --project-root paper_pipeline

python paper_pipeline/upstream_preparation/run_topic_model.py \
  --input paper_pipeline/data/processed/full_corpus_chunked.parquet \
  --results-dir paper_pipeline/data/results \
  --stop-after-embeddings

python paper_pipeline/upstream_preparation/build_three_source_subset.py \
  --output-dir paper_pipeline/data/interim/three_source_oil_gas_metal_mining
```

After this upstream step, point
`paper_pipeline/config/paper_6topic_pipeline_config.example.json` to the
resulting corpus and embedding files before running the six-topic workflow.
