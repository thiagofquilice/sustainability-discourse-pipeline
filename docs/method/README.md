# Method Supplement

This folder contains paper-facing methodological notes that replace the planned
appendix references. It is organized around the stages of the study rather than
around exploratory development history.

## Where To Point From The Paper

Use this repository folder when the paper says:

> Additional details on candidate selection, similarity thresholds, manual
> recovery, and final review criteria are reported in Appendix X.

Point to:

- `docs/method/04_corporate_centered_selection.md`

Use this repository folder when the paper says:

> Further details on embedding, descriptor construction, thresholds, prompts,
> and audit procedures are reported in Appendix X.

Point to:

- `docs/method/02_embedding_descriptor_validation.md`
- `docs/method/03_microtopic_modeling_and_review.md`
- `docs/method/prompts/`

## Files

- `01_method_stages.md`: concise stage-by-stage overview of the full method.
- `02_embedding_descriptor_validation.md`: environmental taxonomy, descriptor
  construction, embedding assignment, thresholds, and validation prompt.
- `03_microtopic_modeling_and_review.md`: BERTopic submodels, topic review,
  merge logic, and audit procedures.
- `04_corporate_centered_selection.md`: corporate anchors, cross-source
  candidate selection, manual recovery, and final review outcomes.
- `05_longitudinal_and_temporal_relation_metrics.md`: annual prevalence,
  longitudinal panels, same-year Spearman, first active-year gaps, and peak-year
  gaps.
- `prompts/`: prompt templates used for binary validation, yearly topic
  summaries, temporal synthesis, and same-issue/discourse-function review.

## Scope

These notes are intended to document the final paper-facing workflow. They do
not include raw data, model folders, embeddings, full representative texts, or
large intermediate outputs.
