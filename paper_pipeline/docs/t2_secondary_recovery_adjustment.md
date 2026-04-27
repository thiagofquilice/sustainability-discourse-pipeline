# T2 Adjustment: `material recovery` + `secondary recovery projects`

This note records the final adjustment we made to `T2` in the 6-topic pipeline, why we made it, and where the canonical artifacts live.

## Why T2 was revised

The original `T2` list was too broad in the cosine stage, especially in `corporate`.

The main problem came from broad terms such as:

- `resource efficiency`
- `material efficiency`
- `water efficiency`
- `energy efficiency`
- `resource recovery`

These terms were pulling in text about:

- generic operational efficiency
- mining and extraction performance
- ore recovery
- reserves and processing
- production/cost language without substantive environmental content

Manual review of the post-Gemma `150` sample confirmed that `T2` was the weakest topic in the new 6-topic taxonomy and needed a clearer environmental-operational boundary.

## Final T2 list used for the latest remap

The latest accepted experimental `T2` variant is:

- `cleaner production`
- `waste reduction`
- `waste minimization`
- `material recycling`
- `recycling of materials`
- `material recovery`
- `wastewater reuse`
- `wastewater treatment and reuse`
- `circular material use`
- `hazardous chemicals management`
- `hazardous waste management`
- `life cycle assessment`
- `industrial waste management`
- `secondary recovery projects`

Important distinction:

- `resource recovery` was removed
- `material recovery` was kept
- `secondary recovery projects` was added after targeted testing

## Why `secondary recovery projects` was added

We tested whether adding `secondary recovery projects` could recover specific `corporate` cases that the stricter `T2` was excluding but that the user considered substantively relevant.

In targeted testing:

- it brought back the `UAVS ... chunk_0042` example
- it did **not** bring back the `NEM ... chunk_0017` example
- it did **not** reactivate the known `corporate no` `T2` examples that had been flagged in the reviewed sample

So the term was accepted as a controlled expansion of `T2`, even though it still needs downstream scrutiny.

## Canonical cosine remap artifacts

The canonical remap for this final `T2` variant is stored in:

- [summary.json](paper_pipeline/outputs/t2_secondary_recovery_cosine_round/summary.json)
- [old_t2_under_secondary_recovery_variant.parquet](paper_pipeline/outputs/t2_secondary_recovery_cosine_round/old_t2_under_secondary_recovery_variant.parquet)
- [transition_summary.csv](paper_pipeline/outputs/t2_secondary_recovery_cosine_round/transition_summary.csv)

Key remap totals from the old `T2` set:

- old `T2` rows: `56,817`
- remain `T2`: `38,882`
- changed to another topic: `5,333`
- changed to `OUTSIDE_SCOPE`: `12,602`

Changed-to-other breakdown:

- `T3`: `2,904`
- `T1`: `1,071`
- `T5`: `653`
- `T6`: `371`
- `T4`: `334`

Changed-to-other by source:

- `academic`: `4,715`
- `media`: `522`
- `corporate`: `96`

## Canonical changed-only Gemma rerun

Only the rows that changed from old `T2` to another topic were sent to the local Gemma rerun.

Canonical changed-only input:

- [old_t2_changed_to_other_topics.parquet](paper_pipeline/outputs/t2_secondary_recovery_cosine_round/old_t2_changed_to_other_topics.parquet)
- [old_t2_changed_to_other_topics.csv](paper_pipeline/outputs/t2_secondary_recovery_cosine_round/old_t2_changed_to_other_topics.csv)
- [old_t2_changed_to_other_topics_summary.csv](paper_pipeline/outputs/t2_secondary_recovery_cosine_round/old_t2_changed_to_other_topics_summary.csv)

Canonical changed-only row count:

- `5,333`

Local Gemma rerun output directory:

- [gemma_local_full](paper_pipeline/outputs/t2_secondary_recovery_cosine_round/gemma_local_full)

Expected files there:

- `validation_output.csv`
- `error_log.csv`
- `run_manifest.json`
- `prompt_used.txt`
- `run.log`

## Adjustment of the full best-only output

Once the changed-only Gemma rerun finishes, the full best-only output is adjusted as follows:

1. keep all original non-`T2` rows unchanged
2. keep old `T2` rows that remain `T2`
3. drop old `T2` rows that become `OUTSIDE_SCOPE`
4. replace old `T2 -> other topic` rows with the changed-only Gemma rerun output

The automated report builder for this step is:

- [data/external/project_repo/codex_staging/build_adjusted_full_report.py](data/external/project_repo/codex_staging/build_adjusted_full_report.py)

The final adjusted outputs are written to:

- [adjusted_with_t2_secondary_recovery](paper_pipeline/outputs/full_run/adjusted_with_t2_secondary_recovery)

Main report files:

- [manual_vs_adjusted_gemma_by_source.csv](paper_pipeline/outputs/full_run/adjusted_with_t2_secondary_recovery/manual_vs_adjusted_gemma_by_source.csv)
- [manual_vs_adjusted_gemma_by_source_topic.csv](paper_pipeline/outputs/full_run/adjusted_with_t2_secondary_recovery/manual_vs_adjusted_gemma_by_source_topic.csv)
- [adjustment_manifest.json](paper_pipeline/outputs/full_run/adjusted_with_t2_secondary_recovery/adjustment_manifest.json)

Build log for the automatic adjustment step:

- [adjusted_with_t2_secondary_recovery_report_build.log](paper_pipeline/outputs/full_run/adjusted_with_t2_secondary_recovery_report_build.log)

## Current baseline comparison before adjustment finishes

From the current reviewed `150` post-Gemma sample:

- `academic`: manual keep rate `98.0%`
- `corporate`: manual keep rate `83.3%`
- `media`: manual keep rate `88.0%`

From the current original full best-only Gemma output:

- `academic`: Gemma keep rate `90.9%`
- `corporate`: Gemma keep rate `91.8%`
- `media`: Gemma keep rate `80.6%`

These are only the baseline pre-adjustment rates. The adjusted report should be treated as the authoritative post-remap comparison.
