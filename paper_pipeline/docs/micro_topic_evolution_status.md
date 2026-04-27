# Micro Topic Evolution Status

Last updated: 2026-04-14

## What this workflow is

This is the new paper-facing interpretation layer built on top of:

- `outputs/bertopic_micro_unsupervised/`

It is designed to explain how selected BERTopic micro-topics evolve over time within each `source x assigned_label` subgroup.

The workflow uses two LLM stages with local Ollama:

1. annual reading of selected `subgroup x micro_topic x year`
2. phase-based synthesis across years for each selected micro-topic

The LLM runtime is:

- `gemma4:e4b`

Method note:

- BERTopic `topics_over_time` is treated as temporal topic tracking / temporal topic representation
- it is **not** described as a strict dynamic topic model with re-fitted topics over time

## Scripts created for this workflow

These scripts now exist in:

- `scripts/build_micro_topic_evolution_inputs.py`
- `scripts/run_micro_topic_year_summaries.py`
- `scripts/run_micro_topic_evolution_synthesis.py`
- `scripts/build_micro_topic_evolution_summary.py`
- `scripts/run_micro_topic_evolution_pipeline.py`
- `scripts/micro_topic_evolution_common.py`

## Current deterministic input layer

Built outputs:

- `outputs/bertopic_micro_evolution/selected_micro_topics.csv`
- `outputs/bertopic_micro_evolution/selected_micro_topics_by_subgroup.csv`
- `outputs/bertopic_micro_evolution/micro_topic_year_evidence.csv`
- `outputs/bertopic_micro_evolution/micro_topic_year_evidence.jsonl`
- `outputs/bertopic_micro_evolution/selection_manifest.json`

Current counts from `selection_manifest.json`:

- `selected_micro_topic_count = 121`
- `annual_evidence_row_count = 2448`
- `subgroup_count = 18`

Selection rule:

- exclude BERTopic outlier topic `-1`
- keep micro-topics until cumulative subgroup coverage reaches `75%`
- cap at `10` topics per subgroup
- floor at `1`

Year-level evidence rule:

- use up to `5` most representative chunks from the same `micro_topic_id` and `year`
- include `topics_over_time.csv` words for that same year

## What has already been validated

### Annual summary stage

The annual runner was validated successfully on a minimal test case.

Files showing a successful 1-row validation:

- `outputs/bertopic_micro_evolution/year_summaries/micro_topic_year_summaries.csv`
- `outputs/bertopic_micro_evolution/year_summaries/raw_responses.jsonl`
- `outputs/bertopic_micro_evolution/year_summaries/run_manifest.json`

Important runtime fixes already applied:

- chunk text is shortened before entering the prompt
- Ollama is called with:
  - `--hidethinking`
  - `--think=false`
  - `--nowordwrap`
- ANSI escape codes are stripped before JSON parsing
- raw model responses are saved even when parsing fails

### Evolution synthesis stage

The synthesis runner was also validated successfully on a minimal 1-row case.

Files:

- `outputs/bertopic_micro_evolution/evolution_summaries/micro_topic_evolution_narratives.csv`
- `outputs/bertopic_micro_evolution/evolution_summaries/raw_responses.jsonl`
- `outputs/bertopic_micro_evolution/evolution_summaries/run_manifest.json`

Important note:

- the synthesis directory currently reflects only the small validation run
- the **full synthesis has not yet been launched**

### Summary / figure layer

The downstream summary builder has already been run and is functional.

Files:

- `outputs/bertopic_micro_evolution/micro_topic_evolution_summary.csv`
- `outputs/bertopic_micro_evolution/subgroup_evolution_coverage_summary.csv`
- `outputs/bertopic_micro_evolution/methods_note.md`
- `outputs/bertopic_micro_evolution/methods_note_accessible_english.md`
- `outputs/bertopic_micro_evolution/figure_descriptions.md`
- `outputs/bertopic_micro_evolution/figure_descriptions_accessible_english.md`
- `outputs/bertopic_micro_evolution/figures/`

These will need to be rebuilt after the **full** annual and synthesis stages complete.

## Continuity handoff

This section reflects the **current recoverable state** and should be used if the session closes.

### Current state snapshot

#### Stage 1 annual summaries

Stage 1 is **complete**.

Files:

- `outputs/bertopic_micro_evolution/year_summaries/micro_topic_year_summaries.csv`
- `outputs/bertopic_micro_evolution/year_summaries/error_log.csv`
- `outputs/bertopic_micro_evolution/year_summaries/run_manifest.json`

Completion status:

- `completed_rows = 2448`
- `error_count = 0`
- expected line count in `micro_topic_year_summaries.csv` = `2448 + 1` header = `2449`

#### Stage 2 evolution synthesis

Stage 2 is **currently running** in `tmux`.

Session:

- `micro_topic_evolution_stage2`

Process:

- `run_micro_topic_evolution_synthesis.py`

Live progress should be checked from:

- `outputs/bertopic_micro_evolution/evolution_summaries/micro_topic_evolution_narratives.csv`

Current situation when this handoff was updated:

- `selected_micro_topic_count = 121`
- `micro_topic_evolution_narratives.csv` had `26` lines total
- this means about `25` completed narratives plus the header row
- rough remaining time estimate was about `4.5` to `5.5` hours

Important note:

- `outputs/bertopic_micro_evolution/evolution_summaries/run_manifest.json` is **not** a reliable live-progress tracker during the run
- use CSV line count instead

#### Cross-source matched-pair branch at threshold 0.65

The deterministic branch for the pair analysis has already been built in:

- `outputs/microtopic_cross_source_pairs_threshold_065/`

This branch is ready now and does **not** need Stage 2 to exist.

Key outputs already materialized:

- `retained_pairs_threshold_065.csv`
- `temporally_viable_pairs_balanced.csv`
- `pair_lag_analysis_summary.csv`
- `pair_precedence_classification.csv`
- `pair_precedence_summary_by_type.csv`
- `pair_precedence_summary_by_dyad.csv`
- `pair_precedence_summary_by_macro_topic.csv`
- `pair_precedence_summary_by_type_macro_topic.csv`
- `pair_join_ready_table.csv`

Current counts in that branch:

- retained semantic pairs at `0.65` = `35`
- balanced viable pairs = `25`
- lag-classified pairs = `25`

Current precedence split:

- `academic_leads_corporate = 4`
- `corporate_leads_academic = 3`
- `media_leads_corporate = 1`
- `synchronous_or_unclear = 17`

#### LLM enrichment status for the 0.65 pair branch

The post-Stage-2 enrichment step is already implemented and has been test-run once.

Files:

- `outputs/microtopic_cross_source_pairs_threshold_065/llm_enriched_pair_table.csv`
- `outputs/microtopic_cross_source_pairs_threshold_065/llm_enriched_pair_table.xlsx`
- `outputs/microtopic_cross_source_pairs_threshold_065/llm_enrichment_manifest.json`

Current status:

- enrichment code works
- current enriched row count is `0`
- this is expected because Stage 2 is still incomplete and there are not yet pairs with narratives on both sides

## How to monitor safely

Useful commands:

```bash
tmux ls
```

```bash
pgrep -af run_micro_topic_evolution_synthesis.py
```

```bash
wc -l paper_pipeline/outputs/bertopic_micro_evolution/evolution_summaries/micro_topic_evolution_narratives.csv
```

```bash
ps -p $(pgrep -f run_micro_topic_evolution_synthesis.py | head -n 1) -o pid,etime,%cpu,%mem,cmd
```

```bash
ollama ps
```

Notes:

- the progress signal is the growth of `micro_topic_evolution_narratives.csv`
- the `tmux` pane may look mostly blank because the runner writes directly to CSV/JSONL
- the synthesis `run_manifest.json` is not updated incrementally

## Exact next steps from this point

When Stage 2 finishes, do these steps in order:

1. verify synthesis completion
2. rerun the LLM enrichment step for the `0.65` pair branch
3. rebuild the final micro-topic evolution summary/figures

### 1. Verify synthesis completion

Check:

- `wc -l outputs/bertopic_micro_evolution/evolution_summaries/micro_topic_evolution_narratives.csv`
- compare against:
  - `selected_micro_topic_count = 121`

Expected:

- CSV line count should equal `121 + 1` header = `122`

Also confirm:

- `outputs/bertopic_micro_evolution/evolution_summaries/error_log.csv`

### 2. Rerun the pair-LLM enrichment step

Command:

```bash
data/external/project_repo/.venv/bin/python \
  paper_pipeline/scripts/enrich_cross_source_pairs_with_llm_evolution.py \
  --output-root paper_pipeline/outputs/microtopic_cross_source_pairs_threshold_065
```

This should refresh:

- `outputs/microtopic_cross_source_pairs_threshold_065/llm_enriched_pair_table.csv`
- `outputs/microtopic_cross_source_pairs_threshold_065/llm_enriched_pair_table.xlsx`
- `outputs/microtopic_cross_source_pairs_threshold_065/README_microtopic_cross_source_pairs_threshold_065.md`

### 3. Rebuild summary and figures

Command:

```bash
data/external/project_repo/.venv/bin/python \
  paper_pipeline/scripts/build_micro_topic_evolution_summary.py
```

This should refresh:

- `micro_topic_evolution_summary.csv`
- `subgroup_evolution_coverage_summary.csv`
- methods notes
- figure descriptions
- figures

## If Stage 2 dies and needs restart

The synthesis runner supports resume through:

- `outputs/bertopic_micro_evolution/evolution_summaries/micro_topic_evolution_narratives.csv`

Restart command:

```bash
data/external/project_repo/.venv/bin/python \
  paper_pipeline/scripts/run_micro_topic_evolution_synthesis.py
```

Current active session name, if still alive:

- `micro_topic_evolution_stage2`

## If the 0.65 pair branch needs rebuild

The deterministic branch can be rebuilt independently with:

```bash
data/external/project_repo/.venv/bin/python \
  paper_pipeline/scripts/run_cross_source_microtopic_pipeline.py \
  --output-root paper_pipeline/outputs/microtopic_cross_source_pairs_threshold_065 \
  --main-threshold 0.65
```

This rebuilds:

- semantic pair matching
- temporal filtering
- annual series
- lag analysis
- precedence classification

It does **not** depend on Stage 2.

## Interpretation caution to preserve

Keep this wording consistent in methods and results:

- BERTopic micro-topics are fixed after fitting
- `topics_over_time` is used to track variation in topic prevalence and word representation over time
- this supports temporal interpretation of topic evolution
- but it should **not** be described as a strict dynamic topic model in the classical sense
