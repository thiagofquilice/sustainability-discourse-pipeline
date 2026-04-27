#!/usr/bin/env python3
"""Generate phase-based micro-topic evolution narratives with Gemma via local Ollama."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from micro_topic_evolution_common import (
    OLLAMA_MODEL,
    OUTPUT_ROOT,
    append_jsonl,
    canonicalize_stage2_payload,
    call_ollama,
    configure_logging,
    ensure_directory,
    extract_json_object,
    is_stage2_narrative_ready,
    resume_completed_keys,
    stage2_missing_core_fields,
    write_json,
)


PROMPT_TEMPLATE = """You are synthesizing the temporal evolution of one BERTopic micro-topic.

Task:
Read the annual summaries and infer 2 to 4 contiguous interpretive phases over time.

Important:
- Use only the evidence provided.
- The micro-topic identity is fixed after BERTopic fitting.
- The yearly words come from BERTopic temporal topic tracking (`topics_over_time`), which shows how word representation and prevalence vary over time.
- Do not describe this as a separately refit dynamic topic model.
- Keep phase ranges contiguous and non-overlapping.
- If there is little change, you may still use 2 phases with a stable interpretation.
- Reply with one JSON object only and no surrounding text.

Return exactly these fields:
- subgroup
- source
- assigned_label
- micro_topic_id
- topic_name_original
- topic_label_refined
- overall_summary
- phase_1_years
- phase_1_summary
- phase_2_years
- phase_2_summary
- phase_3_years
- phase_3_summary
- phase_4_years
- phase_4_summary
- evolution_pattern
- evidence_note

Metadata:
subgroup: {subgroup}
source: {source}
assigned_label: {assigned_label}
micro_topic_id: {micro_topic_id}
topic_name_original: {topic_name_original}
topic_size: {topic_size}
share_of_non_outlier: {share_of_non_outlier}
overall_keywords: {overall_keywords}

Annual evidence:
{annual_block}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-topics", type=Path, default=OUTPUT_ROOT / "selected_micro_topics.csv")
    parser.add_argument("--annual-summaries", type=Path, default=OUTPUT_ROOT / "year_summaries" / "micro_topic_year_summaries.csv")
    parser.add_argument("--year-evidence", type=Path, default=OUTPUT_ROOT / "micro_topic_year_evidence.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "evolution_summaries")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL)
    parser.add_argument("--timeout-seconds", type=int, default=420)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def build_annual_block(summary_rows: pd.DataFrame, evidence_rows: pd.DataFrame) -> str:
    lines = []
    merged = summary_rows.merge(
        evidence_rows[["subgroup", "micro_topic_id", "year", "year_frequency", "year_specific_words"]],
        on=["subgroup", "micro_topic_id", "year"],
        how="left",
    ).sort_values("year")
    for _, row in merged.iterrows():
        lines.append(
            "\n".join(
                [
                    f"Year: {int(row['year'])}",
                    f"Frequency: {int(row['year_frequency']) if pd.notna(row['year_frequency']) else ''}",
                    f"Year-specific words: {row.get('year_specific_words', '')}",
                    f"Primary focus: {row.get('year_focus_primary', '')}",
                    f"Secondary focus: {row.get('year_focus_secondary', '')}",
                    f"Keywords interpreted: {row.get('year_keywords_interpreted', '')}",
                    f"Frame or angle: {row.get('year_frame_or_angle', '')}",
                    f"Evidence note: {row.get('year_evidence_note', '')}",
                ]
            )
        )
    return "\n\n---\n\n".join(lines)


def build_retry_prompt(base_prompt: str) -> str:
    return (
        base_prompt
        + "\n\nFinal instruction:\n"
        + "Return exactly one valid JSON object.\n"
        + "Do not use markdown fences.\n"
        + "Do not add explanations before or after the JSON.\n"
        + "If evidence is sparse, still return the full JSON schema with empty strings where needed."
    )


def build_schema_retry_prompt(base_prompt: str) -> str:
    return (
        base_prompt
        + "\n\nReturn exactly one valid JSON object using this schema and no other text:\n"
        + "{\n"
        + '  "subgroup": "",\n'
        + '  "source": "",\n'
        + '  "assigned_label": "",\n'
        + '  "micro_topic_id": 0,\n'
        + '  "topic_name_original": "",\n'
        + '  "topic_label_refined": "",\n'
        + '  "overall_summary": "",\n'
        + '  "phase_1_years": "",\n'
        + '  "phase_1_summary": "",\n'
        + '  "phase_2_years": "",\n'
        + '  "phase_2_summary": "",\n'
        + '  "phase_3_years": "",\n'
        + '  "phase_3_summary": "",\n'
        + '  "phase_4_years": "",\n'
        + '  "phase_4_summary": "",\n'
        + '  "evolution_pattern": "",\n'
        + '  "evidence_note": ""\n'
        + "}\n"
        + "Copy subgroup, source, assigned_label, micro_topic_id, and topic_name_original exactly from the metadata.\n"
        + "If you cannot infer multiple phases, still fill phase_1_years and phase_1_summary using the full observed period."
    )


def build_ultra_minimal_retry_prompt(base_prompt: str) -> str:
    return (
        base_prompt
        + "\n\nCritical instruction:\n"
        + "Reply with JSON only. No prose before or after.\n"
        + "Keep values short and plain.\n"
        + "You must fill topic_label_refined, overall_summary, phase_1_years, and phase_1_summary.\n"
        + "If uncertain, use one broad phase covering the full observed period.\n"
        + "Do not omit keys.\n"
    )


def safe_int_or_default(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_directory(args.output_dir)

    selected = pd.read_csv(args.selected_topics)
    annual = pd.read_csv(args.annual_summaries)
    evidence = pd.read_csv(args.year_evidence)
    if args.limit is not None:
        selected = selected.head(args.limit).copy()

    output_csv = args.output_dir / "micro_topic_evolution_narratives.csv"
    output_jsonl = args.output_dir / "micro_topic_evolution_narratives.jsonl"
    raw_jsonl = args.output_dir / "raw_responses.jsonl"
    error_csv = args.output_dir / "error_log.csv"
    prompt_path = args.output_dir / "prompt_template.txt"
    prompt_path.write_text(PROMPT_TEMPLATE, encoding="utf-8")

    completed = set()
    if output_csv.exists() and output_csv.stat().st_size > 0:
        existing = pd.read_csv(output_csv)
        if not existing.empty:
            for _, existing_row in existing.iterrows():
                if is_stage2_narrative_ready(existing_row):
                    completed.add((existing_row["subgroup"], int(existing_row["micro_topic_id"])))
    error_rows: list[dict] = []
    wrote_header = output_csv.exists() and output_csv.stat().st_size > 0
    processed = 0

    for _, row in selected.iterrows():
        key = (row["subgroup"], int(row["micro_topic_id"]))
        if key in completed:
            continue
        summary_rows = annual.loc[
            (annual["subgroup"] == row["subgroup"]) & (annual["micro_topic_id"].astype(int) == int(row["micro_topic_id"]))
        ].copy()
        evidence_rows = evidence.loc[
            (evidence["subgroup"] == row["subgroup"]) & (evidence["micro_topic_id"].astype(int) == int(row["micro_topic_id"]))
        ].copy()
        prompt = PROMPT_TEMPLATE.format(
            subgroup=row["subgroup"],
            source=row["source"],
            assigned_label=row["assigned_label"],
            micro_topic_id=int(row["micro_topic_id"]),
            topic_name_original=row["topic_name_original"],
            topic_size=int(row["topic_size"]),
            share_of_non_outlier=float(row["share_of_non_outlier"]),
            overall_keywords=row["overall_keywords"],
            annual_block=build_annual_block(summary_rows, evidence_rows),
        )
        year_values = summary_rows["year"].dropna().astype(int).tolist() if not summary_rows.empty else []
        default_year_span = ""
        if year_values:
            default_year_span = (
                str(min(year_values)) if min(year_values) == max(year_values) else f"{min(year_values)}-{max(year_values)}"
            )
        started_at = time.time()
        raw_response = ""
        try:
            last_error: Exception | None = None
            prompts = [
                prompt,
                build_retry_prompt(prompt),
                build_schema_retry_prompt(prompt),
                build_ultra_minimal_retry_prompt(prompt),
            ]
            for attempt_idx, attempt_prompt in enumerate(prompts[: args.max_attempts], start=1):
                raw_response = call_ollama(args.model, attempt_prompt, timeout_seconds=args.timeout_seconds)
                try:
                    parsed = extract_json_object(raw_response)
                    if not isinstance(parsed, dict):
                        raise ValueError("Parsed model response is not a JSON object.")
                    parsed = canonicalize_stage2_payload(
                        parsed,
                        defaults={
                            "subgroup": row["subgroup"],
                            "source": row["source"],
                            "assigned_label": row["assigned_label"],
                            "micro_topic_id": int(row["micro_topic_id"]),
                            "topic_name_original": row["topic_name_original"],
                            "default_year_span": default_year_span,
                        },
                    )
                    if not is_stage2_narrative_ready(parsed):
                        raise ValueError(
                            "Incomplete Stage 2 narrative schema after normalization: "
                            + ",".join(stage2_missing_core_fields(parsed))
                        )
                    break
                except Exception as exc:  # noqa: PERF203
                    last_error = exc
                    if attempt_idx >= args.max_attempts:
                        raise
                    append_jsonl(
                        raw_jsonl,
                        {
                            "subgroup": row["subgroup"],
                            "micro_topic_id": int(row["micro_topic_id"]),
                            "raw_response": raw_response,
                            "parse_or_runtime_error": str(exc),
                            "attempt": attempt_idx,
                            "note": "retrying_with_stricter_json_instruction",
                        },
                    )
                    raw_response = ""
            else:
                if last_error is not None:
                    raise last_error
            final_row = {
                "subgroup": parsed["subgroup"],
                "source": parsed["source"],
                "assigned_label": parsed["assigned_label"],
                "micro_topic_id": safe_int_or_default(parsed.get("micro_topic_id"), int(row["micro_topic_id"])),
                "topic_name_original": str(parsed["topic_name_original"]),
                "topic_label_refined": str(parsed.get("topic_label_refined", "")).strip(),
                "overall_summary": str(parsed.get("overall_summary", "")).strip(),
                "phase_1_years": str(parsed.get("phase_1_years", "")).strip(),
                "phase_1_summary": str(parsed.get("phase_1_summary", "")).strip(),
                "phase_2_years": str(parsed.get("phase_2_years", "")).strip(),
                "phase_2_summary": str(parsed.get("phase_2_summary", "")).strip(),
                "phase_3_years": str(parsed.get("phase_3_years", "")).strip(),
                "phase_3_summary": str(parsed.get("phase_3_summary", "")).strip(),
                "phase_4_years": str(parsed.get("phase_4_years", "")).strip(),
                "phase_4_summary": str(parsed.get("phase_4_summary", "")).strip(),
                "evolution_pattern": str(parsed.get("evolution_pattern", "")).strip(),
                "evidence_note": str(parsed.get("evidence_note", "")).strip(),
            }
            append_jsonl(output_jsonl, final_row)
            append_jsonl(
                raw_jsonl,
                {
                    "subgroup": row["subgroup"],
                    "micro_topic_id": int(row["micro_topic_id"]),
                    "raw_response": raw_response,
                    "elapsed_seconds": round(time.time() - started_at, 3),
                },
            )
            pd.DataFrame([final_row]).to_csv(output_csv, mode="a", index=False, header=not wrote_header)
            wrote_header = True
            completed.add(key)
            processed += 1
        except Exception as exc:
            if raw_response:
                append_jsonl(
                    raw_jsonl,
                    {
                        "subgroup": row["subgroup"],
                        "micro_topic_id": int(row["micro_topic_id"]),
                        "raw_response": raw_response,
                        "parse_or_runtime_error": str(exc),
                    },
                )
            error_rows.append(
                {
                    "subgroup": row["subgroup"],
                    "micro_topic_id": int(row["micro_topic_id"]),
                    "error": str(exc),
                }
            )

    if error_rows:
        pd.DataFrame(error_rows).to_csv(error_csv, index=False)
    else:
        pd.DataFrame(columns=["subgroup", "micro_topic_id", "error"]).to_csv(error_csv, index=False)

    write_json(
        args.output_dir / "run_manifest.json",
        {
            "model": args.model,
            "timeout_seconds": args.timeout_seconds,
            "max_attempts": args.max_attempts,
            "selected_topics": str(args.selected_topics),
            "annual_summaries": str(args.annual_summaries),
            "processed_rows": processed,
            "completed_rows": len(completed),
            "error_count": len(error_rows),
            "outputs": {
                "evolution_narratives_csv": str(output_csv),
                "evolution_narratives_jsonl": str(output_jsonl),
                "raw_responses": str(raw_jsonl),
                "error_log": str(error_csv),
                "prompt_template": str(prompt_path),
            },
        },
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
