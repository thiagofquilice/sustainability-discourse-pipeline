#!/usr/bin/env python3
"""Compare same-issue candidate pairs using Stage 2 narratives and Gemma via Ollama."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from cross_source_microtopic_common import DEFAULT_DISCOURSE_FUNCTION_OUTPUT_ROOT, load_stage2_narratives
from micro_topic_evolution_common import (
    OLLAMA_MODEL,
    append_jsonl,
    call_ollama,
    configure_logging,
    ensure_directory,
    extract_json_object,
    normalize_stage2_text,
    resume_completed_keys,
    write_json,
)


PROMPT_TEMPLATE = """You are comparing two cross-source BERTopic micro-topics that were matched as candidate discussions of the same issue.

Task:
Decide whether they are actually about the same issue, and whether they perform different discourse functions.

Important:
- Use only the information provided.
- The pair is already restricted to the same macro-topic.
- Focus on substantive issue overlap and discourse function, not only on lexical similarity.
- Treat discourse function as the role each source plays when discussing the issue.
- Reply with one JSON object only and no surrounding text.

Allowed discourse function labels:
- technical_research
- operational_management
- market_strategy
- financial_investor
- policy_governance
- public_political_debate
- legal_compliance
- community_access_justice
- environmental_risk_stewardship
- mixed_or_unclear

Return exactly these fields:
- pair_id
- macro_topic
- dyad
- source_a
- subgroup_a
- microtopic_id_a
- source_b
- subgroup_b
- microtopic_id_b
- same_issue_assessment
- discourse_function_relation
- function_label_a
- function_label_b
- difference_summary
- temporal_interpretation_note
- keep_for_discourse_analysis
- confidence_note

Metadata:
pair_id: {pair_id}
macro_topic: {macro_topic}
macro_topic_name: {macro_topic_name}
dyad: {dyad}
source_a: {source_a}
subgroup_a: {subgroup_a}
microtopic_id_a: {microtopic_id_a}
source_b: {source_b}
subgroup_b: {subgroup_b}
microtopic_id_b: {microtopic_id_b}
cosine_similarity: {cosine_similarity}
best_lag_docs: {best_lag_docs}
best_correlation_docs: {best_correlation_docs}

Side A narrative:
topic_label_refined: {topic_label_refined_a}
overall_summary: {overall_summary_a}
phase_1_years: {phase_1_years_a}
phase_1_summary: {phase_1_summary_a}
phase_2_years: {phase_2_years_a}
phase_2_summary: {phase_2_summary_a}
phase_3_years: {phase_3_years_a}
phase_3_summary: {phase_3_summary_a}
phase_4_years: {phase_4_years_a}
phase_4_summary: {phase_4_summary_a}
evolution_pattern: {evolution_pattern_a}
evidence_note: {evidence_note_a}

Side B narrative:
topic_label_refined: {topic_label_refined_b}
overall_summary: {overall_summary_b}
phase_1_years: {phase_1_years_b}
phase_1_summary: {phase_1_summary_b}
phase_2_years: {phase_2_years_b}
phase_2_summary: {phase_2_summary_b}
phase_3_years: {phase_3_years_b}
phase_3_summary: {phase_3_summary_b}
phase_4_years: {phase_4_years_b}
phase_4_summary: {phase_4_summary_b}
evolution_pattern: {evolution_pattern_b}
evidence_note: {evidence_note_b}
"""

REQUIRED_FIELDS = [
    "pair_id",
    "same_issue_assessment",
    "discourse_function_relation",
    "function_label_a",
    "function_label_b",
    "difference_summary",
    "temporal_interpretation_note",
    "keep_for_discourse_analysis",
    "confidence_note",
]

ALLOWED_SAME_ISSUE = {"same_issue", "adjacent_issue", "weak_match"}
ALLOWED_RELATIONS = {"same_function", "different_function", "unclear"}
ALLOWED_FUNCTION_LABELS = {
    "technical_research",
    "operational_management",
    "market_strategy",
    "financial_investor",
    "policy_governance",
    "public_political_debate",
    "legal_compliance",
    "community_access_justice",
    "environmental_risk_stewardship",
    "mixed_or_unclear",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DISCOURSE_FUNCTION_OUTPUT_ROOT)
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--narratives-csv", type=Path, default=None)
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL)
    parser.add_argument("--timeout-seconds", type=int, default=420)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def canonicalize_same_issue_assessment(value: Any) -> str:
    raw_text = "" if value is None else str(value)
    text = normalize_stage2_text(value).strip().lower()
    if not text and raw_text:
        text = raw_text.strip().lower()
    compact = text.replace("-", "_").replace(" ", "_")
    if compact in ALLOWED_SAME_ISSUE:
        return compact
    if compact in {"yes", "true", "same", "sameissue", "strong_match", "direct_match"}:
        return "same_issue"
    if any(token in compact for token in {"same_issue", "same-topic", "same_topic", "sameproblem", "same_problem"}):
        return "same_issue"
    if any(token in compact for token in {"adjacent", "related", "overlap", "complement", "partial"}):
        return "adjacent_issue"
    if any(token in compact for token in {"weak", "different", "distinct", "mismatch", "no"}):
        return "weak_match"
    return compact or ""


def canonicalize_discourse_relation(value: Any) -> str:
    raw_text = "" if value is None else str(value)
    text = normalize_stage2_text(value).strip().lower()
    if not text and raw_text:
        text = raw_text.strip().lower()
    compact = text.replace("-", "_").replace(" ", "_")
    if compact in ALLOWED_RELATIONS:
        return compact
    if any(token in compact for token in {"different", "complement", "contrast", "distinct", "divergent", "unrelated", "none"}):
        return "different_function"
    if any(token in compact for token in {"same", "similar", "aligned"}):
        return "same_function"
    if any(token in compact for token in {"unclear", "mixed", "partial", "unknown"}):
        return "unclear"
    return compact or ""


def canonicalize_function_label(value: Any) -> str:
    raw_text = "" if value is None else str(value)
    text = normalize_stage2_text(value).strip().lower()
    if not text and raw_text:
        text = raw_text.strip().lower()
    compact = text.replace("-", "_").replace(" ", "_").replace("/", "_")
    compact = "_".join(part for part in compact.split("_") if part)
    if compact in ALLOWED_FUNCTION_LABELS:
        return compact
    keyword_map = [
        ("technical_research", ("technical", "research", "scientific", "engineering", "technology_development")),
        ("operational_management", ("operational", "operations", "management", "implementation", "project_execution")),
        ("market_strategy", ("market", "strategy", "commercial", "consumer", "demand", "branding", "business_strategy")),
        ("financial_investor", ("financial", "investor", "investment", "lender", "financ", "capital_market", "esg_finance")),
        ("policy_governance", ("policy", "governance", "regulatory_design", "institutional", "public_policy")),
        ("public_political_debate", ("public", "political", "media", "debate", "public_discourse")),
        ("legal_compliance", ("legal", "compliance", "litigation", "regulatory_compliance", "disclosure")),
        ("community_access_justice", ("community", "justice", "equity", "access", "livelihood", "local_population")),
        ("environmental_risk_stewardship", ("risk", "stewardship", "environmental", "biodiversity", "restoration", "pollution")),
        ("mixed_or_unclear", ("mixed", "unclear", "hybrid", "ambiguous")),
    ]
    for canonical, tokens in keyword_map:
        if any(token in compact for token in tokens):
            return canonical
    return compact or ""


def normalize_compare_payload(payload: dict[str, Any], row: pd.Series) -> dict[str, Any]:
    temporal_note = normalize_stage2_text(payload.get("temporal_interpretation_note"))
    if not temporal_note:
        raw_temporal = "" if payload.get("temporal_interpretation_note") is None else str(payload.get("temporal_interpretation_note")).strip()
        temporal_note = raw_temporal or "Not applicable"

    normalized = {
        "pair_id": str(row["pair_id"]),
        "macro_topic": str(row["macro_topic"]),
        "dyad": str(row["dyad"]),
        "source_a": str(row["source_a"]),
        "subgroup_a": str(row["subgroup_a"]),
        "microtopic_id_a": int(row["microtopic_id_a"]),
        "source_b": str(row["source_b"]),
        "subgroup_b": str(row["subgroup_b"]),
        "microtopic_id_b": int(row["microtopic_id_b"]),
        "same_issue_assessment": canonicalize_same_issue_assessment(payload.get("same_issue_assessment")),
        "discourse_function_relation": canonicalize_discourse_relation(payload.get("discourse_function_relation")),
        "function_label_a": canonicalize_function_label(payload.get("function_label_a")),
        "function_label_b": canonicalize_function_label(payload.get("function_label_b")),
        "difference_summary": normalize_stage2_text(payload.get("difference_summary")),
        "temporal_interpretation_note": temporal_note,
        "keep_for_discourse_analysis": payload.get("keep_for_discourse_analysis"),
        "confidence_note": normalize_stage2_text(payload.get("confidence_note")),
    }
    keep_value = normalized["keep_for_discourse_analysis"]
    if isinstance(keep_value, bool):
        normalized["keep_for_discourse_analysis"] = keep_value
    else:
        keep_text = normalize_stage2_text(keep_value).lower()
        normalized["keep_for_discourse_analysis"] = keep_text in {"true", "yes", "1", "keep"}
    return normalized


def compare_payload_ready(payload: dict[str, Any]) -> bool:
    base_ready = all(
        payload.get(field) not in ("", None) if field != "keep_for_discourse_analysis" else isinstance(payload.get(field), bool)
        for field in REQUIRED_FIELDS
    )
    if not base_ready:
        return False
    return (
        payload.get("same_issue_assessment") in ALLOWED_SAME_ISSUE
        and payload.get("discourse_function_relation") in ALLOWED_RELATIONS
        and payload.get("function_label_a") in ALLOWED_FUNCTION_LABELS
        and payload.get("function_label_b") in ALLOWED_FUNCTION_LABELS
    )


def build_retry_prompt(base_prompt: str) -> str:
    return (
        base_prompt
        + "\n\nFinal instruction:\nReturn exactly one valid JSON object.\n"
        + "Do not use markdown fences.\nDo not add explanations before or after the JSON.\n"
    )


def build_schema_retry_prompt(base_prompt: str) -> str:
    return (
        base_prompt
        + "\n\nReturn exactly one valid JSON object using this schema and no other text:\n"
        + "{\n"
        + '  "pair_id": "",\n'
        + '  "macro_topic": "",\n'
        + '  "dyad": "",\n'
        + '  "source_a": "",\n'
        + '  "subgroup_a": "",\n'
        + '  "microtopic_id_a": 0,\n'
        + '  "source_b": "",\n'
        + '  "subgroup_b": "",\n'
        + '  "microtopic_id_b": 0,\n'
        + '  "same_issue_assessment": "",\n'
        + '  "discourse_function_relation": "",\n'
        + '  "function_label_a": "",\n'
        + '  "function_label_b": "",\n'
        + '  "difference_summary": "",\n'
        + '  "temporal_interpretation_note": "",\n'
        + '  "keep_for_discourse_analysis": true,\n'
        + '  "confidence_note": ""\n'
        + "}\n"
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_directory(args.output_root)

    candidate_csv = args.candidate_csv or (args.output_root / "candidate_pairs_topk.csv")
    narratives = load_stage2_narratives(args.narratives_csv) if args.narratives_csv else load_stage2_narratives()
    candidates = pd.read_csv(candidate_csv)
    if args.limit is not None:
        candidates = candidates.head(args.limit).copy()

    compare_csv = args.output_root / "same_issue_discourse_compare.csv"
    compare_jsonl = args.output_root / "same_issue_discourse_compare.jsonl"
    raw_jsonl = args.output_root / "same_issue_discourse_raw_responses.jsonl"
    error_csv = args.output_root / "same_issue_discourse_error_log.csv"
    prompt_path = args.output_root / "same_issue_discourse_prompt_template.txt"
    prompt_path.write_text(PROMPT_TEMPLATE, encoding="utf-8")

    completed = resume_completed_keys(compare_csv, ["pair_id"])
    narrative_map = {
        (str(row.subgroup), int(row.micro_topic_id)): row._asdict()
        for row in narratives.loc[narratives["is_narrative_ready"]].itertuples(index=False)
    }

    wrote_header = compare_csv.exists() and compare_csv.stat().st_size > 0
    processed = 0
    error_rows: list[dict[str, Any]] = []

    for _, row in candidates.iterrows():
        pair_id = str(row["pair_id"])
        if (pair_id,) in completed:
            continue
        narr_a = narrative_map[(str(row["subgroup_a"]), int(row["microtopic_id_a"]))]
        narr_b = narrative_map[(str(row["subgroup_b"]), int(row["microtopic_id_b"]))]
        prompt = PROMPT_TEMPLATE.format(
            pair_id=pair_id,
            macro_topic=row["macro_topic"],
            macro_topic_name=row["macro_topic_name"],
            dyad=row["dyad"],
            source_a=row["source_a"],
            subgroup_a=row["subgroup_a"],
            microtopic_id_a=int(row["microtopic_id_a"]),
            source_b=row["source_b"],
            subgroup_b=row["subgroup_b"],
            microtopic_id_b=int(row["microtopic_id_b"]),
            cosine_similarity=row["cosine_similarity"],
            best_lag_docs=row.get("best_lag_docs", ""),
            best_correlation_docs=row.get("best_correlation_docs", ""),
            **{f"{field}_a": narr_a.get(field, "") for field in [
                "topic_label_refined",
                "overall_summary",
                "phase_1_years",
                "phase_1_summary",
                "phase_2_years",
                "phase_2_summary",
                "phase_3_years",
                "phase_3_summary",
                "phase_4_years",
                "phase_4_summary",
                "evolution_pattern",
                "evidence_note",
            ]},
            **{f"{field}_b": narr_b.get(field, "") for field in [
                "topic_label_refined",
                "overall_summary",
                "phase_1_years",
                "phase_1_summary",
                "phase_2_years",
                "phase_2_summary",
                "phase_3_years",
                "phase_3_summary",
                "phase_4_years",
                "phase_4_summary",
                "evolution_pattern",
                "evidence_note",
            ]},
        )
        raw_response = ""
        started_at = time.time()
        try:
            prompts = [prompt, build_retry_prompt(prompt), build_schema_retry_prompt(prompt)]
            last_error: Exception | None = None
            for attempt_idx, attempt_prompt in enumerate(prompts[: args.max_attempts], start=1):
                raw_response = call_ollama(args.model, attempt_prompt, timeout_seconds=args.timeout_seconds)
                try:
                    parsed = extract_json_object(raw_response)
                    if not isinstance(parsed, dict):
                        raise ValueError("Parsed compare response is not a JSON object.")
                    normalized = normalize_compare_payload(parsed, row)
                    if not compare_payload_ready(normalized):
                        raise ValueError("Incomplete same-issue compare payload.")
                    break
                except Exception as exc:  # noqa: PERF203
                    last_error = exc
                    if attempt_idx >= args.max_attempts:
                        raise
                    append_jsonl(
                        raw_jsonl,
                        {
                            "pair_id": pair_id,
                            "raw_response": raw_response,
                            "parse_or_runtime_error": str(exc),
                            "attempt": attempt_idx,
                            "note": "retrying_with_stricter_compare_instruction",
                        },
                    )
                    raw_response = ""
            else:
                raise last_error if last_error else ValueError("No valid compare response.")

            append_jsonl(compare_jsonl, normalized)
            append_jsonl(
                raw_jsonl,
                {
                    "pair_id": pair_id,
                    "raw_response": raw_response,
                    "elapsed_seconds": round(time.time() - started_at, 3),
                },
            )
            pd.DataFrame([normalized]).to_csv(compare_csv, mode="a", index=False, header=not wrote_header)
            wrote_header = True
            completed.add((pair_id,))
            processed += 1
        except Exception as exc:
            if raw_response:
                append_jsonl(
                    raw_jsonl,
                    {
                        "pair_id": pair_id,
                        "raw_response": raw_response,
                        "parse_or_runtime_error": str(exc),
                    },
                )
            error_rows.append({"pair_id": pair_id, "error": str(exc)})

    if error_rows:
        pd.DataFrame(error_rows).to_csv(error_csv, index=False)
    else:
        pd.DataFrame(columns=["pair_id", "error"]).to_csv(error_csv, index=False)

    write_json(
        args.output_root / "same_issue_discourse_compare_manifest.json",
        {
            "candidate_csv": str(candidate_csv),
            "model": args.model,
            "timeout_seconds": args.timeout_seconds,
            "max_attempts": args.max_attempts,
            "processed_rows": processed,
            "completed_rows": len(completed),
            "error_count": len(error_rows),
            "outputs": {
                "compare_csv": str(compare_csv),
                "compare_jsonl": str(compare_jsonl),
                "raw_responses": str(raw_jsonl),
                "error_log": str(error_csv),
                "prompt_template": str(prompt_path),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
