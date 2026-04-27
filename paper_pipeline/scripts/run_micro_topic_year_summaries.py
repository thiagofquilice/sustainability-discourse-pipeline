#!/usr/bin/env python3
"""Generate annual micro-topic summaries with Gemma via local Ollama."""

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
    call_ollama,
    configure_logging,
    ensure_directory,
    extract_json_object,
    resume_completed_keys,
    write_json,
)


PROMPT_TEMPLATE = """You are reading yearly evidence for one BERTopic micro-topic within a source-topic subgroup.

Task:
Summarize what this micro-topic appears to focus on in this specific year.

Important:
- Use only the evidence provided.
- Treat the BERTopic topic as a fixed micro-topic whose wording can vary over time.
- The year-specific words come from BERTopic temporal topic tracking (`topics_over_time`), not from a separately refit dynamic topic model.
- Be concrete and concise.
- If the evidence is sparse, say so plainly.
- Reply with one JSON object only and no surrounding text.

Return exactly these fields:
- subgroup
- source
- assigned_label
- micro_topic_id
- year
- topic_name_original
- year_focus_primary
- year_focus_secondary
- year_keywords_interpreted
- year_frame_or_angle
- year_evidence_note

Metadata:
subgroup: {subgroup}
source: {source}
assigned_label: {assigned_label}
micro_topic_id: {micro_topic_id}
year: {year}
topic_name_original: {topic_name_original}
overall_keywords: {overall_keywords}
year_specific_words: {year_specific_words}
available_chunk_count: {available_chunk_count}
is_sparse_year: {is_sparse_year}

Representative chunks for this year:
{chunk_block}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=OUTPUT_ROOT / "micro_topic_year_evidence.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "year_summaries")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def shorten_text(text: str, max_chars: int = 1800) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def render_chunk_block(row: pd.Series) -> str:
    chunks = json.loads(row["chunk_records_json"])
    if not chunks:
        return "No representative chunks were available for this year."
    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[Chunk {idx}] chunk_id={chunk['chunk_id']} prob={chunk['micro_topic_probability']:.4f}\n{shorten_text(chunk['text'])}"
        )
    return "\n\n".join(lines)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_directory(args.output_dir)

    frame = pd.read_csv(args.input_csv)
    if args.limit is not None:
        frame = frame.head(args.limit).copy()

    output_csv = args.output_dir / "micro_topic_year_summaries.csv"
    output_jsonl = args.output_dir / "micro_topic_year_summaries.jsonl"
    raw_jsonl = args.output_dir / "raw_responses.jsonl"
    error_csv = args.output_dir / "error_log.csv"
    prompt_path = args.output_dir / "prompt_template.txt"
    prompt_path.write_text(PROMPT_TEMPLATE, encoding="utf-8")

    completed = resume_completed_keys(output_csv, ["subgroup", "micro_topic_id", "year"])
    error_rows: list[dict] = []
    wrote_header = output_csv.exists() and output_csv.stat().st_size > 0
    processed = 0

    for _, row in frame.iterrows():
        key = (row["subgroup"], int(row["micro_topic_id"]), int(row["year"]))
        if key in completed:
            continue
        prompt = PROMPT_TEMPLATE.format(
            subgroup=row["subgroup"],
            source=row["source"],
            assigned_label=row["assigned_label"],
            micro_topic_id=int(row["micro_topic_id"]),
            year=int(row["year"]),
            topic_name_original=row["topic_name_original"],
            overall_keywords=row["overall_keywords"],
            year_specific_words=row["year_specific_words"],
            available_chunk_count=int(row["available_chunk_count"]),
            is_sparse_year=str(bool(row["is_sparse_year"])).lower(),
            chunk_block=render_chunk_block(row),
        )
        started_at = time.time()
        raw_response = ""
        try:
            raw_response = call_ollama(args.model, prompt)
            parsed = extract_json_object(raw_response)
            parsed.setdefault("subgroup", row["subgroup"])
            parsed.setdefault("source", row["source"])
            parsed.setdefault("assigned_label", row["assigned_label"])
            parsed.setdefault("micro_topic_id", int(row["micro_topic_id"]))
            parsed.setdefault("year", int(row["year"]))
            parsed.setdefault("topic_name_original", row["topic_name_original"])
            annual_row = {
                "subgroup": parsed["subgroup"],
                "source": parsed["source"],
                "assigned_label": parsed["assigned_label"],
                "micro_topic_id": int(parsed["micro_topic_id"]),
                "year": int(parsed["year"]),
                "topic_name_original": str(parsed["topic_name_original"]),
                "year_focus_primary": str(parsed.get("year_focus_primary", "")).strip(),
                "year_focus_secondary": str(parsed.get("year_focus_secondary", "")).strip(),
                "year_keywords_interpreted": str(parsed.get("year_keywords_interpreted", "")).strip(),
                "year_frame_or_angle": str(parsed.get("year_frame_or_angle", "")).strip(),
                "year_evidence_note": str(parsed.get("year_evidence_note", "")).strip(),
            }
            append_jsonl(output_jsonl, annual_row)
            append_jsonl(
                raw_jsonl,
                {
                    "subgroup": row["subgroup"],
                    "micro_topic_id": int(row["micro_topic_id"]),
                    "year": int(row["year"]),
                    "raw_response": raw_response,
                    "elapsed_seconds": round(time.time() - started_at, 3),
                },
            )
            pd.DataFrame([annual_row]).to_csv(output_csv, mode="a", index=False, header=not wrote_header)
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
                        "year": int(row["year"]),
                        "raw_response": raw_response,
                        "parse_or_runtime_error": str(exc),
                    },
                )
            error_rows.append(
                {
                    "subgroup": row["subgroup"],
                    "micro_topic_id": int(row["micro_topic_id"]),
                    "year": int(row["year"]),
                    "error": str(exc),
                }
            )

    if error_rows:
        pd.DataFrame(error_rows).to_csv(error_csv, index=False)
    else:
        pd.DataFrame(columns=["subgroup", "micro_topic_id", "year", "error"]).to_csv(error_csv, index=False)

    write_json(
        args.output_dir / "run_manifest.json",
        {
            "model": args.model,
            "input_csv": str(args.input_csv),
            "processed_rows": processed,
            "completed_rows": len(completed),
            "error_count": len(error_rows),
            "outputs": {
                "annual_summaries_csv": str(output_csv),
                "annual_summaries_jsonl": str(output_jsonl),
                "raw_responses": str(raw_jsonl),
                "error_log": str(error_csv),
                "prompt_template": str(prompt_path),
            },
        },
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
