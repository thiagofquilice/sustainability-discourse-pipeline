#!/usr/bin/env python3
"""Run Gemma on micro-topic annual summaries to produce evolution narratives in Colab."""

from __future__ import annotations

import argparse
import json
import platform
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm


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

DEFAULT_MODEL_NAME = "google/gemma-4-E4B-it"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CODE_FENCE_RE = re.compile(r"```(?:json)?|```", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-topics", required=True, type=Path)
    parser.add_argument("--annual-summaries", required=True, type=Path)
    parser.add_argument("--year-evidence", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def print_log(message: str) -> None:
    print(f"[{now_utc_iso()}] {message}", flush=True)


def sanitize_model_output(text: str) -> str:
    cleaned = ANSI_ESCAPE_RE.sub("", str(text))
    cleaned = CODE_FENCE_RE.sub("", cleaned)
    return cleaned.strip()


def try_load_json(candidate: str) -> dict[str, Any]:
    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        sanitized = candidate.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        return json.loads(sanitized)


def extract_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = sanitize_model_output(raw_text)
    start = raw_text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found in model response.")
    depth = 0
    for idx in range(start, len(raw_text)):
        char = raw_text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = raw_text[start : idx + 1]
                return try_load_json(candidate)
    fallback_candidate = raw_text[start:].strip()
    missing_closers = fallback_candidate.count("{") - fallback_candidate.count("}")
    if missing_closers > 0:
        fallback_candidate = fallback_candidate + ("}" * missing_closers)
        try:
            return try_load_json(fallback_candidate)
        except Exception:
            pass
    raise ValueError("No complete JSON object found in model response.")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_completed_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path)
    if frame.empty:
        return set()
    return {
        (str(row["subgroup"]), int(row["micro_topic_id"]))
        for _, row in frame[["subgroup", "micro_topic_id"]].iterrows()
    }


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


def build_prompt(topic_row: pd.Series, summary_rows: pd.DataFrame, evidence_rows: pd.DataFrame) -> str:
    return PROMPT_TEMPLATE.format(
        subgroup=topic_row["subgroup"],
        source=topic_row["source"],
        assigned_label=topic_row["assigned_label"],
        micro_topic_id=int(topic_row["micro_topic_id"]),
        topic_name_original=topic_row["topic_name_original"],
        topic_size=int(topic_row["topic_size"]),
        share_of_non_outlier=float(topic_row["share_of_non_outlier"]),
        overall_keywords=topic_row["overall_keywords"],
        annual_block=build_annual_block(summary_rows, evidence_rows),
    )


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


def canonicalize_stage2_payload(payload: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    year_span = str(defaults.get("default_year_span", "")).strip()
    canonical = {
        "subgroup": str(payload.get("subgroup") or defaults["subgroup"]),
        "source": str(payload.get("source") or defaults["source"]),
        "assigned_label": str(payload.get("assigned_label") or defaults["assigned_label"]),
        "micro_topic_id": safe_int_or_default(payload.get("micro_topic_id"), int(defaults["micro_topic_id"])),
        "topic_name_original": str(payload.get("topic_name_original") or defaults["topic_name_original"]),
        "topic_label_refined": str(payload.get("topic_label_refined", "")).strip(),
        "overall_summary": str(payload.get("overall_summary", "")).strip(),
        "phase_1_years": str(payload.get("phase_1_years", "")).strip(),
        "phase_1_summary": str(payload.get("phase_1_summary", "")).strip(),
        "phase_2_years": str(payload.get("phase_2_years", "")).strip(),
        "phase_2_summary": str(payload.get("phase_2_summary", "")).strip(),
        "phase_3_years": str(payload.get("phase_3_years", "")).strip(),
        "phase_3_summary": str(payload.get("phase_3_summary", "")).strip(),
        "phase_4_years": str(payload.get("phase_4_years", "")).strip(),
        "phase_4_summary": str(payload.get("phase_4_summary", "")).strip(),
        "evolution_pattern": str(payload.get("evolution_pattern", "")).strip(),
        "evidence_note": str(payload.get("evidence_note", "")).strip(),
    }
    if not canonical["phase_1_years"] and year_span:
        canonical["phase_1_years"] = year_span
    return canonical


def stage2_missing_core_fields(payload: dict[str, Any]) -> list[str]:
    required = [
        "subgroup",
        "source",
        "assigned_label",
        "micro_topic_id",
        "topic_name_original",
        "topic_label_refined",
        "overall_summary",
        "phase_1_years",
        "phase_1_summary",
    ]
    missing = []
    for field in required:
        value = payload.get(field, "")
        if value is None:
            missing.append(field)
        elif isinstance(value, str) and not value.strip():
            missing.append(field)
    return missing


def is_stage2_narrative_ready(payload: dict[str, Any]) -> bool:
    return not stage2_missing_core_fields(payload)


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Use a Colab GPU runtime.")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model.eval()
    hardware = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "transformers_version": transformers.__version__,
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": int(torch.cuda.device_count()),
    }
    return model, tokenizer, hardware


def generate_one(model, tokenizer, prompt: str, args: argparse.Namespace) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "micro_topic_evolution_narratives.csv"
    output_jsonl = args.output_dir / "micro_topic_evolution_narratives.jsonl"
    raw_jsonl = args.output_dir / "raw_responses.jsonl"
    error_csv = args.output_dir / "error_log.csv"
    prompt_path = args.output_dir / "prompt_template.txt"
    prompt_path.write_text(PROMPT_TEMPLATE, encoding="utf-8")

    selected = pd.read_csv(args.selected_topics)
    annual = pd.read_csv(args.annual_summaries)
    evidence = pd.read_csv(args.year_evidence)
    if args.max_rows is not None:
        selected = selected.head(args.max_rows).copy()

    completed = load_completed_keys(output_csv) if args.resume else set()
    pending = selected.loc[
        ~selected.apply(lambda row: (str(row["subgroup"]), int(row["micro_topic_id"])) in completed, axis=1)
    ].copy()
    pending = pending.reset_index(drop=True)

    print_log(f"Input rows={len(selected)} pending_rows={len(pending)} completed_rows={len(completed)}")
    model, tokenizer, hardware = load_model_and_tokenizer(args)
    print_log(f"Loaded model={args.model_name} gpu={hardware['gpu_name']}")

    error_rows: list[dict[str, Any]] = []
    wrote_header = output_csv.exists() and output_csv.stat().st_size > 0
    processed = 0

    for _, row in tqdm(pending.iterrows(), total=len(pending), desc="Gemma evolution synthesis"):
        key = (str(row["subgroup"]), int(row["micro_topic_id"]))
        summary_rows = annual.loc[
            (annual["subgroup"] == row["subgroup"]) & (annual["micro_topic_id"].astype(int) == int(row["micro_topic_id"]))
        ].copy()
        evidence_rows = evidence.loc[
            (evidence["subgroup"] == row["subgroup"]) & (evidence["micro_topic_id"].astype(int) == int(row["micro_topic_id"]))
        ].copy()
        prompt = build_prompt(row, summary_rows, evidence_rows)
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
                raw_response = sanitize_model_output(generate_one(model, tokenizer, attempt_prompt, args))
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
                except Exception as exc:
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
                    "subgroup": key[0],
                    "micro_topic_id": key[1],
                    "raw_response": raw_response,
                    "elapsed_seconds": round(time.time() - started_at, 3),
                },
            )
            pd.DataFrame([final_row]).to_csv(output_csv, mode="a", index=False, header=not wrote_header)
            wrote_header = True
            processed += 1
        except Exception as exc:
            error_rows.append(
                {
                    "subgroup": key[0],
                    "micro_topic_id": key[1],
                    "error": str(exc),
                }
            )
            append_jsonl(
                raw_jsonl,
                {
                    "subgroup": key[0],
                    "micro_topic_id": key[1],
                    "raw_response": raw_response,
                    "parse_or_runtime_error": str(exc),
                },
            )
            pd.DataFrame(error_rows).to_csv(error_csv, index=False)

    if error_rows:
        pd.DataFrame(error_rows).to_csv(error_csv, index=False)
    else:
        pd.DataFrame(columns=["subgroup", "micro_topic_id", "error"]).to_csv(error_csv, index=False)

    manifest = {
        "model": args.model_name,
        "max_new_tokens": args.max_new_tokens,
        "max_attempts": args.max_attempts,
        "selected_topics": str(args.selected_topics),
        "annual_summaries": str(args.annual_summaries),
        "processed_rows": processed,
        "pending_rows_initial": int(len(pending)),
        "error_count": len(error_rows),
        "hardware": hardware,
        "outputs": {
            "evolution_narratives_csv": str(output_csv),
            "evolution_narratives_jsonl": str(output_jsonl),
            "raw_responses": str(raw_jsonl),
            "error_log": str(error_csv),
            "prompt_template": str(prompt_path),
        },
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print_log(f"Finished processed_rows={processed} error_count={len(error_rows)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
