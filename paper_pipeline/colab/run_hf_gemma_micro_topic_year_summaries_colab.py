#!/usr/bin/env python3
"""Run Gemma on micro-topic annual evidence in Colab using Hugging Face Transformers."""

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

DEFAULT_MODEL_NAME = "google/gemma-4-E4B-it"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CODE_FENCE_RE = re.compile(r"```(?:json)?|```", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def print_log(message: str) -> None:
    print(f"[{now_utc_iso()}] {message}", flush=True)


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


def build_prompt(row: pd.Series) -> str:
    return PROMPT_TEMPLATE.format(
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


def generate_batch(model, tokenizer, rows: list[pd.Series], args: argparse.Namespace) -> list[str]:
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(row)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
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
    prompt_length = input_ids.shape[1]
    outputs = []
    for idx in range(generated.shape[0]):
        new_tokens = generated[idx, prompt_length:]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def load_completed_keys(path: Path) -> set[tuple[str, int, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path)
    if frame.empty:
        return set()
    return {
        (str(row["subgroup"]), int(row["micro_topic_id"]), int(row["year"]))
        for _, row in frame[["subgroup", "micro_topic_id", "year"]].iterrows()
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def flush_rows(output_csv: Path, rows: list[dict[str, Any]], wrote_header: bool) -> bool:
    if not rows:
        return wrote_header
    pd.DataFrame(rows).to_csv(output_csv, mode="a", index=False, header=not wrote_header)
    return True


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "micro_topic_year_summaries.csv"
    output_jsonl = args.output_dir / "micro_topic_year_summaries.jsonl"
    raw_jsonl = args.output_dir / "raw_responses.jsonl"
    error_csv = args.output_dir / "error_log.csv"
    prompt_path = args.output_dir / "prompt_template.txt"
    prompt_path.write_text(PROMPT_TEMPLATE, encoding="utf-8")

    frame = pd.read_csv(args.input_csv)
    if args.max_rows is not None:
        frame = frame.head(args.max_rows).copy()
    completed = load_completed_keys(output_csv) if args.resume else set()
    pending = frame.loc[
        ~frame.apply(
            lambda row: (str(row["subgroup"]), int(row["micro_topic_id"]), int(row["year"])) in completed,
            axis=1,
        )
    ].copy()
    pending = pending.reset_index(drop=True)

    print_log(f"Input rows={len(frame)} pending_rows={len(pending)} completed_rows={len(completed)}")
    model, tokenizer, hardware = load_model_and_tokenizer(args)
    print_log(f"Loaded model={args.model_name} gpu={hardware['gpu_name']}")

    saved_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    wrote_header = output_csv.exists() and output_csv.stat().st_size > 0
    processed = 0

    for start in tqdm(range(0, len(pending), args.batch_size), desc="Gemma annual summaries"):
        batch = pending.iloc[start : start + args.batch_size]
        raw_outputs = generate_batch(model, tokenizer, [row for _, row in batch.iterrows()], args)
        for (_, row), raw_response in zip(batch.iterrows(), raw_outputs):
            raw_response = sanitize_model_output(raw_response)
            key = (str(row["subgroup"]), int(row["micro_topic_id"]), int(row["year"]))
            try:
                parsed = extract_json_object(raw_response)
                annual_row = {
                    "subgroup": str(parsed.get("subgroup", row["subgroup"])),
                    "source": str(parsed.get("source", row["source"])),
                    "assigned_label": str(parsed.get("assigned_label", row["assigned_label"])),
                    "micro_topic_id": int(parsed.get("micro_topic_id", row["micro_topic_id"])),
                    "year": int(parsed.get("year", row["year"])),
                    "topic_name_original": str(parsed.get("topic_name_original", row["topic_name_original"])),
                    "year_focus_primary": str(parsed.get("year_focus_primary", "")).strip(),
                    "year_focus_secondary": str(parsed.get("year_focus_secondary", "")).strip(),
                    "year_keywords_interpreted": str(parsed.get("year_keywords_interpreted", "")).strip(),
                    "year_frame_or_angle": str(parsed.get("year_frame_or_angle", "")).strip(),
                    "year_evidence_note": str(parsed.get("year_evidence_note", "")).strip(),
                }
                saved_rows.append(annual_row)
                append_jsonl(output_jsonl, annual_row)
                append_jsonl(
                    raw_jsonl,
                    {
                        "subgroup": key[0],
                        "micro_topic_id": key[1],
                        "year": key[2],
                        "raw_response": raw_response,
                    },
                )
                processed += 1
            except Exception as exc:
                error_rows.append(
                    {
                        "subgroup": key[0],
                        "micro_topic_id": key[1],
                        "year": key[2],
                        "error": str(exc),
                    }
                )
                append_jsonl(
                    raw_jsonl,
                    {
                        "subgroup": key[0],
                        "micro_topic_id": key[1],
                        "year": key[2],
                        "raw_response": raw_response,
                        "parse_or_runtime_error": str(exc),
                    },
                )

        if len(saved_rows) >= args.save_every:
            wrote_header = flush_rows(output_csv, saved_rows, wrote_header)
            saved_rows = []
            if error_rows:
                pd.DataFrame(error_rows).to_csv(error_csv, index=False)
            else:
                pd.DataFrame(columns=["subgroup", "micro_topic_id", "year", "error"]).to_csv(error_csv, index=False)

    wrote_header = flush_rows(output_csv, saved_rows, wrote_header)
    if error_rows:
        pd.DataFrame(error_rows).to_csv(error_csv, index=False)
    else:
        pd.DataFrame(columns=["subgroup", "micro_topic_id", "year", "error"]).to_csv(error_csv, index=False)

    manifest = {
        "model": args.model_name,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "input_csv": str(args.input_csv),
        "processed_rows": processed,
        "pending_rows_initial": int(len(pending)),
        "error_count": len(error_rows),
        "hardware": hardware,
        "outputs": {
            "annual_summaries_csv": str(output_csv),
            "annual_summaries_jsonl": str(output_jsonl),
            "raw_responses": str(raw_jsonl),
            "error_log": str(error_csv),
            "prompt_template": str(prompt_path),
        },
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print_log(f"Finished processed_rows={processed} error_count={len(error_rows)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
