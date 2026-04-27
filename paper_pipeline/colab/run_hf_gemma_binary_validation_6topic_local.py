#!/usr/bin/env python3
"""Run local Gemma binary validation for the unified 6-topic discourse pipeline."""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm


PROMPT_TEMPLATE = """You are a binary classifier.

Task:
Decide whether the text substantively discusses the specific topic below.

Decision rule:
Answer "yes" only if the text clearly discusses at least one listed topic element, or a clearly equivalent concept, in meaning, in a concrete and topic-specific way.
Answer "no" if the match is only generic sustainability language, generic environmental language, generic risk disclosure, generic finance or investment language, or a nearby but different topic.

Important:
- Do not infer the topic from broad ESG wording alone.
- Do not answer "yes" just because the text mentions climate, sustainability, environment, risk, energy, biodiversity, disclosure, or regulation in general.
- The text must match at least one specific topic element below in meaning, not just through overlapping words.
- Do not answer "yes" based only on partial lexical overlap or nearby vocabulary if the underlying concept is different.
- Do not answer "yes" if the topic appears only as a vague or incidental reference.
- If the text is only loosely related, answer "no".
- When in doubt, prefer "no".

Topic:
{topic_name}

Topic elements:
{topic_elements_bullets}

Text:
{text}

Question:
Does this text substantively discuss this topic?

Reply with only one word: yes or no."""
DEFAULT_MODEL_NAME = "google/gemma-4-E4B-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=2048)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--id-col", type=str, default="chunk_id")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="assigned_label")
    parser.add_argument("--topic-name-col", type=str, default="topic_name")
    parser.add_argument("--label-text-col", type=str, default="assigned_label_text")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--require-gpu", action="store_true", default=True)
    parser.add_argument("--input-format", choices=["auto", "csv", "json", "jsonl", "parquet"], default="auto")
    parser.add_argument("--mirror-output-dir", type=Path, default=None)
    return parser.parse_args()


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def print_log(message: str) -> None:
    print(f"[{now_utc_iso()}] {message}", flush=True)


def normalize_decision(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).strip().lower())
    match = re.search(r"\b(yes|no|sim|nao|não)\b", cleaned)
    if not match:
        raise ValueError(f"Could not parse yes/no from response: {text!r}")
    token = match.group(1)
    return "yes" if token in {"yes", "sim"} else "no"


def infer_input_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".parquet":
        return "parquet"
    raise ValueError(f"Could not infer input format from suffix: {path}")


def load_input_frame(path: Path, file_format: str) -> pd.DataFrame:
    if file_format == "csv":
        return pd.read_csv(path)
    if file_format == "jsonl":
        return pd.read_json(path, lines=True)
    if file_format == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            return pd.DataFrame(payload["records"])
        raise ValueError("JSON input must be a list of objects or a dict with a 'records' list.")
    if file_format == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {file_format}")


def ensure_required_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")


def detect_hardware() -> dict[str, Any]:
    import torch

    hardware = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": None,
        "gpu_total_memory_gb": None,
        "gpu_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        hardware["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        hardware["gpu_total_memory_gb"] = round(props.total_memory / (1024**3), 2)
    return hardware


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    hardware = detect_hardware()
    hardware["transformers_version"] = transformers.__version__
    if args.require_gpu and not hardware["cuda_available"]:
        raise RuntimeError("No CUDA GPU detected in the runtime. This runner is intended for Colab GPU use.")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    return model, tokenizer, hardware


def build_prompt_messages(row: pd.Series, args: argparse.Namespace) -> list[dict[str, str]]:
    topic_name = str(row.get(args.topic_name_col, row.get(args.label_col, "")))
    topic_elements_bullets = str(row.get(args.label_text_col, ""))
    return [
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(
                topic_name=topic_name,
                topic_elements_bullets=topic_elements_bullets,
                text=str(row.get(args.text_col, "")),
            ),
        }
    ]


def generate_batch(model, tokenizer, rows: list[pd.Series], args: argparse.Namespace) -> list[str]:
    import torch

    prompts = [
        tokenizer.apply_chat_template(
            build_prompt_messages(row, args),
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_lengths = attention_mask.sum(dim=1).tolist()
    decisions: list[str] = []
    for idx, length in enumerate(input_lengths):
        new_tokens = generated[idx, int(length):]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        decisions.append(normalize_decision(decoded))
    return decisions


def make_result_row(row: pd.Series, decision: str) -> dict[str, Any]:
    payload = row.to_dict()
    payload["v_gemma"] = decision
    return payload


def make_error_row(item_id: str, error: Exception | str) -> dict[str, Any]:
    message = str(error)
    return {
        "id": item_id,
        "error_type": type(error).__name__ if isinstance(error, Exception) else "RuntimeError",
        "error_message": message,
        "logged_at_utc": now_utc_iso(),
    }


def build_manifest(*, args: argparse.Namespace, hardware: dict[str, Any], total_input_rows: int, rows_processed_this_run: int, rows_in_output: int, resumed_rows: int, error_rows: int, started_at: float) -> dict[str, Any]:
    elapsed_seconds = max(time.time() - started_at, 1e-9)
    rows_per_minute = (rows_processed_this_run / elapsed_seconds) * 60.0 if rows_processed_this_run > 0 else 0.0
    return {
        "generated_at_utc": now_utc_iso(),
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "model_name": args.model_name,
        "criterion": "mention_elements",
        "prompt_template": PROMPT_TEMPLATE,
        "topic_name_column": args.topic_name_col,
        "topic_elements_column": args.label_text_col,
        "batch_size": args.batch_size,
        "save_every": args.save_every,
        "max_new_tokens": args.max_new_tokens,
        "resume": bool(args.resume),
        "total_input_rows": int(total_input_rows),
        "rows_processed_this_run": int(rows_processed_this_run),
        "rows_in_output": int(rows_in_output),
        "resumed_rows": int(resumed_rows),
        "error_rows": int(error_rows),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "rows_per_minute": round(rows_per_minute, 3),
        **hardware,
    }


def persist_outputs(*, output_df: pd.DataFrame, error_df: pd.DataFrame, manifest: dict[str, Any], output_path: Path, error_path: Path, manifest_path: Path, prompt_path: Path, mirror_output_dir: Path | None) -> None:
    output_df.to_csv(output_path, index=False)
    error_df.to_csv(error_path, index=False)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_path.write_text(PROMPT_TEMPLATE, encoding="utf-8")
    if mirror_output_dir is None:
        return
    mirror_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, mirror_output_dir / output_path.name)
    shutil.copy2(error_path, mirror_output_dir / error_path.name)
    shutil.copy2(manifest_path, mirror_output_dir / manifest_path.name)
    shutil.copy2(prompt_path, mirror_output_dir / prompt_path.name)


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    local_error_path = args.output_path.parent / "error_log.csv"
    local_manifest_path = args.output_path.parent / "run_manifest.json"
    local_prompt_path = args.output_path.parent / "prompt_used.txt"

    file_format = infer_input_format(args.input_path, args.input_format)
    frame = load_input_frame(args.input_path, file_format)
    ensure_required_columns(frame, [args.id_col, args.text_col, args.label_col, args.label_text_col])
    frame[args.id_col] = frame[args.id_col].astype(str)
    if args.max_rows is not None:
        frame = frame.head(args.max_rows).copy()

    if args.resume and args.output_path.exists():
        existing_output = pd.read_csv(args.output_path)
        existing_output[args.id_col] = existing_output[args.id_col].astype(str)
    else:
        existing_output = pd.DataFrame(columns=list(frame.columns) + ["v_gemma"])

    if local_error_path.exists():
        error_log = pd.read_csv(local_error_path)
    else:
        error_log = pd.DataFrame(columns=["id", "error_type", "error_message", "logged_at_utc"])

    completed_ids = set(existing_output[args.id_col].astype(str).tolist()) if not existing_output.empty else set()
    pending = frame.loc[~frame[args.id_col].isin(completed_ids)].copy()
    resumed_rows = len(completed_ids)
    total_input_rows = len(frame)

    print_log(f"Rows in input: {total_input_rows}")
    print_log(f"Rows already completed via resume: {resumed_rows}")
    print_log(f"Rows pending in this run: {len(pending)}")

    model, tokenizer, hardware = load_model_and_tokenizer(args)
    print_log(f"Detected hardware: {hardware}")

    started_at = time.time()
    new_rows: list[dict[str, Any]] = []
    new_errors: list[dict[str, Any]] = []
    last_saved_processed = 0
    progress = tqdm(total=len(pending), desc="Gemma validation", unit="rows")

    batch_rows: list[pd.Series] = []
    for row in pending.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        batch_rows.append(row_series)
        if len(batch_rows) < args.batch_size:
            continue
        try:
            decisions = generate_batch(model, tokenizer, batch_rows, args)
            for source_row, decision in zip(batch_rows, decisions):
                new_rows.append(make_result_row(source_row, decision))
                progress.update(1)
        except Exception as batch_exc:
            print_log(f"Batch failed; falling back to row-by-row. Error: {batch_exc}")
            for source_row in batch_rows:
                item_id = str(source_row[args.id_col])
                try:
                    decision = generate_batch(model, tokenizer, [source_row], args)[0]
                    new_rows.append(make_result_row(source_row, decision))
                except Exception as row_exc:
                    new_errors.append(make_error_row(item_id, row_exc))
                progress.update(1)
        batch_rows = []

        processed_now = len(new_rows) + len(new_errors)
        if processed_now and (processed_now - last_saved_processed) >= args.save_every:
            output_df = pd.concat([existing_output, pd.DataFrame(new_rows)], ignore_index=True)
            error_df = pd.concat([error_log, pd.DataFrame(new_errors)], ignore_index=True)
            manifest = build_manifest(
                args=args,
                hardware=hardware,
                total_input_rows=total_input_rows,
                rows_processed_this_run=processed_now,
                rows_in_output=len(output_df),
                resumed_rows=resumed_rows,
                error_rows=len(error_df),
                started_at=started_at,
            )
            persist_outputs(
                output_df=output_df,
                error_df=error_df,
                manifest=manifest,
                output_path=args.output_path,
                error_path=local_error_path,
                manifest_path=local_manifest_path,
                prompt_path=local_prompt_path,
                mirror_output_dir=args.mirror_output_dir,
            )
            last_saved_processed = processed_now
            print_log(f"Checkpoint saved after {processed_now} processed rows in this run.")

    if batch_rows:
        try:
            decisions = generate_batch(model, tokenizer, batch_rows, args)
            for source_row, decision in zip(batch_rows, decisions):
                new_rows.append(make_result_row(source_row, decision))
                progress.update(1)
        except Exception as batch_exc:
            print_log(f"Final partial batch failed; falling back to row-by-row. Error: {batch_exc}")
            for source_row in batch_rows:
                item_id = str(source_row[args.id_col])
                try:
                    decision = generate_batch(model, tokenizer, [source_row], args)[0]
                    new_rows.append(make_result_row(source_row, decision))
                except Exception as row_exc:
                    new_errors.append(make_error_row(item_id, row_exc))
                progress.update(1)
    progress.close()

    output_df = pd.concat([existing_output, pd.DataFrame(new_rows)], ignore_index=True)
    error_df = pd.concat([error_log, pd.DataFrame(new_errors)], ignore_index=True)
    manifest = build_manifest(
        args=args,
        hardware=hardware,
        total_input_rows=total_input_rows,
        rows_processed_this_run=len(new_rows) + len(new_errors),
        rows_in_output=len(output_df),
        resumed_rows=resumed_rows,
        error_rows=len(error_df),
        started_at=started_at,
    )
    persist_outputs(
        output_df=output_df,
        error_df=error_df,
        manifest=manifest,
        output_path=args.output_path,
        error_path=local_error_path,
        manifest_path=local_manifest_path,
        prompt_path=local_prompt_path,
        mirror_output_dir=args.mirror_output_dir,
    )
    print_log(f"Completed. Final output saved to {args.output_path}")


if __name__ == "__main__":
    main()
