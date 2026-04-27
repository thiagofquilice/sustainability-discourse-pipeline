#!/usr/bin/env python3
"""Run the post-Gemma supervised BERTopic workflow for the unified 6-topic pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from bertopic import BERTopic

from workflow_common import configure_logging


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = WORKFLOW_ROOT / "config" / "paper_6topic_pipeline_config.json"
DEFAULT_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs"
DEFAULT_ADJUSTED_DIR = DEFAULT_OUTPUT_ROOT / "full_run" / "adjusted_with_t2_secondary_recovery"
DEFAULT_ADJUSTED_FULL = DEFAULT_ADJUSTED_DIR / "adjusted_full_validation_output.csv"
DEFAULT_ADJUSTED_YES = DEFAULT_ADJUSTED_DIR / "adjusted_full_yes.parquet"
DEFAULT_CATALOG = WORKFLOW_ROOT / "catalog" / "six_topic_discourse_catalog.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--adjusted-full-output", type=Path, default=DEFAULT_ADJUSTED_FULL)
    parser.add_argument("--adjusted-yes-output", type=Path, default=DEFAULT_ADJUSTED_YES)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--save-model", action="store_true", default=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def run_script(script_name: str, args: list[str]) -> None:
    command = [sys.executable, str(WORKFLOW_ROOT / "scripts" / script_name), *args]
    subprocess.run(command, check=True)


def write_config(base_config: dict, output_root: Path, catalog: Path) -> Path:
    bertopic_root = output_root / "bertopic_supervised"
    bertopic_root.mkdir(parents=True, exist_ok=True)
    config = json.loads(json.dumps(base_config))
    config["paths"]["stable_label_catalog_csv"] = str(catalog)
    config["paths"]["stable_label_catalog_json"] = str(catalog.with_suffix(".json"))
    config["paths"]["full_output_root"] = str(bertopic_root)
    config["paths"]["labeled_corpus"] = str(bertopic_root / "gemma_yes_best_only.parquet")
    config["paths"]["positive_labeled_corpus"] = str(bertopic_root / "gemma_yes_best_only.parquet")
    config["paths"]["document_topics"] = str(bertopic_root / "document_topics.parquet")
    config["paths"]["topic_info_csv"] = str(bertopic_root / "topic_info.csv")
    config["paths"]["label_topic_mapping_csv"] = str(bertopic_root / "label_topic_mapping.csv")
    config["paths"]["manifest_json"] = str(bertopic_root / "manifest.json")
    config["paths"]["descriptive_analysis_dir"] = str(bertopic_root / "descriptive_analysis")
    config["paths"]["model_dir"] = str(bertopic_root / "bertopic_model")
    config_path = bertopic_root / "six_topic_supervised_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def ensure_adjusted_inputs(adjusted_full_output: Path, adjusted_yes_output: Path) -> tuple[Path, Path]:
    if adjusted_full_output.exists() and adjusted_yes_output.exists():
        return adjusted_full_output, adjusted_yes_output
    run_script(
        "materialize_adjusted_full_validation_output.py",
        [
            "--output-dir",
            str(DEFAULT_ADJUSTED_DIR),
        ],
    )
    return adjusted_full_output, adjusted_yes_output


def build_gemma_yes_input(adjusted_full_output: Path, adjusted_yes_output: Path, output_root: Path) -> Path:
    bertopic_root = output_root / "bertopic_supervised"
    bertopic_root.mkdir(parents=True, exist_ok=True)
    parquet_candidate = adjusted_yes_output.with_suffix(".parquet")
    if not adjusted_yes_output.exists() and parquet_candidate.exists():
        adjusted_yes_output = parquet_candidate
    if adjusted_yes_output.exists():
        if adjusted_yes_output.suffix.lower() == ".parquet":
            yes_frame = pd.read_parquet(adjusted_yes_output)
        else:
            yes_frame = pd.read_csv(adjusted_yes_output, low_memory=False)
    else:
        frame = pd.read_csv(adjusted_full_output)
        yes_frame = frame.loc[frame["v_gemma"] == "yes"].copy().reset_index(drop=True)
    yes_frame.to_parquet(bertopic_root / "gemma_yes_best_only.parquet", index=False)
    yes_frame.to_csv(bertopic_root / "gemma_yes_best_only.csv", index=False)
    return bertopic_root / "gemma_yes_best_only.parquet"


def run_hierarchical_topics(model_dir: Path, document_topics_path: Path, output_dir: Path) -> None:
    frame = pd.read_parquet(document_topics_path)
    topic_model = BERTopic.load(str(model_dir))
    docs = frame["text"].fillna("").astype(str).tolist()
    hierarchical = topic_model.hierarchical_topics(docs)
    hierarchical.to_csv(output_dir / "hierarchical_topics.csv", index=False)
    try:
        fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical)
        fig.write_html(str(output_dir / "hierarchical_topics.html"))
    except Exception:
        pass


def build_temporal_prep(output_root: Path) -> None:
    bertopic_root = output_root / "bertopic_supervised"
    prep_dir = output_root / "temporal_prep"
    prep_dir.mkdir(parents=True, exist_ok=True)
    topic_info = pd.read_csv(bertopic_root / "topic_info.csv")
    document_topics = pd.read_parquet(bertopic_root / "document_topics.parquet")
    topics_over_time = pd.read_csv(bertopic_root / "topics_over_time_global.csv")

    topic_year_source = (
        document_topics.groupby(["bertopic_topic_id", "year", "source"], dropna=False)
        .size()
        .reset_index(name="document_count")
        .sort_values(["bertopic_topic_id", "year", "source"])
    )
    representative = (
        document_topics.sort_values(["bertopic_topic_id", "year", "bertopic_probability"], ascending=[True, True, False])
        .groupby(["bertopic_topic_id", "year"], dropna=False)
        .head(3)[["bertopic_topic_id", "year", "source", "chunk_id", "source_doc_id", "assigned_label", "topic_name", "text"]]
        .rename(columns={"text": "text_full"})
    )
    topic_info.to_csv(prep_dir / "topic_info_for_llm.csv", index=False)
    topics_over_time.to_csv(prep_dir / "topics_over_time_global_for_llm.csv", index=False)
    topic_year_source.to_csv(prep_dir / "topic_year_source_counts_for_llm.csv", index=False)
    representative.to_csv(prep_dir / "representative_docs_by_topic_year.csv", index=False)
    manifest = {
        "topic_info": str(prep_dir / "topic_info_for_llm.csv"),
        "topics_over_time": str(prep_dir / "topics_over_time_global_for_llm.csv"),
        "topic_year_source_counts": str(prep_dir / "topic_year_source_counts_for_llm.csv"),
        "representative_docs": str(prep_dir / "representative_docs_by_topic_year.csv"),
    }
    (prep_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_descriptive_analysis(output_root: Path, config_path: Path, log_level: str) -> None:
    bertopic_root = output_root / "bertopic_supervised"
    run_script(
        "build_bertopic_supervised_descriptive_analysis.py",
        [
            "--config",
            str(config_path),
            "--document-topics",
            str(bertopic_root / "document_topics.parquet"),
            "--model-dir",
            str(bertopic_root / "bertopic_model"),
            "--output-dir",
            str(bertopic_root / "descriptive_analysis"),
            "--log-level",
            log_level,
        ],
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    config_path = write_config(base_config, args.output_root, args.catalog)
    adjusted_full_output, adjusted_yes_output = ensure_adjusted_inputs(args.adjusted_full_output, args.adjusted_yes_output)
    gemma_yes_input = build_gemma_yes_input(adjusted_full_output, adjusted_yes_output, args.output_root)
    bertopic_root = args.output_root / "bertopic_supervised"

    supervised_args = [
        "--config",
        str(config_path),
        "--input",
        str(gemma_yes_input),
        "--output-dir",
        str(bertopic_root),
        "--model-dir",
        str(bertopic_root / "bertopic_model"),
        "--log-level",
        args.log_level,
    ]
    if args.save_model:
        supervised_args.append("--save-model")
    run_script("run_supervised_bertopic.py", supervised_args)
    run_script(
        "run_topics_per_class.py",
        [
            "--config",
            str(config_path),
            "--document-topics",
            str(bertopic_root / "document_topics.parquet"),
            "--model-dir",
            str(bertopic_root / "bertopic_model"),
            "--output-dir",
            str(bertopic_root),
            "--log-level",
            args.log_level,
        ],
    )
    run_script(
        "run_topics_over_time.py",
        [
            "--config",
            str(config_path),
            "--document-topics",
            str(bertopic_root / "document_topics.parquet"),
            "--model-dir",
            str(bertopic_root / "bertopic_model"),
            "--output-dir",
            str(bertopic_root),
            "--log-level",
            args.log_level,
        ],
    )
    run_hierarchical_topics(bertopic_root / "bertopic_model", bertopic_root / "document_topics.parquet", bertopic_root)
    build_descriptive_analysis(args.output_root, config_path, args.log_level)
    build_temporal_prep(args.output_root)
    print(bertopic_root)


if __name__ == "__main__":
    main()
