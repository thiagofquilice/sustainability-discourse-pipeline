#!/usr/bin/env python3
"""Run top-1 cosine assignment for the unified 6-topic discourse pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from six_topic_discourse_catalog import TOPIC_GROUPS, bullet_list_text, catalog_rows
from workflow_common import (
    add_document_aliases,
    configure_logging,
    cosine_similarity_batch,
    current_run_manifest,
    detect_embedding_model_name,
    ensure_directory,
    load_config,
    load_corpus,
    load_embedding_memmap,
    load_sentence_transformer,
    safe_share,
    save_dataframe,
    top_two_matches,
    write_json,
)


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = WORKFLOW_ROOT / "config" / "paper_6topic_pipeline_config.json"
DEFAULT_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs" / "cosine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def build_catalog() -> pd.DataFrame:
    catalog = pd.DataFrame(catalog_rows()).sort_values("topic_id").reset_index(drop=True)
    catalog["label_embedding_text"] = catalog["topic_embedding_text"].astype(str)
    return catalog


def build_subanchor_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in TOPIC_GROUPS:
        topic_elements_bullets = bullet_list_text(group["elements"])
        for order, element in enumerate(group["elements"], start=1):
            rows.append(
                {
                    "assigned_label": group["topic_code"],
                    "assigned_label_id": group["topic_id"],
                    "topic_name": group["topic_name"],
                    "topic_definition": group["topic_definition"],
                    "assigned_label_text": topic_elements_bullets,
                    "topic_summary_text": group["topic_summary_text"],
                    "topic_embedding_text": group["topic_summary_text"],
                    "sdg_crosswalk": ", ".join(group["sdg_crosswalk"]),
                    "boundary_notes": group["boundary_notes"],
                    "false_positive_notes": group["false_positive_notes"],
                    "subanchor_order": order,
                    "subanchor_text": element,
                }
            )
    return pd.DataFrame(rows)


def summarize_counts(frame: pd.DataFrame, group_cols: list[str], count_name: str) -> pd.DataFrame:
    return (
        frame.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name=count_name)
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    output_root = args.output_root
    ensure_directory(output_root)

    catalog = build_catalog()
    subanchors = build_subanchor_frame()

    corpus = add_document_aliases(load_corpus(Path(config["paths"]["dataset_corpus"])))
    if args.max_docs is not None:
        corpus = corpus.head(args.max_docs).copy()

    full_corpus_rows = len(load_corpus(Path(config["paths"]["dataset_corpus"]), columns=["doc_id"]))
    document_embeddings, _ = load_embedding_memmap(
        Path(config["paths"]["dataset_embeddings"]),
        Path(config["paths"]["dataset_embedding_meta"]),
        expected_rows=full_corpus_rows,
    )
    doc_matrix = np.asarray(document_embeddings[: len(corpus)], dtype="float32")

    model_name = detect_embedding_model_name(config)
    model = load_sentence_transformer(model_name)
    subanchor_embeddings = model.encode(
        subanchors["subanchor_text"].tolist(),
        batch_size=args.embedding_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    np.save(output_root / "subanchor_embeddings.npy", subanchor_embeddings)

    batch_size = int(config["assignment"]["batch_size"])
    threshold = float(config["assignment"]["similarity_threshold"])
    outside_scope_value = str(config["assignment"]["output_label_for_no_match"])

    topic_codes = catalog["topic_code"].tolist()
    topic_meta = catalog.set_index("topic_code")
    topic_to_anchor_idx = {
        topic_code: subanchors.index[subanchors["assigned_label"] == topic_code].to_numpy()
        for topic_code in topic_codes
    }

    topic_score_parts: list[np.ndarray] = []
    for start in range(0, len(corpus), batch_size):
        end = min(start + batch_size, len(corpus))
        subanchor_scores = cosine_similarity_batch(doc_matrix[start:end], subanchor_embeddings)
        topic_scores = np.empty((len(subanchor_scores), len(topic_codes)), dtype="float32")
        for idx, topic_code in enumerate(topic_codes):
            topic_scores[:, idx] = subanchor_scores[:, topic_to_anchor_idx[topic_code]].max(axis=1)
        topic_score_parts.append(topic_scores)

    score_matrix = np.vstack(topic_score_parts)
    best_idx, second_idx = top_two_matches(score_matrix)
    best_scores = score_matrix[np.arange(len(score_matrix)), best_idx]
    second_scores = score_matrix[np.arange(len(score_matrix)), second_idx]
    score_gap = best_scores - second_scores
    n_candidates_above_threshold = (score_matrix >= threshold).sum(axis=1).astype(int)
    positive_mask = best_scores >= threshold

    result = corpus.copy()
    result["embedding_row_index"] = result.index.astype(int)
    result["best_label_id"] = catalog.loc[best_idx, "topic_id"].to_numpy()
    result["best_label_code"] = catalog.loc[best_idx, "topic_code"].to_numpy()
    result["best_topic_name"] = catalog.loc[best_idx, "topic_name"].to_numpy()
    result["best_label_text"] = catalog.loc[best_idx, "topic_elements_bullets"].to_numpy()
    result["best_topic_definition"] = catalog.loc[best_idx, "topic_definition"].to_numpy()
    result["best_topic_summary_text"] = catalog.loc[best_idx, "topic_summary_text"].to_numpy()
    result["best_sdg_crosswalk"] = catalog.loc[best_idx, "sdg_crosswalk"].to_numpy()
    result["best_score"] = best_scores.astype("float64")
    result["second_best_label_id"] = catalog.loc[second_idx, "topic_id"].to_numpy()
    result["second_best_label_code"] = catalog.loc[second_idx, "topic_code"].to_numpy()
    result["second_best_topic_name"] = catalog.loc[second_idx, "topic_name"].to_numpy()
    result["second_best_label_text"] = catalog.loc[second_idx, "topic_elements_bullets"].to_numpy()
    result["second_best_score"] = second_scores.astype("float64")
    result["score_gap"] = score_gap.astype("float64")
    result["score_margin"] = result["score_gap"]
    result["n_candidates_above_threshold"] = n_candidates_above_threshold
    result["assignment_status"] = np.where(positive_mask, "assigned_positive", "below_threshold")
    result["assigned_label"] = np.where(positive_mask, result["best_label_code"], outside_scope_value)
    result["assigned_label_id"] = np.where(positive_mask, result["best_label_id"], -1)
    result["topic_name"] = np.where(positive_mask, result["best_topic_name"], "OUTSIDE_SCOPE")
    result["assigned_label_text"] = np.where(
        positive_mask,
        result["best_label_text"],
        "No topic elements reached the cosine threshold.",
    )
    result["topic_definition"] = np.where(
        positive_mask,
        result["best_topic_definition"],
        "No topic definition because the row remained outside scope.",
    )
    result["topic_summary_text"] = np.where(
        positive_mask,
        result["best_topic_summary_text"],
        "No topic elements reached the cosine threshold.",
    )
    result["sdg_crosswalk"] = np.where(positive_mask, result["best_sdg_crosswalk"], "")

    candidate_rows: list[pd.DataFrame] = []
    for idx, topic_code in enumerate(topic_codes):
        mask = score_matrix[:, idx] >= threshold
        if not mask.any():
            continue
        part = result.loc[
            mask,
            [
                "chunk_id",
                "document_id",
                "doc_id",
                "source_doc_id",
                "text",
                "source",
                "year",
                "industry",
                "embedding_row_index",
                "best_label_code",
                "best_topic_name",
                "best_score",
                "second_best_label_code",
                "second_best_topic_name",
                "second_best_score",
                "score_gap",
                "score_margin",
                "n_candidates_above_threshold",
            ],
        ].copy()
        part["assigned_label"] = topic_code
        part["assigned_label_id"] = int(topic_meta.loc[topic_code, "topic_id"])
        part["topic_name"] = str(topic_meta.loc[topic_code, "topic_name"])
        part["assigned_label_text"] = str(topic_meta.loc[topic_code, "topic_elements_bullets"])
        part["topic_definition"] = str(topic_meta.loc[topic_code, "topic_definition"])
        part["topic_summary_text"] = str(topic_meta.loc[topic_code, "topic_summary_text"])
        part["sdg_crosswalk"] = str(topic_meta.loc[topic_code, "sdg_crosswalk"])
        part["candidate_score"] = score_matrix[mask, idx].astype("float64")
        candidate_rows.append(part)

    candidates = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
    if not candidates.empty:
        candidates["candidate_rank"] = (
            candidates.groupby("chunk_id", dropna=False)["candidate_score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        candidates = candidates.sort_values(["chunk_id", "candidate_rank", "assigned_label"]).reset_index(drop=True)
        if args.candidate_limit is not None:
            candidates = candidates.loc[candidates["candidate_rank"] <= args.candidate_limit].copy()

    positive = result.loc[positive_mask].copy().sort_values(["source", "assigned_label", "chunk_id"]).reset_index(drop=True)

    save_dataframe(result, output_root / "labeled_corpus.parquet")
    save_dataframe(positive, output_root / "best_only_positive.parquet")
    if not candidates.empty:
        save_dataframe(candidates, output_root / "multi_candidate_positive.parquet")

    catalog.to_csv(output_root / "six_topic_catalog_snapshot.csv", index=False)
    subanchors.to_csv(output_root / "six_topic_subanchor_catalog.csv", index=False)
    summarize_counts(positive, ["source"], "positive_rows").to_csv(output_root / "positive_by_source.csv", index=False)
    summarize_counts(positive, ["assigned_label", "topic_name"], "positive_rows").to_csv(output_root / "positive_by_topic.csv", index=False)
    (
        result.groupby("n_candidates_above_threshold", dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("n_candidates_above_threshold")
        .to_csv(output_root / "candidate_count_distribution.csv", index=False)
    )

    diagnostics = {
        "variant": "six_topic_elements_max_without_margin",
        "total_rows": int(len(result)),
        "positive_rows": int(len(positive)),
        "outside_scope_rows": int((result["assigned_label"] == outside_scope_value).sum()),
        "outside_scope_share": float(safe_share((result["assigned_label"] == outside_scope_value).sum(), len(result))),
        "threshold": threshold,
        "candidate_rows": int(len(candidates)),
        "docs_with_multiple_candidates": int((result["n_candidates_above_threshold"] > 1).sum()),
        "docs_with_multiple_candidates_share": float(safe_share((result["n_candidates_above_threshold"] > 1).sum(), len(result))),
        "embedding_model": model_name,
        "topic_codes": topic_codes,
    }
    write_json(output_root / "assignment_diagnostics.json", diagnostics)
    write_json(
        output_root / "manifest.json",
        current_run_manifest(
            config,
            extra={
                "stage": "run_6topic_discourse_cosine",
                "outputs": {
                    "labeled_corpus": str(output_root / "labeled_corpus.parquet"),
                    "best_only_positive": str(output_root / "best_only_positive.parquet"),
                    "multi_candidate_positive": str(output_root / "multi_candidate_positive.parquet"),
                    "diagnostics": str(output_root / "assignment_diagnostics.json"),
                },
            },
        ),
    )
    print(output_root / "best_only_positive.parquet")


if __name__ == "__main__":
    main()
