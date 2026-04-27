#!/usr/bin/env python3
"""Fit supervised BERTopic using upstream-assigned labels from the 6-topic pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from workflow_common import configure_logging, current_run_manifest, load_config, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None, help="Optional labeled corpus path.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def build_supervised_topic_model(config: dict) -> BERTopic:
    classifier_params = config["bertopic"]["classifier_params"]
    classifier = LogisticRegression(
        max_iter=int(classifier_params["max_iter"]),
        class_weight=classifier_params["class_weight"],
    )
    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=int(config["bertopic"]["vectorizer_min_df"]),
        max_df=float(config["bertopic"]["vectorizer_max_df"]),
    )
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    return BERTopic(
        embedding_model=None,
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=classifier,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        top_n_words=int(config["bertopic"]["top_n_words"]),
        verbose=True,
    )


def build_crosswalk(document_topics: pd.DataFrame) -> pd.DataFrame:
    crosswalk = (
        document_topics.groupby(
            ["assigned_label", "assigned_label_id", "assigned_label_text", "bertopic_topic_id"],
            dropna=False,
        )
        .size()
        .rename("document_count")
        .reset_index()
    )
    totals = crosswalk.groupby("assigned_label", dropna=False)["document_count"].transform("sum")
    crosswalk["row_share_within_label"] = crosswalk["document_count"] / totals
    return crosswalk.sort_values(["assigned_label", "document_count"], ascending=[True, False])


def diagnostics_tables(document_topics: pd.DataFrame, output_dir: Path) -> None:
    for field_name in ["assigned_label", "source", "year", "industry"]:
        summary = (
            document_topics.groupby(field_name, dropna=False)
            .size()
            .rename("document_count")
            .reset_index()
            .sort_values("document_count", ascending=False)
        )
        summary.to_csv(output_dir / f"counts_by_{field_name}.csv", index=False)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    output_dir = args.output_dir or Path(config["paths"]["full_output_root"])
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_path = args.input or Path(config["paths"]["labeled_corpus"])
    frame = pd.read_parquet(labeled_path)
    include_outside_scope = bool(config["bertopic"]["include_outside_scope_in_fit"])
    fit_frame = frame.copy()
    if not include_outside_scope:
        fit_frame = fit_frame.loc[fit_frame["assigned_label"] != "OUTSIDE_SCOPE"].copy()
    if fit_frame.empty:
        raise SystemExit("No rows available for supervised BERTopic fit after applying the fit filter.")

    topic_model = build_supervised_topic_model(config)
    docs = fit_frame["text"].fillna("").astype(str).tolist()
    y = pd.to_numeric(fit_frame["assigned_label_id"], errors="raise").astype(int).tolist()
    embeddings_meta = read_json(Path(config["paths"]["dataset_embedding_meta"]))
    dropped_missing_embedding_rows = 0
    if "embedding_row_index" in fit_frame.columns:
        fit_frame["embedding_row_index"] = pd.to_numeric(fit_frame["embedding_row_index"], errors="coerce")
        dropped_missing_embedding_rows = int(fit_frame["embedding_row_index"].isna().sum())
        if dropped_missing_embedding_rows:
            fit_frame = fit_frame.loc[fit_frame["embedding_row_index"].notna()].copy()
            docs = fit_frame["text"].fillna("").astype(str).tolist()
            y = pd.to_numeric(fit_frame["assigned_label_id"], errors="raise").astype(int).tolist()
        fit_indices = fit_frame["embedding_row_index"].astype(int).to_numpy()
    elif len(frame) == int(embeddings_meta["rows"]):
        fit_indices = fit_frame.index.to_numpy()
    else:
        raise SystemExit(
            "The labeled corpus does not include embedding_row_index, so alignment cannot be recovered for a subset run."
        )
    import numpy as np

    memmap = np.memmap(
        Path(config["paths"]["dataset_embeddings"]),
        dtype="float32",
        mode="r",
        shape=(int(embeddings_meta["rows"]), int(embeddings_meta["dims"])),
    )
    fit_embeddings = np.asarray(memmap[fit_indices], dtype="float32")

    topics, probabilities = topic_model.fit_transform(docs, embeddings=fit_embeddings, y=y)

    document_topics = fit_frame.copy()
    document_topics["bertopic_topic_id"] = pd.Series(topics, index=document_topics.index, dtype="Int64")
    if probabilities is None:
        document_topics["bertopic_probability"] = pd.NA
    else:
        document_topics["bertopic_probability"] = probabilities

    document_topics_path = output_dir / "document_topics.parquet"
    document_topics.to_parquet(document_topics_path, index=False)

    topic_info = topic_model.get_topic_info()
    topic_info_path = output_dir / "topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False)

    crosswalk = build_crosswalk(document_topics)
    crosswalk_path = output_dir / "label_topic_mapping.csv"
    crosswalk.to_csv(crosswalk_path, index=False)

    dominant_topic_per_label = (
        crosswalk.sort_values(["assigned_label", "document_count"], ascending=[True, False])
        .drop_duplicates(subset=["assigned_label"])
        .rename(columns={"bertopic_topic_id": "dominant_bertopic_topic_id"})
    )
    dominant_topic_per_label.to_csv(output_dir / "dominant_topic_per_label.csv", index=False)

    diagnostics_tables(document_topics, output_dir)

    if args.save_model:
        model_dir = args.model_dir or (output_dir / "bertopic_model")
        model_dir.mkdir(parents=True, exist_ok=True)
        topic_model.save(
            str(model_dir),
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=False,
        )

    write_json(
        output_dir / "manifest.json",
        current_run_manifest(
            config,
            extra={
                "stage": "run_supervised_bertopic",
                "include_outside_scope_in_fit": include_outside_scope,
                "fit_rows": int(len(fit_frame)),
                "dropped_missing_embedding_rows": dropped_missing_embedding_rows,
                "topic_count": int(topic_info["Topic"].nunique()),
                "outputs": {
                    "document_topics": str(document_topics_path),
                    "topic_info_csv": str(topic_info_path),
                    "label_topic_mapping_csv": str(crosswalk_path),
                },
            },
        ),
    )
    print(f"Saved topic assignments to {document_topics_path}")
    print(f"Saved topic info to {topic_info_path}")
    print(f"Saved label-topic mapping to {crosswalk_path}")


if __name__ == "__main__":
    main()
