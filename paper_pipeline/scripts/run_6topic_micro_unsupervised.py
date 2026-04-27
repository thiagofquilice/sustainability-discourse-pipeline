#!/usr/bin/env python3
"""Fit unsupervised BERTopic models for each source-topic subgroup in the adjusted 6-topic corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from workflow_common import configure_logging, load_sentence_transformer, read_json


WORKFLOW_ROOT = Path("paper_pipeline")
DEFAULT_INPUT = WORKFLOW_ROOT / "outputs" / "full_run" / "adjusted_with_t2_secondary_recovery" / "adjusted_full_yes.csv"
DEFAULT_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs" / "bertopic_micro_unsupervised_multiaspect"
DEFAULT_EMBEDDING_FILE = Path("data/external/filtered_embeddings.f32")
DEFAULT_EMBEDDING_META = Path("data/external/filtered_embeddings.meta.json")
DEFAULT_CATALOG = WORKFLOW_ROOT / "catalog" / "six_topic_discourse_catalog.csv"
SOURCE_ORDER = ["academic", "media", "corporate"]
SPACY_MODEL_NAME = "en_core_web_sm"


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def aspect_terms_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        terms: list[str] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                term = str(item[0]).strip()
                if term:
                    terms.append(term)
            elif isinstance(item, str) and item.strip():
                terms.append(item.strip())
        return terms
    if isinstance(value, tuple) and len(value) >= 1:
        term = str(value[0]).strip()
        return [term] if term else []
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def parse_topic_terms(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text == "nan":
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [part.strip().strip("'").strip('"') for part in text.split(",") if part.strip()]


def plot_heatmap(data: pd.DataFrame, title: str, cbar_label: str, output_base: Path, footnote: str, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    matrix = data.fillna(0.0).to_numpy()
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([str(x) for x in data.columns])
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([str(y) for y in data.index])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.text(0.01, 0.01, footnote, ha="left", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_base.with_suffix(".png"), dpi=200)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--embedding-file", type=Path, default=DEFAULT_EMBEDDING_FILE)
    parser.add_argument("--embedding-meta", type=Path, default=DEFAULT_EMBEDDING_META)
    parser.add_argument("--representation-embedding-model", type=str, default=None)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--subgroups", nargs="*", default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def subgroup_min_topic_size(n_rows: int) -> int:
    return max(5, min(50, int(round(n_rows * 0.02))))


def build_representation_model() -> dict[str, Any]:
    main_representation = KeyBERTInspired()
    aspect_model1 = PartOfSpeech(SPACY_MODEL_NAME)
    aspect_model2 = [KeyBERTInspired(top_n_words=30), MaximalMarginalRelevance(diversity=0.5)]
    return {
        "Main": main_representation,
        "POS": aspect_model1,
        "Aspect2": aspect_model2,
    }


def build_model(n_rows: int, embedding_model_backend: Any) -> BERTopic:
    vectorizer = CountVectorizer(stop_words="english", min_df=1, max_df=0.95)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    return BERTopic(
        embedding_model=embedding_model_backend,
        representation_model=build_representation_model(),
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf_model,
        min_topic_size=subgroup_min_topic_size(n_rows),
        top_n_words=10,
        calculate_probabilities=False,
        verbose=True,
    )


def representative_docs_from_assignments(frame: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    return (
        frame.groupby("micro_topic_id", dropna=False)
        .head(top_n)[["micro_topic_id", "chunk_id", "source_doc_id", "year", "text"]]
        .copy()
    )


def build_topic_aspects_frame(topic_info: pd.DataFrame, topic_aspects: dict[str, Any]) -> pd.DataFrame:
    valid = topic_info.loc[topic_info["Topic"] != -1].copy()
    rows: list[dict[str, Any]] = []
    for row in valid.itertuples(index=False):
        topic_id = int(row.Topic)
        main_terms = parse_topic_terms(getattr(row, "Representation", ""))
        record: dict[str, Any] = {
            "micro_topic_id": topic_id,
            "topic_name_original": str(row.Name),
            "topic_size": int(row.Count),
            "main_terms_display": " | ".join(main_terms),
            "pos_terms_display": "",
            "aspect2_terms_display": "",
            "main_terms_json": json.dumps(main_terms, ensure_ascii=False),
            "pos_terms_json": "[]",
            "aspect2_terms_json": "[]",
        }
        for aspect_name, column_prefix in [("POS", "pos"), ("Aspect2", "aspect2")]:
            raw_value = topic_aspects.get(aspect_name, {}).get(topic_id, [])
            terms = aspect_terms_from_value(raw_value)
            record[f"{column_prefix}_terms_display"] = " | ".join(terms)
            record[f"{column_prefix}_terms_json"] = json.dumps(json_ready(raw_value), ensure_ascii=False)
        rows.append(record)
    return pd.DataFrame(rows)


def write_descriptions(summary_dir: Path) -> None:
    technical = """# BERTopic Micro Figure Descriptions

## Figure 1: Number of discovered BERTopic topics per source-topic subgroup
This heatmap shows the number of non-outlier BERTopic topics discovered in each source-topic subgroup.

## Figure 2: Subgroup size by source-topic
This heatmap shows the number of validated chunk-level rows in each subgroup used to fit the unsupervised BERTopic models.

## Figure 3: Selected subgroup topic evolution
This panel figure shows the annual counts of the largest discovered micro-topics in selected high-volume subgroups.
"""
    accessible = """# BERTopic Micro Figure Descriptions in Accessible English

## Figure 1: Number of discovered topics per subgroup
This heatmap shows how much internal variety BERTopic found inside each source-topic subgroup.

## Figure 2: Subgroup size
This heatmap shows how many validated text chunks were available for each subgroup model.

## Figure 3: Selected subgroup topic evolution
This figure shows how the main micro-topics rise and fall over time inside some of the larger subgroup models.
"""
    (summary_dir / "figure_descriptions.md").write_text(technical, encoding="utf-8")
    (summary_dir / "figure_descriptions_accessible_english.md").write_text(accessible, encoding="utf-8")


def collect_existing_manifests(output_root: Path) -> pd.DataFrame:
    rows = []
    for manifest_path in sorted(output_root.glob("*/manifest.json")):
        if manifest_path.parent.name == "summary":
            continue
        rows.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "source" in frame.columns:
        frame["source"] = pd.Categorical(frame["source"], categories=SOURCE_ORDER, ordered=True)
        frame = frame.sort_values(["source", "assigned_label"]).reset_index(drop=True)
    return frame


def build_summary_outputs(output_root: Path) -> None:
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = summary_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifests = collect_existing_manifests(output_root)
    if manifests.empty:
        return

    top_keywords_paths = sorted(path for path in output_root.glob("*/*topic_info.csv"))
    top_keywords_rows = []
    topic_aspect_frames = []
    representative_paths = sorted(path for path in output_root.glob("*/*representative_docs.csv"))
    representative_frames = []
    for path in top_keywords_paths:
        subgroup_name = path.parent.name
        info = pd.read_csv(path)
        valid = info.loc[info["Topic"] != -1].copy()
        manifest_path = output_root / subgroup_name / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        for _, row in valid.head(5).iterrows():
            top_keywords_rows.append(
                {
                    "subgroup": subgroup_name,
                    "source": manifest.get("source"),
                    "assigned_label": manifest.get("assigned_label"),
                    "micro_topic_id": int(row["Topic"]),
                    "topic_name": row["Name"],
                    "row_count": int(row["Count"]),
                }
            )
        topic_aspects_csv = path.parent / "topic_aspects.csv"
        if topic_aspects_csv.exists():
            aspect_frame = pd.read_csv(topic_aspects_csv)
            aspect_frame["subgroup"] = subgroup_name
            aspect_frame["source"] = manifest.get("source")
            aspect_frame["assigned_label"] = manifest.get("assigned_label")
            topic_aspect_frames.append(aspect_frame)
    for path in representative_paths:
        subgroup_name = path.parent.name
        rep = pd.read_csv(path)
        manifest_path = output_root / subgroup_name / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        rep["subgroup"] = subgroup_name
        rep["source"] = manifest.get("source")
        rep["assigned_label"] = manifest.get("assigned_label")
        representative_frames.append(rep)

    manifests.to_csv(summary_dir / "subgroup_manifest_summary.csv", index=False)
    pd.DataFrame(top_keywords_rows).to_csv(summary_dir / "top_keywords_summary.csv", index=False)
    if topic_aspect_frames:
        pd.concat(topic_aspect_frames, ignore_index=True).to_csv(summary_dir / "topic_aspect_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(summary_dir / "topic_aspect_summary.csv", index=False)
    if representative_frames:
        pd.concat(representative_frames, ignore_index=True).to_csv(summary_dir / "representative_docs_index.csv", index=False)
    else:
        pd.DataFrame().to_csv(summary_dir / "representative_docs_index.csv", index=False)

    topic_count_heatmap = manifests.pivot(index="source", columns="assigned_label", values="n_topics_found").reindex(SOURCE_ORDER)
    plot_heatmap(
        topic_count_heatmap,
        "Discovered BERTopic topics per source-topic subgroup",
        "BERTopic topics",
        figures_dir / "micro_topic_count_heatmap",
        "Base calculation: each cell shows the number of non-outlier BERTopic topics found in one source-topic subgroup.",
        "Blues",
    )

    size_heatmap = manifests.pivot(index="source", columns="assigned_label", values="row_count").reindex(SOURCE_ORDER)
    plot_heatmap(
        size_heatmap,
        "Subgroup size per source-topic model",
        "Validated chunks",
        figures_dir / "micro_subgroup_size_heatmap",
        "Base calculation: each cell shows the number of validated chunk-level rows used in the subgroup model.",
        "YlOrBr",
    )

    evolution_candidates = [(row["subgroup"], int(row["row_count"])) for _, row in manifests.loc[manifests["status"] == "completed"].iterrows()]
    selected = [name for name, _ in sorted(evolution_candidates, key=lambda item: item[1], reverse=True)[:6]]
    if selected:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=False, sharey=False)
        for ax, subgroup_name in zip(axes.flatten(), selected):
            subgroup_dir = output_root / subgroup_name
            subset = pd.read_parquet(subgroup_dir / "document_topics.parquet")
            counts = (
                subset.loc[subset["micro_topic_id"] != -1]
                .groupby(["year", "micro_topic_id"], dropna=False)
                .size()
                .reset_index(name="row_count")
            )
            top_topics = counts.groupby("micro_topic_id", dropna=False)["row_count"].sum().sort_values(ascending=False).head(4).index.tolist()
            for topic_id in top_topics:
                topic_subset = counts.loc[counts["micro_topic_id"] == topic_id]
                ax.plot(topic_subset["year"], topic_subset["row_count"], label=f"{topic_id}", linewidth=1.5)
            ax.set_title(subgroup_name)
            ax.set_xlabel("Year")
            ax.set_ylabel("Rows")
        handles, labels = axes.flatten()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, title="Micro topic")
        fig.suptitle("Selected subgroup topic evolution", y=0.98)
        fig.text(
            0.01,
            0.01,
            "Base calculation: each line shows the annual number of validated chunk rows assigned to a discovered micro-topic within a selected subgroup.",
            ha="left",
            va="bottom",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        fig.savefig(figures_dir / "selected_subgroup_topics_over_time.png", dpi=200)
        fig.savefig(figures_dir / "selected_subgroup_topics_over_time.pdf")
        plt.close(fig)

    low_volume = manifests.loc[manifests["warning"].notna()].copy()
    low_volume.to_csv(summary_dir / "low_volume_exploratory_cells.csv", index=False)
    write_descriptions(summary_dir)

    summary_manifest = {
        "subgroup_count": int(len(manifests)),
        "completed_count": int((manifests["status"] == "completed").sum()),
        "skipped_empty_count": int((manifests["status"] == "skipped_empty").sum()) if "status" in manifests.columns else 0,
        "low_volume_count": int(low_volume.shape[0]),
        "outputs": {
            "subgroup_manifest_summary": str(summary_dir / "subgroup_manifest_summary.csv"),
            "top_keywords_summary": str(summary_dir / "top_keywords_summary.csv"),
            "topic_aspect_summary": str(summary_dir / "topic_aspect_summary.csv"),
            "representative_docs_index": str(summary_dir / "representative_docs_index.csv"),
        },
    }
    (summary_dir / "manifest.json").write_text(json.dumps(summary_manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_dir = args.output_root / "summary"
    if args.input.suffix.lower() == ".parquet":
        frame = pd.read_parquet(args.input).copy()
    else:
        frame = pd.read_csv(args.input, low_memory=False).copy()
    catalog = pd.read_csv(args.catalog)[["topic_code", "topic_name"]]
    meta = read_json(args.embedding_meta)
    embeddings = np.memmap(
        args.embedding_file,
        dtype="float32",
        mode="r",
        shape=(int(meta["rows"]), int(meta["dims"])),
    )
    representation_embedding_model_name = args.representation_embedding_model or str(meta.get("model_name", "")).strip()
    if not representation_embedding_model_name:
        raise ValueError(
            "Could not determine a sentence-transformer model for KeyBERTInspired/MMR representations. "
            "Pass --representation-embedding-model explicitly."
        )
    representation_embedding_backend = load_sentence_transformer(representation_embedding_model_name)

    requested_subgroups = set(args.subgroups or [])
    subgroup_manifests: list[dict] = []

    for source_value in SOURCE_ORDER:
        for topic_code in catalog["topic_code"].tolist():
            subset = frame.loc[(frame["source"] == source_value) & (frame["assigned_label"] == topic_code)].copy()
            subgroup_name = f"{source_value}_{topic_code}"
            if requested_subgroups and subgroup_name not in requested_subgroups:
                continue
            subgroup_dir = args.output_root / subgroup_name
            subgroup_dir.mkdir(parents=True, exist_ok=True)
            warning = None

            if subset.empty:
                manifest = {
                    "subgroup": subgroup_name,
                    "source": source_value,
                    "assigned_label": topic_code,
                    "status": "skipped_empty",
                    "row_count": 0,
                }
                (subgroup_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                subgroup_manifests.append(manifest)
                continue

            if len(subset) < 500:
                warning = "low_volume_exploratory"

            subset["embedding_row_index"] = pd.to_numeric(subset["embedding_row_index"], errors="coerce")
            missing_embedding_rows = int(subset["embedding_row_index"].isna().sum())
            if missing_embedding_rows:
                subset = subset.dropna(subset=["embedding_row_index"]).copy()
                warning = "low_volume_exploratory" if warning is None else f"{warning};missing_embedding_rows"

            if subset.empty:
                manifest = {
                    "subgroup": subgroup_name,
                    "source": source_value,
                    "assigned_label": topic_code,
                    "status": "skipped_missing_embeddings",
                    "row_count": 0,
                    "missing_embedding_rows": missing_embedding_rows,
                }
                (subgroup_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                subgroup_manifests.append(manifest)
                continue

            subgroup_embeddings = np.asarray(
                embeddings[subset["embedding_row_index"].astype(int).to_numpy()],
                dtype="float32",
            )
            model = build_model(len(subset), embedding_model_backend=representation_embedding_backend)
            topics, probabilities = model.fit_transform(subset["text"].fillna("").astype(str).tolist(), embeddings=subgroup_embeddings)

            subset["micro_topic_id"] = pd.Series(topics, index=subset.index, dtype="Int64")
            subset["micro_topic_probability"] = pd.NA if probabilities is None else probabilities
            subset.to_parquet(subgroup_dir / "document_topics.parquet", index=False)
            subset.to_csv(subgroup_dir / "document_topics.csv", index=False)

            topic_info = model.get_topic_info()
            topic_info.to_csv(subgroup_dir / "topic_info.csv", index=False)
            valid_topic_info = topic_info.loc[topic_info["Topic"] != -1].copy()
            topic_aspects = {str(key): value for key, value in getattr(model, "topic_aspects_", {}).items()}
            (subgroup_dir / "topic_aspects.json").write_text(
                json.dumps(json_ready(topic_aspects), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            topic_aspects_frame = build_topic_aspects_frame(topic_info=topic_info, topic_aspects=topic_aspects)
            topic_aspects_frame.to_csv(subgroup_dir / "topic_aspects.csv", index=False)

            representative = representative_docs_from_assignments(subset.sort_values(["micro_topic_id", "year"]), top_n=3)
            representative.to_csv(subgroup_dir / "representative_docs.csv", index=False)

            topics_over_time = model.topics_over_time(
                docs=subset["text"].fillna("").astype(str).tolist(),
                timestamps=subset["year"].astype(int).tolist(),
                topics=subset["micro_topic_id"].astype(int).tolist(),
            )
            topics_over_time.to_csv(subgroup_dir / "topics_over_time.csv", index=False)

            try:
                hierarchical = model.hierarchical_topics(subset["text"].fillna("").astype(str).tolist())
                hierarchical.to_csv(subgroup_dir / "hierarchical_topics.csv", index=False)
            except Exception:
                pass

            outlier_share = float((subset["micro_topic_id"] == -1).mean())
            largest_topic_size = int(valid_topic_info["Count"].max()) if not valid_topic_info.empty else 0
            n_topics_found = int(valid_topic_info["Topic"].nunique())
            manifest = {
                "subgroup": subgroup_name,
                "source": source_value,
                "assigned_label": topic_code,
                "status": "completed",
                "row_count": int(len(subset)),
                "unique_docs": int(subset["source_doc_id"].nunique()),
                "year_min": int(pd.to_numeric(subset["year"]).min()),
                "year_max": int(pd.to_numeric(subset["year"]).max()),
                "year_count": int(pd.to_numeric(subset["year"]).nunique()),
                "min_topic_size": subgroup_min_topic_size(len(subset)),
                "n_topics_found": n_topics_found,
                "largest_topic_size": largest_topic_size,
                "outlier_share": outlier_share,
                "warning": warning,
                "missing_embedding_rows": missing_embedding_rows,
                "representation_mode": "multi_aspect",
                "representation_model_name": "KeyBERTInspired + PartOfSpeech + MMR",
                "spacy_model": SPACY_MODEL_NAME,
                "representation_embedding_model": representation_embedding_model_name,
                "topic_aspects_json": str(subgroup_dir / "topic_aspects.json"),
                "topic_aspects_csv": str(subgroup_dir / "topic_aspects.csv"),
            }
            (subgroup_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            subgroup_manifests.append(manifest)
    build_summary_outputs(args.output_root)
    print(summary_dir)


if __name__ == "__main__":
    main()
