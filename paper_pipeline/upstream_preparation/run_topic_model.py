#!/usr/bin/env python3
"""Train BERTopic on the prepared corpus with persisted batch embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from bertopic import BERTopic
from hdbscan import HDBSCAN
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


LOGGER = logging.getLogger("run_topic_model")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "full_corpus_chunked.parquet"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_EMBEDDINGS_PATH = DEFAULT_RESULTS_DIR / "bge_large_en_v1_5_embeddings.f32"
DEFAULT_EMBEDDINGS_META = DEFAULT_RESULTS_DIR / "bge_large_en_v1_5_embeddings.meta.json"
DEFAULT_CORPUS_WITH_TOPICS = DEFAULT_RESULTS_DIR / "corpus_with_topics.parquet"
DEFAULT_TOPIC_INFO = DEFAULT_RESULTS_DIR / "topic_info.parquet"
DEFAULT_TOPIC_DOCS = DEFAULT_RESULTS_DIR / "topic_top_documents.parquet"
DEFAULT_REP_DOCS = DEFAULT_RESULTS_DIR / "topic_representative_docs.parquet"
DEFAULT_MODEL_DIR = DEFAULT_RESULTS_DIR / "bertopic_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--reuse-embeddings", action="store_true")
    parser.add_argument("--stop-after-embeddings", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--embedding-file", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--embedding-meta", type=Path, default=DEFAULT_EMBEDDINGS_META)
    parser.add_argument("--corpus-output", type=Path, default=DEFAULT_CORPUS_WITH_TOPICS)
    parser.add_argument("--topic-info-output", type=Path, default=DEFAULT_TOPIC_INFO)
    parser.add_argument("--topic-docs-output", type=Path, default=DEFAULT_TOPIC_DOCS)
    parser.add_argument("--representative-docs-output", type=Path, default=DEFAULT_REP_DOCS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--min-cluster-size", type=int, default=500)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--vectorizer-min-df", type=int, default=1)
    parser.add_argument("--vectorizer-max-df", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def configure_runtime_threads() -> None:
    omp_threads = os.environ.get("OMP_NUM_THREADS")
    torch_threads = os.environ.get("TOPICMODEL_TORCH_THREADS") or omp_threads
    interop_threads = os.environ.get("TOPICMODEL_TORCH_INTEROP_THREADS")

    if torch_threads:
        try:
            torch.set_num_threads(int(torch_threads))
        except (TypeError, ValueError) as exc:
            LOGGER.warning("Invalid torch thread setting %r: %s", torch_threads, exc)
    if interop_threads:
        try:
            torch.set_num_interop_threads(int(interop_threads))
        except (TypeError, ValueError, RuntimeError) as exc:
            LOGGER.warning("Invalid torch interop thread setting %r: %s", interop_threads, exc)

    LOGGER.info(
        "Runtime thread config: OMP_NUM_THREADS=%s MKL_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s "
        "NUMEXPR_NUM_THREADS=%s torch_num_threads=%s torch_num_interop_threads=%s",
        os.environ.get("OMP_NUM_THREADS"),
        os.environ.get("MKL_NUM_THREADS"),
        os.environ.get("OPENBLAS_NUM_THREADS"),
        os.environ.get("NUMEXPR_NUM_THREADS"),
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    )


def load_corpus(input_path: Path, sample_size: int | None) -> pd.DataFrame:
    LOGGER.info("Reading corpus from %s", input_path)
    table = pq.read_table(input_path)
    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    df = df.convert_dtypes(dtype_backend="numpy_nullable")
    if sample_size is not None:
        df = df.head(sample_size).copy()
    df["text"] = df["text"].fillna("").astype(str)
    return df


def embedding_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_local_snapshot(model_name: str) -> Path | None:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return candidate

    sanitized = model_name.replace("/", "--")
    hub_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{sanitized}" / "snapshots"
    if not hub_root.exists():
        return None

    snapshots = sorted(path for path in hub_root.iterdir() if path.is_dir())
    return snapshots[-1] if snapshots else None


def load_embedding_model(model_name: str) -> SentenceTransformer:
    device = embedding_device()
    local_snapshot = resolve_local_snapshot(model_name)
    load_target = str(local_snapshot) if local_snapshot is not None else model_name
    local_files_only = local_snapshot is not None
    LOGGER.info(
        "Loading embedding model %s on device=%s using target=%s local_files_only=%s",
        model_name,
        device,
        load_target,
        local_files_only,
    )
    model = SentenceTransformer(load_target, device=device, local_files_only=local_files_only)
    return model


def load_embeddings_from_disk(
    embedding_file: Path,
    embedding_meta: Path,
    expected_rows: int,
) -> np.memmap | None:
    if not embedding_file.exists() or not embedding_meta.exists():
        return None

    meta = json.loads(embedding_meta.read_text(encoding="utf-8"))
    if meta.get("rows") != expected_rows:
        LOGGER.warning(
            "Embedding cache row mismatch: expected %s rows but found %s. Recomputing embeddings.",
            expected_rows,
            meta.get("rows"),
        )
        return None

    dims = int(meta["dims"])
    return np.memmap(embedding_file, dtype="float32", mode="r+", shape=(expected_rows, dims))


def write_embedding_meta(embedding_meta: Path, rows: int, dims: int, model_name: str) -> None:
    embedding_meta.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": rows,
        "dims": dims,
        "model_name": model_name,
        "dtype": "float32",
        "created_at_unix": time.time(),
    }
    embedding_meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_embeddings(
    df: pd.DataFrame,
    model_name: str,
    embedding_file: Path,
    embedding_meta: Path,
    batch_size: int,
    reuse_embeddings: bool,
) -> np.ndarray:
    if reuse_embeddings:
        cached = load_embeddings_from_disk(embedding_file, embedding_meta, len(df))
        if cached is not None:
            LOGGER.info("Reusing cached embeddings from %s", embedding_file)
            return np.asarray(cached)

    model = load_embedding_model(model_name)
    sample_embedding = model.encode(
        [df.iloc[0]["text"]],
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    dims = int(sample_embedding.shape[1])

    embedding_file.parent.mkdir(parents=True, exist_ok=True)
    if embedding_file.exists():
        embedding_file.unlink()
    embeddings = np.memmap(embedding_file, dtype="float32", mode="w+", shape=(len(df), dims))
    embeddings[0:1] = sample_embedding.astype("float32")

    processed = 1
    for start in range(1, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch_texts = df.iloc[start:end]["text"].tolist()
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        embeddings[start:end] = batch_embeddings
        processed = end
        if processed % max(batch_size * 100, 10_000) == 0 or processed == len(df):
            LOGGER.info("Embeddings computed for %s / %s documents", processed, len(df))

    embeddings.flush()
    write_embedding_meta(embedding_meta, rows=len(df), dims=dims, model_name=model_name)
    return np.asarray(embeddings)


def build_topic_model(args: argparse.Namespace) -> BERTopic:
    umap_model = UMAP(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        metric="cosine",
        low_memory=True,
        random_state=args.seed,
        verbose=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=args.vectorizer_min_df,
        max_df=args.vectorizer_max_df,
    )
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True,
        low_memory=True,
        min_topic_size=args.min_cluster_size,
    )
    return topic_model


def append_topics(df: pd.DataFrame, topics, probabilities) -> pd.DataFrame:
    result = df.copy()
    result["topic"] = pd.Series(topics, dtype="Int64")
    if probabilities is None:
        result["probability"] = pd.Series([pd.NA] * len(result), dtype="Float64")
    else:
        result["probability"] = pd.Series(np.asarray(probabilities), dtype="float64")
    return result


def build_topic_info(topic_model: BERTopic, result_df: pd.DataFrame) -> pd.DataFrame:
    topic_info = topic_model.get_topic_info()
    topic_sizes = result_df.groupby("topic").size().rename("observed_size").reset_index()
    topic_info = topic_info.merge(topic_sizes, how="left", left_on="Topic", right_on="topic")
    topic_info = topic_info.drop(columns=["topic"], errors="ignore")
    topic_info["observed_size"] = topic_info["observed_size"].fillna(0).astype(int)
    return topic_info


def build_topic_top_documents(result_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    ordered = result_df.sort_values(
        by=["topic", "probability", "doc_id", "chunk_id"],
        ascending=[True, False, True, True],
        na_position="last",
    ).copy()
    ordered["rank_within_topic"] = ordered.groupby("topic").cumcount() + 1
    return ordered.loc[ordered["rank_within_topic"] <= top_n].copy()


def build_representative_docs(
    topic_model: BERTopic,
    top_documents: pd.DataFrame,
) -> pd.DataFrame:
    representative = topic_model.get_representative_docs()
    rows: list[dict[str, object]] = []

    for topic_id, docs in representative.items():
        docs = docs or []
        for rank, text in enumerate(docs, start=1):
            matches = top_documents[(top_documents["topic"] == topic_id) & (top_documents["text"] == text)]
            if matches.empty:
                rows.append(
                    {
                        "topic": topic_id,
                        "representative_rank": rank,
                        "doc_id": None,
                        "chunk_id": None,
                        "probability": None,
                        "text": text,
                    }
                )
            else:
                row = matches.iloc[0]
                rows.append(
                    {
                        "topic": int(topic_id),
                        "representative_rank": rank,
                        "doc_id": row["doc_id"],
                        "chunk_id": row["chunk_id"],
                        "probability": row["probability"],
                        "text": row["text"],
                    }
                )

    return pd.DataFrame(rows)


def save_outputs(
    result_df: pd.DataFrame,
    topic_info: pd.DataFrame,
    topic_top_documents: pd.DataFrame,
    representative_docs: pd.DataFrame,
    args: argparse.Namespace,
    topic_model: BERTopic,
) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Writing %s", args.corpus_output)
    result_df.to_parquet(args.corpus_output, index=False)

    LOGGER.info("Writing %s", args.topic_info_output)
    topic_info.to_parquet(args.topic_info_output, index=False)

    LOGGER.info("Writing %s", args.topic_docs_output)
    topic_top_documents.to_parquet(args.topic_docs_output, index=False)

    LOGGER.info("Writing %s", args.representative_docs_output)
    representative_docs.to_parquet(args.representative_docs_output, index=False)

    if args.save_model:
        LOGGER.info("Saving BERTopic model to %s", args.model_dir)
        topic_model.save(args.model_dir, serialization="safetensors", save_ctfidf=True)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    configure_runtime_threads()

    input_path = args.input.resolve()
    args.results_dir = args.results_dir.resolve()
    args.embedding_file = args.embedding_file.resolve()
    args.embedding_meta = args.embedding_meta.resolve()
    args.corpus_output = args.corpus_output.resolve()
    args.topic_info_output = args.topic_info_output.resolve()
    args.topic_docs_output = args.topic_docs_output.resolve()
    args.representative_docs_output = args.representative_docs_output.resolve()
    args.model_dir = args.model_dir.resolve()

    df = load_corpus(input_path=input_path, sample_size=args.sample_size)
    LOGGER.info("Loaded %s documents for topic modeling", len(df))
    embeddings = compute_embeddings(
        df=df,
        model_name=args.embedding_model,
        embedding_file=args.embedding_file,
        embedding_meta=args.embedding_meta,
        batch_size=args.batch_size,
        reuse_embeddings=args.reuse_embeddings,
    )

    if args.stop_after_embeddings:
        LOGGER.info(
            "Embeddings were computed and saved to %s. Stopping before BERTopic as requested.",
            args.embedding_file,
        )
        return 0

    topic_model = build_topic_model(args)
    LOGGER.info("Fitting BERTopic")
    topics, probabilities = topic_model.fit_transform(df["text"].tolist(), embeddings)
    result_df = append_topics(df, topics, probabilities)

    topic_info = build_topic_info(topic_model, result_df)
    topic_top_documents = build_topic_top_documents(result_df, top_n=50)
    representative_docs = build_representative_docs(topic_model, topic_top_documents)

    save_outputs(
        result_df=result_df,
        topic_info=topic_info,
        topic_top_documents=topic_top_documents,
        representative_docs=representative_docs,
        args=args,
        topic_model=topic_model,
    )
    LOGGER.info("Topic modeling completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
