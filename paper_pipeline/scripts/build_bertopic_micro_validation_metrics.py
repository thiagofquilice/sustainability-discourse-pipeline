#!/usr/bin/env python3
"""Build diagnostic validation metrics for the 18 reviewed micro BERTopic models."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score

from workflow_common import load_embedding_memmap, read_json, write_json


WORKFLOW_ROOT = Path("paper_pipeline")
DEFAULT_MODEL_ROOT = WORKFLOW_ROOT / "outputs" / "bertopic_micro_merged_multiaspect_reviewed"
DEFAULT_VALIDATION_ROOT = DEFAULT_MODEL_ROOT / "validation"
DEFAULT_EMBEDDING_FILE = Path("data/external/filtered_embeddings.f32")
DEFAULT_EMBEDDING_META = Path("data/external/filtered_embeddings.meta.json")
SOURCE_ORDER = {"academic": 0, "media": 1, "corporate": 2}
TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")
REFERENCES = [
    {
        "citation": "Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure.",
        "url": "https://arxiv.org/abs/2203.05794",
    },
    {
        "citation": "BERTopic documentation. Topics over Time.",
        "url": "https://maartengr.github.io/BERTopic/getting_started/topicsovertime/topicsovertime.html",
    },
    {
        "citation": "Roder, M., Both, A., and Hinneburg, A. (2015). Exploring the space of topic coherence measures.",
        "url": "https://doi.org/10.1145/2684822.2685324",
    },
    {
        "citation": "Dieng, A. B., Ruiz, F. J. R., and Blei, D. M. (2020). Topic Modeling in Embedding Spaces.",
        "url": "https://aclanthology.org/2020.tacl-1.29/",
    },
    {
        "citation": "Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.",
        "url": "https://doi.org/10.1016/0377-0427(87)90125-7",
    },
    {
        "citation": "Chang, J., Gerrish, S., Wang, C., Boyd-Graber, J., and Blei, D. (2009). Reading Tea Leaves.",
        "url": "https://papers.nips.cc/paper_files/paper/2009/hash/f92586a25bb3145facd64ab20fd554ff-Abstract.html",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--embedding-file", type=Path, default=DEFAULT_EMBEDDING_FILE)
    parser.add_argument("--embedding-meta", type=Path, default=DEFAULT_EMBEDDING_META)
    parser.add_argument("--top-n-words", type=int, default=10)
    parser.add_argument("--max-coherence-docs", type=int, default=20_000)
    parser.add_argument("--max-silhouette-docs", type=int, default=5_000)
    parser.add_argument("--manual-sample-target", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_terms(value: Any, top_n: int | None = None) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        terms = [str(item).strip() for item in value if str(item).strip()]
        return terms[:top_n] if top_n else terms
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    parsed: Any | None = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            break
        except Exception:
            parsed = None
    if isinstance(parsed, list):
        terms = []
        for item in parsed:
            if isinstance(item, (list, tuple)) and item:
                term = str(item[0]).strip()
            else:
                term = str(item).strip()
            if term:
                terms.append(term)
        return terms[:top_n] if top_n else terms
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    terms = [part.strip().strip("'").strip('"') for part in text.split(",") if part.strip()]
    return terms[:top_n] if top_n else terms


def tokenize(text: Any) -> list[str]:
    tokens = TOKEN_RE.findall(str(text).lower())
    return [token for token in tokens if token not in ENGLISH_STOP_WORDS]


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def gini(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    sorted_values = np.sort(values.astype(float))
    if sorted_values.sum() == 0:
        return 0.0
    n = sorted_values.size
    numerator = np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_values)
    return float(numerator / (n * sorted_values.sum()))


def jaccard(left: set[str], right: set[str]) -> float | None:
    if not left and not right:
        return None
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def sort_subgroups(paths: list[Path]) -> list[Path]:
    def key(path: Path) -> tuple[int, str]:
        source = path.name.split("_", 1)[0]
        return (SOURCE_ORDER.get(source, 99), path.name)

    return sorted(paths, key=key)


def discover_subgroups(model_root: Path) -> list[Path]:
    paths = []
    for path in model_root.iterdir():
        if not path.is_dir() or path.name in {"summary", "validation"}:
            continue
        required = ["document_topics.parquet", "topic_info.csv", "representative_docs.csv", "topics_over_time.csv"]
        if all((path / name).exists() for name in required):
            paths.append(path)
    return sort_subgroups(paths)


def topic_column(frame: pd.DataFrame) -> str:
    for candidate in ["micro_topic_id", "Topic", "merged_micro_topic_id"]:
        if candidate in frame.columns:
            return candidate
    raise ValueError("Could not find a topic id column in document topics.")


def sample_frame(frame: pd.DataFrame, max_rows: int, seed: int, stratify_col: str | None = None) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    if stratify_col is None or stratify_col not in frame.columns:
        return frame.sample(n=max_rows, random_state=seed).copy()
    pieces = []
    counts = frame[stratify_col].value_counts()
    total = float(len(frame))
    allocations = {}
    for label, count in counts.items():
        allocations[label] = max(1, int(round(max_rows * (count / total))))
    while sum(allocations.values()) > max_rows:
        label = max(allocations, key=allocations.get)
        if allocations[label] > 1:
            allocations[label] -= 1
        else:
            break
    while sum(allocations.values()) < max_rows:
        label = counts.idxmax()
        allocations[label] += 1
    for label, n_rows in allocations.items():
        group = frame.loc[frame[stratify_col] == label]
        pieces.append(group.sample(n=min(n_rows, len(group)), random_state=seed))
    sampled = pd.concat(pieces, ignore_index=False)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=seed)
    return sampled.copy()


def compute_coherence(
    texts: pd.Series,
    topic_terms: dict[int, list[str]],
    max_docs: int,
    seed: int,
) -> tuple[dict[int, dict[str, float | None]], dict[str, Any]]:
    result = {topic_id: {"coherence_cv": None, "coherence_npmi": None} for topic_id in topic_terms}
    status: dict[str, Any] = {"status": "not_run", "sampled_docs": 0, "reason": ""}
    sampled = texts.dropna().astype(str)
    if len(sampled) > max_docs:
        sampled = sampled.sample(n=max_docs, random_state=seed)
    tokenized = [tokens for tokens in sampled.map(tokenize).tolist() if tokens]
    if not tokenized:
        status.update({"status": "skipped", "reason": "no tokenized texts"})
        return result, status
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
    except Exception as exc:
        fallback, fallback_status = compute_document_coherence_fallback(tokenized, topic_terms)
        status.update(
            {
                "status": "fallback_document_cooccurrence",
                "sampled_docs": len(tokenized),
                "reason": f"gensim import failed: {exc}",
                **fallback_status,
            }
        )
        return fallback, status

    if not tokenized:
        status.update({"status": "skipped", "reason": "no tokenized texts"})
        return result, status

    ordered_topic_ids = [topic_id for topic_id, terms in topic_terms.items() if terms]
    ordered_terms = [topic_terms[topic_id] for topic_id in ordered_topic_ids]
    dictionary = Dictionary(tokenized)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
    status.update(
        {
            "status": "ok",
            "sampled_docs": len(tokenized),
            "reason": "",
            "empty_topic_term_count": len(topic_terms) - len(ordered_topic_ids),
        }
    )
    if not ordered_topic_ids:
        status.update({"status": "skipped", "reason": "no topics with parsed top words"})
        return result, status

    for coherence_name, output_key in [("c_v", "coherence_cv"), ("c_npmi", "coherence_npmi")]:
        try:
            model = CoherenceModel(
                topics=ordered_terms,
                texts=tokenized,
                corpus=corpus,
                dictionary=dictionary,
                coherence=coherence_name,
            )
            values = model.get_coherence_per_topic()
            for topic_id, value in zip(ordered_topic_ids, values, strict=True):
                result[topic_id][output_key] = safe_float(value)
        except Exception as exc:
            status[f"{output_key}_reason"] = str(exc)
    return result, status


def compute_document_coherence_fallback(
    tokenized: list[list[str]],
    topic_terms: dict[int, list[str]],
) -> tuple[dict[int, dict[str, float | None]], dict[str, Any]]:
    """Approximate c_npmi and c_v from document-level word co-occurrence.

    This fallback is used only when Gensim is unavailable. It is intentionally
    deterministic and conservative: c_npmi is the mean pairwise NPMI of top
    words, and c_v is a document-level proxy based on cosine similarity among
    positive NPMI context vectors for each topic's top words.
    """
    normalized_terms_by_topic = {
        topic_id: [term.lower().strip() for term in terms if term.lower().strip()]
        for topic_id, terms in topic_terms.items()
    }
    vocabulary = sorted({term for terms in normalized_terms_by_topic.values() for term in terms})
    vocabulary_set = set(vocabulary)
    result = {
        topic_id: {"coherence_cv": None, "coherence_npmi": None}
        for topic_id in topic_terms
    }
    if not vocabulary_set:
        return result, {"fallback_reason": "no top-word vocabulary"}

    doc_count = len(tokenized)
    word_doc_freq: dict[str, int] = defaultdict(int)
    pair_doc_freq: dict[tuple[str, str], int] = defaultdict(int)
    for tokens in tokenized:
        present = sorted(set(tokens) & vocabulary_set)
        for term in present:
            word_doc_freq[term] += 1
        for left, right in combinations(present, 2):
            pair_doc_freq[(left, right)] += 1

    def pair_npmi(left: str, right: str) -> float:
        if left == right:
            return 1.0
        key = (left, right) if left < right else (right, left)
        pair_count = pair_doc_freq.get(key, 0)
        left_count = word_doc_freq.get(left, 0)
        right_count = word_doc_freq.get(right, 0)
        if pair_count == 0 or left_count == 0 or right_count == 0:
            return -1.0
        p_pair = pair_count / doc_count
        p_left = left_count / doc_count
        p_right = right_count / doc_count
        pmi = math.log(p_pair / (p_left * p_right))
        return float(pmi / (-math.log(p_pair)))

    for topic_id, terms in normalized_terms_by_topic.items():
        unique_terms = list(dict.fromkeys(terms))
        if len(unique_terms) < 2:
            continue
        npmi_values = [pair_npmi(left, right) for left, right in combinations(unique_terms, 2)]
        result[topic_id]["coherence_npmi"] = safe_float(np.mean(npmi_values)) if npmi_values else None

        vectors = []
        for term in unique_terms:
            vector = np.array([max(pair_npmi(term, other), 0.0) for other in unique_terms], dtype="float64")
            norm = np.linalg.norm(vector)
            if norm > 0:
                vectors.append(vector / norm)
        if vectors:
            matrix = np.vstack(vectors)
            aggregate = matrix.mean(axis=0)
            aggregate_norm = np.linalg.norm(aggregate)
            if aggregate_norm > 0:
                aggregate = aggregate / aggregate_norm
                result[topic_id]["coherence_cv"] = safe_float(np.mean(matrix @ aggregate))
    return result, {"fallback_reason": "used document-level co-occurrence fallback for c_npmi and c_v proxy"}


def representative_doc_map(path: Path) -> dict[int, list[dict[str, Any]]]:
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    topic_col = topic_column(frame)
    frame[topic_col] = pd.to_numeric(frame[topic_col], errors="coerce")
    frame = frame.dropna(subset=[topic_col]).copy()
    frame[topic_col] = frame[topic_col].astype(int)
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        topic_id = int(getattr(row, topic_col))
        result[topic_id].append(
            {
                "chunk_id": getattr(row, "chunk_id", ""),
                "source_doc_id": getattr(row, "source_doc_id", ""),
                "year": getattr(row, "year", ""),
                "text": str(getattr(row, "text", "")),
            }
        )
    return result


def process_temporal(
    subgroup: str,
    source: str,
    macro_topic: str,
    topics_over_time: pd.DataFrame,
    topic_terms: dict[int, list[str]],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    aggregates: dict[int, dict[str, Any]] = {}
    if topics_over_time.empty:
        return rows, aggregates

    frame = topics_over_time.copy()
    frame["Topic"] = pd.to_numeric(frame["Topic"], errors="coerce")
    frame["Timestamp"] = pd.to_numeric(frame["Timestamp"], errors="coerce")
    frame["Frequency"] = pd.to_numeric(frame["Frequency"], errors="coerce").fillna(0).astype(int)
    frame = frame.dropna(subset=["Topic", "Timestamp"]).copy()
    frame["Topic"] = frame["Topic"].astype(int)
    frame["Timestamp"] = frame["Timestamp"].astype(int)
    frame = frame.loc[frame["Topic"].isin(topic_terms)].copy()

    for topic_id, group in frame.sort_values(["Topic", "Timestamp"]).groupby("Topic"):
        global_terms = set(topic_terms.get(int(topic_id), []))
        previous_terms: set[str] | None = None
        previous_year: int | None = None
        word_jaccards = []
        global_jaccards = []
        frequencies = group["Frequency"].to_numpy(dtype=float)
        for row in group.itertuples(index=False):
            year_terms = set(parse_terms(getattr(row, "Words", ""), top_n=None))
            global_j = jaccard(year_terms, global_terms)
            previous_j = jaccard(previous_terms, year_terms) if previous_terms is not None else None
            if previous_j is not None:
                word_jaccards.append(previous_j)
            if global_j is not None:
                global_jaccards.append(global_j)
            rows.append(
                {
                    "subgroup": subgroup,
                    "source": source,
                    "macro_topic": macro_topic,
                    "topic_id": int(topic_id),
                    "year": int(getattr(row, "Timestamp")),
                    "frequency": int(getattr(row, "Frequency")),
                    "is_sparse_year": bool(int(getattr(row, "Frequency")) < 5),
                    "year_words": ", ".join(sorted(year_terms)),
                    "previous_observed_year": previous_year,
                    "previous_year_word_jaccard": previous_j,
                    "global_word_jaccard": global_j,
                }
            )
            previous_terms = year_terms
            previous_year = int(getattr(row, "Timestamp"))
        aggregates[int(topic_id)] = {
            "active_year_count": int(len(group)),
            "frequency_mean": safe_float(np.mean(frequencies)) if frequencies.size else None,
            "frequency_median": safe_float(np.median(frequencies)) if frequencies.size else None,
            "frequency_min": safe_float(np.min(frequencies)) if frequencies.size else None,
            "frequency_max": safe_float(np.max(frequencies)) if frequencies.size else None,
            "sparse_year_share": safe_float(np.mean(frequencies < 5)) if frequencies.size else None,
            "mean_consecutive_year_jaccard": safe_float(np.mean(word_jaccards)) if word_jaccards else None,
            "mean_global_year_word_jaccard": safe_float(np.mean(global_jaccards)) if global_jaccards else None,
        }
    return rows, aggregates


def compute_embedding_metrics(
    docs: pd.DataFrame,
    topics: list[int],
    topic_col: str,
    embeddings: np.memmap,
    max_silhouette_docs: int,
    seed: int,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metrics = {
        topic_id: {
            "embedding_rows": 0,
            "mean_own_centroid_cosine": None,
            "mean_intra_topic_cosine_distance": None,
            "nearest_centroid_topic_id": None,
            "nearest_centroid_cosine": None,
            "nearest_centroid_margin": None,
        }
        for topic_id in topics
    }
    pair_rows: list[dict[str, Any]] = []
    status: dict[str, Any] = {
        "silhouette_cosine": None,
        "silhouette_sampled_docs": 0,
        "silhouette_skip_reason": "",
        "embedding_skip_reason": "",
    }
    if "embedding_row_index" not in docs.columns:
        status["embedding_skip_reason"] = "document_topics has no embedding_row_index column"
        status["silhouette_skip_reason"] = status["embedding_skip_reason"]
        return metrics, pair_rows, status

    valid = docs.loc[docs[topic_col].isin(topics), [topic_col, "embedding_row_index"]].copy()
    valid["embedding_row_index"] = pd.to_numeric(valid["embedding_row_index"], errors="coerce")
    valid = valid.dropna(subset=["embedding_row_index"]).copy()
    valid["embedding_row_index"] = valid["embedding_row_index"].astype(int)
    valid = valid.loc[(valid["embedding_row_index"] >= 0) & (valid["embedding_row_index"] < embeddings.shape[0])]
    if valid.empty:
        status["embedding_skip_reason"] = "no valid embedding row indices"
        status["silhouette_skip_reason"] = status["embedding_skip_reason"]
        return metrics, pair_rows, status

    centroids = {}
    own_similarity_by_topic = {}
    for topic_id, group in valid.groupby(topic_col):
        indices = group["embedding_row_index"].to_numpy(dtype=int)
        matrix = normalize_rows(np.asarray(embeddings[indices], dtype="float32"))
        centroid = matrix.mean(axis=0, dtype="float64")
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        centroids[int(topic_id)] = centroid.astype("float32")
        own_sim = matrix @ centroids[int(topic_id)]
        own_similarity_by_topic[int(topic_id)] = own_sim
        metrics[int(topic_id)]["embedding_rows"] = int(len(indices))
        metrics[int(topic_id)]["mean_own_centroid_cosine"] = safe_float(np.mean(own_sim))
        if metrics[int(topic_id)]["mean_own_centroid_cosine"] is not None:
            metrics[int(topic_id)]["mean_intra_topic_cosine_distance"] = 1.0 - metrics[int(topic_id)]["mean_own_centroid_cosine"]

    centroid_topic_ids = sorted(centroids)
    if len(centroid_topic_ids) >= 2:
        centroid_matrix = np.vstack([centroids[topic_id] for topic_id in centroid_topic_ids])
        centroid_similarity = centroid_matrix @ centroid_matrix.T
        for i, j in combinations(range(len(centroid_topic_ids)), 2):
            topic_a = centroid_topic_ids[i]
            topic_b = centroid_topic_ids[j]
            pair_rows.append(
                {
                    "topic_id_a": topic_a,
                    "topic_id_b": topic_b,
                    "centroid_cosine": safe_float(centroid_similarity[i, j]),
                    "high_centroid_similarity_flag": bool(centroid_similarity[i, j] >= 0.90),
                }
            )
        for i, topic_id in enumerate(centroid_topic_ids):
            similarities = centroid_similarity[i].copy()
            similarities[i] = -np.inf
            nearest_index = int(np.argmax(similarities))
            nearest_topic_id = centroid_topic_ids[nearest_index]
            nearest_similarity = safe_float(similarities[nearest_index])
            own_mean = metrics[topic_id]["mean_own_centroid_cosine"]
            metrics[topic_id]["nearest_centroid_topic_id"] = nearest_topic_id
            metrics[topic_id]["nearest_centroid_cosine"] = nearest_similarity
            metrics[topic_id]["nearest_centroid_margin"] = (
                safe_float(own_mean - nearest_similarity)
                if own_mean is not None and nearest_similarity is not None
                else None
            )
    else:
        status["silhouette_skip_reason"] = "single non-outlier topic"

    if len(centroid_topic_ids) < 2:
        return metrics, pair_rows, status

    silhouette_frame = valid.loc[valid[topic_col].isin(centroid_topic_ids)].copy()
    topic_counts = silhouette_frame[topic_col].value_counts()
    keep_topics = topic_counts.loc[topic_counts >= 2].index.tolist()
    silhouette_frame = silhouette_frame.loc[silhouette_frame[topic_col].isin(keep_topics)]
    if silhouette_frame[topic_col].nunique() < 2:
        status["silhouette_skip_reason"] = "fewer than two topics with at least two documents"
        return metrics, pair_rows, status
    silhouette_sample = sample_frame(silhouette_frame, max_silhouette_docs, seed, stratify_col=topic_col)
    if silhouette_sample[topic_col].nunique() < 2:
        status["silhouette_skip_reason"] = "sample retained fewer than two topics"
        return metrics, pair_rows, status
    try:
        indices = silhouette_sample["embedding_row_index"].to_numpy(dtype=int)
        matrix = normalize_rows(np.asarray(embeddings[indices], dtype="float32"))
        labels = silhouette_sample[topic_col].astype(int).to_numpy()
        status["silhouette_cosine"] = safe_float(silhouette_score(matrix, labels, metric="cosine"))
        status["silhouette_sampled_docs"] = int(len(silhouette_sample))
    except Exception as exc:
        status["silhouette_skip_reason"] = str(exc)
    return metrics, pair_rows, status


def process_subgroup(
    path: Path,
    embeddings: np.memmap,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    subgroup = path.name
    source, macro_topic = subgroup.split("_", 1)
    docs = pd.read_parquet(path / "document_topics.parquet")
    topic_info = pd.read_csv(path / "topic_info.csv")
    topics_over_time = pd.read_csv(path / "topics_over_time.csv")
    reps = representative_doc_map(path / "representative_docs.csv")
    manifest = read_json(path / "manifest.json") if (path / "manifest.json").exists() else {}
    doc_topic_col = topic_column(docs)
    docs[doc_topic_col] = pd.to_numeric(docs[doc_topic_col], errors="coerce")

    valid_topic_info = topic_info.loc[pd.to_numeric(topic_info["Topic"], errors="coerce") != -1].copy()
    valid_topic_info["Topic"] = pd.to_numeric(valid_topic_info["Topic"], errors="raise").astype(int)
    topics = valid_topic_info["Topic"].astype(int).tolist()
    topic_terms = {
        int(row.Topic): parse_terms(row.Representation, top_n=args.top_n_words)
        for row in valid_topic_info.itertuples(index=False)
    }
    non_outlier_docs = docs.loc[docs[doc_topic_col].isin(topics)].copy()
    topic_counts = non_outlier_docs[doc_topic_col].value_counts().sort_index()
    sizes = topic_counts.to_numpy(dtype=float)
    total_rows = int(len(docs))
    non_outlier_rows = int(len(non_outlier_docs))
    outlier_rows = int(total_rows - non_outlier_rows)
    proportions = sizes / sizes.sum() if sizes.size and sizes.sum() else np.array([])
    entropy = safe_float(-np.sum(proportions * np.log(proportions))) if proportions.size else None
    all_terms = [term for terms in topic_terms.values() for term in terms]
    unique_terms = set(all_terms)
    topic_diversity = (len(unique_terms) / len(all_terms)) if all_terms else None

    coherence_metrics, coherence_status = compute_coherence(
        docs["text"] if "text" in docs.columns else pd.Series([], dtype=str),
        topic_terms,
        args.max_coherence_docs,
        args.seed,
    )
    embedding_metrics, embedding_pair_rows, embedding_status = compute_embedding_metrics(
        docs,
        topics,
        doc_topic_col,
        embeddings,
        args.max_silhouette_docs,
        args.seed,
    )
    temporal_rows, temporal_aggregates = process_temporal(
        subgroup,
        source,
        macro_topic,
        topics_over_time,
        topic_terms,
    )

    lexical_pair_rows = []
    for topic_a, topic_b in combinations(topics, 2):
        terms_a = set(topic_terms.get(topic_a, []))
        terms_b = set(topic_terms.get(topic_b, []))
        lexical_pair_rows.append(
            {
                "topic_id_a": topic_a,
                "topic_id_b": topic_b,
                "top_word_jaccard": jaccard(terms_a, terms_b),
                "high_top_word_overlap_flag": bool((jaccard(terms_a, terms_b) or 0.0) >= 0.50),
            }
        )

    embedding_pair_map = {
        (min(row["topic_id_a"], row["topic_id_b"]), max(row["topic_id_a"], row["topic_id_b"])): row
        for row in embedding_pair_rows
    }
    pair_rows = []
    for row in lexical_pair_rows:
        key = (min(row["topic_id_a"], row["topic_id_b"]), max(row["topic_id_a"], row["topic_id_b"]))
        emb = embedding_pair_map.get(key, {})
        combined = {
            "subgroup": subgroup,
            "source": source,
            "macro_topic": macro_topic,
            **row,
            "centroid_cosine": emb.get("centroid_cosine"),
            "high_centroid_similarity_flag": bool(emb.get("high_centroid_similarity_flag", False)),
        }
        combined["is_flagged"] = bool(
            combined["high_top_word_overlap_flag"] or combined["high_centroid_similarity_flag"]
        )
        pair_rows.append(combined)

    high_overlap_by_topic: dict[int, list[str]] = defaultdict(list)
    for row in pair_rows:
        if row["is_flagged"]:
            reason = []
            if row["high_top_word_overlap_flag"]:
                reason.append("high_top_word_overlap")
            if row["high_centroid_similarity_flag"]:
                reason.append("high_centroid_similarity")
            high_overlap_by_topic[int(row["topic_id_a"])].extend(reason)
            high_overlap_by_topic[int(row["topic_id_b"])].extend(reason)

    topic_rows = []
    for row in valid_topic_info.itertuples(index=False):
        topic_id = int(row.Topic)
        terms = topic_terms.get(topic_id, [])
        topic_size = int(topic_counts.get(topic_id, int(getattr(row, "Count", 0))))
        temporal = temporal_aggregates.get(topic_id, {})
        coherence = coherence_metrics.get(topic_id, {})
        embedding = embedding_metrics.get(topic_id, {})
        flags = []
        if coherence.get("coherence_cv") is not None and coherence["coherence_cv"] < 0.35:
            flags.append("low_cv_coherence")
        if embedding.get("nearest_centroid_margin") is not None and embedding["nearest_centroid_margin"] < 0.05:
            flags.append("low_centroid_margin")
        flags.extend(sorted(set(high_overlap_by_topic.get(topic_id, []))))
        if temporal.get("active_year_count") is not None and temporal["active_year_count"] < 5:
            flags.append("short_temporal_coverage")
        if (
            temporal.get("mean_consecutive_year_jaccard") is not None
            and temporal["mean_consecutive_year_jaccard"] < 0.10
            and temporal.get("active_year_count", 0) >= 3
        ):
            flags.append("low_temporal_word_stability")
        rep_docs = reps.get(topic_id, [])[:3]
        topic_rows.append(
            {
                "subgroup": subgroup,
                "source": source,
                "macro_topic": macro_topic,
                "topic_id": topic_id,
                "topic_size": topic_size,
                "topic_share_non_outlier": topic_size / non_outlier_rows if non_outlier_rows else None,
                "topic_name": getattr(row, "Name", ""),
                "final_merge_group_id": getattr(row, "final_merge_group_id", ""),
                "group_size": getattr(row, "group_size", None),
                "top_words": ", ".join(terms),
                "top_words_json": json.dumps(terms, ensure_ascii=False),
                "coherence_cv": coherence.get("coherence_cv"),
                "coherence_npmi": coherence.get("coherence_npmi"),
                **embedding,
                **temporal,
                "representative_doc_count": len(rep_docs),
                "representative_doc_1": rep_docs[0]["text"][:1000] if len(rep_docs) > 0 else "",
                "representative_doc_2": rep_docs[1]["text"][:1000] if len(rep_docs) > 1 else "",
                "representative_doc_3": rep_docs[2]["text"][:1000] if len(rep_docs) > 2 else "",
                "validation_flags": ";".join(sorted(set(flags))),
                "is_flagged_for_review": bool(flags),
            }
        )

    summary_row = {
        "subgroup": subgroup,
        "source": source,
        "macro_topic": macro_topic,
        "row_count": total_rows,
        "non_outlier_rows": non_outlier_rows,
        "outlier_rows": outlier_rows,
        "outlier_share": outlier_rows / total_rows if total_rows else None,
        "n_topics_non_outlier": len(topics),
        "largest_topic_size": int(sizes.max()) if sizes.size else 0,
        "largest_topic_share_non_outlier": safe_float(sizes.max() / sizes.sum()) if sizes.size and sizes.sum() else None,
        "median_topic_size": safe_float(np.median(sizes)) if sizes.size else None,
        "mean_topic_size": safe_float(np.mean(sizes)) if sizes.size else None,
        "p90_topic_size": safe_float(np.percentile(sizes, 90)) if sizes.size else None,
        "topic_size_entropy": entropy,
        "effective_number_of_topics": safe_float(math.exp(entropy)) if entropy is not None else None,
        "topic_size_hhi": safe_float(np.sum(proportions * proportions)) if proportions.size else None,
        "topic_size_gini": gini(sizes),
        "topic_diversity_top_n": topic_diversity,
        "top_n_words": args.top_n_words,
        "mean_coherence_cv": safe_float(np.nanmean([r["coherence_cv"] for r in topic_rows if r["coherence_cv"] is not None]))
        if any(r["coherence_cv"] is not None for r in topic_rows)
        else None,
        "mean_coherence_npmi": safe_float(np.nanmean([r["coherence_npmi"] for r in topic_rows if r["coherence_npmi"] is not None]))
        if any(r["coherence_npmi"] is not None for r in topic_rows)
        else None,
        "silhouette_cosine": embedding_status["silhouette_cosine"],
        "silhouette_sampled_docs": embedding_status["silhouette_sampled_docs"],
        "silhouette_skip_reason": embedding_status["silhouette_skip_reason"],
        "embedding_skip_reason": embedding_status["embedding_skip_reason"],
        "coherence_status": coherence_status["status"],
        "coherence_sampled_docs": coherence_status["sampled_docs"],
        "coherence_skip_reason": coherence_status.get("reason", ""),
        "high_overlap_pair_count": int(sum(bool(row["is_flagged"]) for row in pair_rows)),
        "manifest_status": manifest.get("status", ""),
    }
    return [summary_row], topic_rows, pair_rows, temporal_rows, {
        "subgroup": subgroup,
        "coherence": coherence_status,
        "embedding": embedding_status,
    }


def build_manual_sample(topic_metrics: pd.DataFrame, pair_flags: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    if topic_metrics.empty:
        return topic_metrics.copy()
    frame = topic_metrics.copy()
    n_topics_by_subgroup = frame.groupby("subgroup")["topic_id"].transform("nunique")
    reasons = []
    for row, subgroup_n in zip(frame.itertuples(index=False), n_topics_by_subgroup, strict=True):
        row_reasons = []
        if row.source == "corporate":
            row_reasons.append("all_corporate_topics")
        if subgroup_n <= 6:
            row_reasons.append("small_model_all_topics")
        if bool(row.is_flagged_for_review):
            row_reasons.append("metric_flag")
        reasons.append(";".join(row_reasons))
    frame["selection_reason"] = reasons
    selected = frame.loc[frame["selection_reason"] != ""].copy()
    selected_keys = set(zip(selected["subgroup"], selected["topic_id"]))
    remaining = frame.loc[
        ~frame.apply(lambda row: (row["subgroup"], row["topic_id"]) in selected_keys, axis=1)
    ].copy()
    if len(selected) < target and not remaining.empty:
        largest = (
            remaining.sort_values(["topic_size", "subgroup", "topic_id"], ascending=[False, True, True])
            .groupby("subgroup", group_keys=False)
            .head(5)
            .copy()
        )
        largest["selection_reason"] = "large_topic_fill"
        selected = pd.concat([selected, largest], ignore_index=True)
        selected_keys = set(zip(selected["subgroup"], selected["topic_id"]))
        remaining = frame.loc[
            ~frame.apply(lambda row: (row["subgroup"], row["topic_id"]) in selected_keys, axis=1)
        ].copy()
    if len(selected) < target and not remaining.empty:
        needed = min(target - len(selected), len(remaining))
        random_fill = remaining.sample(n=needed, random_state=seed).copy()
        random_fill["selection_reason"] = "stratified_random_fill"
        selected = pd.concat([selected, random_fill], ignore_index=True)
    output_columns = [
        "selection_reason",
        "subgroup",
        "source",
        "macro_topic",
        "topic_id",
        "topic_size",
        "topic_share_non_outlier",
        "topic_name",
        "final_merge_group_id",
        "group_size",
        "top_words",
        "coherence_cv",
        "coherence_npmi",
        "mean_intra_topic_cosine_distance",
        "nearest_centroid_topic_id",
        "nearest_centroid_cosine",
        "nearest_centroid_margin",
        "active_year_count",
        "sparse_year_share",
        "mean_consecutive_year_jaccard",
        "mean_global_year_word_jaccard",
        "validation_flags",
        "representative_doc_1",
        "representative_doc_2",
        "representative_doc_3",
    ]
    return selected.sort_values(["source", "macro_topic", "topic_size"], ascending=[True, True, False])[
        [column for column in output_columns if column in selected.columns]
    ].reset_index(drop=True)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    topic_metrics: pd.DataFrame,
    pair_flags: pd.DataFrame,
    manual_sample: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# BERTopic Micro-Model Validation Report",
        "",
        "This diagnostic report validates the 18 reviewed micro BERTopic models without refitting or changing any model output.",
        "",
        "## Scope",
        "",
        f"- Models processed: {len(summary)}",
        f"- Topic rows evaluated: {len(topic_metrics)}",
        f"- Pairwise topic comparisons: {len(pair_flags)}",
        f"- Manual review sample rows: {len(manual_sample)}",
        f"- Seed: {manifest['parameters']['seed']}",
        "",
        "## Summary Metrics",
        "",
    ]
    if not summary.empty:
        lines.extend(
            [
                f"- Total chunks across models: {int(summary['row_count'].sum())}",
                f"- Total non-outlier topic assignments: {int(summary['non_outlier_rows'].sum())}",
                f"- Mean outlier share: {summary['outlier_share'].mean():.3f}",
                f"- Mean topic diversity, top-n words: {summary['topic_diversity_top_n'].mean():.3f}",
                f"- Models with silhouette skipped: {int((summary['silhouette_skip_reason'].fillna('') != '').sum())}",
                f"- High-overlap topic pairs flagged: {int(pair_flags['is_flagged'].sum()) if 'is_flagged' in pair_flags else 0}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- Topic coherence (`c_v`, `c_npmi`) is a lexical diagnostic based on top words and sampled subgroup texts.",
            "- Topic diversity and Jaccard overlap identify redundant word-level representations.",
            "- Embedding centroid metrics evaluate semantic compactness and separation using the original chunk embeddings.",
            "- Silhouette is reported only where it is meaningful and computationally feasible.",
            "- Temporal metrics use BERTopic `topics_over_time` outputs: global topics are held fixed while year-specific lexical representations are recalculated.",
            "- These metrics are diagnostic supports for the paper; final validity still depends on qualitative interpretation of labels, top words, and representative documents.",
            "",
            "## Outputs",
            "",
            "- `subgroup_validation_summary.csv`: one row per source-topic model.",
            "- `topic_validation_metrics.csv`: one row per non-outlier topic.",
            "- `topic_pair_overlap_flags.csv`: pairwise lexical and centroid overlap within each model.",
            "- `temporal_validation_metrics.csv`: one row per topic-year observed in `topics_over_time.csv`.",
            "- `manual_validation_sample.csv`: targeted review sample with flags and representative documents.",
            "- `validation_manifest.json`: parameters, paths, and skip reasons.",
            "",
            "## References",
            "",
        ]
    )
    for reference in REFERENCES:
        lines.append(f"- {reference['citation']} {reference['url']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    subgroups = discover_subgroups(args.model_root)
    if len(subgroups) != 18:
        raise ValueError(f"Expected 18 subgroup models, found {len(subgroups)} in {args.model_root}.")

    embeddings, embedding_meta = load_embedding_memmap(args.embedding_file, args.embedding_meta)
    all_summary: list[dict[str, Any]] = []
    all_topics: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    all_temporal: list[dict[str, Any]] = []
    subgroup_statuses: list[dict[str, Any]] = []

    for subgroup_path in subgroups:
        summary, topics, pairs, temporal, status = process_subgroup(subgroup_path, embeddings, args)
        all_summary.extend(summary)
        all_topics.extend(topics)
        all_pairs.extend(pairs)
        all_temporal.extend(temporal)
        subgroup_statuses.append(status)

    summary_df = pd.DataFrame(all_summary)
    topics_df = pd.DataFrame(all_topics)
    pairs_df = pd.DataFrame(all_pairs)
    temporal_df = pd.DataFrame(all_temporal)
    manual_df = build_manual_sample(topics_df, pairs_df, args.manual_sample_target, args.seed)

    summary_df.to_csv(args.output_root / "subgroup_validation_summary.csv", index=False)
    topics_df.to_csv(args.output_root / "topic_validation_metrics.csv", index=False)
    pairs_df.to_csv(args.output_root / "topic_pair_overlap_flags.csv", index=False)
    temporal_df.to_csv(args.output_root / "temporal_validation_metrics.csv", index=False)
    manual_df.to_csv(args.output_root / "manual_validation_sample.csv", index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_root": str(args.model_root),
        "output_root": str(args.output_root),
        "embedding_file": str(args.embedding_file),
        "embedding_meta": str(args.embedding_meta),
        "embedding_meta_payload": embedding_meta,
        "parameters": {
            "top_n_words": args.top_n_words,
            "max_coherence_docs": args.max_coherence_docs,
            "max_silhouette_docs": args.max_silhouette_docs,
            "manual_sample_target": args.manual_sample_target,
            "seed": args.seed,
        },
        "subgroups_processed": [path.name for path in subgroups],
        "row_counts": {
            "subgroup_validation_summary": int(len(summary_df)),
            "topic_validation_metrics": int(len(topics_df)),
            "topic_pair_overlap_flags": int(len(pairs_df)),
            "temporal_validation_metrics": int(len(temporal_df)),
            "manual_validation_sample": int(len(manual_df)),
        },
        "subgroup_statuses": subgroup_statuses,
        "references": REFERENCES,
    }
    write_json(args.output_root / "validation_manifest.json", manifest)
    write_report(args.output_root / "validation_report.md", summary_df, topics_df, pairs_df, manual_df, manifest)

    expected_topics = 0
    for subgroup_path in subgroups:
        info = pd.read_csv(subgroup_path / "topic_info.csv")
        expected_topics += int((pd.to_numeric(info["Topic"], errors="coerce") != -1).sum())
    if len(topics_df) != expected_topics:
        raise ValueError(f"Topic metric row mismatch: expected {expected_topics}, wrote {len(topics_df)}")
    print(f"Processed {len(subgroups)} subgroups")
    print(f"Wrote validation outputs to {args.output_root}")
    print(f"Topic rows: {len(topics_df)}")


if __name__ == "__main__":
    main()
