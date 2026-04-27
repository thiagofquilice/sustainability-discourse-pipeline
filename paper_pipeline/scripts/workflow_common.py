#!/usr/bin/env python3
"""Shared helpers for the supervised BERTopic workflow."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


LOGGER = logging.getLogger("supervised_bertopic")
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = WORKFLOW_ROOT / "config" / "main_config.json"
DEFAULT_PROJECT_REPO = Path("data/external/project_repo")
DEFAULT_SUBSET_ROOT = (
    DEFAULT_PROJECT_REPO / "data" / "results" / "subsets" / "corporate_oil_gas_metal_mining"
)
DEFAULT_LABEL_SOURCE_CSV = WORKFLOW_ROOT / "1_Labels" / "bertopic_supervised_labels.csv"
DEFAULT_LABEL_SOURCE_JSON = WORKFLOW_ROOT / "1_Labels" / "bertopic_supervised_labels.json"
REQUIRED_LABEL_COLUMNS = [
    "label_id",
    "label_code",
    "sdg_number",
    "sdg_title",
    "bullet_order_within_sdg",
    "label_text",
]
REQUIRED_CORPUS_COLUMNS = [
    "doc_id",
    "chunk_id",
    "text",
    "source",
    "year",
    "industry",
    "source_doc_id",
]


def configure_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Workflow config not found at {path}. Run scripts/prepare_inputs.py first."
        )
    return read_json(path)


def write_config(config: dict[str, Any], config_path: Path | None = None) -> Path:
    path = config_path or CONFIG_PATH
    write_json(path, config)
    return path


def symlink_or_replace(link_path: Path, target_path: Path) -> None:
    ensure_parent(link_path)
    if link_path.is_symlink():
        current_target = Path(os.readlink(link_path))
        if current_target == target_path:
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"Refusing to replace existing non-symlink path: {link_path}")
    link_path.symlink_to(target_path)


def normalize_label_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_LABEL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Label catalog missing required columns: {missing}")

    labels = frame[REQUIRED_LABEL_COLUMNS].copy()
    labels["label_id"] = pd.to_numeric(labels["label_id"], errors="raise").astype(int)
    labels["sdg_number"] = pd.to_numeric(labels["sdg_number"], errors="raise").astype(int)
    labels["bullet_order_within_sdg"] = pd.to_numeric(
        labels["bullet_order_within_sdg"], errors="raise"
    ).astype(int)
    for column in ["label_code", "sdg_title", "label_text"]:
        labels[column] = labels[column].fillna("").astype(str).str.strip()
    if labels["label_id"].duplicated().any():
        raise ValueError("Label catalog contains duplicated label_id values.")
    if labels["label_code"].duplicated().any():
        raise ValueError("Label catalog contains duplicated label_code values.")
    return labels.sort_values(["sdg_number", "bullet_order_within_sdg", "label_id"]).reset_index(
        drop=True
    )


def load_label_catalog(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        frame = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported label catalog format: {path}")
    return normalize_label_catalog(frame)


def load_corpus(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=columns)
    required = REQUIRED_CORPUS_COLUMNS if columns is None else columns
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Corpus is missing required columns: {missing}")
    return frame


def load_embedding_memmap(
    embedding_file: Path,
    embedding_meta: Path,
    expected_rows: int | None = None,
) -> tuple[np.memmap, dict[str, Any]]:
    meta = read_json(embedding_meta)
    rows = int(meta["rows"])
    dims = int(meta["dims"])
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(
            f"Embedding row mismatch: expected {expected_rows} rows but metadata reports {rows}."
        )
    array = np.memmap(embedding_file, dtype="float32", mode="r", shape=(rows, dims))
    return array, meta


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


def embedding_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sentence_transformer(model_name: str) -> SentenceTransformer:
    local_snapshot = resolve_local_snapshot(model_name)
    load_target = str(local_snapshot) if local_snapshot is not None else model_name
    local_files_only = local_snapshot is not None
    device = embedding_device()
    LOGGER.info(
        "Loading sentence-transformer model=%s device=%s local_files_only=%s",
        model_name,
        device,
        local_files_only,
    )
    return SentenceTransformer(load_target, device=device, local_files_only=local_files_only)


def detect_embedding_model_name(config: dict[str, Any]) -> str:
    explicit_name = config.get("embedding", {}).get("model_name")
    if explicit_name:
        return str(explicit_name)
    meta_path = Path(config["paths"]["dataset_embedding_meta"])
    if meta_path.exists():
        meta = read_json(meta_path)
        model_name = str(meta.get("model_name", "")).strip()
        if model_name:
            return model_name
    raise ValueError(
        "Could not detect the embedding model automatically. Set embedding.model_name in config."
    )


def build_label_embedding_text(row: pd.Series) -> str:
    return (
        f"{row['label_code']}. "
        f"SDG {row['sdg_number']}: {row['sdg_title']}. "
        f"Bullet: {row['label_text']}"
    )


def cosine_similarity_batch(
    document_embeddings: np.ndarray,
    label_embeddings: np.ndarray,
) -> np.ndarray:
    return np.matmul(document_embeddings, label_embeddings.T)


def top_two_matches(score_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if score_matrix.shape[1] < 2:
        raise ValueError("Need at least two labels to compute best and second-best matches.")
    top_two = np.argpartition(score_matrix, kth=-2, axis=1)[:, -2:]
    top_two_scores = np.take_along_axis(score_matrix, top_two, axis=1)
    order = np.argsort(top_two_scores, axis=1)
    second_idx = top_two[np.arange(len(top_two)), order[:, 0]]
    best_idx = top_two[np.arange(len(top_two)), order[:, 1]]
    return best_idx, second_idx


def save_dataframe(
    frame: pd.DataFrame,
    path: Path,
    *,
    index: bool = False,
) -> None:
    ensure_parent(path)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=index)
    elif path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=index)
    else:
        raise ValueError(f"Unsupported output format for dataframe: {path}")


def select_time_field(frame: pd.DataFrame) -> str:
    if "year" in frame.columns and frame["year"].notna().any():
        return "year"
    for candidate in ["filing_date", "date", "published_at"]:
        if candidate in frame.columns and frame[candidate].notna().any():
            return candidate
    raise ValueError("No usable time field found in the dataframe.")


def safe_share(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series | float:
    if isinstance(denominator, pd.Series):
        denominator = denominator.replace(0, np.nan)
    elif denominator == 0:
        return 0.0
    result = numerator / denominator
    if isinstance(result, pd.Series):
        return result.fillna(0.0)
    return float(result) if result == result else 0.0


def add_document_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "document_id" not in result.columns and "doc_id" in result.columns:
        result["document_id"] = result["doc_id"]
    return result


def current_run_manifest(
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "paths": config.get("paths", {}),
        "assignment": config.get("assignment", {}),
        "bertopic": config.get("bertopic", {}),
        "descriptive_analysis": config.get("descriptive_analysis", {}),
    }
    if extra:
        payload.update(extra)
    return payload
