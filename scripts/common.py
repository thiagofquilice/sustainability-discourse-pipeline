from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


MACRO_TOPIC_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
MACRO_TOPIC_NAMES = {
    "T1": "Clean energy transition",
    "T2": "Operational sustainability and circular production",
    "T3": "Sustainable products, services and consumption",
    "T4": "Climate strategy, carbon governance and disclosure",
    "T5": "Climate risk, adaptation and resilience",
    "T6": "Ecosystems, pollution and environmental stewardship",
}

STATUS_ORDER = ["aligned", "external_relevant_unpaired", "excluded"]
PRECEDENCE_ORDER = [
    "academic_leads_corporate",
    "media_leads_corporate",
    "corporate_leads_academic",
    "corporate_leads_media",
    "synchronous_or_unclear",
]

TOPIC_KEY = [
    "macro_topic",
    "subgroup_noncorporate",
    "final_merge_group_id_noncorporate",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(input_dir: Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(input_dir / filename)


def write_csv(frame: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    ensure_dir(output_dir)
    path = output_dir / filename
    frame.to_csv(path, index=False)
    return path


def macro_name(topic: str, value: object = "") -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return text or MACRO_TOPIC_NAMES.get(str(topic), str(topic))


def normalize_decision(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().lower()
    if text == "delete":
        return "delete"
    if text == "not related":
        return "not related"
    return ""


def decision_to_status(value: str) -> str:
    if value == "delete":
        return "excluded"
    if value == "not related":
        return "external_relevant_unpaired"
    return "aligned"


def topic_identifier(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["macro_topic"].astype(str)
        + "__"
        + frame["subgroup_noncorporate"].astype(str)
        + "__"
        + frame["final_merge_group_id_noncorporate"].astype(str)
    )


def load_review_frame(input_dir: Path) -> pd.DataFrame:
    master = read_csv(input_dir, "corporate_focus_master_review.csv")
    decisions = read_csv(input_dir, "corporate_focus_commented_decision_rows.csv")

    decision_source = (
        decisions["_normalized_review_decision"]
        if "_normalized_review_decision" in decisions.columns
        else decisions["review_decision"]
    )
    decisions = decisions[TOPIC_KEY].copy().assign(
        _normalized_review_decision=decision_source.map(normalize_decision)
    )
    decisions = decisions[decisions["_normalized_review_decision"] != ""]
    decisions = decisions.drop_duplicates(TOPIC_KEY, keep="last")

    review = master.merge(decisions, on=TOPIC_KEY, how="left")
    review["_normalized_review_decision"] = review["_normalized_review_decision"].fillna("")
    review["paper_status"] = review["_normalized_review_decision"].map(decision_to_status)
    review["macro_topic_name"] = [
        macro_name(topic, name)
        for topic, name in zip(review["macro_topic"], review.get("macro_topic_name", ""))
    ]
    return review


def unique_noncorporate_topics(review: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "macro_topic",
        "macro_topic_name",
        "subgroup_noncorporate",
        "source_noncorporate",
        "final_merge_group_id_noncorporate",
        "inclusion_reason",
        "paper_status",
    ]
    topics = review[columns].drop_duplicates(TOPIC_KEY).copy()
    topics["topic_id"] = topic_identifier(topics)
    return topics


def included_corporate_by_macro(input_dir: Path) -> pd.DataFrame:
    corporate = read_csv(input_dir, "included_corporate_groups.csv")
    macro_column = "assigned_label" if "assigned_label" in corporate.columns else "macro_topic"
    corporate = corporate.rename(columns={macro_column: "macro_topic"})
    corporate["macro_topic_name"] = [
        macro_name(topic, name)
        for topic, name in zip(corporate["macro_topic"], corporate.get("macro_topic_name", ""))
    ]
    return corporate.drop_duplicates(["subgroup", "final_merge_group_id"])


def expand_aggregate_pairs(aggregate_pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in aggregate_pairs.to_dict(orient="records"):
        group_ids = json.loads(row["constituent_noncorporate_group_ids_json"])
        for group_id in group_ids:
            rows.append(
                {
                    "aggregate_pair_id": row["aggregate_pair_id"],
                    "macro_topic": row["macro_topic"],
                    "macro_topic_name": macro_name(row["macro_topic"], row.get("macro_topic_name", "")),
                    "dyad": row["dyad"],
                    "matched_corporate_group_id": str(row["matched_corporate_group_id"]),
                    "noncorp_group_id": str(group_id),
                    "precedence_type_label": row["precedence_type_label"],
                    "is_stable_leading_pair": bool(row["is_stable_leading_pair"]),
                }
            )
    return pd.DataFrame(rows)


def precedence_phrase(label: str, compact: bool = False) -> str:
    if label == "academic_leads_corporate":
        return "academic-leading" if compact else "Academic discourse appears to lead corporate discourse"
    if label == "media_leads_corporate":
        return "media-leading" if compact else "Media discourse appears to lead corporate discourse"
    if label == "corporate_leads_academic":
        return "corporate-leading academic" if compact else "Corporate discourse appears to lead academic discourse"
    if label == "corporate_leads_media":
        return "corporate-leading media" if compact else "Corporate discourse appears to lead media discourse"
    return "synchronous or unclear" if compact else "Temporal ordering is mostly synchronous or unclear"


def infer_temporal_pattern(counts: dict[str, int], stable_counts: dict[str, int]) -> str:
    non_zero = {label: count for label, count in counts.items() if count}
    if not non_zero:
        return "No robust temporal signal remains after final review."

    dominant_label = max(
        non_zero,
        key=lambda label: (non_zero[label], -PRECEDENCE_ORDER.index(label)),
    )
    stable_total = sum(stable_counts.values())
    synchronous_count = counts.get("synchronous_or_unclear", 0)

    if dominant_label == "synchronous_or_unclear":
        leading_labels = [
            label
            for label in PRECEDENCE_ORDER
            if label != "synchronous_or_unclear" and counts.get(label, 0) > 0
        ]
        if not leading_labels:
            return "Mostly synchronous or inconclusive temporal ordering."
        lead = precedence_phrase(leading_labels[0], compact=True)
        if stable_total:
            suffix = "case" if stable_total == 1 else "cases"
            return f"Mostly synchronous/co-evolving, with isolated {lead} episodes ({stable_total} stable leading {suffix})."
        return f"Mostly synchronous/co-evolving, with isolated {lead} episodes."

    phrase = precedence_phrase(dominant_label)
    if synchronous_count:
        return f"{phrase}, but with residual synchronous or unclear cases."
    return f"{phrase}."


def complete_status_grid(frame: pd.DataFrame, value_columns: Iterable[str]) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product(
        [MACRO_TOPIC_ORDER, STATUS_ORDER],
        names=["macro_topic", "paper_status"],
    ).to_frame(index=False)
    result = grid.merge(frame, on=["macro_topic", "paper_status"], how="left")
    result["macro_topic_name"] = [macro_name(topic, name) for topic, name in zip(result["macro_topic"], result.get("macro_topic_name", ""))]
    for column in value_columns:
        result[column] = result[column].fillna(0).astype(int)
    return result[["macro_topic", "macro_topic_name", "paper_status", *value_columns]]
