#!/usr/bin/env python3
"""Interactive explorer for the paper's sustainability discourse results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = APP_ROOT / "data" / "app_data"

MACRO_TOPIC_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
MACRO_TOPIC_NAMES = {
    "T1": "Clean energy transition",
    "T2": "Operational sustainability and circular production",
    "T3": "Sustainable products, services and consumption",
    "T4": "Climate strategy, carbon governance and disclosure",
    "T5": "Climate risk, adaptation and resilience",
    "T6": "Ecosystems, pollution and environmental stewardship",
}
SOURCE_LABELS = {
    "corporate": "Corporate",
    "academic": "Academic",
    "media": "Media",
}
STATUS_LABELS = {
    "corporate_anchor": "Corporate anchor",
    "aligned": "Aligned counterpart",
    "external_relevant_unpaired": "Relevant unpaired external topic",
    "excluded": "Excluded after review",
    "reviewed_external": "Reviewed external topic",
}
SERIES_LABELS = {
    "corporate_anchor": "Corporate anchor",
    "academic_aggregate": "Academic aggregate",
    "media_aggregate": "Media aggregate",
}
SERIES_COLORS = {
    "Corporate anchor": "#2f4858",
    "Academic aggregate": "#2a9d8f",
    "Media aggregate": "#e76f51",
    "Academic": "#2a9d8f",
    "Media": "#e76f51",
    "Corporate": "#2f4858",
}
RELATION_LABELS = {
    "external_leads": "External leads",
    "corporate_leads": "Corporate leads",
    "synchronous_or_unclear": "Synchronous or unclear",
}
PREVALENCE_AXIS_LABEL = "Annual document prevalence (% of source-domain documents in year)"
MERGE_COMPONENT_COLUMNS = [
    "topic_id",
    "macro_topic",
    "source",
    "subgroup",
    "final_merge_group_id",
    "component_micro_topic_id",
    "component_topic_name",
    "component_count",
    "component_representation",
    "component_order",
]


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def numeric(value: object, default: float = 0.0) -> float:
    converted = pd.to_numeric(value, errors="coerce")
    if pd.isna(converted):
        return default
    return float(converted)


def integer_text(value: object) -> str:
    number = numeric(value)
    return f"{int(number):,}" if number else "0"


def decimal_text(value: object, digits: int = 3) -> str:
    converted = pd.to_numeric(value, errors="coerce")
    if pd.isna(converted):
        return "n/a"
    return f"{float(converted):.{digits}f}"


def lag_text(value: object) -> str:
    converted = pd.to_numeric(value, errors="coerce")
    if pd.isna(converted):
        return "n/a"
    lag = int(converted)
    if lag > 0:
        return f"+{lag} external leads"
    if lag < 0:
        return f"{lag} corporate leads"
    return "0 synchronous"


def parse_json(value: object, default: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def topic_label(row: pd.Series) -> str:
    label = clean_text(row.get("display_label", ""))
    fallback = clean_text(row.get("topic_name_original", ""))
    return label or fallback or clean_text(row.get("topic_id", ""))


def format_topic_option(row: pd.Series) -> str:
    status = STATUS_LABELS.get(clean_text(row.get("paper_status", "")), clean_text(row.get("paper_status", "")))
    docs = clean_text(row.get("group_size", "")) or clean_text(row.get("topic_size", ""))
    suffix = f" · {integer_text(docs)} docs" if docs else ""
    return f"{topic_label(row)} · {status}{suffix}"


def macro_option(macro_topic: str) -> str:
    return f"{macro_topic} — {MACRO_TOPIC_NAMES.get(macro_topic, macro_topic)}"


@st.cache_data(show_spinner=False)
def load_app_data(data_dir_text: str) -> tuple[dict[str, Any], list[str]]:
    data_dir = Path(data_dir_text)
    required_files = [
        "topics.csv",
        "year_series.csv",
        "year_evidence.csv",
        "relations.csv",
        "panel_series.csv",
        "unpaired_series.csv",
        "source_relation_aggregate_summary.csv",
        "source_relation_individual_summary.csv",
        "source_relation_aggregate_lag_details.csv",
        "source_relation_individual_lag_details.csv",
        "panel_interpretations.json",
        "app_data_manifest.json",
    ]
    missing = [name for name in required_files if not (data_dir / name).exists()]
    if missing:
        return {}, missing

    data = {
        "topics": pd.read_csv(data_dir / "topics.csv").fillna(""),
        "year_series": pd.read_csv(data_dir / "year_series.csv").fillna(""),
        "year_evidence": pd.read_csv(data_dir / "year_evidence.csv").fillna(""),
        "relations": pd.read_csv(data_dir / "relations.csv").fillna(""),
        "panel_series": pd.read_csv(data_dir / "panel_series.csv").fillna(""),
        "unpaired_series": pd.read_csv(data_dir / "unpaired_series.csv").fillna(""),
        "source_relation_aggregate_summary": pd.read_csv(
            data_dir / "source_relation_aggregate_summary.csv"
        ).fillna(""),
        "source_relation_individual_summary": pd.read_csv(
            data_dir / "source_relation_individual_summary.csv"
        ).fillna(""),
        "source_relation_aggregate_lag_details": pd.read_csv(
            data_dir / "source_relation_aggregate_lag_details.csv"
        ).fillna(""),
        "source_relation_individual_lag_details": pd.read_csv(
            data_dir / "source_relation_individual_lag_details.csv"
        ).fillna(""),
        "interpretations": json.loads((data_dir / "panel_interpretations.json").read_text(encoding="utf-8")),
        "manifest": json.loads((data_dir / "app_data_manifest.json").read_text(encoding="utf-8")),
    }
    merge_components_path = data_dir / "topic_merge_components.csv"
    if merge_components_path.exists():
        data["topic_merge_components"] = pd.read_csv(merge_components_path).fillna("")
    else:
        data["topic_merge_components"] = pd.DataFrame(columns=MERGE_COMPONENT_COLUMNS)
    return data, []


def normalize_series(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    normalized = frame.copy()
    normalized["year"] = pd.to_numeric(normalized["year"], errors="coerce").astype("Int64")
    count_fallback = (
        normalized["year_frequency"] if "year_frequency" in normalized.columns else pd.Series(0, index=normalized.index)
    )
    if "year_document_count" in normalized.columns:
        normalized["year_document_count"] = pd.to_numeric(normalized["year_document_count"], errors="coerce").fillna(0)
    else:
        normalized["year_document_count"] = pd.to_numeric(count_fallback, errors="coerce").fillna(0)
    normalized["year_frequency"] = pd.to_numeric(
        count_fallback if "year_frequency" in normalized.columns else normalized["year_document_count"], errors="coerce"
    ).fillna(0)
    if "annual_source_macro_document_count" in normalized.columns:
        normalized["annual_source_macro_document_count"] = pd.to_numeric(
            normalized["annual_source_macro_document_count"], errors="coerce"
        ).fillna(0)
    else:
        normalized["annual_source_macro_document_count"] = 0
    if "annual_document_prevalence" in normalized.columns:
        normalized["annual_document_prevalence"] = pd.to_numeric(
            normalized["annual_document_prevalence"], errors="coerce"
        ).fillna(0)
    else:
        relative_fallback = (
            normalized["relative_share_of_macro_topic_source_docs"]
            if "relative_share_of_macro_topic_source_docs" in normalized.columns
            else pd.Series(0, index=normalized.index)
        )
        normalized["annual_document_prevalence"] = pd.to_numeric(
            relative_fallback, errors="coerce"
        ).fillna(0)
    normalized["relative_share_percent"] = normalized["annual_document_prevalence"] * 100
    return normalized


def line_chart(frame: pd.DataFrame, title: str, color_column: str | None = None) -> None:
    frame = normalize_series(frame)
    if frame.empty:
        st.info("No temporal series is available for this selection.")
        return
    if color_column and color_column in frame.columns:
        figure = px.line(
            frame,
            x="year",
            y="relative_share_percent",
            color=color_column,
            markers=True,
            color_discrete_map=SERIES_COLORS,
            labels={
                "year": "Year",
                "relative_share_percent": PREVALENCE_AXIS_LABEL,
                color_column: "Series",
                "year_document_count": "Topic documents",
                "annual_source_macro_document_count": "Source-domain documents",
            },
            hover_data={
                "year_document_count": ":,",
                "annual_source_macro_document_count": ":,",
                "relative_share_percent": ":.3f",
            },
            title=title,
        )
    else:
        figure = px.line(
            frame,
            x="year",
            y="relative_share_percent",
            markers=True,
            labels={
                "year": "Year",
                "relative_share_percent": PREVALENCE_AXIS_LABEL,
                "year_document_count": "Topic documents",
                "annual_source_macro_document_count": "Source-domain documents",
            },
            hover_data={
                "year_document_count": ":,",
                "annual_source_macro_document_count": ":,",
                "relative_share_percent": ":.3f",
            },
            title=title,
        )
    figure.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
        legend_title_text="",
        hovermode="x unified",
    )
    figure.update_yaxes(rangemode="tozero")
    st.plotly_chart(figure, use_container_width=True)


def render_topic_metrics(topic: pd.Series, series: pd.DataFrame) -> None:
    series = normalize_series(series)
    total_documents = (
        int(series["year_document_count"].sum()) if not series.empty else int(numeric(topic.get("group_size", 0)))
    )
    if not series.empty and series["year_document_count"].max() > 0:
        peak_row = series.sort_values(["relative_share_percent", "year_document_count"], ascending=False).iloc[0]
        peak_text = f"{int(peak_row['year'])} ({peak_row['relative_share_percent']:.3f}%)"
    else:
        peak_text = "No yearly signal"
    active_min = clean_text(topic.get("active_year_min", ""))
    active_max = clean_text(topic.get("active_year_max", ""))
    active_text = f"{active_min}-{active_max}" if active_min and active_max else "n/a"

    col1, col2, col3 = st.columns(3)
    col1.metric("Topic documents", f"{total_documents:,}")
    col2.metric("Peak prevalence year", peak_text)
    col3.metric("Active years", active_text)


def render_topic_descriptions(topic: pd.Series) -> None:
    summary = clean_text(topic.get("overall_summary", ""))
    pattern = clean_text(topic.get("evolution_pattern", ""))
    evidence_note = clean_text(topic.get("evidence_note", ""))

    if summary:
        st.markdown("**General description**")
        st.write(summary)
    if pattern:
        st.markdown("**Longitudinal summary**")
        st.write(pattern)
    if evidence_note:
        with st.expander("Evidence note"):
            st.write(evidence_note)

    phase_rows = []
    for idx in range(1, 5):
        years = clean_text(topic.get(f"phase_{idx}_years", ""))
        text = clean_text(topic.get(f"phase_{idx}_summary", ""))
        if years or text:
            phase_rows.append((f"Phase {idx}", years, text))
    if phase_rows:
        st.markdown("**Period descriptions**")
        for phase, years, text in phase_rows:
            with st.expander(f"{phase}: {years or 'years not specified'}", expanded=False):
                st.write(text or "No description available.")


def render_merge_composition(merge_components: pd.DataFrame, topic: pd.Series) -> None:
    topic_id = clean_text(topic.get("topic_id", ""))
    group_size = int(numeric(topic.get("group_size", 0)))
    with st.expander("Merged topic composition", expanded=False):
        if merge_components.empty:
            st.info("Merge-composition data is not available in the current app data package.")
            return
        components = merge_components[merge_components["topic_id"] == topic_id].copy()
        if components.empty:
            st.info("No merge-composition rows were found for this topic.")
            return
        components["component_order"] = pd.to_numeric(components["component_order"], errors="coerce").fillna(0)
        components["component_count"] = pd.to_numeric(components["component_count"], errors="coerce")
        components = components.sort_values("component_order", kind="mergesort")
        if group_size <= 1 or len(components) <= 1:
            st.info("Singleton topic: this consolidated topic contains one original microtopic.")
        else:
            st.caption(f"{len(components):,} original microtopics compose this consolidated topic.")
        table = components[
            [
                "component_micro_topic_id",
                "component_topic_name",
                "component_count",
                "component_representation",
            ]
        ].rename(
            columns={
                "component_micro_topic_id": "Original microtopic ID",
                "component_topic_name": "Original BERTopic label/name",
                "component_count": "Count",
                "component_representation": "Top words / representation",
            }
        )
        st.dataframe(table, hide_index=True, use_container_width=True)


def evidence_for_year(evidence: pd.DataFrame, topic_id: str, selected_year: int | None) -> pd.Series | None:
    topic_evidence = evidence[evidence["topic_id"] == topic_id].copy()
    if topic_evidence.empty:
        return None
    topic_evidence["year"] = pd.to_numeric(topic_evidence["year"], errors="coerce").astype("Int64")
    if selected_year is not None:
        exact = topic_evidence[topic_evidence["year"] == selected_year]
        if not exact.empty:
            return exact.iloc[0]
    return topic_evidence.sort_values("year").iloc[-1]


def render_evidence(evidence: pd.DataFrame, topic_id: str, series: pd.DataFrame) -> None:
    st.markdown("**Top words and representative snippets**")
    series = normalize_series(series)
    available_years = sorted(
        int(year)
        for year in pd.to_numeric(evidence[evidence["topic_id"] == topic_id]["year"], errors="coerce").dropna().unique()
    )
    if not available_years:
        st.info("No representative snippet data is available for this topic.")
        return
    if not series.empty and series["year_document_count"].max() > 0:
        peak_year = int(
            series.sort_values(["relative_share_percent", "year_document_count"], ascending=False).iloc[0]["year"]
        )
    else:
        peak_year = available_years[-1]
    default_index = available_years.index(peak_year) if peak_year in available_years else len(available_years) - 1
    selected_year = st.selectbox("Evidence year", available_years, index=default_index)
    row = evidence_for_year(evidence, topic_id, selected_year)
    if row is None:
        st.info("No evidence was found for this year.")
        return

    words = parse_json(row.get("top_words_json", ""), [])
    if words:
        st.caption("Top words")
        st.write(" · ".join(str(word) for word in words[:14]))

    snippets = parse_json(row.get("representative_docs_json", ""), [])
    if not snippets:
        st.info("No representative snippets were exported for this year.")
        return
    st.caption("Representative snippets, public-safe")
    for index, doc in enumerate(snippets[:3], start=1):
        metadata = [
            f"year: {doc.get('year', selected_year)}",
            f"source: {doc.get('source', '')}",
            f"chunk_id: {doc.get('chunk_id', '')}",
        ]
        document_id = clean_text(doc.get("source_doc_id", "")) or clean_text(doc.get("document_id", ""))
        if document_id:
            metadata.append(f"doc_id: {document_id}")
        link = clean_text(doc.get("source_link", ""))
        with st.container(border=True):
            st.markdown(f"**Snippet {index}**")
            st.caption(" | ".join(item for item in metadata if item))
            st.write(clean_text(doc.get("snippet", "")))
            if link:
                st.link_button("Open source link", link)


def render_topic_detail(
    topic: pd.Series,
    year_series: pd.DataFrame,
    year_evidence: pd.DataFrame,
    topic_merge_components: pd.DataFrame,
    override_series: pd.DataFrame | None = None,
) -> None:
    topic_id = clean_text(topic.get("topic_id", ""))
    series = override_series if override_series is not None else year_series[year_series["topic_id"] == topic_id]
    st.subheader(topic_label(topic))
    status = STATUS_LABELS.get(clean_text(topic.get("paper_status", "")), clean_text(topic.get("paper_status", "")))
    source = SOURCE_LABELS.get(clean_text(topic.get("source", "")), clean_text(topic.get("source", "")))
    st.caption(f"{source} · {status} · {clean_text(topic.get('final_merge_group_id', ''))}")
    render_topic_metrics(topic, series)
    render_merge_composition(topic_merge_components, topic)
    line_chart(series, "Temporal evolution")
    render_topic_descriptions(topic)
    render_evidence(year_evidence, topic_id, series)


def render_topic_mode(data: dict[str, Any], macro_topic: str, source: str) -> None:
    topics = data["topics"]
    filtered = topics[(topics["macro_topic"] == macro_topic) & (topics["source"] == source)].copy()
    if filtered.empty:
        st.info("No topics are available for this source and macro-topic.")
        return

    if source != "corporate":
        statuses = sorted(filtered["paper_status"].dropna().unique().tolist())
        selected_statuses = st.multiselect(
            "Review status",
            statuses,
            default=statuses,
            format_func=lambda value: STATUS_LABELS.get(value, value),
        )
        filtered = filtered[filtered["paper_status"].isin(selected_statuses)]
    filtered = filtered.sort_values(["paper_status", "display_label"], kind="mergesort")
    if filtered.empty:
        st.info("No topics remain after the status filter.")
        return

    left, right = st.columns([0.36, 0.64], gap="large")
    with left:
        st.markdown(f"**{SOURCE_LABELS[source]} topics**")
        selected_id = st.radio(
            "Select a topic",
            filtered["topic_id"].tolist(),
            format_func=lambda topic_id: format_topic_option(filtered[filtered["topic_id"] == topic_id].iloc[0]),
            label_visibility="collapsed",
        )
    with right:
        topic = filtered[filtered["topic_id"] == selected_id].iloc[0]
        render_topic_detail(
            topic,
            data["year_series"],
            data["year_evidence"],
            data["topic_merge_components"],
        )


def render_relation_chart(panel_series: pd.DataFrame, corporate_topic_id: str) -> None:
    panel = panel_series[panel_series["corporate_topic_id"] == corporate_topic_id].copy()
    if panel.empty:
        st.info("No relation panel series is available for this corporate anchor.")
        return
    panel["series"] = panel["series_role"].map(SERIES_LABELS).fillna(panel["series_role"])
    line_chart(panel, "Corporate anchor and external aggregate series", color_column="series")


def render_relation_metrics(panel_series: pd.DataFrame, corporate_topic_id: str) -> None:
    panel = panel_series[panel_series["corporate_topic_id"] == corporate_topic_id].copy()
    if panel.empty:
        return
    rows = []
    for role, label in SERIES_LABELS.items():
        role_frame = panel[panel["series_role"] == role]
        if role_frame.empty:
            rows.append((label, 0))
            continue
        if role == "corporate_anchor":
            count = role_frame["corporate_unique_document_count"].replace("", 0).astype(float).max()
        else:
            count = role_frame["series_unique_document_count"].replace("", 0).astype(float).max()
        rows.append((label, int(count)))
    cols = st.columns(3)
    for col, (label, count) in zip(cols, rows):
        col.metric(label, f"{count:,} docs")


def relation_interpretation(data: dict[str, Any], label: str) -> str:
    interpretations = data["interpretations"].get("anchors", {})
    if label in interpretations:
        return interpretations[label]
    for key, value in interpretations.items():
        if clean_text(key).lower() == clean_text(label).lower():
            return value
    return ""


def render_relation_tables(data: dict[str, Any], corporate_topic_id: str) -> pd.DataFrame:
    relations = data["relations"]
    matches = relations[relations["corporate_topic_id"] == corporate_topic_id].copy()
    if matches.empty:
        st.info("This corporate anchor has no retained academic or media aggregate counterpart.")
        return matches
    table = matches[
        [
            "source_noncorporate",
            "topic_label_refined_noncorporate",
            "external_unique_document_count",
            "best_cosine_similarity",
            "direct_pair_count",
        ]
    ].rename(
        columns={
            "source_noncorporate": "Source",
            "topic_label_refined_noncorporate": "External topic",
            "external_unique_document_count": "Documents",
            "best_cosine_similarity": "Best cosine",
            "direct_pair_count": "Direct pairs",
        }
    )
    st.dataframe(table, hide_index=True, use_container_width=True)
    return matches


def render_anchor_evidence_picker(data: dict[str, Any], anchor: pd.Series, matches: pd.DataFrame) -> None:
    topics = data["topics"]
    evidence_options = [clean_text(anchor["topic_id"])] + matches["external_topic_id"].dropna().astype(str).tolist()
    evidence_options = [topic_id for topic_id in dict.fromkeys(evidence_options) if topic_id]
    option_rows = topics[topics["topic_id"].isin(evidence_options)].copy()
    if option_rows.empty:
        return
    st.markdown("**Evidence snippets for relation components**")
    selected = st.selectbox(
        "Topic evidence",
        option_rows["topic_id"].tolist(),
        format_func=lambda topic_id: format_topic_option(option_rows[option_rows["topic_id"] == topic_id].iloc[0]),
    )
    topic = option_rows[option_rows["topic_id"] == selected].iloc[0]
    series = data["year_series"][data["year_series"]["topic_id"] == selected]
    render_merge_composition(data["topic_merge_components"], topic)
    render_evidence(data["year_evidence"], selected, series)


def render_unpaired_section(data: dict[str, Any], macro_topic: str) -> None:
    st.divider()
    st.subheader("Relevant external topics without corporate counterpart")
    interpretation = data["interpretations"].get("unpaired", {}).get(macro_topic, "")
    if interpretation:
        st.write(interpretation)

    topics = data["topics"]
    unpaired = topics[
        (topics["macro_topic"] == macro_topic) & (topics["paper_status"] == "external_relevant_unpaired")
    ].copy()
    if unpaired.empty:
        st.info("No relevant unpaired external topic remained after final review for this macro-topic.")
        return
    selected = st.selectbox(
        "Unpaired external topic",
        unpaired["topic_id"].tolist(),
        format_func=lambda topic_id: format_topic_option(unpaired[unpaired["topic_id"] == topic_id].iloc[0]),
    )
    topic = unpaired[unpaired["topic_id"] == selected].iloc[0]
    series = data["unpaired_series"][data["unpaired_series"]["topic_id"] == selected]
    render_topic_detail(
        topic,
        data["year_series"],
        data["year_evidence"],
        data["topic_merge_components"],
        override_series=series,
    )


def render_relations_mode(data: dict[str, Any], macro_topic: str) -> None:
    topics = data["topics"]
    corporate = topics[(topics["macro_topic"] == macro_topic) & (topics["source"] == "corporate")].copy()
    if corporate.empty:
        st.info("No corporate anchors are available for this macro-topic.")
        return
    corporate = corporate.sort_values("display_label", kind="mergesort")
    left, right = st.columns([0.34, 0.66], gap="large")
    with left:
        st.markdown("**Corporate anchors**")
        selected_anchor = st.radio(
            "Select corporate anchor",
            corporate["topic_id"].tolist(),
            format_func=lambda topic_id: format_topic_option(corporate[corporate["topic_id"] == topic_id].iloc[0]),
            label_visibility="collapsed",
        )
    anchor = corporate[corporate["topic_id"] == selected_anchor].iloc[0]
    with right:
        st.subheader(topic_label(anchor))
        st.caption(f"Corporate anchor · {clean_text(anchor.get('final_merge_group_id', ''))}")
        render_relation_metrics(data["panel_series"], selected_anchor)
        render_relation_chart(data["panel_series"], selected_anchor)
        render_merge_composition(data["topic_merge_components"], anchor)

        interpretation = relation_interpretation(data, topic_label(anchor))
        if interpretation:
            st.markdown("**Longitudinal interpretation**")
            st.write(interpretation)

        st.markdown("**External aggregate topics retained as counterparts**")
        matches = render_relation_tables(data, selected_anchor)
        render_anchor_evidence_picker(data, anchor, matches)

    render_unpaired_section(data, macro_topic)


def relation_option(frame: pd.DataFrame, relation_id: str) -> str:
    row = frame[frame["relation_id"] == relation_id].iloc[0]
    external_label = clean_text(row.get("external_topic_label", "")) or clean_text(row.get("external_relation_label", ""))
    corporate_label = clean_text(row.get("corporate_topic_label", ""))
    source = SOURCE_LABELS.get(clean_text(row.get("external_source", "")), clean_text(row.get("external_source", "")))
    return (
        f"{corporate_label} x {source}: {external_label} · "
        f"{lag_text(row.get('best_lag_spearman'))} · "
        f"Spearman={decimal_text(row.get('best_spearman_r'), 2)}"
    )


def relation_display_table(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    columns = [
        "corporate_topic_label",
        "external_source",
        "external_relation_label",
        "external_topic_count",
        "corporate_unique_document_count",
        "external_unique_document_count",
        "best_lag_spearman",
        "best_spearman_r",
        "best_spearman_q_table",
        "pearson_r_at_best_spearman_lag",
        "first_active_year_gap",
        "peak_year_gap",
        "temporal_relation_label",
        "ambiguous_best_lag_flag",
    ]
    if level == "individual":
        columns.insert(3, "external_topic_label")
        columns.extend(["best_cosine_similarity", "direct_pair_count"])
    available = [column for column in columns if column in frame.columns]
    table = frame[available].copy()
    rename = {
        "corporate_topic_label": "Corporate anchor",
        "external_source": "External source",
        "external_relation_label": "External relation",
        "external_topic_label": "External topic",
        "external_topic_count": "External topics",
        "corporate_unique_document_count": "Corporate docs",
        "external_unique_document_count": "External docs",
        "best_lag_spearman": "Best lag",
        "best_spearman_r": "Spearman",
        "best_spearman_q_table": "Spearman q",
        "pearson_r_at_best_spearman_lag": "Pearson at lag",
        "first_active_year_gap": "First-year gap",
        "peak_year_gap": "Peak-year gap",
        "temporal_relation_label": "Temporal label",
        "ambiguous_best_lag_flag": "Ambiguous lag",
        "best_cosine_similarity": "Cosine",
        "direct_pair_count": "Direct pairs",
    }
    for column in [
        "best_spearman_r",
        "best_spearman_q_table",
        "pearson_r_at_best_spearman_lag",
        "first_active_year_gap",
        "peak_year_gap",
        "best_cosine_similarity",
    ]:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(3)
    if "temporal_relation_label" in table.columns:
        table["temporal_relation_label"] = table["temporal_relation_label"].map(
            lambda value: RELATION_LABELS.get(clean_text(value), clean_text(value))
        )
    if "external_source" in table.columns:
        table["external_source"] = table["external_source"].map(
            lambda value: SOURCE_LABELS.get(clean_text(value), clean_text(value))
        )
    return table.rename(columns=rename)


def render_relation_summary_metrics(row: pd.Series) -> None:
    cols = st.columns(5)
    cols[0].metric("Best lag", lag_text(row.get("best_lag_spearman")))
    cols[1].metric("Spearman", decimal_text(row.get("best_spearman_r")))
    cols[2].metric("Spearman q", decimal_text(row.get("best_spearman_q_table")))
    cols[3].metric("Pearson at lag", decimal_text(row.get("pearson_r_at_best_spearman_lag")))
    cols[4].metric(
        "Temporal label",
        RELATION_LABELS.get(clean_text(row.get("temporal_relation_label", "")), "n/a"),
    )


def render_lag_detail_chart(details: pd.DataFrame, relation_id: str) -> None:
    relation_details = details[details["relation_id"] == relation_id].copy()
    if relation_details.empty:
        st.info("No lag-detail rows are available for this relation.")
        return
    for column in ["lag", "spearman_r", "pearson_r", "n_overlap_years"]:
        relation_details[column] = pd.to_numeric(relation_details[column], errors="coerce")
    plot_frame = relation_details[["lag", "spearman_r", "pearson_r"]].melt(
        id_vars="lag",
        value_vars=["spearman_r", "pearson_r"],
        var_name="Statistic",
        value_name="Correlation",
    )
    plot_frame = plot_frame.dropna(subset=["Correlation"])
    if plot_frame.empty:
        st.info("The available series were too sparse or constant for valid lag correlations.")
        return
    plot_frame["Statistic"] = plot_frame["Statistic"].map(
        {"spearman_r": "Spearman", "pearson_r": "Pearson"}
    )
    figure = px.line(
        plot_frame,
        x="lag",
        y="Correlation",
        color="Statistic",
        markers=True,
        labels={
            "lag": "Lag (positive = external source leads corporate)",
            "Correlation": "Correlation",
        },
        title="Lag profile",
    )
    figure.add_hline(y=0, line_dash="dash", line_color="#999999")
    figure.update_xaxes(dtick=1)
    figure.update_yaxes(range=[-1.05, 1.05])
    figure.update_layout(height=330, margin=dict(l=10, r=10, t=45, b=10), hovermode="x unified")
    st.plotly_chart(figure, use_container_width=True)

    detail_table = relation_details[
        [
            "lag",
            "n_overlap_years",
            "spearman_r",
            "spearman_p",
            "spearman_q_table",
            "pearson_r",
            "pearson_p",
            "pearson_q_table",
            "low_information_flag",
        ]
    ].copy()
    for column in ["spearman_r", "spearman_p", "spearman_q_table", "pearson_r", "pearson_p", "pearson_q_table"]:
        detail_table[column] = pd.to_numeric(detail_table[column], errors="coerce").round(3)
    with st.expander("Lag-detail table", expanded=False):
        st.dataframe(detail_table, hide_index=True, use_container_width=True)


def filter_relation_frame(frame: pd.DataFrame, macro_topic: str, key_prefix: str) -> pd.DataFrame:
    filtered = frame[frame["macro_topic"] == macro_topic].copy()
    if filtered.empty:
        return filtered
    sources = sorted(filtered["external_source"].dropna().astype(str).unique().tolist())
    selected_sources = st.multiselect(
        "External source",
        sources,
        default=sources,
        format_func=lambda value: SOURCE_LABELS.get(value, value),
        key=f"{key_prefix}_sources",
    )
    filtered = filtered[filtered["external_source"].isin(selected_sources)]
    labels = sorted(filtered["temporal_relation_label"].dropna().astype(str).unique().tolist())
    selected_labels = st.multiselect(
        "Temporal label",
        labels,
        default=labels,
        format_func=lambda value: RELATION_LABELS.get(value, value),
        key=f"{key_prefix}_labels",
    )
    return filtered[filtered["temporal_relation_label"].isin(selected_labels)].copy()


def render_relation_summary_tab(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    macro_topic: str,
    level: str,
) -> None:
    filtered = filter_relation_frame(summary, macro_topic, level)
    if filtered.empty:
        st.info("No temporal-relation rows are available for this macro-topic and filter.")
        return
    filtered = filtered.sort_values(
        ["external_source", "corporate_topic_label", "external_relation_label"],
        kind="mergesort",
    )
    st.dataframe(relation_display_table(filtered, level), hide_index=True, use_container_width=True)
    selected_relation = st.selectbox(
        "Inspect relation",
        filtered["relation_id"].tolist(),
        format_func=lambda relation_id: relation_option(filtered, relation_id),
        key=f"{level}_relation_select",
    )
    row = filtered[filtered["relation_id"] == selected_relation].iloc[0]
    st.markdown("**Selected temporal relation**")
    render_relation_summary_metrics(row)
    st.caption(
        "Positive lags mean the academic/media series precedes the corporate series. "
        "The statistic uses annual document prevalence, not raw document counts."
    )
    render_lag_detail_chart(details, selected_relation)


def render_temporal_relations_mode(data: dict[str, Any], macro_topic: str) -> None:
    aggregate = data["source_relation_aggregate_summary"]
    individual = data["source_relation_individual_summary"]
    aggregate_details = data["source_relation_aggregate_lag_details"]
    individual_details = data["source_relation_individual_lag_details"]

    aggregate_macro = aggregate[aggregate["macro_topic"] == macro_topic].copy()
    individual_macro = individual[individual["macro_topic"] == macro_topic].copy()
    col1, col2, col3 = st.columns(3)
    col1.metric("Aggregate relations", f"{len(aggregate_macro):,}")
    col2.metric("Individual topic relations", f"{len(individual_macro):,}")
    significant = pd.to_numeric(individual_macro.get("best_spearman_q_table", pd.Series(dtype=float)), errors="coerce")
    col3.metric("Individual q < .05", f"{int((significant < 0.05).sum()):,}")
    st.caption(
        "These tables summarize lagged Spearman correlations from -3 to +3 years. "
        "They are exploratory association diagnostics, not causal estimates."
    )

    aggregate_tab, individual_tab = st.tabs(["Aggregate source relations", "Individual topic relations"])
    with aggregate_tab:
        render_relation_summary_tab(aggregate, aggregate_details, macro_topic, "aggregate")
    with individual_tab:
        render_relation_summary_tab(individual, individual_details, macro_topic, "individual")


def render_public_safe_note(manifest: dict[str, Any]) -> None:
    public_safe = manifest.get("public_safe", True)
    snippet_chars = manifest.get("snippet_chars", "")
    if public_safe:
        st.sidebar.success(f"Public-safe evidence mode · snippets capped at {snippet_chars} chars")
    else:
        st.sidebar.warning("Local full-text mode is active.")


def main() -> None:
    st.set_page_config(
        page_title="Sustainability Discourse Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Sustainability Discourse Explorer")
    st.caption("Corporate-centered comparison of academic, media, and corporate sustainability discourse.")

    data_dir = Path(os.environ.get("SUSTAINABILITY_APP_DATA", DEFAULT_DATA_DIR))
    data, missing = load_app_data(str(data_dir))
    if missing:
        st.error("The Streamlit app data files are missing.")
        st.code(
            "python scripts/build_streamlit_app_data.py "
            "--pipeline-root /path/to/paper_6topic_discourse_pipeline "
            "--output-dir data/app_data --public-safe",
            language="bash",
        )
        st.write(f"Missing files in `{data_dir}`: {', '.join(missing)}")
        st.stop()

    render_public_safe_note(data["manifest"])
    macro_topic = st.sidebar.selectbox(
        "Macro topic",
        MACRO_TOPIC_ORDER,
        format_func=macro_option,
    )
    view_mode = st.sidebar.segmented_control(
        "View",
        ["Corporate", "Academic", "Media", "Relations", "Temporal Relations"],
        default="Corporate",
    )

    st.header(macro_option(macro_topic))
    if view_mode == "Corporate":
        render_topic_mode(data, macro_topic, "corporate")
    elif view_mode == "Academic":
        render_topic_mode(data, macro_topic, "academic")
    elif view_mode == "Media":
        render_topic_mode(data, macro_topic, "media")
    elif view_mode == "Relations":
        render_relations_mode(data, macro_topic)
    else:
        render_temporal_relations_mode(data, macro_topic)


if __name__ == "__main__":
    main()
