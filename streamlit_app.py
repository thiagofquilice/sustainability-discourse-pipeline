#!/usr/bin/env python3
"""Reader companion app for the paper's sustainability discourse results."""

from __future__ import annotations

import html
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
MACRO_DEFINITIONS = {
    "T1": "Cleaner energy systems, renewable electricity, grid integration, storage, and access to modern clean energy services.",
    "T2": "Operational practices that reduce material, water, energy, chemical, or pollution impacts.",
    "T3": "Lower-impact products, services, materials, and market-facing consumption choices.",
    "T4": "Targets, disclosure, governance, pricing, and formal strategic positioning around climate and carbon.",
    "T5": "Physical and transition climate risk, adaptation, resilience planning, and operational exposure.",
    "T6": "Protection, restoration, compliance, and pollution-prevention tied to biodiversity, land, coasts, and ecosystems.",
}
DOMAIN_PATTERNS = {
    "T1": "Concentrated translation",
    "T2": "External spillover",
    "T3": "Broad incorporation",
    "T4": "Broad but bounded incorporation",
    "T5": "Compressed incorporation",
    "T6": "Broad stewardship incorporation",
}
DOMAIN_READING = {
    "T1": "Corporate disclosure engages clean energy mainly where transition issues become demand, infrastructure, supply-chain, competitive, or financing exposure.",
    "T2": "Operational sustainability shows several external issues that remain relevant outside the corporate frame, especially circularity and reuse trajectories.",
    "T3": "Product and consumption issues show the broadest incorporation pattern, with many academic and media topics absorbed into corporate-facing market, product, and stakeholder frames.",
    "T4": "Climate strategy and disclosure are widely incorporated, but mainly through recognizable corporate categories such as regulation, governance, carbon markets, and reporting.",
    "T5": "Climate risk is compressed into a small number of corporate risk anchors, even though academic and media discourse contains broader adaptation and resilience trajectories.",
    "T6": "Environmental stewardship is strongly present in corporate disclosure through compliance, liability, habitat protection, biodiversity, and pollution-prevention frames.",
}
SOURCE_LABELS = {
    "corporate": "Corporate",
    "academic": "Academic",
    "media": "Media",
}
SOURCE_COLORS = {
    "Corporate": "#2f4858",
    "Academic": "#2a9d8f",
    "Media": "#e76f51",
}
STATUS_LABELS = {
    "corporate_anchor": "Corporate anchor",
    "aligned": "Aligned counterpart",
    "external_relevant_unpaired": "Relevant but unpaired",
    "excluded": "Excluded after review",
    "reviewed_external": "Reviewed external topic",
}
SERIES_LABELS = {
    "corporate": "Corporate anchor",
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
FINAL_SAMPLE_METRICS = {
    "Corporate anchors": 27,
    "Non-corporate topics reviewed": 214,
    "Aligned counterparts": 150,
    "Relevant but unpaired": 20,
    "Excluded after review": 44,
    "Final unique documents": 116_356,
}
WORKFLOW_STEPS = [
    ("1", "Classify environmental domains", "Texts were assigned to six environmental macro-topics using phrase-level embedding anchors."),
    ("2", "Model microtopics", "BERTopic was estimated separately for each source-domain subgroup."),
    ("3", "Review and merge", "Fragmented or overlapping microtopics were consolidated before interpretation."),
    ("4", "Anchor on corporate disclosure", "Corporate microtopics were treated as anchors for the comparative sample."),
    ("5", "Compare trajectories", "Annual document prevalence and qualitative summaries were used to interpret cross-source evolution."),
]
PREVALENCE_AXIS_LABEL = "Annual document prevalence (% of retained source-domain documents in year)"
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


def parse_json(value: object, default: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def macro_option(macro_topic: str) -> str:
    return f"{macro_topic} - {MACRO_TOPIC_NAMES.get(macro_topic, macro_topic)}"


def topic_label(row: pd.Series) -> str:
    label = clean_text(row.get("display_label", ""))
    fallback = clean_text(row.get("topic_name_original", ""))
    return label or fallback or clean_text(row.get("topic_id", ""))


def topic_doc_count(year_series: pd.DataFrame, topic_id: str) -> int:
    series = year_series[year_series["topic_id"] == topic_id].copy()
    if series.empty:
        return 0
    return int(pd.to_numeric(series["year_document_count"], errors="coerce").fillna(0).sum())


def format_topic_option(row: pd.Series, year_series: pd.DataFrame | None = None) -> str:
    status = STATUS_LABELS.get(clean_text(row.get("paper_status", "")), clean_text(row.get("paper_status", "")))
    docs = topic_doc_count(year_series, clean_text(row.get("topic_id", ""))) if year_series is not None else 0
    suffix = f" · {docs:,} docs" if docs else ""
    return f"{topic_label(row)} · {status}{suffix}"


def escape(value: object) -> str:
    return html.escape(clean_text(value))


def apply_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
        h1, h2, h3 {letter-spacing: 0;}
        .paper-hero {
            border: 1px solid #d9e2ec;
            background: linear-gradient(180deg, #f7fbfc 0%, #ffffff 100%);
            padding: 1.25rem 1.35rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .paper-hero h1 {font-size: 2.05rem; margin: 0 0 .35rem 0;}
        .paper-hero p {font-size: 1rem; line-height: 1.5; margin: 0; color: #405261;}
        .metric-card, .note-card, .topic-card {
            border: 1px solid #d9e2ec;
            background: #ffffff;
            border-radius: 8px;
            padding: .85rem .95rem;
            height: 100%;
        }
        .metric-label {font-size: .78rem; text-transform: uppercase; color: #617384; letter-spacing: .03rem;}
        .metric-value {font-size: 1.45rem; font-weight: 700; color: #1f3442; margin-top: .15rem;}
        .note-card {background: #f8fafc;}
        .note-title {font-weight: 700; color: #1f3442; margin-bottom: .25rem;}
        .small-muted {font-size: .86rem; color: #5b6b7a;}
        .source-pill {
            display: inline-block;
            padding: .16rem .48rem;
            border-radius: 999px;
            background: #eef3f7;
            font-size: .78rem;
            margin-right: .25rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: .7rem .85rem;
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: object, help_text: str = "") -> None:
    caption = f"<div class='small-muted'>{escape(help_text)}</div>" if help_text else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
            {caption}
        </div>
        """,
        unsafe_allow_html=True,
    )


def note_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="note-card">
            <div class="note-title">{escape(title)}</div>
            <div class="small-muted">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        "source_relation_aggregate_same_year.csv",
        "source_relation_individual_same_year.csv",
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
        "source_relation_aggregate_same_year": pd.read_csv(
            data_dir / "source_relation_aggregate_same_year.csv"
        ).fillna(""),
        "source_relation_individual_same_year": pd.read_csv(
            data_dir / "source_relation_individual_same_year.csv"
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
        fallback = (
            normalized["relative_share_of_macro_topic_source_docs"]
            if "relative_share_of_macro_topic_source_docs" in normalized.columns
            else pd.Series(0, index=normalized.index)
        )
        normalized["annual_document_prevalence"] = pd.to_numeric(fallback, errors="coerce").fillna(0)
    normalized["relative_share_percent"] = normalized["annual_document_prevalence"] * 100
    return normalized


def line_chart(frame: pd.DataFrame, title: str, color_column: str | None = None, height: int = 360) -> None:
    frame = normalize_series(frame)
    if frame.empty:
        st.info("No temporal series is available for this selection.")
        return
    labels = {
        "year": "Year",
        "relative_share_percent": PREVALENCE_AXIS_LABEL,
        "year_document_count": "Topic documents",
        "annual_source_macro_document_count": "Source-domain documents",
    }
    hover_data = {
        "year_document_count": ":,",
        "annual_source_macro_document_count": ":,",
        "relative_share_percent": ":.3f",
    }
    if color_column and color_column in frame.columns:
        labels[color_column] = "Series"
        figure = px.line(
            frame,
            x="year",
            y="relative_share_percent",
            color=color_column,
            markers=True,
            color_discrete_map=SERIES_COLORS,
            labels=labels,
            hover_data=hover_data,
            title=title,
        )
    else:
        figure = px.line(
            frame,
            x="year",
            y="relative_share_percent",
            markers=True,
            labels=labels,
            hover_data=hover_data,
            title=title,
        )
    figure.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        legend_title_text="",
        hovermode="x unified",
        font=dict(family="Arial", size=13),
    )
    figure.update_yaxes(rangemode="tozero")
    st.plotly_chart(figure, width="stretch")


def domain_summary(data: dict[str, Any]) -> pd.DataFrame:
    topics = data["topics"]
    relations = data["relations"]
    rows: list[dict[str, object]] = []
    for macro_topic in MACRO_TOPIC_ORDER:
        macro_topics = topics[topics["macro_topic"] == macro_topic]
        corporate = macro_topics[macro_topics["source"] == "corporate"]
        related_anchor_count = relations[relations["macro_topic"] == macro_topic]["corporate_topic_id"].nunique()
        aligned = macro_topics[(macro_topics["source"] != "corporate") & (macro_topics["paper_status"] == "aligned")]
        unpaired = macro_topics[
            (macro_topics["source"] != "corporate") & (macro_topics["paper_status"] == "external_relevant_unpaired")
        ]
        excluded = macro_topics[(macro_topics["source"] != "corporate") & (macro_topics["paper_status"] == "excluded")]
        rows.append(
            {
                "macro_topic": macro_topic,
                "macro_topic_name": MACRO_TOPIC_NAMES[macro_topic],
                "definition": MACRO_DEFINITIONS[macro_topic],
                "main_pattern": DOMAIN_PATTERNS[macro_topic],
                "how_to_read": DOMAIN_READING[macro_topic],
                "corporate_anchor_coverage": f"{related_anchor_count}/{len(corporate)}",
                "corporate_anchors": int(len(corporate)),
                "anchors_with_counterparts": int(related_anchor_count),
                "aligned_external_topics": int(len(aligned)),
                "relevant_unpaired_topics": int(len(unpaired)),
                "excluded_topics": int(len(excluded)),
            }
        )
    return pd.DataFrame(rows)


def render_public_safe_note(manifest: dict[str, Any]) -> None:
    public_safe = manifest.get("public_safe", True)
    snippet_chars = manifest.get("snippet_chars", "")
    if public_safe:
        st.sidebar.success(f"Public-safe snippets: {snippet_chars} characters")
    else:
        st.sidebar.warning("Local full-text mode is active.")


def render_app_guide() -> None:
    with st.sidebar.expander("How to use this app", expanded=False):
        st.markdown(
            """
            This is a companion to the paper, not a full data repository.

            Start with **Paper overview** for the study logic. Use **Domain overview** to understand each macro-topic. Use **Browse topics** for source-specific microtopics. Use **Corporate-external relations** to see how corporate anchors connect to academic and media discourse. **Advanced timing diagnostics** keeps only the simplified diagnostics reported in the paper.
            """
        )


def render_paper_overview(data: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="paper-hero">
            <h1>Sustainability Discourse Companion</h1>
            <p>
            This app helps readers inspect how corporate sustainability disclosure relates to academic
            and media discourse. It follows the paper's corporate-centered logic: corporate microtopics
            are anchors; external topics are interpreted as aligned counterparts, relevant unpaired
            signals, or excluded topics after review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for index, (label, value) in enumerate(FINAL_SAMPLE_METRICS.items()):
        with cols[index % 3]:
            metric_card(label, f"{value:,}")

    st.subheader("Reading path")
    step_cols = st.columns(len(WORKFLOW_STEPS))
    for col, (number, title, body) in zip(step_cols, WORKFLOW_STEPS):
        with col:
            note_card(f"{number}. {title}", body)

    st.subheader("Domain patterns at a glance")
    summary = domain_summary(data)
    display = summary[
        [
            "macro_topic",
            "macro_topic_name",
            "corporate_anchor_coverage",
            "aligned_external_topics",
            "relevant_unpaired_topics",
            "excluded_topics",
            "main_pattern",
        ]
    ].rename(
        columns={
            "macro_topic": "Code",
            "macro_topic_name": "Environmental domain",
            "corporate_anchor_coverage": "Anchor coverage",
            "aligned_external_topics": "Aligned external topics",
            "relevant_unpaired_topics": "Relevant unpaired",
            "excluded_topics": "Excluded",
            "main_pattern": "Main pattern",
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption(
        "Anchor coverage means corporate anchors with at least one retained academic or media counterpart over all corporate anchors in the domain."
    )


def render_domain_overview(data: dict[str, Any], macro_topic: str) -> None:
    summary = domain_summary(data)
    row = summary[summary["macro_topic"] == macro_topic].iloc[0]
    st.header(macro_option(macro_topic))
    st.caption(row["definition"])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Corporate anchor coverage", row["corporate_anchor_coverage"])
    col2.metric("Aligned external topics", f"{row['aligned_external_topics']:,}")
    col3.metric("Relevant unpaired", f"{row['relevant_unpaired_topics']:,}")
    col4.metric("Excluded after review", f"{row['excluded_topics']:,}")
    note_card(row["main_pattern"], row["how_to_read"])

    st.subheader("What this means in the paper")
    st.write(
        "The domain-level interpretation focuses on the form of corporate incorporation, not only on the number of topic matches. "
        "A domain may show broad incorporation, concentrated translation into a few corporate anchors, or external issues that remain visible outside the corporate frame."
    )

    st.subheader("Topics by review outcome")
    topics = data["topics"]
    filtered = topics[topics["macro_topic"] == macro_topic].copy()
    table = (
        filtered.groupby(["source", "paper_status"], dropna=False)
        .size()
        .reset_index(name="topics")
        .sort_values(["source", "paper_status"])
    )
    table["source"] = table["source"].map(lambda value: SOURCE_LABELS.get(value, value))
    table["paper_status"] = table["paper_status"].map(lambda value: STATUS_LABELS.get(value, value))
    st.dataframe(
        table.rename(columns={"source": "Source", "paper_status": "Review outcome", "topics": "Topics"}),
        hide_index=True,
        width="stretch",
    )


def render_topic_metrics(topic: pd.Series, series: pd.DataFrame) -> None:
    series = normalize_series(series)
    total_documents = int(series["year_document_count"].sum()) if not series.empty else 0
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


def render_topic_overview(topic: pd.Series) -> None:
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
        with st.expander("Evidence note", expanded=False):
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


def merge_components_table(merge_components: pd.DataFrame, topic: pd.Series) -> pd.DataFrame:
    topic_id = clean_text(topic.get("topic_id", ""))
    if merge_components.empty:
        return pd.DataFrame()
    components = merge_components[merge_components["topic_id"] == topic_id].copy()
    if components.empty:
        return pd.DataFrame()
    components["component_order"] = pd.to_numeric(components["component_order"], errors="coerce").fillna(0)
    components["component_count"] = pd.to_numeric(components["component_count"], errors="coerce")
    components = components.sort_values("component_order", kind="mergesort")
    return components[
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


def render_merge_composition(merge_components: pd.DataFrame, topic: pd.Series) -> None:
    table = merge_components_table(merge_components, topic)
    if table.empty:
        st.info("Merge-composition data is not available for this topic.")
        return
    group_size = int(numeric(topic.get("group_size", 0)))
    if group_size <= 1 or len(table) <= 1:
        st.info("Singleton topic: this consolidated topic contains one original microtopic.")
    else:
        st.caption(f"{len(table):,} original microtopics compose this consolidated topic.")
    st.dataframe(table, hide_index=True, width="stretch")


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


def render_evidence(evidence: pd.DataFrame, topic_id: str, series: pd.DataFrame, key_prefix: str = "") -> None:
    series = normalize_series(series)
    available_years = sorted(
        int(year)
        for year in pd.to_numeric(evidence[evidence["topic_id"] == topic_id]["year"], errors="coerce").dropna().unique()
    )
    if not available_years:
        st.info("No representative snippet data is available for this topic.")
        return
    if not series.empty and series["year_document_count"].max() > 0:
        peak_year = int(series.sort_values(["relative_share_percent", "year_document_count"], ascending=False).iloc[0]["year"])
    else:
        peak_year = available_years[-1]
    default_index = available_years.index(peak_year) if peak_year in available_years else len(available_years) - 1
    selected_year = st.selectbox("Evidence year", available_years, index=default_index, key=f"{key_prefix}_evidence_year")
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
    key_prefix: str = "topic",
) -> None:
    topic_id = clean_text(topic.get("topic_id", ""))
    series = override_series if override_series is not None else year_series[year_series["topic_id"] == topic_id]
    source = SOURCE_LABELS.get(clean_text(topic.get("source", "")), clean_text(topic.get("source", "")))
    status = STATUS_LABELS.get(clean_text(topic.get("paper_status", "")), clean_text(topic.get("paper_status", "")))
    st.subheader(topic_label(topic))
    st.caption(f"{source} · {status} · {clean_text(topic.get('final_merge_group_id', ''))}")
    render_topic_metrics(topic, series)
    overview_tab, temporal_tab, evidence_tab, merge_tab = st.tabs(
        ["Overview", "Temporal evolution", "Evidence", "Merged composition"]
    )
    with overview_tab:
        render_topic_overview(topic)
    with temporal_tab:
        st.caption("Annual prevalence is topic documents in a year divided by all documents from the same source and macro-topic in that year.")
        line_chart(series, "Temporal evolution")
    with evidence_tab:
        render_evidence(year_evidence, topic_id, series, key_prefix=key_prefix)
    with merge_tab:
        render_merge_composition(topic_merge_components, topic)


def topic_picker(frame: pd.DataFrame, data: dict[str, Any], key_prefix: str) -> pd.Series | None:
    if frame.empty:
        st.info("No topics are available for this selection.")
        return None
    query = st.text_input("Search topic labels", key=f"{key_prefix}_search")
    filtered = frame.copy()
    if query.strip():
        query_lower = query.strip().lower()
        filtered = filtered[
            filtered["display_label"].astype(str).str.lower().str.contains(query_lower, regex=False)
            | filtered["overall_summary"].astype(str).str.lower().str.contains(query_lower, regex=False)
        ]
    if filtered.empty:
        st.info("No topics match the search.")
        return None
    filtered = filtered.sort_values(["paper_status", "display_label"], kind="mergesort")
    lookup = {row.topic_id: row for row in filtered.itertuples(index=False)}
    selected_id = st.selectbox(
        "Select topic",
        filtered["topic_id"].tolist(),
        format_func=lambda topic_id: format_topic_option(pd.Series(lookup[topic_id]._asdict()), data["year_series"]),
        key=f"{key_prefix}_select",
    )
    return filtered[filtered["topic_id"] == selected_id].iloc[0]


def render_source_topic_browser(data: dict[str, Any], macro_topic: str, source: str) -> None:
    topics = data["topics"]
    filtered = topics[(topics["macro_topic"] == macro_topic) & (topics["source"] == source)].copy()
    if filtered.empty:
        st.info("No topics are available for this source and macro-topic.")
        return
    if source != "corporate":
        available_statuses = sorted(filtered["paper_status"].dropna().unique().tolist())
        default_statuses = [status for status in available_statuses if status != "excluded"] or available_statuses
        selected_statuses = st.multiselect(
            "Review outcome",
            available_statuses,
            default=default_statuses,
            format_func=lambda value: STATUS_LABELS.get(value, value),
            key=f"{macro_topic}_{source}_status",
        )
        filtered = filtered[filtered["paper_status"].isin(selected_statuses)]
    cols = st.columns(3)
    cols[0].metric("Topics shown", f"{len(filtered):,}")
    cols[1].metric("Aligned", f"{int((filtered['paper_status'] == 'aligned').sum()):,}")
    cols[2].metric("Unpaired", f"{int((filtered['paper_status'] == 'external_relevant_unpaired').sum()):,}")
    topic = topic_picker(filtered, data, f"{macro_topic}_{source}")
    if topic is not None:
        render_topic_detail(
            topic,
            data["year_series"],
            data["year_evidence"],
            data["topic_merge_components"],
            key_prefix=f"{macro_topic}_{source}_{clean_text(topic.get('topic_id', '')).replace(':', '_')}",
        )


def render_browse_topics(data: dict[str, Any], macro_topic: str) -> None:
    st.header("Browse topics")
    st.caption("Inspect corporate anchors and reviewed external topics source by source.")
    corporate_tab, academic_tab, media_tab = st.tabs(["Corporate", "Academic", "Media"])
    with corporate_tab:
        render_source_topic_browser(data, macro_topic, "corporate")
    with academic_tab:
        render_source_topic_browser(data, macro_topic, "academic")
    with media_tab:
        render_source_topic_browser(data, macro_topic, "media")


def render_relation_chart(panel_series: pd.DataFrame, corporate_topic_id: str) -> None:
    panel = panel_series[panel_series["corporate_topic_id"] == corporate_topic_id].copy()
    if panel.empty:
        st.info("No relation panel series is available for this corporate anchor.")
        return
    panel["series"] = panel["series_role"].map(SERIES_LABELS).fillna(panel["series_role"])
    line_chart(panel, "Corporate anchor and external aggregate series", color_column="series", height=390)


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
        if role == "corporate":
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


def render_counterpart_cards(matches: pd.DataFrame, source: str) -> None:
    source_matches = matches[matches["source_noncorporate"] == source].copy()
    label = SOURCE_LABELS[source]
    st.markdown(f"**{label} counterparts**")
    if source_matches.empty:
        st.info(f"No retained {label.lower()} counterpart for this corporate anchor.")
        return
    source_matches = source_matches.sort_values("topic_label_refined_noncorporate", kind="mergesort")
    for row in source_matches.itertuples(index=False):
        title = f"{row.topic_label_refined_noncorporate} ({int(numeric(row.external_unique_document_count)):,} docs)"
        with st.expander(title, expanded=False):
            st.write(clean_text(row.overall_summary_noncorporate))
            col1, col2 = st.columns(2)
            col1.metric("Best cosine", decimal_text(row.best_cosine_similarity))
            col2.metric("Direct pairs", integer_text(row.direct_pair_count))


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
        format_func=lambda topic_id: format_topic_option(
            option_rows[option_rows["topic_id"] == topic_id].iloc[0],
            data["year_series"],
        ),
        key=f"relation_evidence_{clean_text(anchor['topic_id'])}",
    )
    topic = option_rows[option_rows["topic_id"] == selected].iloc[0]
    series = data["year_series"][data["year_series"]["topic_id"] == selected]
    render_evidence(data["year_evidence"], selected, series, key_prefix=f"relation_{selected.replace(':', '_')}")


def render_unpaired_section(data: dict[str, Any], macro_topic: str) -> None:
    st.divider()
    st.subheader("Relevant external signals outside the corporate frame")
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
    source_filter = st.multiselect(
        "External source",
        sorted(unpaired["source"].unique().tolist()),
        default=sorted(unpaired["source"].unique().tolist()),
        format_func=lambda value: SOURCE_LABELS.get(value, value),
        key=f"{macro_topic}_unpaired_sources",
    )
    unpaired = unpaired[unpaired["source"].isin(source_filter)]
    topic = topic_picker(unpaired, data, f"{macro_topic}_unpaired")
    if topic is None:
        return
    series = data["unpaired_series"][data["unpaired_series"]["topic_id"] == topic["topic_id"]]
    render_topic_detail(
        topic,
        data["year_series"],
        data["year_evidence"],
        data["topic_merge_components"],
        override_series=series,
        key_prefix=f"unpaired_{clean_text(topic['topic_id']).replace(':', '_')}",
    )


def render_relations_mode(data: dict[str, Any], macro_topic: str) -> None:
    st.header("Corporate-external relations")
    st.caption("Select a corporate anchor to see the academic and media topics retained as counterparts.")
    topics = data["topics"]
    corporate = topics[(topics["macro_topic"] == macro_topic) & (topics["source"] == "corporate")].copy()
    if corporate.empty:
        st.info("No corporate anchors are available for this macro-topic.")
        return
    corporate = corporate.sort_values("display_label", kind="mergesort")
    selected_anchor = st.selectbox(
        "Corporate anchor",
        corporate["topic_id"].tolist(),
        format_func=lambda topic_id: format_topic_option(
            corporate[corporate["topic_id"] == topic_id].iloc[0],
            data["year_series"],
        ),
        key=f"{macro_topic}_relation_anchor",
    )
    anchor = corporate[corporate["topic_id"] == selected_anchor].iloc[0]
    matches = data["relations"][data["relations"]["corporate_topic_id"] == selected_anchor].copy()

    st.subheader(topic_label(anchor))
    st.caption(f"Corporate anchor · {clean_text(anchor.get('final_merge_group_id', ''))}")
    if matches.empty:
        st.warning("This is a corporate-only anchor in the final reviewed sample.")
    render_relation_metrics(data["panel_series"], selected_anchor)
    render_relation_chart(data["panel_series"], selected_anchor)

    interpretation = relation_interpretation(data, topic_label(anchor))
    if interpretation:
        with st.expander("Longitudinal interpretation", expanded=True):
            st.write(interpretation)

    col1, col2 = st.columns(2)
    with col1:
        render_counterpart_cards(matches, "academic")
    with col2:
        render_counterpart_cards(matches, "media")
    if not matches.empty:
        render_anchor_evidence_picker(data, anchor, matches)
    render_unpaired_section(data, macro_topic)


def relation_option(frame: pd.DataFrame, relation_id: str) -> str:
    row = frame[frame["relation_id"] == relation_id].iloc[0]
    external_label = clean_text(row.get("external_topic_label", "")) or clean_text(row.get("external_relation_label", ""))
    corporate_label = clean_text(row.get("corporate_topic_label", ""))
    source = SOURCE_LABELS.get(clean_text(row.get("external_source", "")), clean_text(row.get("external_source", "")))
    return (
        f"{corporate_label} x {source}: {external_label} · "
        f"Spearman={decimal_text(row.get('spearman_r'), 2)} · "
        f"peak gap={decimal_text(row.get('peak_year_gap'), 0)}"
    )


def relation_display_table(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    columns = [
        "corporate_topic_label",
        "external_source",
        "external_relation_label",
        "corporate_unique_document_count",
        "external_unique_document_count",
        "first_active_year_gap",
        "peak_year_gap",
        "spearman_r",
        "spearman_p",
    ]
    if level == "individual":
        columns.insert(3, "external_topic_label")
    available = [column for column in columns if column in frame.columns]
    table = frame[available].copy()
    for column in ["first_active_year_gap", "peak_year_gap", "spearman_r", "spearman_p"]:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(3)
    if "external_source" in table.columns:
        table["external_source"] = table["external_source"].map(
            lambda value: SOURCE_LABELS.get(clean_text(value), clean_text(value))
        )
    return table.rename(
        columns={
            "corporate_topic_label": "Corporate anchor",
            "external_source": "External source",
            "external_relation_label": "External relation",
            "external_topic_label": "External topic",
            "corporate_unique_document_count": "Corporate docs",
            "external_unique_document_count": "External docs",
            "first_active_year_gap": "First active-year gap",
            "peak_year_gap": "Peak-year gap",
            "spearman_r": "Same-year Spearman r",
            "spearman_p": "Spearman p",
        }
    )


def render_relation_method_guide() -> None:
    with st.expander("How to read these diagnostics", expanded=True):
        st.markdown(
            """
            These diagnostics are descriptive, not causal tests.

            **Same-year Spearman** measures whether the corporate and external annual prevalence series move in the same direction in the same year.

            **First active-year gap** is `corporate first active year - external first active year`. Positive values mean the external topic appeared earlier.

            **Peak-year gap** is `corporate peak year - external peak year`. Positive values mean the external topic peaked earlier.
            """
        )


def selected_relation_plain_language(row: pd.Series) -> str:
    spearman = numeric(row.get("spearman_r"), default=float("nan"))
    first_gap = numeric(row.get("first_active_year_gap"), default=float("nan"))
    peak_gap = numeric(row.get("peak_year_gap"), default=float("nan"))
    if pd.isna(spearman):
        movement = "Same-year co-movement could not be estimated because the series were too sparse or constant."
    elif spearman > 0:
        movement = "The two series tend to move in the same direction in the same year."
    elif spearman < 0:
        movement = "The two series tend to move in opposite directions in the same year."
    else:
        movement = "The two series show no monotonic same-year association."

    timing_parts = []
    if not pd.isna(first_gap):
        if first_gap > 0:
            timing_parts.append(f"the external topic appeared {int(first_gap)} year(s) before corporate")
        elif first_gap < 0:
            timing_parts.append(f"corporate appeared {int(abs(first_gap))} year(s) before the external topic")
        else:
            timing_parts.append("both topics first appeared in the same year")
    if not pd.isna(peak_gap):
        if peak_gap > 0:
            timing_parts.append(f"the external topic peaked {int(peak_gap)} year(s) before corporate")
        elif peak_gap < 0:
            timing_parts.append(f"corporate peaked {int(abs(peak_gap))} year(s) before the external topic")
        else:
            timing_parts.append("both topics peaked in the same year")
    timing = "Timing: " + "; ".join(timing_parts) + "." if timing_parts else "Timing diagnostics are unavailable."
    return f"{movement} {timing}"


def render_selected_relation_details(row: pd.Series) -> None:
    st.markdown("**Plain-language reading**")
    st.write(selected_relation_plain_language(row))
    detail_columns = [
        "corporate_first_active_year",
        "external_first_active_year",
        "first_active_year_gap",
        "corporate_peak_year",
        "external_peak_year",
        "peak_year_gap",
        "spearman_r",
        "spearman_p",
    ]
    details = pd.DataFrame(
        [{"Metric": column, "Value": row.get(column, "")} for column in detail_columns if column in row.index]
    )
    st.dataframe(details, hide_index=True, width="stretch")
    st.caption("Gap variables are computed as corporate year minus external year. Positive gaps mean external-before-corporate.")


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
    return filtered[filtered["external_source"].isin(selected_sources)].copy()


def render_relation_summary_tab(summary: pd.DataFrame, macro_topic: str, level: str) -> None:
    filtered = filter_relation_frame(summary, macro_topic, level)
    if filtered.empty:
        st.info("No timing-diagnostic rows are available for this macro-topic and filter.")
        return
    filtered = filtered.sort_values(
        ["external_source", "corporate_topic_label", "external_relation_label"],
        kind="mergesort",
    )
    if level == "aggregate":
        st.caption("Aggregate rows compare a corporate anchor with the union of all retained academic or media counterparts.")
    else:
        st.caption("Individual rows compare a corporate anchor with one specific academic or media counterpart.")
    st.dataframe(relation_display_table(filtered, level), hide_index=True, width="stretch")
    selected_relation = st.selectbox(
        "Inspect relation",
        filtered["relation_id"].tolist(),
        format_func=lambda relation_id: relation_option(filtered, relation_id),
        key=f"{level}_relation_select",
    )
    row = filtered[filtered["relation_id"] == selected_relation].iloc[0]
    cols = st.columns(4)
    cols[0].metric("Same-year Spearman r", decimal_text(row.get("spearman_r")))
    cols[1].metric("Spearman p", decimal_text(row.get("spearman_p")))
    cols[2].metric("First active-year gap", decimal_text(row.get("first_active_year_gap"), 0))
    cols[3].metric("Peak-year gap", decimal_text(row.get("peak_year_gap"), 0))
    render_selected_relation_details(row)


def render_temporal_relations_mode(data: dict[str, Any], macro_topic: str) -> None:
    st.header("Advanced timing diagnostics")
    st.caption("Simplified timing diagnostics retained for the paper: same-year Spearman, first active-year gap, and peak-year gap.")
    aggregate = data["source_relation_aggregate_same_year"]
    individual = data["source_relation_individual_same_year"]
    aggregate_macro = aggregate[aggregate["macro_topic"] == macro_topic].copy()
    individual_macro = individual[individual["macro_topic"] == macro_topic].copy()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Aggregate relations", f"{len(aggregate_macro):,}")
    col2.metric("Individual relations", f"{len(individual_macro):,}")
    aggregate_p = pd.to_numeric(aggregate_macro.get("spearman_p", pd.Series(dtype=float)), errors="coerce")
    individual_p = pd.to_numeric(individual_macro.get("spearman_p", pd.Series(dtype=float)), errors="coerce")
    col3.metric("Aggregate p < .10", f"{int((aggregate_p < 0.10).sum()):,}")
    col4.metric("Individual p < .10", f"{int((individual_p < 0.10).sum()):,}")
    render_relation_method_guide()
    aggregate_tab, individual_tab = st.tabs(["Aggregate source relations", "Individual topic relations"])
    with aggregate_tab:
        render_relation_summary_tab(aggregate, macro_topic, "aggregate")
    with individual_tab:
        render_relation_summary_tab(individual, macro_topic, "individual")


def main() -> None:
    st.set_page_config(
        page_title="Sustainability Discourse Companion",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_style()

    data_dir = Path(os.environ.get("SUSTAINABILITY_APP_DATA", DEFAULT_DATA_DIR))
    data, missing = load_app_data(str(data_dir))
    if missing:
        st.error("The Streamlit app data files are missing.")
        st.code(
            "python scripts/build_streamlit_app_data.py "
            "--pipeline-root /path/to/final_pipeline_workspace "
            "--output-dir data/app_data --public-safe",
            language="bash",
        )
        st.write(f"Missing files in `{data_dir}`: {', '.join(missing)}")
        st.stop()

    render_public_safe_note(data["manifest"])
    render_app_guide()
    view_mode = st.sidebar.radio(
        "Reader path",
        [
            "Paper overview",
            "Domain overview",
            "Browse topics",
            "Corporate-external relations",
            "Advanced timing diagnostics",
        ],
    )
    needs_macro = view_mode != "Paper overview"
    macro_topic = "T1"
    if needs_macro:
        macro_topic = st.sidebar.selectbox("Macro-topic", MACRO_TOPIC_ORDER, format_func=macro_option)

    if view_mode == "Paper overview":
        render_paper_overview(data)
    elif view_mode == "Domain overview":
        render_domain_overview(data, macro_topic)
    elif view_mode == "Browse topics":
        st.header(macro_option(macro_topic))
        render_browse_topics(data, macro_topic)
    elif view_mode == "Corporate-external relations":
        st.header(macro_option(macro_topic))
        render_relations_mode(data, macro_topic)
    else:
        st.header(macro_option(macro_topic))
        render_temporal_relations_mode(data, macro_topic)


if __name__ == "__main__":
    main()
