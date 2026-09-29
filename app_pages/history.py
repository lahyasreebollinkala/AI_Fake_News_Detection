"""Prediction log -- every analysis this session has recorded, on disk."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.cards import kpi_strip
from app.components.charts import history_trend, label_split_chart
from app.components.layout import empty_state, render_page_header
from app.services.history import (
    clear_history,
    filter_history,
    load_history,
    remove_entry,
    summarize,
    to_csv_bytes,
)
from src.config import PREDICTION_HISTORY

render_page_header(
    "Prediction log",
    "A local audit trail of every article you have analysed, newest first. It lives in "
    f"`{PREDICTION_HISTORY.name}` and never leaves this machine.",
)

records = load_history()

if records.empty:
    empty_state(
        "history",
        "No analyses recorded yet",
        "Run a prediction from **Analyze article** or **Batch analysis** and it will appear "
        "here automatically.",
        hint="The log holds the most recent 5,000 rows and is trimmed on every write.",
    )
    st.stop()

stats = summarize(records)

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
kpi_strip(
    [
        ("Analyses", f"{stats['total']:,}"),
        ("Flagged fake", f"{stats['fake']:,}"),
        ("Cleared real", f"{stats['real']:,}"),
        ("Mean confidence", f"{stats['mean_confidence']:.1f}%"),
    ],
    caption=(
        f"{stats['fake_share']:.1f}% of the log is flagged as fabricated and "
        f"{stats['low_confidence']:,} entries sit below the 80% confidence threshold. "
        f"Storage: {PREDICTION_HISTORY}"
    ),
)

st.space("small")
trend_left, trend_right = st.columns([1, 2], gap="large")

with trend_left:
    with st.container(border=True):
        label_split_chart(
            {"Real": stats["real"], "Fake": stats["fake"]},
            title="Logged verdicts",
        )

with trend_right:
    with st.container(border=True):
        st.markdown("**Confidence over time**")
        history_trend(records)
        st.caption("Every logged analysis, oldest to newest.")

# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------
st.subheader("Entries")

verdict_filter = st.segmented_control(
    "Verdict", ["All", "Real", "Fake", "Low confidence"], default="All", key="log_filter"
) or "All"

filter_left, filter_right = st.columns([1, 2], gap="medium")

with filter_left:
    source_filter = st.selectbox(
        "Source", ["All"] + sorted(records["Source"].dropna().unique().tolist()), key="log_source"
    )
with filter_right:
    search = st.text_input("Search", placeholder="Filter by headline or source", key="log_search")

view = filter_history(
    records,
    query=search,
    verdicts=None if verdict_filter == "All" else [verdict_filter],
)

if verdict_filter == "Low confidence":
    view = view[pd.to_numeric(view["Confidence"], errors="coerce").fillna(0) < 80]

if source_filter != "All":
    view = view[view["Source"] == source_filter]

st.caption(f"Showing {len(view):,} of {len(records):,} entries.")

# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------
display = view.copy()
display["Title"] = display["Title"].fillna("Untitled").str.slice(0, 90)

event = st.dataframe(
    display,
    hide_index=True,
    height=430,
    on_select="rerun",
    selection_mode="single-row",
    key="log_table",
    column_config={
        "Confidence": st.column_config.ProgressColumn(
            "Confidence", min_value=0, max_value=100, format="%.1f%%"
        ),
        "Real score": st.column_config.NumberColumn("P (real)", format="%.1f%%"),
        "Fake score": st.column_config.NumberColumn("P (fake)", format="%.1f%%"),
        "Words": st.column_config.NumberColumn(format="%d"),
        "Title": st.column_config.TextColumn("Headline", width="large"),
    },
)

selected = list(event.selection.rows) if event and event.selection else []

# --------------------------------------------------------------------------
# Detail + maintenance
# --------------------------------------------------------------------------
detail_left, action_left = st.columns([3, 1], gap="large")

with detail_left:
    with st.container(border=True):
        st.markdown("**Selected entry**")
        if selected:
            row = view.iloc[selected[0]]
            label = str(row["Prediction"])
            st.badge(label, color="red" if label == "Fake" else "green")
            st.markdown(f"#### {row['Title'] or 'Untitled'}")
            st.caption(
                f"{row['Timestamp']} · {row['Source']} · {row['Model']} · "
                f"{int(pd.to_numeric(pd.Series([row['Words']]), errors='coerce').fillna(0).iloc[0]):,} words"
            )
            bar_left, bar_right = st.columns(2)
            with bar_left:
                st.progress(
                    float(row["Real score"]) / 100, text=f"P (real) {float(row['Real score']):.1f}%"
                )
            with bar_right:
                st.progress(
                    float(row["Fake score"]) / 100, text=f"P (fake) {float(row['Fake score']):.1f}%"
                )
        else:
            st.caption("Select a row above to inspect the probabilities it recorded.")

with action_left:
    with st.container(border=True):
        st.markdown("**Export & maintenance**")
        st.download_button(
            "Download full log",
            data=to_csv_bytes(records),
            file_name="veritas_prediction_log.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
            key="log_download",
        )
        if selected:
            st.button(
                "Delete selected",
                icon=":material/delete:",
                width="stretch",
                on_click=remove_entry,
                args=(view.index[selected[0]],),
                key="log_delete",
            )
        st.button(
            "Clear all entries",
            icon=":material/delete_sweep:",
            width="stretch",
            on_click=clear_history,
            key="log_clear",
        )
        st.caption("Deletion is immediate and cannot be undone.")

# --------------------------------------------------------------------------
# Full log
# --------------------------------------------------------------------------
with st.expander("Show the complete log as a table", icon=":material/table_rows:"):
    st.dataframe(records, hide_index=True, height=320)
    st.caption(f"{len(records):,} rows · {PREDICTION_HISTORY}")
