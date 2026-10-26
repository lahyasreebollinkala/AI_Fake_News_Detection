"""Batch analysis -- score a CSV of articles in one pass."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.cards import kpi_strip
from app.components.charts import histogram, label_split_chart
from app.components.layout import empty_state, render_page_header
from app.services.artifacts import load_bundle
from app.services.inference import (
    LOW_CONFIDENCE_BAND,
    MIN_WORDS_FOR_PREDICTION,
    analyze_frame,
    prepare_batch_frame,
)

bundle = load_bundle()

render_page_header(
    "Batch analysis",
    "Upload a CSV, map the columns, and score every row. Results stay on screen and can be "
    "downloaded as a new CSV.",
)

MAX_ROWS = 20_000

# Suggested text columns, in the order they are offered first.
TEXT_HINTS = ("text", "article", "content", "body", "full_text", "headline", "title")
TITLE_HINTS = ("title", "headline", "headline_text")


def pick_column(columns: list[str], hints: tuple[str, ...], label: str, optional: bool = False):
    """Default to the first column whose name matches a hint."""
    lowered = {column.lower(): column for column in columns}
    default = None
    for hint in hints:
        if hint in lowered:
            default = lowered[hint]
            break
    options = ([None] if optional else []) + columns
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index, format_func=lambda c: c or "None")


def score(frame: pd.DataFrame, text_column: str, title_column: str | None):
    """Run the batch job inside a status block so progress is visible."""
    with st.status("Scoring articles…", expanded=True) as status:
        st.write(f"Vectorising and classifying {len(frame):,} rows.")
        result = analyze_frame(bundle, frame, text_column, title_column)
        status.update(label="Scored", state="complete")
    return result


uploaded = st.file_uploader(
    "Articles CSV",
    type=["csv"],
    help="Any CSV with one column of article text. An optional second column can hold headlines.",
    key="batch_upload",
)

result = None
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as error:  # noqa: BLE001 - surfaced verbatim in the UI
        st.error(f"That file could not be parsed as CSV: {error}", icon=":material/error:")
        st.stop()

    if raw.empty:
        st.warning("The uploaded file has a header but no rows.", icon=":material/table_rows:")
        st.stop()

    if raw.shape[1] < 1:
        st.error("The uploaded file has no columns.", icon=":material/error:")
        st.stop()

    st.caption(
        f"Read **{len(raw):,} rows** and **{raw.shape[1]} columns** "
        f"({uploaded.name}, {uploaded.size / 1024:.0f} KB)."
    )

    if len(raw) > MAX_ROWS:
        st.warning(
            f"Only the first {MAX_ROWS:,} rows will be scored. "
            "Narrow the file before uploading for a full pass.",
            icon=":material/filter_alt:",
        )

    mapping_left, mapping_right, mapping_run = st.columns([1, 1, 1], gap="medium", vertical_alignment="bottom")

    with mapping_left:
        text_column = pick_column(list(raw.columns), TEXT_HINTS, "Text column")
    with mapping_right:
        title_column = pick_column(list(raw.columns), TITLE_HINTS, "Headline column", optional=True)
    with mapping_run:
        run = st.button(
            "Score articles", type="primary", icon=":material/rocket_launch:", key="batch_run"
        )

    st.caption(
        "The headline column is optional. When supplied it is prepended to the body before "
        "cleaning, matching how the model was trained."
    )

    if run:
        try:
            prepared, skipped = prepare_batch_frame(
                raw.head(MAX_ROWS), text_column, title_column
            )
        except KeyError as error:
            st.error(str(error), icon=":material/error:")
            st.stop()

        if skipped:
            st.warning(
                f"{skipped:,} row(s) contained fewer than {MIN_WORDS_FOR_PREDICTION} "
                "words and were not scored. Empty and near-empty records are skipped "
                "instead of being given an unreliable verdict.",
                icon=":material/filter_alt:",
            )

        if prepared.empty:
            st.error(
                f"None of the uploaded rows contain at least {MIN_WORDS_FOR_PREDICTION} "
                "words of article text, so there is nothing to score.",
                icon=":material/error:",
            )
            st.stop()

        with st.spinner("Scoring…"):
            result = score(prepared, text_column, title_column)
        st.session_state["batch_result"] = result
        st.toast(f"Scored {result.total:,} articles", icon=":material/task_alt:")

elif "batch_result" in st.session_state:
    result = st.session_state["batch_result"]
    st.caption("Showing the previous batch run. Upload a new file to replace it.")

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
if result is not None:
    st.subheader("Outcome")
    kpi_strip(
        [
            ("Articles scored", f"{result.total:,}"),
            ("Real", f"{result.real:,}"),
            ("Fake", f"{result.fake:,}"),
            ("Mean confidence", f"{result.mean_confidence:.1f}%"),
        ],
        caption=(
            f"{result.low_confidence:,} article(s) landed below the "
            f"{LOW_CONFIDENCE_BAND:.0f}% confidence threshold and are flagged for review."
        ),
    )

    st.space("small")
    label_split_chart(
        {"Real": result.real, "Fake": result.fake}, title="Predicted split"
    )

    st.subheader("Inspect the scored rows")

    verdict_filter = st.segmented_control(
        "Verdict", ["All", "Real", "Fake", "Low confidence"], default="All", key="batch_filter"
    )
    search = st.text_input(
        "Search", placeholder="Filter by any text column", key="batch_search", label_visibility="collapsed"
    )

    view = result.frame
    if verdict_filter == "Real":
        view = view[view["Prediction"] == "Real"]
    elif verdict_filter == "Fake":
        view = view[view["Prediction"] == "Fake"]
    elif verdict_filter == "Low confidence":
        view = view[view["Verdict band"] == "Low"]

    needle = search.strip()
    if needle:
        haystack = view.astype(str).agg(" ".join, axis=1)
        view = view[haystack.str.contains(needle, case=False, regex=False)]

    st.caption(f"Showing {len(view):,} of {result.total:,} rows.")

    st.dataframe(
        view,
        hide_index=True,
        height=440,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Prediction": st.column_config.TextColumn("Verdict"),
            "Verdict band": st.column_config.TextColumn("Review?"),
        },
    )

    st.download_button(
        "Download scored CSV",
        data=result.frame.to_csv(index=False).encode("utf-8"),
        file_name="veritas_batch_predictions.csv",
        mime="text/csv",
        icon=":material/download:",
        key="batch_download",
    )

    st.space("small")
    with st.expander("Confidence distribution", icon=":material/query_stats:"):
        histogram(
            result.frame["Confidence"],
            title="How decisive the model was across the batch",
            x_title="Confidence (%)",
            bins=20,
        )

elif uploaded is None:
    empty_state(
        "upload_file",
        "No file loaded",
        "Drop a CSV above to score it. The file never leaves your machine — parsing and "
        "classification both happen in this process.",
        hint="A good batch file has one text column; a headline column improves accuracy.",
    )
