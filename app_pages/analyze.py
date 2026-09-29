"""Analyze article -- the primary detection flow."""

from __future__ import annotations

import json

import streamlit as st

from app.components.cards import (
    evidence_columns,
    signal_table,
    text_statistics_table,
    term_attribution_chart,
    verdict_panel,
)
from app.components.layout import empty_state, render_page_header
from app.samples import SAMPLES
from app.services.artifacts import load_bundle, load_dataset
from app.services.history import append_record, build_record
from app.services.inference import analyze

bundle = load_bundle()

render_page_header(
    "Analyze an article",
    "Paste a headline and body, or load an example. The verdict appears together with the "
    "evidence that produced it.",
)


# --------------------------------------------------------------------------
# Rendering helpers (declared before use -- pages run top to bottom)
# --------------------------------------------------------------------------
def analysis_payload(verdict, model_name: str) -> str:
    """Serialise one analysis for download."""
    return json.dumps(
        {
            "model": model_name,
            "prediction": verdict.label_name,
            "confidence": round(verdict.confidence, 4),
            "probabilities": {name: round(value, 4) for name, value in verdict.probabilities.items()},
            "statistics": verdict.statistics,
            "signals": verdict.signals,
            "top_terms": verdict.terms,
        },
        indent=2,
    )


def render_result(verdict, model_name: str, reference_row) -> None:
    """Verdict card, attribution, document statistics and ground-truth check."""
    st.subheader("Result")
    verdict_panel(verdict, model_name)

    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            st.markdown("**Why the model decided this**")
            term_attribution_chart(verdict)

    with right:
        with st.container(border=True):
            st.markdown("**What the text looks like**")
            stats_tab, signals_tab, cleaned_tab = st.tabs(["Statistics", "Signals", "Cleaned text"])

            with stats_tab:
                st.dataframe(text_statistics_table(verdict.statistics), hide_index=True)

            with signals_tab:
                st.dataframe(signal_table(verdict.signals), hide_index=True)

            with cleaned_tab:
                st.caption("Exactly the string handed to the TF-IDF vectoriser.")
                st.code(verdict.cleaned_text[:2500] or "(empty)", language="text")

    if reference_row is not None:
        truth = "Fake" if int(reference_row.get("label", 0)) == 1 else "Real"
        agreed = truth == verdict.label_name
        (st.success if agreed else st.error)(
            f"Corpus ground truth: **{truth}** — the model {'agreed' if agreed else 'disagreed'}."
        )

    with st.expander("Editorial signals against the corpus average", icon=":material/insights:"):
        evidence_columns(verdict)

    st.download_button(
        "Download this analysis as JSON",
        data=analysis_payload(verdict, model_name),
        file_name="veritas_analysis.json",
        mime="application/json",
        icon=":material/download:",
        key="analyze_download",
    )


# --------------------------------------------------------------------------
# Input source
# --------------------------------------------------------------------------
source = st.segmented_control(
    "Article source",
    ["Write or paste", "Curated example", "Labelled corpus row"],
    default="Write or paste",
    label_visibility="collapsed",
    key="analyze_source",
)

reference_row = None

if source == "Curated example":
    keys = [sample.key for sample in SAMPLES]
    choice = st.selectbox(
        "Example article",
        options=range(len(keys)),
        format_func=lambda index: keys[index],
        key="analyze_example",
    )
    sample = SAMPLES[choice]

    preview, expectation = st.columns([3, 2])
    with preview:
        st.markdown(f"**{sample.title}**")
        st.caption(sample.body[:280] + ("…" if len(sample.body) > 280 else ""))
    with expectation:
        st.markdown("**What to look for**")
        st.caption(sample.note)

    st.caption(":gray-badge[Illustrative sample written for demonstration, not a real article]")
    st.session_state["draft_title"] = sample.title
    st.session_state["draft_body"] = sample.body
    st.session_state["last_source"] = f"Example · {sample.key}"

elif source == "Labelled corpus row":
    try:
        corpus = load_dataset()
    except FileNotFoundError:
        corpus = None

    if corpus is None or corpus.empty:
        empty_state(
            "dataset",
            "No corpus available",
            "Run `python -m src.prepare_dataset` and `python -m src.preprocessing` to build "
            "the training corpus, then reload this page.",
        )
        st.stop()

    row_index = st.selectbox(
        "Corpus article",
        options=range(min(len(corpus), 500)),
        format_func=lambda index: str(corpus.iloc[index].get("title", ""))[:88],
        key="analyze_corpus",
    )
    reference_row = corpus.iloc[row_index]

    st.caption(
        "The true label stays hidden until you analyse, so a run doubles as an accuracy check."
    )
    st.session_state["draft_title"] = str(reference_row.get("title", ""))
    st.session_state["draft_body"] = str(reference_row.get("text", ""))
    st.session_state["last_source"] = "Corpus row"

# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------
headline_input, body_input = st.columns([1, 2], gap="medium")

with headline_input:
    title = st.text_input(
        "Headline",
        key="draft_title",
        placeholder="Optional, but headlines carry real signal",
        help="Leave empty to score the body alone.",
    )

with body_input:
    body = st.text_area(
        "Article body",
        key="draft_body",
        height=230,
        placeholder="Paste the full text of the article you want to verify…",
    )

action_left, action_right = st.columns([1, 3], gap="medium", vertical_alignment="center")
with action_left:
    submitted = st.button(
        "Analyze", type="primary", icon=":material/travel_explore:", key="analyze_run"
    )
with action_right:
    st.caption(
        "The model reads the headline and body together. Inputs under five words are "
        "rejected rather than guessed at."
    )

# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------
st.space("medium")

if not submitted:
    previous = st.session_state.get("last_verdict")
    if previous is not None:
        st.caption("Showing your previous analysis. Press **Analyze** to refresh it.")
        render_result(previous, bundle.best_model_name, None)
    else:
        empty_state(
            "search",
            "No analysis yet",
            "Add an article on the left and press **Analyze**. The verdict, the class split, "
            "the editorial signals and the term attribution appear here.",
            hint="Switch the source to *Curated example* to watch the model work on a known case.",
        )
    st.stop()

verdict = analyze(bundle, body, title)

if verdict is None:
    st.warning(
        "Add at least a few sentences of article text before analysing.",
        icon=":material/short_text:",
    )
    st.stop()

st.session_state["last_verdict"] = verdict
append_record(
    build_record(
        source=st.session_state.get("last_source", "Manual"),
        title=title,
        prediction=verdict.label_name,
        confidence=verdict.confidence,
        real_score=verdict.probabilities["Real"],
        fake_score=verdict.probabilities["Fake"],
        words=verdict.statistics.get("words", 0),
        model=bundle.best_model_name,
    )
)
st.toast(f"Logged as {verdict.label_name.lower()} news", icon=":material/bookmark_added:")

render_result(verdict, bundle.best_model_name, reference_row)
