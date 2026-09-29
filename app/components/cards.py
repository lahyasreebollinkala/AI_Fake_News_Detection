"""Card-style compositions used across the pages.

Each helper renders a complete, self-contained unit (KPI strip, verdict panel,
signal table, pipeline strip) so page files stay short and consistent.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import (
    probability_chart,
    signal_radar,
    term_contribution_chart,
)
from app.components.layout import empty_state
from app.services.inference import LOW_CONFIDENCE_BAND, LABEL_NAMES, Verdict

SIGNAL_LABELS = {
    "clickbait_score": "Clickbait pressure",
    "emotional_word_ratio": "Emotional language",
    "sentiment_score": "Tone",
    "readability_score": "Readability",
    "source_trust_score": "Source trust",
}

SIGNAL_NOTES = {
    "clickbait_score": "Headline phrases, shouting and exclamation marks",
    "emotional_word_ratio": "Share of loaded, alarmist words",
    "sentiment_score": "Negative values read as hostile framing",
    "readability_score": "Flesch Reading Ease; higher is plainer",
    "source_trust_score": "Neutral prior -- the corpus has no publisher identity",
}


def kpi_strip(items: list[tuple[str, str]], *, caption: str = "") -> None:
    """Responsive KPI row. Each item is ``(label, value)``."""
    with st.container(horizontal=True):
        for label, value in items:
            st.metric(label, value, border=True)
    if caption:
        st.caption(caption)


def pipeline_strip(representation: str = "TF-IDF") -> None:
    """The five transformation stages, shown as a numbered flow."""
    stages = [
        ("Normalise", "lowercase, strip URLs, HTML, digits, punctuation"),
        ("Tokenise", "drop stopwords and sub-3-character tokens"),
        ("Lemmatise", "reduce words to a shared dictionary form"),
        ("Represent", f"{representation} over the training vocabulary"),
        ("Classify", "model confidence for real vs fake"),
    ]

    with st.container(horizontal=True):
        for index, (name, detail) in enumerate(stages, start=1):
            with st.container(border=True):
                st.caption(f"Step {index}")
                st.markdown(f"**{name}**")
                st.caption(detail)


def verdict_panel(verdict: Verdict, model_name: str) -> None:
    """The primary result card: badge, confidence, split, evidence terms."""
    badge = f":{verdict.badge}-badge[{verdict.label_name} news]"

    with st.container(border=True):
        head, confidence, length = st.columns([2, 1, 1])

        with head:
            st.markdown("### Verdict")
            st.markdown(badge)
            st.caption(f"Scored by {model_name}")

        with confidence:
            st.metric("Confidence", f"{verdict.confidence * 100:.1f}%", border=False)
            if verdict.confidence * 100 < LOW_CONFIDENCE_BAND:
                st.caption("Low confidence -- the model is close to undecided.")

        with length:
            st.metric("Words", f"{verdict.statistics.get('words', 0):,}", border=False)
            st.caption(f"{verdict.statistics.get('sentences', 0):,} sentences")

        if verdict.warning:
            st.warning(verdict.warning, icon=":material/warning:")

        probability_chart(verdict.probabilities)


def term_attribution_chart(verdict: Verdict) -> None:
    """Diverging bar chart of the terms behind the verdict."""
    if not verdict.terms:
        empty_state(
            "search_off",
            "No attributable terms",
            "Every token in this document was removed during cleaning, so there is "
            "nothing left for the model to attribute the decision to.",
        )
        return

    term_contribution_chart(verdict.terms)


def evidence_columns(verdict: Verdict) -> None:
    """Signals and term attribution side by side."""
    terms, signals = st.columns(2)

    with terms:
        with st.container(border=True):
            st.markdown("**Terms that moved the decision**")
            if not verdict.terms:
                st.caption("No interpretable terms survived cleaning for this document.")
            else:
                frame = pd.DataFrame(verdict.terms)
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Term": frame["term"],
                            "Pulls toward": frame["direction"],
                            "Weight": frame["weight"].abs().round(4),
                        }
                    ),
                    hide_index=True,
                    column_config={"Weight": st.column_config.NumberColumn(format="%.4f")},
                )

    with signals:
        with st.container(border=True):
            st.markdown("**Editorial signals**")
            signal_radar(verdict.signals)
            st.caption(
                "Rule-based indicators computed from the raw text. They are shown "
                "for context -- the verdict itself comes from the trained model."
            )


def signal_table(signals: dict) -> pd.DataFrame:
    """Tidy frame of the editorial signals for display and download."""
    rows = []
    for key, label in SIGNAL_LABELS.items():
        if key not in signals:
            continue
        rows.append(
            {
                "Signal": label,
                "Value": round(float(signals[key]), 3),
                "How it is computed": SIGNAL_NOTES[key],
            }
        )
    return pd.DataFrame(rows)


def text_statistics_table(statistics: dict) -> pd.DataFrame:
    """Word / character / sentence counts for the analysed document."""
    return pd.DataFrame(
        [
            {"Measure": "Words", "Value": f"{statistics.get('words', 0):,}"},
            {"Measure": "Characters", "Value": f"{statistics.get('characters', 0):,}"},
            {"Measure": "Sentences", "Value": f"{statistics.get('sentences', 0):,}"},
            {
                "Measure": "Words per sentence",
                "Value": f"{statistics.get('avg_words_per_sentence', 0)}",
            },
        ]
    )


def confusion_frame(matrix: list[list[int]]) -> pd.DataFrame:
    """Reshape a stored confusion matrix into a display frame."""
    labels = [LABEL_NAMES[0], LABEL_NAMES[1]]
    return pd.DataFrame(
        [
            {
                "Actual": labels[row],
                "Predicted real": int(matrix[row][0]),
                "Predicted fake": int(matrix[row][1]),
            }
            for row in range(len(labels))
        ]
    )


def dataset_table(frame: pd.DataFrame, *, title: str | None = None) -> None:
    """Scrollable, formatted corpus browser."""
    if title:
        st.markdown(f"**{title}**")
    st.dataframe(frame, hide_index=True, height=420)
