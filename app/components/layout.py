"""Reusable layout fragments: page headers, the sidebar shell and empty states.

The visual system is expressed with native Streamlit primitives (bordered
containers, metric cards, badges, Material icons) so light and dark mode both
work without a single line of custom CSS.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

APP_NAME = "Veritas"
APP_TAGLINE = "News authenticity analyzer"
APP_VERSION = "3.0"


def render_brand() -> None:
    """Product mark shown at the top of the sidebar."""
    st.markdown(f"### :material/travel_explore: {APP_NAME}")
    st.caption(APP_TAGLINE)


def render_page_header(title: str, subtitle: str) -> None:
    """Standard page heading. Pages do not call ``st.title`` themselves."""
    st.title(title)
    st.caption(subtitle)


def render_section(label: str, icon: str = "") -> None:
    """A small, quiet section label used instead of heavy dividers."""
    text = f":material/{icon}: {label}" if icon else label
    st.markdown(f"**{text}**")


def empty_state(icon: str, title: str, body: str, *, hint: str | None = None) -> None:
    """Friendly placeholder for a panel that has nothing to show yet."""
    with st.container(border=True):
        st.markdown(f":material/{icon}:")
        st.markdown(f"**{title}**")
        st.caption(body)
        if hint:
            st.caption(f":material/lightbulb: {hint}")


def artifact_error(status: list[dict]) -> None:
    """Rendered when the trained artifacts are missing -- states the fix."""
    missing = sum(1 for row in status if not row["Present"])

    if missing:
        st.error(
            f"**{missing} of {len(status)} required artifacts are missing.** "
            "Veritas cannot score text until the pipeline has been built.",
            icon=":material/build_circle:",
        )
    else:
        st.info(
            "Every artifact is present, but the trained pipeline failed to load. "
            "Re-run the training stage to regenerate it.",
            icon=":material/error:",
        )

    with st.container(border=True):
        st.caption(
            "Veritas needs the trained pipeline before it can score text. "
            "Run these stages from the project root, then reload this page. "
            "The first stage takes about a minute; the rest are seconds."
        )
        st.code(
            "python -m src.prepare_dataset\n"
            "python -m src.preprocessing\n"
            "python -m src.feature_engineering\n"
            "python -m src.train_models",
            language="console",
        )

    st.dataframe(
        pd.DataFrame(status),
        hide_index=True,
        column_config={"Present": st.column_config.TextColumn("Status")},
    )


def sidebar_model_card(bundle, scores: dict) -> None:
    """Compact model summary that lives in the sidebar."""
    with st.container(border=True):
        st.markdown("**Active model**")
        st.markdown(f":material/model_training: {bundle.best_model_name}")
        accuracy = scores.get("accuracy")
        if accuracy is not None:
            st.caption(f"Accuracy {accuracy * 100:.2f}%  ·  F1 {scores.get('f1_score', 0) * 100:.2f}%")
        representation = getattr(bundle, "representation", None)
        label = f"{representation} features" if representation else "features"
        st.caption(f"{bundle.vocabulary_size:,} {label}")


def render_footer() -> None:
    """Project footer rendered at the bottom of every page."""
    st.caption(
        f"{APP_NAME} v{APP_VERSION}  ·  Python · Streamlit · scikit-learn · NLTK · Altair  ·  "
        "Built for educational and research use."
    )
