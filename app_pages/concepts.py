"""Concepts -- how AI, machine learning, NLP and evaluation fit together.

A companion page to the model lab: this one explains the ideas, without needing
the trained artifacts, so it stays readable even on a machine where the pipeline
has not been built yet.
"""

from __future__ import annotations

import streamlit as st

from app.components.layout import render_page_header
from src import concepts

render_page_header(
    "Concepts",
    "How the pieces of this project relate: the problem is an AI task, the solution is "
    "supervised machine learning, the input to that solution is natural language processing, "
    "and the only way to trust the result is honest evaluation.",
)

tabs = st.tabs(["Artificial intelligence", "Machine learning", "NLP", "Evaluation"])

with tabs[0]:
    st.markdown(concepts.AI_OVERVIEW)
    st.markdown(
        """
        #### Why this is a useful application of AI

        A rule-based detector would need someone to enumerate the phrases that indicate
        fabrication. That list is unwritable: the signals are distributional rather than
        lexical, and any fixed list is both incomplete and trivially evaded by rewording.
        A learned model instead picks up the regularity from the data itself, which is why
        the same pipeline keeps working on articles written after it was trained.
        """
    )

with tabs[1]:
    st.markdown(concepts.ML_OVERVIEW)
    st.markdown(
        """
        #### Supervised, and why the split matters

        Every label in this project is supplied by the source dataset, which makes the task
        *supervised*: the model is given the answer for each training article and learns the
        mapping from article to answer. The single most consequential engineering decision in
        the whole project is where that boundary between training and evaluation is drawn.
        Fitting the vocabulary or the embedding basis on the full corpus before splitting
        would leak information across the boundary, because the model would already have
        seen the test articles' vocabulary when it is scored on them. The reported numbers
        would then describe memorisation rather than generalisation, and they would be
        optimistic by an unknown amount. This pipeline therefore splits first and fits the
        representation second, and selects the model on a validation split so the test split
        is scored exactly once.
        """
    )

with tabs[2]:
    st.markdown(concepts.NLP_OVERVIEW)

    st.markdown("#### The preprocessing chain")
    for name, text in concepts.PREPROCESSING_STEPS:
        with st.expander(name):
            st.markdown(text)

    st.markdown("#### The three representations")
    for name, text in concepts.FEATURE_EXTRACTION_NOTES:
        with st.expander(name):
            st.markdown(text)

    st.caption(
        "NLTK supplies the tokeniser, the stopword list and the lemmatiser. "
        "`src/nlp_resources.py` downloads them on first use and falls back to static word "
        "lists and regular-expression patterns when the corpora are unavailable, so the "
        "pipeline is reproducible offline."
    )

with tabs[3]:
    st.markdown(concepts.EVALUATION_OVERVIEW)

    st.markdown("#### Why a single number is not enough")
    st.markdown(
        """
        A model that flagged every article as fabricated would reach 58% accuracy on this
        corpus, because 58% of the articles are fabricated. Accuracy alone would rank that
        useless model above one that misses a third of the fabrications. Precision and recall
        pull in opposite directions as the decision threshold moves, which is exactly why
        F1 -- their harmonic mean -- is used as the selection criterion, and why the ROC curve
        is reported alongside it: the curve shows the whole trade-off rather than a single
        operating point on it.
        """
    )

    st.markdown("#### What this benchmark can and cannot tell you")
    for item in concepts.LIMITS:
        st.markdown(f"- {item}")
