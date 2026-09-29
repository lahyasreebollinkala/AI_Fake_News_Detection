"""Veritas -- AI-powered news authenticity analyzer.

Entry point for the Streamlit app. Run with::

    streamlit run streamlit_app.py

This module owns the app shell: page configuration, shared session state, the
sidebar, and the navigation map. Each page lives in ``app_pages/`` and is a
plain script that only renders UI -- all business logic sits in ``app/services``
and the ML pipeline in ``src``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st  # noqa: E402

from app.components.layout import (  # noqa: E402
    artifact_error,
    render_brand,
    render_footer,
    sidebar_model_card,
)
from app.navigation import ROUTES, grouped  # noqa: E402
from app.services.artifacts import (  # noqa: E402
    artifact_status,
    load_bundle,
    model_ready,
    trained_on,
)

st.set_page_config(
    page_title="Veritas · News authenticity analyzer",
    page_icon=":material/travel_explore:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Shared session state -- initialised once, here
# --------------------------------------------------------------------------
st.session_state.setdefault("last_verdict", None)
st.session_state.setdefault("last_source", "Manual")
st.session_state.setdefault("draft_title", "")
st.session_state.setdefault("draft_body", "")
st.session_state.setdefault("corpus_choice", 0)
st.session_state.setdefault("sample_choice", 0)

# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------
page = st.navigation(
    {
        group: [
            st.Page(
                route.path,
                title=route.title,
                icon=f":material/{route.icon}:",
                default=route.default,
            )
            for route in routes
        ]
        for group, routes in grouped().items()
        if routes
    },
    position="sidebar",
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
status = artifact_status()
ready = model_ready()
bundle = load_bundle() if ready else None

with st.sidebar:
    render_brand()

    if bundle is not None:
        sidebar_model_card(bundle=bundle, scores=bundle.scores)
        st.caption(f"Model trained {trained_on()}.")


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
if ready:
    page.run()
else:
    route = next((r for r in ROUTES if r.title == page.title), None)
    if route is not None and not route.needs_model:
        page.run()
    elif route is not None:
        st.title(page.title)
        st.caption("Build the pipeline once, then this page becomes fully available.")
        artifact_error(status)
    else:
        st.title(page.title)
        st.warning("The trained model is required for this page.", icon=":material/build_circle:")
        artifact_error(status)

render_footer()
