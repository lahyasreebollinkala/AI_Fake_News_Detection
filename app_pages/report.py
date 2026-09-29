"""Report -- the full IEEE-formatted write-up, rendered inside the app.

The same sections are written to ``docs/ieee_report.md`` and
``docs/ieee_report.tex`` by ``python -m src.report``. Both are built from
``src.report_sections.build_sections``, so what is read here and what is in the
file are guaranteed to be the same document.
"""

from __future__ import annotations

import streamlit as st

from app.components.layout import empty_state, render_page_header
from src.config import METRICS_FILE, REPORT_HTML, REPORT_LATEX, REPORT_MARKDOWN, REPORT_PDF
from src.data_loader import load_profile
from src.report_sections import INDEX_TERMS, abstract, build_sections

metrics = {}
if METRICS_FILE.exists():
    import json

    with open(METRICS_FILE, encoding="utf-8") as handle:
        metrics = json.load(handle)

profile = load_profile()
sections = build_sections(metrics, profile)

render_page_header(
    "Report",
    "The complete write-up in IEEE structure: abstract, eight numbered sections, tables "
    "read from `results/`, and a figure gallery. Regenerate the files with "
    "`python -m src.report`.",
)

if not metrics:
    empty_state(
        "description",
        "Design documented, results pending",
        "No `results/model_metrics.json` yet. Run `python -m src.train_models` to add the "
        "measured tables; the design sections below are already complete.",
        icon=":material/description:",
    )

# --------------------------------------------------------------------------
# Abstract
# --------------------------------------------------------------------------
summary = abstract(metrics)
if summary:
    st.subheader("Abstract")
    st.markdown(summary.replace(" -- ", " — "))
    st.caption(f"**Index Terms** — {INDEX_TERMS}")

st.divider()

# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
TOC = " · ".join(f"{s['roman']}. {s['title']}" for s in sections)
st.caption(TOC)

for section in sections:
    with st.expander(f"{section['roman']}. {section['title']}", expanded=section["id"] == "introduction"):
        for block in section["blocks"]:
            kind = block["kind"]
            if kind == "prose":
                st.markdown(block["text"])
            elif kind == "list":
                st.markdown(f"**{block['title']}**")
                for item in block["items"]:
                    st.markdown(f"- {item}")
            elif kind == "table":
                st.caption(block["caption"])
                st.dataframe(block["rows"], width="stretch", hide_index=True)
            elif kind == "metric":
                st.info(block["caption"], icon=":material/insights:")

# --------------------------------------------------------------------------
# Figure gallery
# --------------------------------------------------------------------------
from src.report import FIGURE_GALLERY  # noqa: E402
from src.config import IMAGE_DIR  # noqa: E402

gallery = [(name, caption) for name, caption in FIGURE_GALLERY if (IMAGE_DIR / name).exists()]
if gallery:
    st.divider()
    st.subheader("Figure gallery")
    for name, caption in gallery:
        st.image(str(IMAGE_DIR / name), width="stretch")
        st.caption(caption)

# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
st.divider()
st.subheader("Source files")

downloads = []
if REPORT_MARKDOWN.exists():
    downloads.append(
        ("ieee_report.md", REPORT_MARKDOWN.read_bytes(), REPORT_MARKDOWN.name, "text/markdown")
    )
if REPORT_LATEX.exists():
    downloads.append(
        ("ieee_report.tex", REPORT_LATEX.read_bytes(), REPORT_LATEX.name, "text/x-tex")
    )
if REPORT_HTML.exists():
    downloads.append(
        ("ieee_report.html", REPORT_HTML.read_bytes(), REPORT_HTML.name, "text/html")
    )
if REPORT_PDF.exists():
    downloads.append(("ieee_report.pdf", REPORT_PDF.read_bytes(), REPORT_PDF.name, "application/pdf"))

if downloads:
    for label, data, filename, mime in downloads:
        st.download_button(f"Download {label}", data=data, file_name=filename, mime=mime)
    st.caption(
        "`ieee_report.tex` is written in the `IEEEtran` class; `python -m src.report` "
        "compiles it with `pdflatex` when a TeX distribution is installed, and otherwise "
        "prints the two-column HTML build with a headless browser. Either way all four "
        "renderings come from the same sections."
    )
else:
    st.caption("Run `python -m src.report` to write the report to `docs/`.")
