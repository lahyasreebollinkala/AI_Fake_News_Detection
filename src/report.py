"""Render the IEEE-formatted report from the artifacts on disk.

Run with ``python -m src.report``. The output is two files in ``docs/``:

* ``ieee_report.md``  -- Markdown, also rendered inside the application
* ``ieee_report.tex`` -- LaTeX in the ``IEEEtran`` class

Both are generated from :func:`src.report_sections.build_sections`, so the report,
the dashboard and the last real training run always agree. Nothing here invents a
number: every statistic is read from ``results/``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from src.config import (
    DOC_DIR,
    IMAGE_DIR,
    METRICS_FILE,
    RANDOM_STATE,
    REPORT_HTML,
    REPORT_LATEX,
    REPORT_MARKDOWN,
    REPORT_PDF,
    RESULT_DIR,
)
from src.data_loader import load_profile
from src.report_sections import build_sections
from src import concepts
from src import report_pdf
from src import report_sections

TITLE = "AI-Powered Fake News Detection Using Text Classification"
AUTHORS = "Department of Computer Science and Engineering"

#: Figures embedded in the results section, in order.
FIGURE_GALLERY = [
    ("confusion_matrix.png", "Selected model confusion matrix"),
    ("confusion_matrices.png", "Confusion matrix for every algorithm"),
    ("roc_curve.png", "ROC curve for the selected model"),
    ("roc_comparison.png", "ROC curves for all four algorithms"),
    ("metric_comparison.png", "Hold-out metric comparison"),
    ("representation_comparison.png", "Representation x algorithm F1 grid"),
    ("feature_importance.png", "Strongest features of the selected model"),
    ("term_separation.png", "Terms that separate the two classes"),
    ("label_distribution.png", "Class distribution"),
    ("length_distribution.png", "Article length distribution"),
    ("average_length.png", "Mean length per class"),
    ("word_count_box.png", "Word count spread per class"),
    ("subject_distribution.png", "Articles per news desk"),
    ("engineered_features.png", "Engineered editorial signals per class"),
    ("feature_correlation.png", "Engineered feature correlation"),
]

_PARA_SPLIT = re.compile(r"\n\s*\n")


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

def _markdown(sections: list[dict], metrics: dict) -> str:
    out: list[str] = [
        f"# {TITLE}",
        "",
        f"*{AUTHORS}*",
        "",
        f"_Generated {datetime.now().strftime('%d %B %Y')} by "
        "`python -m src.report` from the artifacts in `results/`._",
        "",
    ]

    summary = report_sections.abstract(metrics)
    if summary:
        out += [
            "## Abstract",
            "",
            summary,
            "",
            f"**Index Terms** -- {report_sections.INDEX_TERMS}",
            "",
        ]
    for section in sections:
        out += [f"## {section['roman']}. {section['title']}", ""]
        for block in section["blocks"]:
            kind = block["kind"]
            if kind == "prose":
                out += [block["text"], ""]
            elif kind == "list":
                out.append(f"**{block['title']}**")
                out.append("")
                out += [f"- {item}" for item in block["items"]]
                out.append("")
            elif kind == "table":
                out += [
                    f"*{block['caption']}*",
                    "",
                    "| " + " | ".join(block["columns"]) + " |",
                    "|" + "|".join("---" for _ in block["columns"]) + "|",
                ]
                out += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in block["rows"]]
                out.append("")
            elif kind == "metric":
                out += [f"> {block['caption']}", ""]

    gallery = [
        (name, caption)
        for name, caption in FIGURE_GALLERY
        if (IMAGE_DIR / name).exists()
    ]
    if gallery:
        out += ["## Figure gallery", ""]
        for name, caption in gallery:
            out += [f"![{caption}](../images/{name})", "", f"*{caption}*", ""]

    out += [
        "## Reproduction",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python -m src.prepare_dataset",
        "python -m src.preprocessing",
        "python -m src.feature_engineering",
        "python -m src.train_models",
        "python -m src.evaluate_models",
        "python -m src.eda",
        "python -m src.eda_visualization",
        "python -m src.report",
        "streamlit run streamlit_app.py",
        "python -m pytest tests",
        "```",
        "",
        f"All randomness is seeded with `random_state={RANDOM_STATE}`, so a re-run on "
        f"the same corpus reproduces every figure in this document exactly.",
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------
# LaTeX (IEEEtran)
# --------------------------------------------------------------------------

def _tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def _tex_md(text: str) -> str:
    """Convert the small Markdown subset used by the report into LaTeX."""
    text = text.replace("**", "").replace("`", "").replace("*", "")
    lines = []
    for paragraph in _PARA_SPLIT.split(text.strip()):
        paragraph = paragraph.strip()
        if paragraph:
            lines.append(_tex_escape(paragraph))
    return "\n\n".join(lines)


def _tex_table(block: dict) -> list[str]:
    """Render one table block.

    Wide tables (six or more columns) are promoted to a two-column ``table*``
    so the numbers stay readable instead of overflowing the IEEE measure.
    """
    columns = len(block["columns"])
    wide = columns >= 6
    environment = "table*" if wide else "table"
    size = r"\footnotesize" if wide else r"\small"
    out = [
        f"\\begin{{{environment}}}[!t]",
        r"  \centering",
        f"  {size}",
        f"  \\caption{{{_tex_escape(block['caption'])}}}",
        "  \\label{tab:" + re.sub(r"[^a-z0-9]+", "", block["caption"].lower())[:28] + "}",
        "  \\begin{tabular}{" + "l" * columns + "}",
        "    \\toprule",
        "    " + " & ".join(_tex_escape(c) for c in block["columns"]) + r" \\",
        "    \\midrule",
    ]
    for row in block["rows"]:
        out.append("    " + " & ".join(_tex_escape(cell) for cell in row) + r" \\")
    out += ["    \\bottomrule", "  \\end{tabular}", f"\\end{{{environment}}}", ""]
    return out


def _tex_gallery() -> list[str]:
    """Emit every generated figure, as full-width floats."""
    gallery = [(name, caption) for name, caption in FIGURE_GALLERY if (IMAGE_DIR / name).exists()]
    if not gallery:
        return []

    out = ["% ---------------------------------------------------------------", "% Figure gallery", "% ---------------------------------------------------------------", ""]
    for name, caption in gallery:
        out += [
            r"\begin{figure*}[!t]",
            r"  \centering",
            "  \\includegraphics[width=0.9\\textwidth]{" + name + "}",
            "  \\caption{" + _tex_escape(caption) + "}",
            f"  \\label{{fig:{name.rsplit('.', 1)[0].replace('_', '')}}}",
            r"\end{figure*}",
            "",
        ]
    return out


def _tex(sections: list[dict], metrics: dict) -> str:
    out = [
        r"\documentclass[conference,a4paper]{IEEEtran}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{multirow}",
        r"\usepackage{hyperref}",
        r"\graphicspath{{../images/}{images/}}",
        "",
        f"\\title{{{_tex_escape(TITLE)}}}",
        f"\\author{{{_tex_escape(AUTHORS)}}}",
        "",
        r"\begin{document}",
        "",
        r"\maketitle",
        "",
        r"\begin{abstract}",
    ]

    summary = report_sections.abstract(metrics)
    if summary:
        out += [summary, ""]
    out += [
        r"\end{abstract}",
        "",
        r"\begin{IEEEkeywords}",
        _tex_escape(report_sections.INDEX_TERMS),
        r"\end{IEEEkeywords}",
        "",
    ]

    body: list[str] = []

    for section in sections:
        if section["id"] == "references":
            # Rendered from ``concepts.REFERENCES`` below, after the gallery.
            continue

        body += [f"\\section{{{_tex_escape(section['title'])}}}", ""]
        for block in section["blocks"]:
            kind = block["kind"]
            if kind == "prose":
                body += [_tex_md(block["text"]), ""]
            elif kind == "list":
                body += [
                    f"\\paragraph{{{_tex_escape(block['title'])}}}",
                    "",
                    r"\begin{itemize}",
                ]
                body += [f"  \\item {_tex_md(item)}" for item in block["items"]]
                body += [r"\end{itemize}", ""]
            elif kind == "table":
                body += _tex_table(block)
            elif kind == "metric":
                body += [_tex_escape(block["caption"]), ""]

    out += body

    out += _tex_gallery()

    out += [r"\begin{thebibliography}{99}", r"\small"]
    for index, entry in enumerate(concepts.REFERENCES, 1):
        out += [
            f"\\bibitem{{ref{index}}}",
            _tex_md(entry).replace("[Online]. Available:", "Available:"),
            "",
        ]
    out += [r"\end{thebibliography}", "", r"\end{document}", ""]

    return "\n".join(out)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def compile_pdf(tex_path: Path = REPORT_LATEX) -> Path | None:
    """Compile the LaTeX source when a TeX toolchain is installed.

    Thin wrapper over :func:`src.report_pdf.compile_pdf` so the report entry
    point stays readable. Returns ``None`` when no compiler is installed; the
    Markdown, LaTeX and HTML renderings are written either way.
    """
    return report_pdf.compile_pdf(tex_path)


def main() -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {}
    if METRICS_FILE.exists():
        with open(METRICS_FILE, encoding="utf-8") as handle:
            metrics = json.load(handle)
    else:
        print("! results/model_metrics.json is missing -- run `python -m src.train_models` first.")
        print("  the report will describe the design only.")

    profile = load_profile()
    sections = build_sections(metrics, profile)

    REPORT_MARKDOWN.write_text(_markdown(sections, metrics), encoding="utf-8")
    REPORT_LATEX.write_text(_tex(sections, metrics), encoding="utf-8")

    print("=" * 66)
    print("IEEE REPORT")
    print("=" * 66)
    for section in sections:
        kinds = {block["kind"] for block in section["blocks"]}
        print(
            f"  {section['roman']:>3}. {section['title']:<24} "
            f"{len(section['blocks']):>2} blocks  [{', '.join(sorted(kinds))}]"
        )
    tables = sum(
        1 for section in sections for block in section["blocks"] if block["kind"] == "table"
    )
    print(f"\n  tables       : {tables}")
    print(f"  sections     : {len(sections)}")

    html_path = report_pdf.build_html()
    for path in (REPORT_MARKDOWN, REPORT_LATEX, html_path):
        print(f"  wrote {path.relative_to(REPORT_MARKDOWN.parents[1])}")

    pdf = compile_pdf()
    if pdf is not None:
        print(f"  wrote {pdf.relative_to(REPORT_MARKDOWN.parents[1])}  (LaTeX)")
    elif report_pdf.build_pdf(html_path, REPORT_PDF):
        pdf = REPORT_PDF
        print(f"  wrote {pdf.relative_to(REPORT_MARKDOWN.parents[1])}  (headless browser)")
    else:
        print("  pdf          : skipped, no LaTeX or headless-browser toolchain available")
    print("=" * 66)


if __name__ == "__main__":
    main()
