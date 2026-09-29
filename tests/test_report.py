"""The generated report must stay honest and self-consistent.

Every assertion here guards a defect that was actually present at some point:
a figure gallery that silently vanished from the LaTeX build, an abstract copied
into three renderers by hand, a "calibrated confidence" claim with no
calibration behind it, and a stale page count in the appendix.
"""

from __future__ import annotations

import json
import re

import pytest

from src.config import IMAGE_DIR, REPORT_HTML, REPORT_LATEX, REPORT_MARKDOWN, REPORT_PDF
from src.report_sections import (
    INDEX_TERMS,
    SECTION_TITLES,
    TABLE_REFS,
    abstract,
    build_sections,
)

pytestmark = pytest.mark.skipif(
    not REPORT_MARKDOWN.exists(),
    reason="report not generated; run `python -m src.report`",
)


@pytest.fixture(scope="module")
def markdown() -> str:
    return REPORT_MARKDOWN.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tex() -> str:
    return REPORT_LATEX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def metrics() -> dict:
    import json

    from src.config import METRICS_FILE

    with open(METRICS_FILE, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------

def test_every_section_is_present(markdown: str) -> None:
    for roman in ("I", "II", "III", "IV", "V", "VI", "VII", "VIII"):
        assert re.search(rf"^## {roman}\. ", markdown, re.M), f"missing section {roman}"
    assert len(SECTION_TITLES) == 8


def test_abstract_and_index_terms_are_rendered(markdown: str) -> None:
    assert "## Abstract" in markdown
    assert INDEX_TERMS.split(",")[0] in markdown


def test_headline_numbers_come_from_the_metrics(markdown: str, metrics: dict) -> None:
    scores = metrics["models"][metrics["best_model_name"]]
    assert f"{scores['accuracy']:.4f}" in markdown
    assert f"{scores['f1_score']:.4f}" in markdown
    assert f"{metrics['dataset']['test_rows']}" in markdown


# --------------------------------------------------------------------------
# Honesty
# --------------------------------------------------------------------------

def test_no_unimplemented_calibration_claim(metrics: dict) -> None:
    """The pipeline has no calibration stage, so the report must not claim one."""
    summary = abstract(metrics)
    assert "not a calibrated probability" in summary
    assert "calibrated confidence" not in summary.lower() or "not a calibrated" in summary.lower()


def test_report_does_not_promise_calibration_anywhere(markdown: str) -> None:
    for match in re.finditer(r"[^.]*[Cc]alibrat[^.]*\.", markdown):
        sentence = match.group(0)
        assert "not a calibrated" in sentence or "Calibrate the output" in sentence, sentence


def test_confidence_is_labelled_as_a_model_score(markdown: str) -> None:
    assert "not a calibrated probability" in markdown


# --------------------------------------------------------------------------
# Single source of truth
# --------------------------------------------------------------------------

def test_abstract_is_one_shared_string(metrics: dict) -> None:
    """The app and both document renderers must call the same builder."""
    from app_pages import report as report_page
    from src import report as report_module

    summary = abstract(metrics)
    assert summary
    # The renderers must not hold their own private copy of the prose.
    for module in (report_module, report_page):
        source = open(module.__file__, encoding="utf-8").read()
        assert "Fabricated news spreads faster" not in source, module.__name__


def test_appendix_route_count_is_live(markdown: str) -> None:
    from app.navigation import page_count

    assert f"all {page_count()} routes render" in markdown


# --------------------------------------------------------------------------
# LaTeX build
# --------------------------------------------------------------------------

def test_latex_contains_every_generated_figure(tex: str) -> None:
    """Regression: the References block used to return early, dropping the gallery."""
    available = [
        path.name
        for path in sorted(IMAGE_DIR.glob("*.png"))
        if path.name in tex
    ]
    assert len(available) == len(list(IMAGE_DIR.glob("*.png"))), available
    assert tex.count(r"\begin{figure*}") == len(available)


def test_latex_resolves_images_relative_to_the_document(tex: str) -> None:
    assert r"\graphicspath{{../images/}{images/}}" in tex
    assert "../images/" not in tex.split(r"\begin{document}")[1]


def test_latex_closes_after_the_bibliography(tex: str) -> None:
    assert tex.index(r"\begin{thebibliography}") < tex.index(r"\end{document}")
    assert r"\end{thebibliography}" in tex
    assert tex.count(r"\begin{document}") == 1


def test_wide_tables_span_both_columns(tex: str) -> None:
    assert r"\begin{table*}" in tex


# --------------------------------------------------------------------------
# Other renderings
# --------------------------------------------------------------------------

def test_html_rendering_is_two_column_and_embeds_figures() -> None:
    if not REPORT_HTML.exists():
        pytest.skip("html rendering not generated")
    document = REPORT_HTML.read_text(encoding="utf-8")
    assert "column-count: 2" in document
    assert document.count("<img ") == len(list(IMAGE_DIR.glob("*.png")))
    assert "<table>" in document


def test_pdf_is_a_valid_document() -> None:
    if not REPORT_PDF.exists():
        pytest.skip("no PDF toolchain available on this machine")
    from pypdf import PdfReader

    reader = PdfReader(REPORT_PDF)
    assert len(reader.pages) > 0
    text = reader.pages[0].extract_text()
    assert "AI-Powered Fake News Detection" in text
    assert "ABSTRACT" in text.upper()


# --------------------------------------------------------------------------
# Section builder
# --------------------------------------------------------------------------

def test_sections_build_without_metrics() -> None:
    """A missing benchmark must still produce the design-only document."""
    sections = build_sections({}, {})
    assert len(sections) == len(SECTION_TITLES)
    assert abstract({}) == ""


def test_every_table_row_matches_its_header(metrics: dict) -> None:
    from src.data_loader import load_profile

    for section in build_sections(metrics, load_profile()):
        for block in section["blocks"]:
            if block["kind"] != "table":
                continue
            width = len(block["columns"])
            for row in block["rows"]:
                assert len(row) == width, (block["caption"], row)


# --------------------------------------------------------------------------
# Prose stays tied to the run
# --------------------------------------------------------------------------

def test_no_unresolved_placeholders_survive(markdown: str) -> None:
    """``build_sections`` raises on an unknown token, so nothing may leak through."""
    for token in ("{table_", "{bow_", "{train_rows}", "{test_rows}", "{sample_rows}"):
        assert token not in markdown, token


def test_table_numbers_are_assigned_in_reading_order(metrics: dict) -> None:
    from src.data_loader import load_profile

    sections = build_sections(metrics, load_profile())
    tables = [
        block
        for section in sections
        for block in section["blocks"]
        if block["kind"] == "table"
    ]
    assert [block["number"] for block in tables] == list(range(1, len(tables) + 1))
    assert {block["ref"] for block in tables} == set(TABLE_REFS)


def test_prose_cross_references_point_at_the_right_table(metrics: dict) -> None:
    """A caption or a sentence quoting a literal table number is drift waiting."""
    from src.data_loader import load_profile

    sections = build_sections(metrics, load_profile())
    numbers = {
        block["ref"]: block["number"]
        for section in sections
        for block in section["blocks"]
        if block["kind"] == "table"
    }
    for section in sections:
        for block in section["blocks"]:
            if block["kind"] == "prose":
                for quote in re.findall(r"Table (\d+)", block["text"]):
                    assert int(quote) in numbers.values(), block["text"][:80]


def test_dimensions_and_row_counts_come_from_the_run(metrics: dict) -> None:
    from src.data_loader import load_profile

    from src.config import MAX_FEATURES

    sections = build_sections(metrics, load_profile())
    body = json.dumps(sections)
    assert f"{MAX_FEATURES:,}-dimensional" in body
    assert f"{metrics['dataset']['train_rows']:,}-row" in body
    # The un-formatted spellings would mean a placeholder was filled as a bare
    # int, and the thousands separator is what the rest of the document uses.
    assert f"{MAX_FEATURES}-dimensional" not in body
    assert f"{metrics['dataset']['train_rows']}-row" not in body


def test_missing_token_raises_rather_than_silently_passing() -> None:
    """A typo in a placeholder must break the build, not ship a literal brace."""
    from src.report_sections import _fill

    with pytest.raises(ValueError, match="unresolved placeholder"):
        _fill("a {not_a_real_token} b", {})
