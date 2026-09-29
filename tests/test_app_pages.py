"""Every Streamlit route must render without raising, in both artifact states."""

from __future__ import annotations

import pytest

from app.navigation import ROUTES, page_count

PAGES = ["streamlit_app.py", *[route.path for route in ROUTES]]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, app_test):
    at = app_test(page)
    assert not at.exception, f"{page} raised: {[e.message for e in at.exception]}"
    assert at.title, f"{page} rendered no page title"


def test_analyze_runs_a_real_prediction(app_test):
    """The headline flow: text in, REAL/FAKE verdict out, from the trained model."""
    at = app_test("app_pages/analyze.py")
    at.text_area(key="draft_body").set_value(
        "The finance ministry published a three-year plan on Tuesday that allocates funding "
        "for broadband equipment in public schools, and a parliamentary committee will review "
        "progress each year."
    )
    at.button(key="analyze_run").click().run()

    assert not at.exception, [e.message for e in at.exception]
    body = " ".join(str(m.value) for m in at.markdown)
    assert "Verdict" in body, "No verdict panel was rendered"


def test_concepts_page_renders_without_artifacts(app_test):
    """Concepts is explanatory, so it must work even with no trained pipeline."""
    at = app_test("app_pages/concepts.py")
    assert not at.exception, [e.message for e in at.exception]
    assert at.title, "Concepts page rendered no title"


def test_report_page_renders_the_eight_sections(app_test):
    at = app_test("app_pages/report.py")
    assert not at.exception, [e.message for e in at.exception]

    labels = " ".join(str(e.label) for e in at.expander)
    for section in (
        "Introduction",
        "Dataset Description",
        "Methodology",
        "Results",
        "Discussion",
        "Conclusion",
        "Appendix",
        "References",
    ):
        assert section in labels, f"Report page is missing the {section} section"


def test_navigation_exposes_every_route():
    from pathlib import Path

    declared = {route.path for route in ROUTES}
    on_disk = {
        f"app_pages/{p.name}"
        for p in (Path(__file__).resolve().parent.parent / "app_pages").glob("*.py")
    }

    assert declared == on_disk, f"Unrouted pages: {on_disk ^ declared}"
    assert page_count() == len(declared)
    assert sum(1 for route in ROUTES if route.default) == 1, "exactly one default route expected"


def test_navigation_titles_are_unique():
    titles = [route.title for route in ROUTES]
    assert len(titles) == len(set(titles)), f"Duplicate page titles: {titles}"
