"""The navigation map, declared once and shared by the shell and the tests.

Keeping the map here means ``streamlit_app.py`` and the test that checks every
route is reachable read the same list, so a page can never be added to the
sidebar without a test noticing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    """One navigation entry."""

    path: str
    title: str
    icon: str
    default: bool = False
    group: str = ""
    #: Routes that need the trained artifacts to render anything useful.
    needs_model: bool = True


#: Declared order is the sidebar order within each group.
ROUTES: list[Route] = [
    Route("app_pages/overview.py", "Overview", "home", default=True, group="", needs_model=False),
    Route("app_pages/analyze.py", "Analyze article", "search", group="Detection"),
    Route("app_pages/batch.py", "Batch analysis", "upload_file", group="Detection"),
    Route("app_pages/model_lab.py", "Model lab", "insights", group="Evidence"),
    Route("app_pages/dataset.py", "Dataset explorer", "dataset", group="Evidence"),
    Route("app_pages/concepts.py", "Concepts", "school", group="Evidence", needs_model=False),
    Route("app_pages/history.py", "Prediction log", "history", group="Records"),
    Route("app_pages/report.py", "Report", "description", group="Records", needs_model=False),
    Route("app_pages/pipeline.py", "Pipeline & setup", "account_tree", group="Records"),
]

#: Group order, with "" first for the top-level entries.
GROUP_ORDER: list[str] = ["", "Detection", "Evidence", "Records"]

GROUPS: list[str] = GROUP_ORDER


def grouped() -> dict[str, list[Route]]:
    """Routes bucketed by group, in declared order."""
    buckets: dict[str, list[Route]] = {group: [] for group in GROUP_ORDER}
    for route in ROUTES:
        buckets.setdefault(route.group, []).append(route)
    return buckets


def page_count() -> int:
    return len(ROUTES)
