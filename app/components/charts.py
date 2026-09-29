"""Vega-Lite chart builders.

Altair is used throughout instead of Plotly: it is bundled with Streamlit, it
inherits the theme's colour tokens, and every chart here returns a single
``Chart`` object so pages never touch chart configuration inline.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

REAL_COLOR = "#0E9F6E"
FAKE_COLOR = "#E11D48"
MUTED_COLOR = "#8B90A6"
GRID_COLOR = "#E4E4EC"
TEXT_COLOR = "#4A5068"
TITLE_COLOR = "#1F2333"

LABEL_COLORS = {"Real": REAL_COLOR, "Fake": FAKE_COLOR}
LABEL_ORDER = ["Real", "Fake"]

#: Signals are normalised to 0..1 so they can share one scale.
SIGNAL_SCALE = {
    "clickbait_score": ("Clickbait pressure", 1.0),
    "emotional_word_ratio": ("Emotional language", 1.0),
    "sentiment_score": ("Tone", 1.0),
    "readability_score": ("Readability", 100.0),
    "source_trust_score": ("Source trust", 1.0),
}


def _style(chart: alt.Chart) -> alt.Chart:
    """Apply the shared axis and view styling to a finished chart.

    Altair's top-level ``configure_*`` methods have to be called on the chart
    itself, so every builder passes its result through this helper before
    handing it to Streamlit.
    """
    return chart.configure_view(strokeWidth=0).configure_axis(
        labelColor=TEXT_COLOR,
        titleColor=TEXT_COLOR,
        gridColor=GRID_COLOR,
        domainColor=GRID_COLOR,
        tickColor=GRID_COLOR,
        labelFontSize=11,
        titleFontSize=12,
        titleFontWeight=500,
    )


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def probability_chart(probabilities: dict[str, float], *, title: str = "Class probability") -> None:
    """Two-bar horizontal chart of P(real) vs P(fake)."""
    frame = pd.DataFrame(
        [{"Class": name, "Probability": value} for name, value in probabilities.items()]
    )
    frame["Order"] = frame["Class"].map({name: i for i, name in enumerate(LABEL_ORDER)})

    chart = (
        alt.Chart(frame)
        .mark_bar(height=34, cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            y=alt.Y("Class:N", sort=alt.EncodingSortField("Order"), title=None),
            x=alt.X(
                "Probability:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="percent", tickCount=5),
                title="Probability",
            ),
            color=alt.Color(
                "Class:N",
                scale=alt.Scale(domain=list(LABEL_ORDER), range=[REAL_COLOR, FAKE_COLOR]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Class:N", title="Class"),
                alt.Tooltip("Probability:Q", title="Probability", format=".2%"),
            ],
        )
        .properties(height=110, title=alt.TitleParams(text=title, fontSize=13, anchor="start", color=TITLE_COLOR))
    )

    st.altair_chart(_style(chart), width="stretch")


def term_contribution_chart(terms: list[dict], *, title: str = "Term attribution") -> None:
    """Diverging bars: left = toward real, right = toward fake."""
    if not terms:
        return

    frame = pd.DataFrame(terms)
    frame["Term"] = frame["term"].astype(str)
    frame["Weight"] = frame["weight"].astype(float)

    limit = max(float(frame["Weight"].abs().max()) * 1.15, 1e-6)
    frame["Order"] = range(len(frame))

    base = alt.Chart(frame).encode(y=alt.Y("Term:N", sort=alt.EncodingSortField("Order"), title=None))

    bars = base.mark_bar(height=20).encode(
        x=alt.X(
            "Weight:Q",
            scale=alt.Scale(domain=[-limit, limit]),
            axis=alt.Axis(format=".2f", title="Contribution to P(fake)  (left = real, right = fake)"),
            title="Contribution to P(fake)",
        ),
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(domain=list(LABEL_ORDER), range=[REAL_COLOR, FAKE_COLOR]),
            legend=alt.Legend(title=None, orient="top"),
        ),
        tooltip=[
            alt.Tooltip("Term:N", title="Term"),
            alt.Tooltip("direction:N", title="Pulls toward"),
            alt.Tooltip("Weight:Q", title="Weight", format=".4f"),
        ],
    )

    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED_COLOR, strokeWidth=1).encode(x="x:Q")

    chart = (zero + bars).properties(
        height=max(150, 34 * len(frame) + 60),
        title=alt.TitleParams(text=title, fontSize=13, anchor="start", color=TITLE_COLOR),
    )

    st.altair_chart(_style(chart), width="stretch")


def signal_radar(signals: dict, *, title: str = "Editorial signals") -> None:
    """Grouped bars comparing one article against the corpus average."""
    from app.services.corpus_stats import corpus_signal_means

    means = corpus_signal_means()

    rows = []
    for key, (label, scale) in SIGNAL_SCALE.items():
        if key not in signals:
            continue
        rows.append(
            {
                "Signal": label,
                "This article": min(max(float(signals[key]) / scale, 0.0), 1.0),
            }
        )

    if not rows:
        st.caption("No editorial signals available for this document.")
        return

    if means:
        for key, (label, scale) in SIGNAL_SCALE.items():
            if key in means:
                rows.append(
                    {
                        "Signal": label,
                        "Corpus average": min(max(float(means[key]) / scale, 0.0), 1.0),
                    }
                )

    frame = pd.DataFrame(rows).melt(
        id_vars=["Signal"],
        value_vars=["This article", "Corpus average"],
        var_name="Series",
        value_name="Value",
    )
    frame = frame.dropna(subset=["Value"])
    if frame.empty:
        st.caption("No editorial signals available for this document.")
        return
    signal_order = {label: index for index, (label, _) in enumerate(SIGNAL_SCALE.values())}
    frame["Order"] = frame["Signal"].map(signal_order)

    chart = (
        alt.Chart(frame)
        .mark_bar(height=13, cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X(
                "Value:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="percent", title=None),
                title=None,
            ),
            y=alt.Y("Signal:N", sort=alt.EncodingSortField("Order"), title=None),
            yOffset=alt.YOffset("Series:N", sort=["This article", "Corpus average"]),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=["This article", "Corpus average"], range=["#4F46E5", "#C9CCDD"]),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
            tooltip=[
                alt.Tooltip("Signal:N", title="Signal"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("Value:Q", title="Share of scale", format=".1%"),
            ],
        )
        .properties(
            height=190,
            title=alt.TitleParams(text=title, fontSize=13, anchor="start", color=TITLE_COLOR),
        )
    )

    st.altair_chart(_style(chart), width="stretch")


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------

def metric_comparison_chart(metrics: dict, *, key: str = "f1_score", title: str = "F1 by model") -> None:
    """Grouped bar chart comparing every candidate on one metric."""
    rows = [
        {"Model": name, "Score": scores.get(key)}
        for name, scores in metrics.get("models", {}).items()
        if scores.get(key) is not None
    ]
    if not rows:
        return

    frame = pd.DataFrame(rows)
    best = metrics.get("best_model_name")
    frame["Model"] = frame["Model"] + frame["Model"].eq(best).map({True: "  ★", False: ""})

    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, height=120)
        .encode(
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="percent"), title=None),
            y=alt.Y("Model:N", sort="-x", title=None),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(
                    domain=list(frame["Model"]),
                    range=["#4F46E5" if name == best else "#B9BBD0" for name in frame["Model"]],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip("Score:Q", title=title, format=".2%"),
            ],
        )
        .properties(height=180, title=alt.TitleParams(text=title, fontSize=13, anchor="start", color=TITLE_COLOR))
    )

    st.altair_chart(_style(chart), width="stretch")


def metric_grid_chart(metrics: dict) -> None:
    """All four headline metrics for every candidate in one grouped view."""
    keys = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1_score", "F1")]
    rows = [
        {"Model": name, "Metric": label, "Score": scores.get(key)}
        for name, scores in metrics.get("models", {}).items()
        for key, label in keys
        if scores.get(key) is not None
    ]
    if not rows:
        return

    frame = pd.DataFrame(rows)
    palette = ["#4F46E5", "#8B5CF6", "#0EA5E9", "#E11D48"]

    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, height=34)
        .encode(
            x=alt.X("Score:Q", scale=alt.Scale(domain=[max(0.8, frame["Score"].min() - 0.05), 1.0]), axis=alt.Axis(format=".1%"), title=None),
            y=alt.Y("Model:N", sort="-x", title=None),
            yOffset=alt.YOffset("Metric:N", sort=[label for _, label in keys]),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(domain=[label for _, label in keys], range=palette),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip("Metric:N", title="Metric"),
                alt.Tooltip("Score:Q", title="Score", format=".2%"),
            ],
        )
        .properties(height=210, title=alt.TitleParams(text="All candidates, all metrics", fontSize=13, anchor="start", color=TITLE_COLOR))
    )

    st.altair_chart(_style(chart), width="stretch")


def representation_grid_chart(
    metrics: dict, *, key: str = "f1_score", title: str = "Representation x algorithm"
) -> None:
    """Grouped bars over the full representation x algorithm grid."""
    rows = []
    best_repr = metrics.get("best_representation")
    best_model = metrics.get("best_model_name")
    for representation, block in metrics.get("representations", {}).items():
        for model, scores in block.get("models", {}).items():
            if scores.get(key) is None:
                continue
            rows.append(
                {
                    "Representation": representation,
                    "Model": model,
                    "Score": scores[key],
                    "selected": representation == best_repr and model == best_model,
                }
            )
    if not rows:
        return

    frame = pd.DataFrame(rows)
    palette = ["#4F46E5", "#0EA5E9", "#F59E0B", "#0E9F6E"]

    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, height=54)
        .encode(
            x=alt.X("Representation:N", title=None, sort=list(metrics.get("representations", {}))),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="percent"), title=None),
            xOffset=alt.XOffset("Model:N", sort=None),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(
                    domain=list(frame["Model"].unique()),
                    range=[
                        "#4F46E5" if model == best_model else "#C7CAE0"
                        for model in frame["Model"].unique()
                    ],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Representation:N", title="Representation"),
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip("Score:Q", title=title, format=".2%"),
            ],
            stroke=alt.Stroke(
                "selected:Q",
                scale=alt.Scale(domain=[True, False], range=["#0E172A", "transparent"]),
                legend=None,
            ),
            strokeWidth=alt.StrokeWidth("selected:Q", scale=alt.Scale(domain=[True, False], range=[2, 0])),
        )
        .properties(
            height=250,
            title=alt.TitleParams(
                text=f"{title} — {best_repr} selected", fontSize=13, anchor="start", color=TITLE_COLOR
            ),
        )
    )

    st.altair_chart(_style(chart), width="stretch")


def representation_effect_chart(metrics: dict) -> None:
    """One line per algorithm across the three representations."""
    rows = []
    for representation, block in metrics.get("representations", {}).items():
        for model, scores in block.get("models", {}).items():
            if scores.get("f1_score") is not None:
                rows.append(
                    {"Representation": representation, "Model": model, "F1 score": scores["f1_score"]}
                )
    if not rows:
        return

    frame = pd.DataFrame(rows)
    order = list(metrics.get("representations", {}))
    frame["Rank"] = frame["Representation"].map({name: i for i, name in enumerate(order)})
    palette = ["#4F46E5", "#0EA5E9", "#F59E0B", "#0E9F6E"]
    models = list(frame["Model"].unique())

    line = (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(size=64), strokeWidth=2.2)
        .encode(
            x=alt.X("Representation:N", sort=order, title=None),
            y=alt.Y("F1 score:Q", scale=alt.Scale(domain=[0.5, 1.0]), axis=alt.Axis(format=".1%"), title=None),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(domain=models, range=palette[: len(models)]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Representation:N", title="Representation"),
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip("F1 score:Q", title="F1", format=".2%"),
            ],
        )
    )

    points = line.mark_line(stroke=None).mark_point(size=90, filled=True)
    labels = (
        alt.Chart(frame)
        .mark_text(dy=-12, fontSize=10, color=TEXT_COLOR)
        .encode(
            x=alt.X("Representation:N", sort=order, title=None),
            y=alt.Y("F1 score:Q", scale=alt.Scale(domain=[0.5, 1.0]), axis=alt.Axis(format=".1%"), title=None),
            text=alt.Text("Model:N"),
        )
    )

    chart = (
        alt.layer(line, points, labels)
        .properties(
            height=250,
            title=alt.TitleParams(
                text="Sensitivity to the choice of representation",
                fontSize=13,
                anchor="start",
                color=TITLE_COLOR,
            ),
        )
    )

    st.altair_chart(_style(chart), width="stretch")


def confusion_matrix_chart(matrix: list[list[int]], *, title: str = "Confusion matrix") -> None:
    """Heatmap of a 2x2 confusion matrix, annotated with counts."""
    if not matrix or len(matrix) != 2:
        return

    total = sum(sum(row) for row in matrix)
    correct = matrix[0][0] + matrix[1][1]
    frame = pd.DataFrame(
        [
            {
                "True class": LABEL_ORDER[i],
                "Predicted class": LABEL_ORDER[j],
                "Count": matrix[i][j],
                "Label": f"{matrix[i][j]:,}",
                "Share of all": matrix[i][j] / total if total else 0,
            }
            for i in range(2)
            for j in range(2)
        ]
    )
    frame["Value"] = frame["Count"].where(
        frame["True class"].ne(frame["Predicted class"]), 1
    )

    cells = (
        alt.Chart(frame)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X("Predicted class:N", title="Predicted", sort=LABEL_ORDER),
            y=alt.Y("True class:N", title="Actual", sort=LABEL_ORDER),
            color=alt.Color(
                "Value:Q",
                scale=alt.Scale(domain=[0, 1], range=["#E4E4EC", "#4F46E5"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("True class:N", title="Actual"),
                alt.Tooltip("Predicted class:N", title="Predicted"),
                alt.Tooltip("Count:Q", title="Articles"),
            ],
        )
    )
    text = (
        alt.Chart(frame)
        .mark_text(fontSize=17, color=TITLE_COLOR)
        .encode(x=alt.X("Predicted class:N", sort=LABEL_ORDER), y=alt.Y("True class:N", sort=LABEL_ORDER), text=alt.Text("Label:N"))
    )

    chart = alt.layer(cells, text).properties(
        height=190,
        title=alt.TitleParams(
            text=f"{title} — {correct:,}/{total:,} correct", fontSize=13, anchor="start", color=TITLE_COLOR
        ),
    )

    st.altair_chart(_style(chart), width="stretch")


def roc_chart(roc: dict, *, title: str = "ROC curve") -> None:
    """ROC curve from the stored points, with the chance diagonal."""
    fpr = roc.get("fpr") or []
    tpr = roc.get("tpr") or []
    if not fpr or not tpr:
        return

    frame = pd.DataFrame(
        [{"False positive rate": x, "True positive rate": y, "Series": "Model"} for x, y in zip(fpr, tpr)]
        + [{"False positive rate": 0, "True positive rate": 0, "Series": "Chance"},
           {"False positive rate": 1, "True positive rate": 1, "Series": "Chance"}]
    )

    auc = roc.get("auc")
    chart = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2.4)
        .encode(
            x=alt.X("False positive rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="percent"), title="False positive rate"),
            y=alt.Y("True positive rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="percent"), title="True positive rate"),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=["Model", "Chance"], range=["#4F46E5", "#C9CCDD"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            strokeDash=alt.StrokeDash(
                "Series:N",
                scale=alt.Scale(domain=["Model", "Chance"], range=[[1, 0], [4, 4]]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("False positive rate:Q", title="FPR", format=".3f"),
                alt.Tooltip("True positive rate:Q", title="TPR", format=".3f"),
            ],
        )
        .properties(
            height=280,
            title=alt.TitleParams(
                text=f"{title} · AUC {auc:.3f}" if auc else title,
                fontSize=13, anchor="start", color=TITLE_COLOR,
            ),
        )
    )

    st.altair_chart(_style(chart), width="stretch")


# --------------------------------------------------------------------------
# Dataset & history
# --------------------------------------------------------------------------

def label_split_chart(counts: dict[str, int], *, title: str = "Verdict split") -> None:
    """Real vs fake counts as a single stacked bar."""
    frame = pd.DataFrame([{"Class": name, "Articles": counts.get(name, 0)} for name in LABEL_ORDER])
    if int(frame["Articles"].sum()) == 0:
        return

    chart = (
        alt.Chart(frame)
        .mark_bar(
            height=40,
            cornerRadiusTopLeft=4,
            cornerRadiusTopRight=4,
            cornerRadiusBottomLeft=4,
            cornerRadiusBottomRight=4,
        )
        .encode(
            x=alt.X("Articles:Q", stack="zero", axis=alt.Axis(format=","), title=None),
            y=alt.Y(
                "Articles:Q",
                stack="zero",
                axis=alt.Axis(labels=False, ticks=False, domain=False),
                title=None,
            ),
            color=alt.Color(
                "Class:N",
                scale=alt.Scale(domain=list(LABEL_ORDER), range=[REAL_COLOR, FAKE_COLOR]),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
            tooltip=[
                alt.Tooltip("Class:N", title="Class"),
                alt.Tooltip("Articles:Q", title="Articles", format=","),
            ],
        )
        .properties(height=80, title=alt.TitleParams(text=title, fontSize=13, anchor="start", color=TITLE_COLOR))
    )

    st.altair_chart(_style(chart), width="stretch")


def class_mean_chart(means: pd.Series, *, title: str, x_title: str) -> None:
    """Horizontal comparison of one feature's mean across the two classes."""
    frame = means.reindex(LABEL_ORDER).dropna().rename("Value").reset_index()
    frame.columns = ["Class", "Value"]
    if frame.empty:
        return

    chart = (
        alt.Chart(frame)
        .mark_bar(height=26, cornerRadiusEnd=4)
        .encode(
            x=alt.X("Value:Q", title=x_title),
            y=alt.Y("Class:N", sort=list(LABEL_ORDER), title=None),
            color=alt.Color(
                "Class:N",
                scale=alt.Scale(domain=list(LABEL_ORDER), range=[REAL_COLOR, FAKE_COLOR]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Class:N", title="Class"),
                alt.Tooltip("Value:Q", title=x_title, format=".3f"),
            ],
        )
        .properties(
            height=120,
            title=alt.TitleParams(text=title, fontSize=13, anchor="start", color=TITLE_COLOR),
        )
    )

    st.altair_chart(_style(chart), width="stretch")


def histogram(
    values: pd.Series,
    *,
    title: str,
    x_title: str,
    bins: int = 20,
    color: str = MUTED_COLOR,
) -> None:
    """Single-series histogram of any numeric column, split by class when present."""
    source = pd.DataFrame({"Value": pd.to_numeric(values, errors="coerce")}).dropna()
    if source.empty:
        return

    chart = (
        alt.Chart(source)
        .mark_bar(color=color, opacity=0.85, binSpacing=1)
        .encode(
            x=alt.X("Value:Q", bin=alt.Bin(maxbins=bins), title=x_title),
            y=alt.Y("count():Q", title="Articles"),
            tooltip=[alt.Tooltip("count():Q", title="Articles", format=",")],
        )
        .properties(
            height=240,
            title=alt.TitleParams(text=title, fontSize=13, anchor="start", color=TITLE_COLOR),
        )
    )

    st.altair_chart(_style(chart), width="stretch")


def length_histogram(frame: pd.DataFrame, column: str, *, title: str, x_title: str) -> None:
    """Overlapping histograms of a numeric column, split by class."""
    if frame.empty or column not in frame.columns:
        return

    source = frame[[column, "label"]].rename(columns={column: "Value", "label": "Class"}).dropna()
    if source.empty:
        return

    source["Class"] = source["Class"].map({0: "Real", 1: "Fake"})
    bins = max(10, min(40, int(len(source) ** 0.5)))

    chart = (
        alt.Chart(source)
        .mark_bar(opacity=0.62, binSpacing=1)
        .encode(
            x=alt.X("Value:Q", bin=alt.Bin(maxbins=bins), title=x_title),
            y=alt.Y("count():Q", stack=None, title="Articles"),
            color=alt.Color(
                "Class:N",
                scale=alt.Scale(domain=list(LABEL_ORDER), range=[REAL_COLOR, FAKE_COLOR]),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
            tooltip=[
                alt.Tooltip("Class:N", title="Class"),
                alt.Tooltip("count():Q", title="Articles"),
            ],
        )
        .properties(height=250, title=alt.TitleParams(text=title, fontSize=13, anchor="start", color=TITLE_COLOR))
    )

    st.altair_chart(_style(chart), width="stretch")


def subject_chart(counts: pd.Series, *, title: str = "Articles per news desk", top: int = 10) -> None:
    """Horizontal bars for the most common categories."""
    if counts.empty:
        return

    frame = counts.head(top).rename_axis("Desk").reset_index(name="Articles")
    frame["Order"] = range(len(frame))

    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=18)
        .encode(
            x=alt.X("Articles:Q", axis=alt.Axis(format=","), title=None),
            y=alt.Y("Desk:N", sort=alt.EncodingSortField("Order"), title=None),
            color=alt.value("#4F46E5"),
            tooltip=[alt.Tooltip("Desk:N"), alt.Tooltip("Articles:Q", format=",")],
        )
        .properties(height=40 * len(frame) + 40, title=alt.TitleParams(text=title, fontSize=13, anchor="start", color=TITLE_COLOR))
    )

    st.altair_chart(_style(chart), width="stretch")


def history_trend(history: pd.DataFrame) -> None:
    """Rolling mean confidence over the log, newest last."""
    if history.empty or len(history) < 2:
        return

    frame = history.copy()
    frame["Confidence"] = pd.to_numeric(frame["Confidence"], errors="coerce")
    frame = frame.dropna(subset=["Confidence"]).reset_index(drop=True)
    if frame.empty:
        return

    frame["Fake share"] = (frame["Prediction"] == "Fake").astype(float) * 100

    left = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2, color="#4F46E5", interpolate="monotone")
        .encode(
            x=alt.X("index:Q", title="Prediction number"),
            y=alt.Y(
                "Confidence:Q",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(title="Confidence %"),
            ),
            tooltip=[
                alt.Tooltip("index:Q", title="Prediction"),
                alt.Tooltip("Confidence:Q", title="Confidence", format=".1f"),
            ],
        )
    )

    right = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2, color=FAKE_COLOR, interpolate="monotone")
        .encode(
            x=alt.X("index:Q"),
            y=alt.Y("Fake share:Q", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(title="Fake share %")),
            tooltip=[
                alt.Tooltip("index:Q", title="Prediction"),
                alt.Tooltip("Fake share:Q", title="Fake share", format=".0f"),
            ],
        )
    )

    combined = alt.layer(left, right).resolve_scale(y="independent")

    chart = combined.properties(
        height=260,
        title=alt.TitleParams(
            text=f"Confidence and fake-share over the last {len(frame)} predictions",
            fontSize=13, anchor="start", color=TITLE_COLOR,
        ),
    )

    st.altair_chart(_style(chart), width="stretch")
