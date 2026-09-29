"""Model lab -- benchmark, representation grid, diagnostics and discussion.

Every number on this page is read from ``results/model_metrics.json``; the prose
that explains *why* the numbers look the way they do lives in ``src/concepts.py``
so this page and the generated report can never disagree.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.cards import confusion_frame, kpi_strip
from app.components.charts import (
    confusion_matrix_chart,
    metric_comparison_chart,
    metric_grid_chart,
    representation_effect_chart,
    representation_grid_chart,
    roc_chart,
)
from app.components.layout import empty_state, render_page_header
from app.services.artifacts import figure_path, load_metrics
from src import concepts

metrics = load_metrics()

render_page_header(
    "Model lab",
    f"{len(metrics.get('representations', {}))} representations x "
    f"{len(metrics.get('models', {}))} algorithms, scored once on the hold-out. "
    "Every figure here is read from `results/model_metrics.json`.",
)

if not metrics:
    empty_state(
        "insights",
        "No benchmark recorded",
        "Run `python -m src.train_models` from the project root, then reload this page.",
    )
    st.stop()

models = metrics.get("models", {})
representations = metrics.get("representations", {})
best_name = metrics.get("best_model_name", "")
best_repr = metrics.get("best_representation", "")
best = models.get(best_name, {})
split = metrics.get("dataset", {})
selection = metrics.get("selection", {})

# --------------------------------------------------------------------------
# Headline
# --------------------------------------------------------------------------
st.subheader("Selected model")

kpi_strip(
    [
        ("Accuracy", f"{best.get('accuracy', 0) * 100:.2f}%"),
        ("Precision", f"{best.get('precision', 0) * 100:.2f}%"),
        ("Recall", f"{best.get('recall', 0) * 100:.2f}%"),
        ("F1 score", f"{best.get('f1_score', 0) * 100:.2f}%"),
        ("ROC-AUC", f"{best.get('roc_auc') or 0:.4f}"),
    ],
    caption=(
        f"{best_repr} + {best_name} · trained in {best.get('train_seconds', 0):.2f}s · "
        f"evaluated on {split.get('test_rows', 0):,} held-out articles "
        f"({split.get('test_real', 0):,} real, {split.get('test_fake', 0):,} fake). "
        + (
            f"Chosen by F1 {selection.get('validation_f1', 0):.4f} on a validation split "
            f"of {selection.get('validation_rows', 0):,} articles carved out of the training side."
            if selection
            else ""
        )
    ),
)

st.info(
    concepts.SELECTION_RATIONALE,
    icon=":material/balance:",
)

# --------------------------------------------------------------------------
# Representation x algorithm grid
# --------------------------------------------------------------------------
st.subheader("Representation x algorithm grid")

st.caption(
    "All "
    f"{len(representations) * len(models)} combinations of "
    f"{len(representations)} representations ({', '.join(representations)}) and "
    f"{len(models)} algorithms ({', '.join(models)}), each scored on the same hold-out split."
)

grid_col, effect_col = st.columns([3, 2], gap="large")

with grid_col:
    with st.container(border=True):
        metric_choice = st.segmented_control(
            "Grid metric",
            ["F1 score", "Accuracy", "Precision", "Recall", "ROC-AUC"],
            default="F1 score",
            key="lab_grid_metric",
        ) or "F1 score"
        key = {
            "F1 score": "f1_score",
            "Accuracy": "accuracy",
            "Precision": "precision",
            "Recall": "recall",
            "ROC-AUC": "roc_auc",
        }[metric_choice]
        representation_grid_chart(metrics, key=key, title=metric_choice)

with effect_col:
    with st.container(border=True):
        representation_effect_chart(metrics)

rows = []
for representation, block in representations.items():
    for model, scores in block.get("models", {}).items():
        rows.append(
            {
                "Representation": representation,
                "Model": model,
                "Accuracy": scores.get("accuracy"),
                "Precision": scores.get("precision"),
                "Recall": scores.get("recall"),
                "F1": scores.get("f1_score"),
                "ROC-AUC": scores.get("roc_auc"),
                "Fit time (s)": scores.get("train_seconds"),
                "Selected": bool(scores.get("selected")),
            }
        )

if rows:
    grid = pd.DataFrame(rows).sort_values("F1", ascending=False)
    st.dataframe(
        grid,
        hide_index=True,
        column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.4f"),
            "Precision": st.column_config.NumberColumn(format="%.4f"),
            "Recall": st.column_config.NumberColumn(format="%.4f"),
            "F1": st.column_config.NumberColumn("F1", format="%.4f"),
            "ROC-AUC": st.column_config.NumberColumn(format="%.4f"),
            "Fit time (s)": st.column_config.NumberColumn(format="%.2f"),
            "Selected": st.column_config.CheckboxColumn(
                "Selected", help="Chosen on the validation split and therefore deployed"
            ),
        },
    )

# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------
st.subheader("Benchmark")

st.caption(
    "One row per algorithm, each shown on the representation that scored best for it. "
    "The deployed row is the one selected on validation data, which is also the best "
    "scoring combination overall here."
)

table = (
    pd.DataFrame(models)
    .T.reset_index()
    .rename(columns={"index": "Model"})
    .sort_values("f1_score", ascending=False)
)
if "family" in table.columns:
    table.insert(1, "Selected", table["Model"] == best_name)
    table.insert(2, "Family", table["family"])
    table.insert(3, "Representation", table["representation"])
else:
    table.insert(1, "Selected", table["Model"] == best_name)

st.dataframe(
    table,
    hide_index=True,
    column_config={
        "Selected": st.column_config.CheckboxColumn("Selected", help="Highest validation F1 wins the pipeline"),
        "accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
        "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
        "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
        "f1_score": st.column_config.NumberColumn("F1", format="%.4f"),
        "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
        "train_seconds": st.column_config.NumberColumn("Train time (s)", format="%.2f"),
    },
)

rank_col, gridview_col = st.columns(2, gap="large")
with rank_col:
    with st.container(border=True):
        metric_comparison_chart(metrics, key="f1_score", title="F1 by model")
with gridview_col:
    with st.container(border=True):
        metric_grid_chart(metrics)

# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
st.subheader("Diagnostics")

view = st.segmented_control(
    "Diagnostic",
    ["Confusion matrix", "Confusion matrix per algorithm", "ROC curve"],
    default="Confusion matrix per algorithm",
    key="lab_diag",
) or "Confusion matrix per algorithm"

if view == "Confusion matrix":
    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            matrix = metrics.get("best_model", {}).get("confusion_matrix", {}).get("matrix")
            if matrix:
                confusion_matrix_chart(matrix, title=f"{best_name} on the hold-out")
                total = sum(sum(row) for row in matrix)
                correct = matrix[0][0] + matrix[1][1]
                st.caption(
                    f"{correct:,} of {total:,} hold-out articles classified correctly "
                    f"({correct / total * 100:.2f}%)."
                )
            else:
                st.caption("No confusion matrix recorded.")
    with right:
        with st.container(border=True):
            st.markdown("**Generated figure**")
            path = figure_path("confusion_matrix.png")
            if path:
                st.image(str(path), width="stretch")
            else:
                st.caption("Run `python -m src.evaluate_models` to render the figure.")

elif view == "Confusion matrix per algorithm":
    st.caption(
        "One confusion matrix per required algorithm, each on the representation that "
        "scored best for it. Rows are the true class, columns the predicted class."
    )
    matrices = metrics.get("confusion_matrices", {})
    for index, (name, block) in enumerate(matrices.items()):
        matrix = block.get("matrix", [])
        if not matrix:
            continue
        chart_col, table_col = st.columns([3, 2], gap="large")
        with chart_col:
            with st.container(border=True):
                confusion_matrix_chart(
                    matrix,
                    title=f"{name} · {block.get('representation', 'n/a')}",
                )
        with table_col:
            with st.container(border=True):
                frame = confusion_frame(matrix)
                st.dataframe(
                    frame,
                    hide_index=True,
                    column_config={
                        "Predicted real": st.column_config.NumberColumn("Predicted real"),
                        "Predicted fake": st.column_config.NumberColumn("Predicted fake"),
                    },
                )
                total = sum(sum(row) for row in matrix)
                correct = matrix[0][0] + matrix[1][1]
                st.caption(
                    f"Accuracy {correct / total * 100:.2f}% · "
                    f"{total - correct} article{'s' if total - correct != 1 else ''} misclassified."
                )
        if index == 0:
            path = figure_path("confusion_matrices.png")
            if path:
                st.caption("**Static figure**")
                st.image(str(path), width="stretch")

else:
    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            roc = metrics.get("best_model", {}).get("roc_curve", {})
            if roc.get("fpr"):
                roc_chart(roc, title=f"ROC curve · {best_name}")
                st.caption(
                    "A curve hugging the top-left corner means the model separates the two "
                    "classes cleanly; the dashed line is random guessing."
                )
            else:
                st.caption("No ROC curve recorded.")
    with right:
        with st.container(border=True):
            path = figure_path("roc_comparison.png")
            if path:
                st.image(str(path), width="stretch")
            else:
                st.caption("Run `python -m src.evaluate_models` to render the figure.")

# --------------------------------------------------------------------------
# Parametric vs non-parametric
# --------------------------------------------------------------------------
st.subheader("Parametric vs non-parametric")

st.markdown(concepts.PARAMETRIC_VS_NONPARAMETRIC)

family_rows = []
for name, note in concepts.MODEL_NOTES.items():
    scores = models.get(name, {})
    family_rows.append(
        {
            "Model": name,
            "Family": note["family"],
            "Best representation": scores.get("representation", "n/a"),
            "F1": scores.get("f1_score"),
            "Train time (s)": scores.get("train_seconds"),
            "Prediction cost": (
                "grows with corpus size"
                if name == "K-Nearest Neighbors"
                else "fixed per document"
            ),
        }
    )

st.dataframe(
    pd.DataFrame(family_rows),
    hide_index=True,
    column_config={
        "F1": st.column_config.NumberColumn("F1", format="%.4f"),
        "Train time (s)": st.column_config.NumberColumn(format="%.2f"),
    },
)

# --------------------------------------------------------------------------
# Per-model roles
# --------------------------------------------------------------------------
st.subheader("Model roles, strengths and limits")

for name, note in concepts.MODEL_NOTES.items():
    scores = models.get(name, {})
    with st.expander(f"{name} · {note['family']} · F1 {scores.get('f1_score', 0):.4f}"):
        st.markdown(note["summary"])
        strength_col, limit_col = st.columns(2, gap="large")
        with strength_col:
            with st.container(border=True):
                st.markdown("**Strengths**")
                for item in note["strengths"]:
                    st.markdown(f"- {item}")
        with limit_col:
            with st.container(border=True):
                st.markdown("**Limitations**")
                for item in note["limitations"]:
                    st.markdown(f"- {item}")
        st.caption(
            f"Hold-out: accuracy {scores.get('accuracy', 0):.4f} · "
            f"precision {scores.get('precision', 0):.4f} · "
            f"recall {scores.get('recall', 0):.4f} · "
            f"F1 {scores.get('f1_score', 0):.4f} · "
            f"ROC-AUC {scores.get('roc_auc') or 0:.4f} on "
            f"{scores.get('representation', 'n/a')}."
        )

# --------------------------------------------------------------------------
# Feature representations
# --------------------------------------------------------------------------
st.subheader("What each representation measures")

for name, text in concepts.FEATURE_EXTRACTION_NOTES:
    with st.expander(name):
        st.markdown(text)

# --------------------------------------------------------------------------
# Terms
# --------------------------------------------------------------------------
st.subheader("What the model leans on")

terms_left, terms_right = st.columns([1, 1], gap="large")

with terms_left:
    with st.container(border=True):
        top = pd.DataFrame(metrics.get("best_model", {}).get("top_features", []))
        if not top.empty:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Term": top["feature"],
                        "Points toward": top["direction"],
                        "Weight": top["weight"].abs().round(4),
                    }
                ),
                hide_index=True,
                column_config={"Weight": st.column_config.NumberColumn(format="%.4f")},
            )
            st.caption(
                f"Highest-weighted {best_repr} terms. With a tree ensemble the weight is an "
                "importance share rather than a signed coefficient, so the direction is "
                "supplied by the difference in class means rather than by the model itself."
            )
        else:
            st.caption("No feature attribution recorded.")

with terms_right:
    with st.container(border=True):
        path = figure_path("feature_importance.png")
        if path:
            st.image(str(path), width="stretch")
        else:
            st.caption("Run `python -m src.evaluate_models` to render the figure.")
