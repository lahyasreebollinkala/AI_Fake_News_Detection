"""Overview -- what Veritas is, how it works and what the model can do."""

import streamlit as st

from app.components.cards import confusion_frame, kpi_strip, pipeline_strip
from app.components.charts import label_split_chart, roc_chart
from app.components.layout import empty_state, render_page_header
from app.services.artifacts import load_dataset, load_metrics, load_profile

render_page_header(
    "Veritas",
    "An NLP and machine-learning workbench for judging whether a news article is authentic.",
)
st.markdown(
        "Paste a news article, get a **real / fake** verdict, the model confidence behind it, "
    "and the terms that drove the decision."
)

# --------------------------------------------------------------------------
# Headline numbers
# --------------------------------------------------------------------------
metrics = load_metrics()
profile = load_profile()

if metrics:
    scores = metrics.get("models", {}).get(metrics.get("best_model_name", ""), {})
    split = metrics.get("dataset", {})
    selection = metrics.get("selection", {})
    kpi_strip(
        [
            ("Hold-out accuracy", f"{scores.get('accuracy', 0) * 100:.2f}%"),
            ("F1 score", f"{scores.get('f1_score', 0) * 100:.2f}%"),
            ("ROC-AUC", f"{scores.get('roc_auc') or 0:.4f}"),
            (
                f"{metrics.get('best_representation', 'Features')} features",
                f"{metrics.get('features', {}).get('vocabulary_size', 0):,}",
            ),
            ("Test articles", f"{split.get('test_rows', 0):,}"),
        ],
        caption=(
            f"{metrics.get('best_representation', 'n/a')} + {metrics.get('best_model_name', 'n/a')}, "
            f"selected on a {selection.get('validation_rows', 0):,}-article validation split and "
            f"scored once on a stratified {split.get('test_rows', 0):,} article hold-out drawn "
            f"from {split.get('train_rows', 0) + split.get('test_rows', 0):,} documents."
        ),
    )
else:
    empty_state(
        "analytics",
        "No benchmark available",
        "Run `python -m src.train_models` to populate the headline numbers.",
    )

# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
st.subheader("How a verdict is produced")
pipeline_strip(metrics.get("best_representation", "TF-IDF") if metrics else "TF-IDF")
st.caption(
    "Every stage lives in `src/utils.py`, `src/feature_extraction.py` and "
    "`src/feature_engineering.py`. The same code path runs during training and at inference "
    "time, so what the app shows is what the model saw. The full step-by-step method, with the "
    "exact hyper-parameters, is on the **Pipeline & setup** page and in the **Report**."
)

# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------
st.subheader("The evidence behind the numbers")

evidence_left, evidence_right = st.columns(2)

with evidence_left:
    with st.container(border=True):
        st.markdown("**How often the model is right**")
        matrix = metrics.get("best_model", {}).get("confusion_matrix", {}).get("matrix")
        if matrix:
            st.dataframe(confusion_frame(matrix), hide_index=True)
            st.caption(
                f"Hold-out confusion matrix for {metrics.get('best_model_name', 'the selected model')}. "
                "Off-diagonal cells are the articles it still gets wrong."
            )
        else:
            st.caption("Run `python -m src.train_models` to generate the confusion matrix.")

with evidence_right:
    with st.container(border=True):
        st.markdown("**Separation quality**")
        roc = metrics.get("best_model", {}).get("roc_curve", {})
        if roc.get("fpr"):
            roc_chart(roc)
        else:
            st.caption("Run `python -m src.train_models` to generate the ROC curve.")

# --------------------------------------------------------------------------
# Corpus composition
# --------------------------------------------------------------------------
st.subheader("Training corpus")

if profile:
    labels = profile.get("labels", {})
    label_split_chart(
        {"Real": labels.get("real", 0), "Fake": labels.get("fake", 0)},
        title=f"{labels.get('total', 0):,} labelled articles",
    )
    st.caption(
        f"Median length {profile.get('characters', {}).get('median', 0):,} characters · "
        f"mean {profile.get('words', {}).get('mean', 0):,} words per article."
    )
else:
    try:
        frame = load_dataset()
        counts = frame["label"].value_counts()
        label_split_chart(
            {"Real": int(counts.get(0, 0)), "Fake": int(counts.get(1, 0))},
            title=f"{len(frame):,} labelled articles",
        )
    except FileNotFoundError:
        empty_state(
            "dataset",
            "Dataset not built",
            "Run `python -m src.prepare_dataset` to create `data/GlobalFakeNews_Research2026_v1.csv`.",
        )

# --------------------------------------------------------------------------
# Capabilities
# --------------------------------------------------------------------------
st.subheader("What you can do here")

capabilities = [
    (
        "search",
        "Analyze an article",
        "Paste a headline and body, or pull a labelled row from the corpus, then inspect the "
        "verdict, the class split, the editorial signals and the term attribution.",
    ),
    (
        "upload_file",
        "Run a batch",
        "Upload a CSV, map the text column, and score thousands of rows in one pass with "
        "downloadable results.",
    ),
    (
        "insights",
        "Compare models",
        f"{len(metrics.get('models', {}))} classifiers over "
        f"{len(metrics.get('representations', {}))} representations "
        f"({len(metrics.get('representations', {})) * len(metrics.get('models', {}))} combinations), "
        "with a confusion matrix each, plus the parametric versus non-parametric discussion.",
    ),
    (
        "dataset",
        "Explore the corpus",
        "Distributions, category breakdowns, engineered-feature profiles and a searchable "
        "record browser.",
    ),
    (
        "history",
        "Keep a log",
        "Every analysis is appended to a persistent CSV with its confidence, ready to filter "
        "and download.",
    ),
    (
        "school",
        "Learn the ideas",
        "How AI, supervised machine learning, natural language processing and evaluation fit "
        "together, and why the split is drawn where it is.",
    ),
    (
        "description",
        "Read the report",
        "The full IEEE-formatted write-up -- abstract, eight sections, measured tables and a "
        "figure gallery -- generated from the same artifacts this app renders.",
    ),
    (
        "account_tree",
        "Read the method",
        "Each pipeline stage, the exact hyper-parameters and the commands to rebuild every "
        "artifact from scratch.",
    ),
]

cards = st.columns(3)
for index, (icon, title, body) in enumerate(capabilities):
    with cards[index % 3]:
        with st.container(border=True):
            st.markdown(f":material/{icon}:")
            st.markdown(f"**{title}**")
            st.caption(body)
