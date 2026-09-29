"""Pipeline & setup -- methodology, artifact health, and how to rebuild."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.cards import dataset_table, kpi_strip, pipeline_strip
from app.components.charts import label_split_chart
from app.components.layout import empty_state, render_page_header
from app.navigation import page_count
from app.services.artifacts import artifact_status, figure_path, load_dataset, load_metrics
from src import concepts
from src.config import (
    CLEAN_DATA,
    EMBEDDING_COMPONENTS,
    IMAGE_DIR,
    LABEL_NAMES,
    MAX_FEATURES,
    METRICS_FILE,
    MIN_DF,
    NGRAM_RANGE,
    RANDOM_STATE,
    REPRESENTATIONS,
    SAMPLE_DATA,
    TEST_SIZE,
    VALIDATION_SIZE,
)

render_page_header(
    "Pipeline & setup",
    "How Veritas goes from raw news to a verdict, what each artefact is for, and how to "
    "rebuild any of it.",
)

# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
status = pd.DataFrame(artifact_status())
status["Ready"] = status["Present"].map({True: "Ready", False: "Missing"})

st.subheader("Artefact health")

missing = status[status["Ready"] == "Missing"]

if missing.empty:
    st.success(
        f"Every required artefact is present. All {page_count()} pages are live.",
        icon=":material/check_circle:",
    )
else:
    st.error(
        f"{len(missing)} artefact(s) missing: "
        + ", ".join(missing["Artifact"].tolist())
        + ". Run the commands below to rebuild them.",
        icon=":material/error:",
    )

st.dataframe(
    status[["Artifact", "Path", "Ready"]],
    hide_index=True,
    column_config={
        "Path": st.column_config.TextColumn("Path", width="large"),
        "Ready": st.column_config.TextColumn("State"),
    },
)

# --------------------------------------------------------------------------
# Methodology
# --------------------------------------------------------------------------
st.subheader("Methodology")

metrics = load_metrics()
deployed_repr = metrics.get("best_representation", "TF-IDF")
selection = metrics.get("selection", {})

pipeline_strip(deployed_repr)

st.caption(
    f"Representations: `BagOfWords(CountVectorizer(max_features={MAX_FEATURES}, "
    f"ngram_range={NGRAM_RANGE}, min_df={MIN_DF}))`, "
    f"`TfidfRepresentation(TfidfVectorizer(same arguments, sublinear_tf=True))` and "
    f"`LsaEmbeddings(n_components={EMBEDDING_COMPONENTS})` over "
    f"{', '.join(REPRESENTATIONS)}. "
    f"Split: `train_test_split(test_size={TEST_SIZE}, stratify=y, random_state={RANDOM_STATE})`, "
    f"then a `{VALIDATION_SIZE:.0%}` validation split of the training side for selection. "
    f"All {len(REPRESENTATIONS) * 4} combinations are benchmarked; the highest validation F1 "
    "is persisted as `models/best_model.pkl` together with the representation it was "
    "fitted on."
)

if selection:
    st.caption(
        f"Last run selected **{selection.get('winner', {}).get('representation')} + "
        f"{selection.get('winner', {}).get('model')}** at validation F1 "
        f"{selection.get('validation_f1', 0):.4f} on {selection.get('validation_rows', 0):,} "
        "articles, then scored it once on the hold-out."
    )

with st.container(border=True):
    st.markdown("**Label convention**")
    left, right = st.columns(2, gap="medium")
    with left:
        st.caption(f"`0` → **{LABEL_NAMES[0]}** — reported by an established outlet.")
    with right:
        st.caption(f"`1` → **{LABEL_NAMES[1]}** — fabricated or machine-generated.")

st.markdown("**Text preprocessing**")
for name, text in concepts.PREPROCESSING_STEPS:
    st.caption(f"**{name}** — {text}")

st.markdown("**Feature extraction**")
for name, text in concepts.FEATURE_EXTRACTION_NOTES:
    st.caption(f"**{name}** — {text}")

with st.expander("Full pipeline, step by step", icon=":material/account_tree:", expanded=True):
    steps = [
        (
            "Acquire",
            "Load `data/Fake.csv` and `data/True.csv` when present, otherwise fall back to "
            "`data/sample_dataset.csv`. Both schemas are normalised to "
            "`title / text / subject / date / label`, and duplicate headline-plus-body pairs "
            "are removed.",
        ),
        (
            "Clean",
            "Lowercase, fold Unicode, strip URLs, HTML, digits and punctuation, drop stopwords "
            "and lemmatise. The cleaned body is written to `data/cleaned_news.csv` alongside "
            "the five engineered editorial signals.",
        ),
        (
            "Split",
            "Draw the stratified 80/20 hold-out and write the documents and labels to disk "
            "**before** any representation is fitted, so the vocabulary and the embedding "
            "basis can never see a test article.",
        ),
        (
            "Represent",
            f"Fit all three required representations on the training documents only: "
            f"Bag-of-Words, TF-IDF and {EMBEDDING_COMPONENTS}-component latent-semantic "
            "embeddings.",
        ),
        (
            "Benchmark and select",
            "Train K-Nearest Neighbours, Logistic Regression, Random Forest and an MLP on "
            "every representation, score them on a validation split of the training side, and "
            "choose the highest F1.",
        ),
        (
            "Score once",
            "Write accuracy, precision, recall, F1, ROC-AUC, the confusion matrix and the "
            "per-class breakdown for every candidate to `results/model_metrics.json`.",
        ),
        (
            "Persist",
            "Save the winner to `models/best_model.pkl` together with the fitted representation "
            "and the ordered feature names.",
        ),
        (
            "Serve",
            "The app reloads the bundle, transforms the incoming text and returns class "
            "probabilities. Linear models are explained by coefficient, tree ensembles by "
            "term occlusion on the feature matrix.",
        ),
    ]
    for number, (title, body) in enumerate(steps, start=1):
        st.markdown(f"**{number}. {title}**")
        st.caption(body)

# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
st.subheader("Rebuild from scratch")

commands = [
    ("python -m src.prepare_dataset", "Merge, de-duplicate and engineer the corpus."),
    ("python -m src.preprocessing", "Write `data/cleaned_news.csv` with the cleaned text."),
    ("python -m src.feature_engineering", "Draw the stratified train/test split and persist it."),
    ("python -m src.train_models", "Benchmark 3 x 4 combinations and persist the winner."),
    ("python -m src.evaluate_models", "Render the evaluation figures into `images/`."),
    ("python -m src.eda", "Print the exploratory report and cache `results/dataset_profile.json`."),
    ("python -m src.eda_visualization", "Render the exploratory charts into `images/`."),
    ("python -m src.report", "Write `docs/ieee_report.md` and `docs/ieee_report.tex`."),
    ("python -m pytest tests", "Run the pipeline and route tests."),
    ("streamlit run streamlit_app.py", "Start this application."),
]

for command, description in commands:
    code_left, code_right = st.columns([1, 1], gap="medium", vertical_alignment="center")
    with code_left:
        st.code(command, language="powershell")
    with code_right:
        st.caption(description)

with st.container(border=True):
    st.caption(
        f"Figures are written to `{IMAGE_DIR}`. Benchmark results to `{METRICS_FILE.name}`. "
        "The app only needs `best_model.pkl` and `vectorizer.pkl` to return a verdict; "
        "everything else is supporting evidence."
    )

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
st.subheader("Data at a glance")

try:
    frame = load_dataset()
except FileNotFoundError:
    frame = None

if frame is None or frame.empty:
    empty_state(
        "dataset",
        "Dataset not built",
        f"Expected `{CLEAN_DATA.name}`. Run `python -m src.preprocessing` to create it.",
    )
else:
    counts = frame["label"].value_counts()
    data_left, data_right = st.columns([1, 2], gap="large")

    with data_left:
        with st.container(border=True):
            label_split_chart(
                {"Real": int(counts.get(0, 0)), "Fake": int(counts.get(1, 0))},
                title="Class balance",
            )

    with data_right:
        with st.container(border=True):
            kpi_strip(
                [
                    ("Rows", f"{len(frame):,}"),
                    ("Columns", f"{frame.shape[1]}"),
                    ("Subjects", f"{frame['subject'].nunique()}"),
                    ("Median words", f"{int(frame['full_text'].astype(str).str.split().str.len().median()):,}"),
                ]
            )
            dataset_table(frame[["title", "subject", "label"]].head(8))

    st.caption(
        f"Bundled corpus: `{SAMPLE_DATA.name}`. Cleaned output: `{CLEAN_DATA.name}`. "
        "Adding `Fake.csv` and `True.csv` to `data/` and re-running the pipeline switches the "
        "application to the full dataset automatically."
    )

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
st.subheader("Recorded run")

if metrics:
    figure_left, figure_right = st.columns(2, gap="large")
    with figure_left:
        st.caption(f"Last benchmark: `{metrics.get('generated_at', 'unknown')}`")
        st.caption(
            f"Train {metrics.get('dataset', {}).get('train_rows', 0):,} rows · "
            f"test {metrics.get('dataset', {}).get('test_rows', 0):,} rows · "
            f"vocabulary {metrics.get('features', {}).get('vocabulary_size', 0):,} terms"
        )
    with figure_right:
        path = figure_path("metric_comparison.png")
        if path:
            st.image(str(path), width="stretch")
        else:
            st.caption("Run `python -m src.evaluate_models` to render the comparison figure.")
