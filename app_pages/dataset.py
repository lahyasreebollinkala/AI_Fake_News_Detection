"""Dataset explorer -- distributions, editorial signals and a record browser."""

from __future__ import annotations

import streamlit as st

from app.components.cards import kpi_strip
from app.components.charts import class_mean_chart, label_split_chart, length_histogram, subject_chart
from app.components.layout import empty_state, render_page_header
from app.services.artifacts import figure_path, load_dataset, load_profile
from src.config import ENGINEERED_FEATURES, IMAGE_DIR

render_page_header(
    "Dataset explorer",
    "The corpus the model was fitted on: composition, length behaviour, the engineered "
    "editorial signals, and every individual record.",
)

try:
    frame = load_dataset()
except FileNotFoundError:
    empty_state(
        "dataset",
        "Dataset not built",
        "Run `python -m src.prepare_dataset` and `python -m src.preprocessing` from the "
        "project root, then reload this page.",
    )
    st.stop()

frame = frame.copy()
frame["label_name"] = frame["label"].map({0: "Real", 1: "Fake"})
frame["Characters"] = frame["full_text"].astype(str).str.len()
frame["Words"] = frame["full_text"].astype(str).str.split().str.len()

profile = load_profile()

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
counts = frame["label_name"].value_counts()

kpi_strip(
    [
        ("Articles", f"{len(frame):,}"),
        ("Real", f"{int(counts.get('Real', 0)):,}"),
        ("Fake", f"{int(counts.get('Fake', 0)):,}"),
        ("Median words", f"{int(frame['Words'].median()):,}"),
        ("News desks", f"{frame['subject'].nunique() if 'subject' in frame else 0:,}"),
    ]
)

st.space("small")

composition_left, composition_right = st.columns([1, 2], gap="large")

with composition_left:
    with st.container(border=True):
        label_split_chart(
            {"Real": int(counts.get("Real", 0)), "Fake": int(counts.get("Fake", 0))},
            title="Class balance",
        )
        st.caption(
            f"The corpus is {counts.get('Fake', 0) / max(len(frame), 1) * 100:.1f}% fabricated. "
            "A stratified split keeps that ratio identical in the training and test sets."
        )

with composition_right:
    with st.container(border=True):
        subject_chart(frame["subject"].value_counts(), title="Articles per news desk")
        st.caption("The `subject` column comes from the source corpus and is not a model feature.")

# --------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------
st.subheader("Length behaviour")

length_left, length_right = st.columns(2, gap="large")

with length_left:
    with st.container(border=True):
        length_histogram(frame, "Words", title="Words per article", x_title="Words")

with length_right:
    with st.container(border=True):
        length_histogram(frame, "Characters", title="Characters per article", x_title="Characters")
        st.caption(
            "Fabricated articles in this corpus are, on average, "
            f"{frame.loc[frame['label'] == 1, 'Words'].mean() - frame.loc[frame['label'] == 0, 'Words'].mean():.0f} "
            "words longer than authentic ones. The model exploits far more than length alone."
        )

# --------------------------------------------------------------------------
# Engineered signals
# --------------------------------------------------------------------------
available = [column for column in ENGINEERED_FEATURES if column in frame.columns]
if available:
    st.subheader("Engineered editorial signals")

    st.caption(
        "Five rule-based features computed during `src/prepare_dataset.py`. They are not fed to "
        "the classifier — the model runs on TF-IDF only — but they make the corpus auditable."
    )

    signal_frame = (
        frame.groupby("label_name")[["Words", *available]]
        .mean()
        .round(3)
        .reset_index()
    )

    st.dataframe(
        signal_frame,
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(column.replace("_", " ").capitalize(), format="%.3f")
            for column in ["Words", *available]
        },
    )

    column = st.selectbox("Compare feature", available, key="dataset_feature")

    with st.container(border=True):
        class_mean_chart(
            frame.groupby("label_name")[column].mean(),
            title=f"Mean {column.replace('_', ' ')} per class",
            x_title="Mean value",
        )
        st.caption("Longer bars mean a stronger signal.")

# --------------------------------------------------------------------------
# Record browser
# --------------------------------------------------------------------------
st.subheader("Record browser")

search = st.text_input(
    "Search headlines and bodies", placeholder="Type a word or phrase", key="dataset_search"
)
desk = st.selectbox(
    "News desk", ["All"] + sorted(frame["subject"].dropna().unique().tolist()), key="dataset_desk"
)
verdict = st.segmented_control("Label", ["All", "Real", "Fake"], default="All", key="dataset_label") or "All"

view = frame
if desk != "All":
    view = view[view["subject"] == desk]
if verdict != "All":
    view = view[view["label_name"] == verdict]

needle = search.strip()
if needle:
    haystack = (
        view["title"].fillna("").astype(str) + " " + view["text"].fillna("").astype(str)
    )
    view = view[haystack.str.contains(needle, case=False, regex=False)]

st.caption(f"Showing {len(view):,} of {len(frame):,} articles.")

columns = ["title", "subject", "label_name", "Words", "Characters"]
st.dataframe(
    view[columns].rename(
        columns={"title": "Headline", "subject": "Desk", "label_name": "Label", "Words": "Words", "Characters": "Characters"}
    ),
    hide_index=True,
    height=420,
    column_config={
        "Headline": st.column_config.TextColumn("Headline", width="large"),
        "Label": st.column_config.TextColumn("Label"),
    },
)

with st.expander("Full text of the selected article", icon=":material/article:"):
    if view.empty:
        st.caption("No article matches the current filters.")
    else:
        position = st.selectbox(
            "Article", options=list(range(min(len(view), 200))),
            format_func=lambda index: f"{index + 1}. {str(view.iloc[index]['title'])[:80]}",
            key="dataset_record",
        )
        record = view.iloc[position]
        st.markdown(f"**{record['title']}**")
        st.caption(f"{record['subject']} · labelled {record['label_name'].lower()}")
        st.write(str(record.get("text", ""))[:4000])

st.download_button(
    "Download the filtered slice",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name="veritas_dataset_slice.csv",
    mime="text/csv",
    icon=":material/download:",
    key="dataset_download",
)

if profile:
    st.caption(
        "Profile cached by `python -m src.eda` in `results/dataset_profile.json`."
    )

# --------------------------------------------------------------------------
# Exploratory figures
# --------------------------------------------------------------------------
GALLERY = [
    ("label_distribution.png", "Class balance"),
    ("subject_distribution.png", "Articles per news desk"),
    ("length_distribution.png", "Length distribution"),
    ("word_count_box.png", "Word count by class"),
    ("average_length.png", "Average length by desk"),
    ("engineered_features.png", "Engineered signals by class"),
    ("feature_correlation.png", "Feature correlation"),
]

available = [(name, caption) for name, caption in GALLERY if figure_path(name)]

if available:
    st.subheader("Exploratory figures")

    st.caption(
        f"Static figures rendered by `python -m src.eda_visualization` into `{IMAGE_DIR.name}/`. "
        "They are the same analysis shown above, exported for reports."
    )

    columns = st.columns(2, gap="large")
    for index, (name, caption) in enumerate(available):
        with columns[index % 2]:
            with st.container(border=True):
                st.image(str(figure_path(name)), caption=caption, width="stretch")
