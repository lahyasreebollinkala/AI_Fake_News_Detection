import os
import sys
import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -------------------------------------------------------
# Project Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import clean_text
from src.config import (
    BEST_MODEL,
    VECTORIZER,
    METRICS_FILE,
    RAW_DATA,
    PREDICTION_HISTORY,
    IMAGE_DIR,
)

from sample_news import REAL_NEWS, FAKE_NEWS

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------

st.set_page_config(
    page_title="AI Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# -------------------------------------------------------
# Load CSS
# -------------------------------------------------------

css_file = PROJECT_ROOT / "app" / "style.css"

if css_file.exists():
    with open(css_file, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(BEST_MODEL)
    vectorizer = joblib.load(VECTORIZER)

    metrics = {}

    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, encoding="utf-8") as f:
            metrics = json.load(f)

    return model, vectorizer, metrics


try:

    model, vectorizer, metrics = load_model()

except Exception as e:

    st.error(f"Unable to load model:\n\n{e}")

    st.stop()

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

with st.sidebar:

    st.title("🤖 AI Dashboard")

    st.markdown("---")

    st.subheader("Best Model")

    st.success(metrics.get("best_model_name", "Unknown"))

    st.metric(
        "Best F1 Score",
        f"{metrics.get('best_f1_score',0)*100:.2f}%"
    )

    st.markdown("---")

    st.write("### Pipeline")

    st.write("""
- Text Cleaning
- TF-IDF (5000 Features)
- Machine Learning
- Binary Classification
""")

    st.markdown("---")

    st.caption("AI Fake News Detection")

# -------------------------------------------------------
# Hero
# -------------------------------------------------------

st.title("📰 AI Fake News Detection Dashboard")

st.caption(
    "Real-time Fake News Classification using TF-IDF and Machine Learning"
)

# -------------------------------------------------------
# Tabs
# -------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🎯 Detector",
        "📁 Batch",
        "📊 Models",
        "📈 Dataset",
        "📜 History",
    ]
)

# ==========================================================
# TAB 1 : REAL TIME DETECTOR
# ==========================================================

with tab1:

    st.header("🎯 Real-Time Fake News Detection")

    st.write(
        "Paste a news article below and click **Predict**."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "📰 Load Real Sample",
            use_container_width=True,
        ):
            st.session_state["title"] = "Government Education Policy"
            st.session_state["article"] = REAL_NEWS

    with col2:

        if st.button(
            "🚨 Load Fake Sample",
            use_container_width=True,
        ):
            st.session_state["title"] = "Miracle Discovery"
            st.session_state["article"] = FAKE_NEWS

    article_title = st.text_input(
        "Article Title",
        value=st.session_state.get("title", "")
    )

    article_text = st.text_area(
        "Article Content",
        value=st.session_state.get("article", ""),
        height=250,
    )

    article_text = article_text or ""

    st.divider()

    if st.button(
        "🚀 Predict",
        use_container_width=True,
        type="primary"
    ):

       article_text = article_text or ""

       if article_text.strip() == "":

            st.warning(
                "Please enter article text."
            )

    else:

            cleaned = clean_text(article_text)

            X = vectorizer.transform(
                [cleaned]
            )

            prediction = model.predict(
                X
            )[0]

            if hasattr(
                model,
                "predict_proba"
            ):

                proba = model.predict_proba(
                    X
                )[0]

            else:

                proba = [0.5,0.5]

            confidence = (
                np.max(proba)
                *100
            )

            verdict = (
                "REAL"
                if prediction==0
                else
                "FAKE"
            )

            st.divider()

            c1,c2,c3=st.columns(3)

            with c1:

                if verdict=="REAL":

                    st.success(
                        "✅ REAL NEWS"
                    )

                else:

                    st.error(
                        "❌ FAKE NEWS"
                    )

            with c2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            with c3:

                st.metric(
                    "Best Model",
                    metrics.get(
                        "best_model_name",
                        "Unknown"
                    )
                )

            st.divider()

            st.subheader(
                "📊 Prediction Probability"
            )

            fig = go.Figure()

            fig.add_trace(

                go.Bar(

                    x=[
                        "REAL",
                        "FAKE"
                    ],

                    y=[
                        proba[0],
                        proba[1]
                    ],

                    marker_color=[
                        "#16a34a",
                        "#dc2626"
                    ]

                )

            )

            fig.update_layout(

                height=350,

                template="plotly_dark",

                yaxis_title="Probability"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.divider()

            st.subheader(
                "📝 Text Statistics"
            )

            s1,s2,s3=st.columns(3)

            with s1:

                st.metric(
                    "Words",
                    len(
                        article_text.split()
                    )
                )

            with s2:

                st.metric(
                    "Characters",
                    len(
                        article_text
                    )
                )

            with s3:

                st.metric(
                    "Clean Tokens",
                    len(
                        cleaned.split()
                    )
                )

            st.divider()

            st.subheader(
                "🧹 Cleaned Text"
            )

            st.code(cleaned)

            history = pd.DataFrame({

                "Timestamp":[
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                ],

                "Title":[
                    article_title
                ],

                "Prediction":[
                    verdict
                ],

                "Confidence":[
                    f"{confidence:.2f}%"
                ]

            })

            if os.path.exists(
                PREDICTION_HISTORY
            ):

                history.to_csv(

                    PREDICTION_HISTORY,

                    mode="a",

                    header=False,

                    index=False

                )

            else:

                history.to_csv(

                    PREDICTION_HISTORY,

                    index=False

                )

                # ==========================================================
# TAB 2 : BATCH PREDICTION
# ==========================================================

with tab2:

    st.header("📁 Batch Fake News Detection")

    st.write(
        "Upload a CSV file containing news articles."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(uploaded_file)

            st.success(
                f"Loaded {len(batch_df)} rows."
            )

            st.dataframe(
                batch_df.head(),
                use_container_width=True
            )

            st.divider()

            text_column = st.selectbox(
                "Select Article Text Column",
                batch_df.columns
            )

            title_column = st.selectbox(
                "Select Title Column (Optional)",
                ["None"] + batch_df.columns.tolist()
            )

            if st.button(
                "🚀 Run Batch Prediction",
                use_container_width=True,
                type="primary"
            ):

                predictions = []
                confidences = []

                progress = st.progress(0)

                for idx, (_, row) in enumerate(batch_df.iterrows()):

                    article = str(
                        row[text_column]
                    )

                    cleaned = clean_text(
                        article
                    )

                    X = vectorizer.transform(
                        [cleaned]
                    )

                    pred = model.predict(
                        X
                    )[0]

                    if hasattr(
                        model,
                        "predict_proba"
                    ):

                        proba = model.predict_proba(
                            X
                        )[0]

                    else:

                        proba = [0.5,0.5]

                    label = (
                        "REAL"
                        if pred==0
                        else
                        "FAKE"
                    )

                    predictions.append(label)

                    confidences.append(
                        round(
                            np.max(proba)*100,
                            2
                        )
                    )

                    progress.progress(
                        (idx+1)/len(batch_df)
                    )

                batch_df["Prediction"] = predictions
                batch_df["Confidence"] = confidences

                st.success(
                    "Batch Prediction Completed!"
                )

                st.divider()

                c1,c2,c3,c4 = st.columns(4)

                with c1:

                    st.metric(
                        "Articles",
                        len(batch_df)
                    )

                with c2:

                    st.metric(
                        "REAL",
                        predictions.count("REAL")
                    )

                with c3:

                    st.metric(
                        "FAKE",
                        predictions.count("FAKE")
                    )

                with c4:

                    st.metric(
                        "Average Confidence",
                        f"{np.mean(confidences):.2f}%"
                    )

                st.divider()

                pie = px.pie(

                    names=["REAL","FAKE"],

                    values=[
                        predictions.count("REAL"),
                        predictions.count("FAKE")
                    ],

                    title="Prediction Distribution",

                    color=["REAL","FAKE"],

                    color_discrete_map={
                        "REAL":"green",
                        "FAKE":"red"
                    }

                )

                st.plotly_chart(
                    pie,
                    use_container_width=True
                )

                st.divider()

                st.dataframe(
                    batch_df,
                    use_container_width=True
                )

                csv = batch_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(

                    "📥 Download Results",

                    csv,

                    "Batch_Predictions.csv",

                    "text/csv",

                    use_container_width=True

                )

        except Exception as e:

            st.error(
                str(e)
            )

            # ==========================================================
# TAB 3 : MODEL BENCHMARKS
# ==========================================================

with tab3:

    st.header("📊 Model Performance Dashboard")

    if not metrics:

        st.warning("Metrics file not found.")

    else:

        models = metrics.get("models", {})

        if len(models) == 0:

            st.warning("No model metrics available.")

        else:

            metric_df = pd.DataFrame(models).T.reset_index()

            metric_df.rename(
                columns={"index": "Model"},
                inplace=True
            )

            metric_df = metric_df.round(4)

            st.subheader("📋 Performance Table")

            st.dataframe(
                metric_df,
                use_container_width=True
            )

            st.divider()

            c1, c2 = st.columns(2)

            with c1:

                fig = px.bar(

                    metric_df,

                    x="Model",

                    y="accuracy",

                    text="accuracy",

                    color="Model",

                    title="Accuracy Comparison"

                )

                fig.update_traces(
                    texttemplate="%{text:.3f}",
                    textposition="outside"
                )

                fig.update_layout(
                    height=450
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with c2:

                fig = px.bar(

                    metric_df,

                    x="Model",

                    y="f1_score",

                    text="f1_score",

                    color="Model",

                    title="F1 Score Comparison"

                )

                fig.update_traces(
                    texttemplate="%{text:.3f}",
                    textposition="outside"
                )

                fig.update_layout(
                    height=450
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.divider()

            c1, c2 = st.columns(2)

            with c1:

                fig = px.bar(

                    metric_df,

                    x="Model",

                    y="precision",

                    text="precision",

                    color="Model",

                    title="Precision Comparison"

                )

                fig.update_traces(
                    texttemplate="%{text:.3f}",
                    textposition="outside"
                )

                fig.update_layout(
                    height=450
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with c2:

                fig = px.bar(

                    metric_df,

                    x="Model",

                    y="recall",

                    text="recall",

                    color="Model",

                    title="Recall Comparison"

                )

                fig.update_traces(
                    texttemplate="%{text:.3f}",
                    textposition="outside"
                )

                fig.update_layout(
                    height=450
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.divider()

            if "roc_auc" in metric_df.columns:

                fig = px.line(

                    metric_df,

                    x="Model",

                    y="roc_auc",

                    markers=True,

                    title="ROC-AUC Comparison"

                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.divider()

            st.subheader("🏆 Best Model")

            best_model = metrics.get(
                "best_model_name",
                "Unknown"
            )

            best = models.get(best_model, {})

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Accuracy",
                    f"{best.get('accuracy',0)*100:.2f}%"
                )

            with c2:

                st.metric(
                    "Precision",
                    f"{best.get('precision',0)*100:.2f}%"
                )

            with c3:

                st.metric(
                    "Recall",
                    f"{best.get('recall',0)*100:.2f}%"
                )

            with c4:

                st.metric(
                    "F1 Score",
                    f"{best.get('f1_score',0)*100:.2f}%"
                )

            st.success(
                f"🏆 Best Performing Model : {best_model}"
            )

            st.divider()

            st.subheader("🖼 Generated Evaluation Figures")

            c1, c2 = st.columns(2)

            with c1:

                cm_path = os.path.join(
                    IMAGE_DIR,
                    "confusion_matrix.png"
                )

                if os.path.exists(cm_path):

                    st.image(
                        cm_path,
                        caption="Confusion Matrix",
                        use_container_width=True
                    )

            with c2:

                roc_path = os.path.join(
                    IMAGE_DIR,
                    "roc_curve.png"
                )

                if os.path.exists(roc_path):

                    st.image(
                        roc_path,
                        caption="ROC Curve",
                        use_container_width=True
                    )

            fi_path = os.path.join(
                IMAGE_DIR,
                "feature_importance.png"
            )

            if os.path.exists(fi_path):

                st.divider()

                st.image(
                    fi_path,
                    caption="Top TF-IDF Features",
                    use_container_width=True
                )

                # ==========================================================
# TAB 4 : DATASET EXPLORER
# ==========================================================

with tab4:

    st.header("📈 Dataset Explorer")

    if os.path.exists(RAW_DATA):

        df = pd.read_csv(RAW_DATA)

        st.success(f"Dataset Loaded Successfully ({len(df)} records)")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Total Articles", len(df))

        with c2:
            st.metric(
                "Real Articles",
                int((df["label"] == 0).sum())
            )

        with c3:
            st.metric(
                "Fake Articles",
                int((df["label"] == 1).sum())
            )

        st.divider()

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.divider()

        st.subheader("Real vs Fake Distribution")

        fig = px.histogram(
            df,
            x="label",
            color="label",
            text_auto=True,
            color_discrete_map={
                0: "#16a34a",
                1: "#dc2626"
            }
        )

        fig.update_layout(
            xaxis_title="Label",
            yaxis_title="Count",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        st.subheader("Article Length Distribution")

        df["Article_Length"] = (
            df["full_text"]
            .astype(str)
            .apply(len)
        )

        fig = px.histogram(
            df,
            x="Article_Length",
            nbins=40,
            color="label",
            color_discrete_map={
                0:"#16a34a",
                1:"#dc2626"
            }
        )

        fig.update_layout(
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        st.subheader("Average Article Length")

        avg = df.groupby(
            "label"
        )["Article_Length"].mean().reset_index()

        fig = px.bar(
            avg,
            x="label",
            y="Article_Length",
            text="Article_Length",
            color="label",
            color_discrete_map={
                0:"#16a34a",
                1:"#dc2626"
            }
        )

        fig.update_traces(
            texttemplate="%{text:.0f}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Dataset not found."
        )


# ==========================================================
# TAB 5 : PREDICTION HISTORY
# ==========================================================

with tab5:

    st.header("📜 Prediction History")

    if os.path.exists(PREDICTION_HISTORY):

        history = pd.read_csv(
            PREDICTION_HISTORY
        )

        if len(history):

            st.success(
                f"{len(history)} Predictions Found"
            )

            search = st.text_input(
                "🔍 Search Title"
            )

            if search.strip():

                history = history[
                    history["Title"]
                    .str.contains(
                        search,
                        case=False,
                        na=False
                    )
                ]

            st.dataframe(
                history,
                use_container_width=True
            )

            st.divider()

            c1,c2 = st.columns(2)

            with c1:

                csv = history.to_csv(
                    index=False
                ).encode()

                st.download_button(

                    "📥 Download History",

                    csv,

                    "Prediction_History.csv",

                    "text/csv",

                    use_container_width=True

                )

            with c2:

                if st.button(

                    "🗑 Clear History",

                    use_container_width=True

                ):

                    os.remove(
                        PREDICTION_HISTORY
                    )

                    st.rerun()

            st.divider()

            st.subheader(
                "Prediction Distribution"
            )

            fig = px.pie(

                history,

                names="Prediction",

                title="History Summary",

                color="Prediction",

                color_discrete_map={
                    "REAL":"green",
                    "FAKE":"red"
                }

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "History is empty."
            )

    else:

        st.info(
            "No history found."
        )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown(
"""
<div style="text-align:center;padding:20px">

### 🤖 AI Fake News Detection Dashboard

Developed using

• Python

• Streamlit

• Scikit-Learn

• TF-IDF

• Plotly

<hr>

**Version 2.0**

</div>
""",
unsafe_allow_html=True
)