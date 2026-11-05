"""Streamlit front-end: score an uploaded test CSV with pre-trained classifiers.

No model is (re)trained here - saved pipelines are loaded from model/.
"""
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from common import (
    CLASS_NAMES, MODEL_FILES, TARGET,
    compute_metrics, encode_target, load_csv, report_and_matrix,
)

HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Student Outcome Predictor", page_icon="🎓", layout="wide")


@st.cache_resource
def get_model(name: str):
    return joblib.load(os.path.join(HERE, MODEL_FILES[name]))


def score(name: str, X: pd.DataFrame, y):
    model = get_model(name)
    X = X[list(model.feature_names_in_)]  # align column order with training
    pred = model.predict(X)
    proba = model.predict_proba(X)
    return pred, compute_metrics(y, pred, proba)


# ---------------------------------------------------------------- header
st.title("🎓 Student Dropout & Academic Success Classifier")
st.caption(
    "Predicts whether a student will **Dropout**, remain **Enrolled** or **Graduate** "
    "(UCI dataset, 36 features, 3 classes). Five pre-trained models are compared on the "
    "test file you upload."
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1 · Upload test data")
    uploaded = st.file_uploader("CSV with all features + `Target` column", type="csv")
    use_sample = False
    if uploaded is None and os.path.exists(os.path.join(HERE, "test_data.csv")):
        use_sample = st.checkbox("Use bundled test_data.csv", value=False)

    st.header("2 · Choose model")
    model_name = st.selectbox("Classifier", list(MODEL_FILES))
    st.markdown("---")
    st.caption("Upload only the held-out test split (free-tier memory is limited).")

source = uploaded if uploaded is not None else (
    os.path.join(HERE, "test_data.csv") if use_sample else None
)
if source is None:
    st.info("⬅️ Upload a test CSV (or tick the bundled sample) to begin.")
    st.stop()

# ---------------------------------------------------------------- validate
df = load_csv(source)
if TARGET not in df.columns:
    st.error(f"Column `{TARGET}` is missing - the app needs it to compute metrics.")
    st.stop()

needed = list(get_model(model_name).feature_names_in_)
missing = [c for c in needed if c not in df.columns]
if missing:
    st.error(f"Missing feature columns: {missing}")
    st.stop()

y_true = encode_target(df[TARGET])
if pd.isna(y_true).any():
    st.error(f"`{TARGET}` must only contain: {', '.join(CLASS_NAMES)}.")
    st.stop()
X = df.drop(columns=[TARGET])

with st.expander(f"Data preview - {len(df)} rows", expanded=False):
    st.dataframe(df.head(20), width="stretch")
    st.write("Class distribution:", df[TARGET].value_counts().to_dict())

# ---------------------------------------------------------------- selected model
pred, metrics = score(model_name, X, y_true)
st.subheader(f"Results · {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")
st.caption("Precision / Recall / F1 are macro-averaged; AUC is one-vs-rest macro.")

left, right = st.columns(2)
cm, report_df = report_and_matrix(y_true, pred)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    st.dataframe(report_df.round(3), width="stretch")

# ---------------------------------------------------------------- all models
st.subheader("All models on this test file")
rows = {n: score(n, X, y_true)[1] for n in MODEL_FILES}
comp = pd.DataFrame(rows).T.round(4)
comp.index.name = "ML Model Name"
st.dataframe(comp.style.highlight_max(axis=0, color="#d9f2d9"), width="stretch")
st.success(f"Best by MCC on this data: **{comp['MCC'].idxmax()}**")
