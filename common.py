"""Shared constants and helpers used by both train_models.py and app.py."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET = "Target"
CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]  # fixed order -> label ids 0,1,2

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}


def load_csv(source) -> pd.DataFrame:
    """Read the UCI csv (semicolon separated) or a comma separated export of it."""
    df = pd.read_csv(source, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Multi-class metrics: macro averaged P/R/F1, one-vs-rest macro AUC."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=[0, 1, 2]
        ),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def report_and_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    rep = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
        zero_division=0, output_dict=True,
    )
    return cm, pd.DataFrame(rep).T


def encode_target(series: pd.Series) -> np.ndarray:
    mapping = {name: i for i, name in enumerate(CLASS_NAMES)}
    return series.str.strip().map(mapping).to_numpy()
