"""Train the five classifiers on the UCI 'Predict Students' Dropout and Academic
Success' dataset, evaluate on a held-out split, and persist everything the
Streamlit app needs (models, test_data.csv, metrics).

Run:  python train_models.py
"""
import json
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import CLASS_NAMES, MODEL_FILES, TARGET, compute_metrics, encode_target, load_csv

SEED = 42
RAW_PATH = "data/students_dropout_full.csv"


def build_models() -> dict:
    # Scaling is baked into each pipeline so the saved artefact is self-contained.
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=SEED)),
        ]),
        "KNN": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=11, weights="distance")),
        ]),
        "Naive Bayes": Pipeline([
            ("scale", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)),
        ]),
    }


def main():
    df = load_csv(RAW_PATH)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1] - 1} features; classes:\n{df[TARGET].value_counts()}\n")

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=SEED
    )
    # The file uploaded to the app: features + string target, never used for fitting.
    test_df.to_csv("test_data.csv", index=False)

    X_train = train_df.drop(columns=[TARGET])
    y_train = encode_target(train_df[TARGET])
    X_test = test_df.drop(columns=[TARGET])
    y_test = encode_target(test_df[TARGET])

    os.makedirs("model", exist_ok=True)
    rows = {}
    for name, pipe in build_models().items():
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODEL_FILES[name])
        rows[name] = compute_metrics(y_test, pipe.predict(X_test), pipe.predict_proba(X_test))
        print(f"trained + saved: {name}")

    table = pd.DataFrame(rows).T.round(4)
    table.index.name = "ML Model Name"
    print("\nTest-set comparison (macro averages, AUC = one-vs-rest macro):\n")
    print(table.to_string())

    table.to_csv("model/metrics_summary.csv")
    with open("model/metrics_summary.json", "w") as fh:
        json.dump({"classes": CLASS_NAMES, "metrics": table.to_dict(orient="index")}, fh, indent=2)
    with open("model/metrics_table.md", "w") as fh:
        cols = ["ML Model Name"] + list(table.columns)
        fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for name, r in table.iterrows():
            fh.write("| " + name + " | " + " | ".join(f"{v:.4f}" for v in r) + " |\n")


if __name__ == "__main__":
    main()
