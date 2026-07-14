"""
train_model.py
--------------
Loads the collected LDR dataset, engineers features, trains a
Random Forest Classifier, evaluates it, and saves the model.

Usage:
    python python/train_model.py --data data/sample_data.csv

Outputs:
    python/model.pkl          – serialised trained model
    Printed evaluation report + confusion matrix
    Plots: confusion matrix heat-map, feature importance bar chart
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# Add project root to path so utils is importable
sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_PATH, FEATURE_NAMES, MODEL_PATH, encode_labels, extract_features


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_ESTIMATORS  = 100
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
CV_FOLDS      = 5


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"timestamp", "ldr_value"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}. Found: {set(df.columns)}")
    df.dropna(subset=["timestamp", "ldr_value"], inplace=True)
    df["ldr_value"] = pd.to_numeric(df["ldr_value"], errors="coerce")
    df.dropna(subset=["ldr_value"], inplace=True)
    return df


def train(csv_path: str, model_output: str) -> None:
    # ---- Load & preprocess ----
    print(f"[train] Loading data from '{csv_path}' ...")
    df = load_data(csv_path)
    print(f"[train] {len(df)} samples loaded.")

    X = extract_features(df)
    y = encode_labels(df)

    class_counts = y.value_counts().rename({0: "Dark", 1: "Light"})
    print(f"[train] Class distribution:\n{class_counts.to_string()}\n")

    # ---- Train / test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ---- Model ----
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ---- Cross-validation ----
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"[train] {CV_FOLDS}-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ---- Hold-out evaluation ----
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[train] Test accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
    print("[train] Classification report:")
    print(classification_report(y_test, y_pred, target_names=["Dark", "Light"]))

    # ---- Confusion matrix plot ----
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Dark", "Light"],
        yticklabels=["Dark", "Light"],
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # ---- Feature importance plot ----
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    axes[1].bar(
        range(len(FEATURE_NAMES)),
        importances[indices],
        color="steelblue",
        edgecolor="white",
    )
    axes[1].set_xticks(range(len(FEATURE_NAMES)))
    axes[1].set_xticklabels([FEATURE_NAMES[i] for i in indices], rotation=30, ha="right")
    axes[1].set_title("Feature Importances")
    axes[1].set_ylabel("Importance")

    plt.tight_layout()
    Path("assets").mkdir(parents=True, exist_ok=True)
    plt.savefig("assets/model_evaluation.png", dpi=150)
    print("[train] Evaluation plots saved to 'assets/model_evaluation.png'.")
    plt.show()

    # ---- Save model ----
    Path(model_output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_output)
    print(f"[train] Model saved to '{model_output}'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Random Forest classifier on LDR sensor data."
    )
    parser.add_argument(
        "--data",
        default=DATA_PATH,
        help=f"Path to CSV dataset (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help=f"Where to save the trained model (default: {MODEL_PATH})",
    )
    args = parser.parse_args()
    train(csv_path=args.data, model_output=args.model)
