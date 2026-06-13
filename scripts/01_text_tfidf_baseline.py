"""
Text TF-IDF Baseline on Fixed Splits
====================================

Scop:
- folosește spliturile fixe create anterior;
- antrenează modele ML clasice pe case_text;
- folosește TF-IDF ca reprezentare textuală;
- selectează modelul pe validation;
- evaluează modelul selectat pe test;
- salvează metrici, rapoarte, predicții și grafice.

Input:
data/splits/train.csv
data/splits/valid.csv
data/splits/test.csv

Output:
outputs/01_text_tfidf_baseline/
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================

TRAIN_PATH = Path("data/splits/train.csv")
VALID_PATH = Path("data/splits/valid.csv")
TEST_PATH = Path("data/splits/test.csv")

OUTPUT_DIR = Path("outputs/01_text_tfidf_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "case_text"
LABEL_COL = "label"

RANDOM_STATE = 42
TOP_K = 5


# =========================================================
# HELPERS
# =========================================================

def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {split_name} split: {path}")

    df = pd.read_csv(path)

    required_cols = [TEXT_COL, LABEL_COL]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"{split_name} split is missing columns: {missing}")

    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()

    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower()

    df = df[df[TEXT_COL].str.len() > 0].copy()
    df = df[df[LABEL_COL].str.len() > 0].copy()

    return df


def validate_label_coverage(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    train_labels = set(train_df[LABEL_COL].unique())
    valid_labels = set(valid_df[LABEL_COL].unique())
    test_labels = set(test_df[LABEL_COL].unique())

    missing_valid = valid_labels - train_labels
    missing_test = test_labels - train_labels

    print_section("LABEL COVERAGE")
    print(f"Train classes: {len(train_labels)}")
    print(f"Valid classes: {len(valid_labels)}")
    print(f"Test classes:  {len(test_labels)}")

    if missing_valid:
        raise ValueError(
            f"Validation contains labels not present in train: {sorted(missing_valid)}"
        )

    if missing_test:
        raise ValueError(
            f"Test contains labels not present in train: {sorted(missing_test)}"
        )


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def build_models() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=3000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),

        "Logistic Regression Balanced": LogisticRegression(
            C=1.0,
            max_iter=3000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Linear SVM Calibrated": CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
            cv=3,
        ),

        "SGD Classifier": SGDClassifier(
            loss="log_loss",
            alpha=0.0001,
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Complement Naive Bayes": ComplementNB(
            alpha=0.1,
        ),
    }


def get_scores(model: Any, X: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    raise ValueError("Model has neither predict_proba nor decision_function.")


def compute_metrics(model: Any, X: Any, y_true: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X)
    scores = get_scores(model, X)

    k = min(TOP_K, len(model.classes_))

    return {
        "accuracy": accuracy_score(y_true, y_pred),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        f"top{TOP_K}_accuracy": top_k_accuracy_score(
            y_true,
            scores,
            k=k,
            labels=model.classes_,
        ),
    }


def build_predictions_df(
    model: Any,
    X: Any,
    df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    y_pred = model.predict(X)
    scores = get_scores(model, X)

    classes = np.array(model.classes_)
    k = min(TOP_K, len(classes))

    top_idx = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    top_labels = classes[top_idx]
    top_scores = np.take_along_axis(scores, top_idx, axis=1)

    result = pd.DataFrame({
        "split": split_name,
        "case_text": df[TEXT_COL].values,
        "true_label": df[LABEL_COL].values,
        "predicted_label": y_pred,
        "correct": df[LABEL_COL].values == y_pred,
    })

    for i in range(k):
        result[f"top_{i + 1}_label"] = top_labels[:, i]
        result[f"top_{i + 1}_score"] = top_scores[:, i]

    return result


def save_classification_report(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n\n")
        f.write(report)


def save_confusion_outputs(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
) -> None:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    cm_df.to_csv(
        OUTPUT_DIR / "best_model_test_confusion_matrix.csv",
        index=True,
    )

    confusion_rows = []

    for true_label in labels:
        for predicted_label in labels:
            value = cm_df.loc[true_label, predicted_label]

            if true_label != predicted_label and value > 0:
                confusion_rows.append({
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": int(value),
                })

    confusion_pairs = pd.DataFrame(confusion_rows)

    if not confusion_pairs.empty:
        confusion_pairs = confusion_pairs.sort_values(
            by="count",
            ascending=False,
        )

        confusion_pairs.to_csv(
            OUTPUT_DIR / "best_model_test_top_confusions.csv",
            index=False,
        )


def plot_validation_comparison(results_df: pd.DataFrame) -> None:
    plot_df = results_df.sort_values(
        by="valid_macro_f1",
        ascending=True,
    )

    metrics = [
        "valid_accuracy",
        "valid_macro_f1",
        "valid_weighted_f1",
        f"valid_top{TOP_K}_accuracy",
    ]

    for metric in metrics:
        plt.figure(figsize=(12, 6))

        plt.barh(
            plot_df["model"],
            plot_df[metric],
        )

        plt.xlabel(metric)
        plt.title(f"TF-IDF Baseline - {metric}")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{metric}.png",
            dpi=300,
        )

        plt.close()

    x = np.arange(len(results_df))
    width = 0.2

    plt.figure(figsize=(14, 7))

    plt.bar(
        x - 1.5 * width,
        results_df["valid_accuracy"],
        width,
        label="Accuracy",
    )

    plt.bar(
        x - 0.5 * width,
        results_df["valid_macro_f1"],
        width,
        label="Macro F1",
    )

    plt.bar(
        x + 0.5 * width,
        results_df["valid_weighted_f1"],
        width,
        label="Weighted F1",
    )

    plt.bar(
        x + 1.5 * width,
        results_df[f"valid_top{TOP_K}_accuracy"],
        width,
        label=f"Top-{TOP_K}",
    )

    plt.xticks(
        x,
        results_df["model"],
        rotation=25,
        ha="right",
    )

    plt.ylabel("Score")
    plt.title("TF-IDF Baseline - Validation Model Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "validation_model_comparison_all_metrics.png",
        dpi=300,
    )

    plt.close()


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    print_section("LOAD FIXED SPLITS")

    train_df = load_split(TRAIN_PATH, "train")
    valid_df = load_split(VALID_PATH, "valid")
    test_df = load_split(TEST_PATH, "test")

    print(f"Train rows: {len(train_df)}")
    print(f"Valid rows: {len(valid_df)}")
    print(f"Test rows:  {len(test_df)}")

    validate_label_coverage(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )

    y_train = train_df[LABEL_COL]
    y_valid = valid_df[LABEL_COL]
    y_test = test_df[LABEL_COL]

    print_section("TF-IDF VECTORIZATION")

    vectorizer = build_vectorizer()

    X_train = vectorizer.fit_transform(train_df[TEXT_COL])
    X_valid = vectorizer.transform(valid_df[TEXT_COL])
    X_test = vectorizer.transform(test_df[TEXT_COL])

    print(f"Train matrix: {X_train.shape}")
    print(f"Valid matrix: {X_valid.shape}")
    print(f"Test matrix:  {X_test.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    models = build_models()

    validation_results = []
    fitted_models = {}

    print_section("TRAIN AND VALIDATE MODELS")

    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")

        clf = clone(model)
        clf.fit(X_train, y_train)

        fitted_models[model_name] = clf

        valid_metrics = compute_metrics(
            model=clf,
            X=X_valid,
            y_true=y_valid,
        )

        row = {
            "model": model_name,
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_weighted_f1": valid_metrics["weighted_f1"],
            f"valid_top{TOP_K}_accuracy": valid_metrics[f"top{TOP_K}_accuracy"],
        }

        validation_results.append(row)

        print(
            f"Accuracy={row['valid_accuracy']:.4f} | "
            f"Macro F1={row['valid_macro_f1']:.4f} | "
            f"Weighted F1={row['valid_weighted_f1']:.4f} | "
            f"Top-{TOP_K}={row[f'valid_top{TOP_K}_accuracy']:.4f}"
        )

        valid_predictions = build_predictions_df(
            model=clf,
            X=X_valid,
            df=valid_df,
            split_name="valid",
        )

        valid_predictions.to_csv(
            OUTPUT_DIR / f"valid_predictions_{safe_name(model_name)}.csv",
            index=False,
        )

    results_df = pd.DataFrame(validation_results)

    results_df = results_df.sort_values(
        by=[
            "valid_macro_f1",
            "valid_accuracy",
            f"valid_top{TOP_K}_accuracy",
        ],
        ascending=False,
    ).reset_index(drop=True)

    results_df.to_csv(
        OUTPUT_DIR / "validation_model_results.csv",
        index=False,
    )

    print_section("VALIDATION RESULTS")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    print_section("BEST MODEL SELECTED")
    print(f"Best model: {best_model_name}")

    print_section("FINAL TEST EVALUATION")

    test_metrics = compute_metrics(
        model=best_model,
        X=X_test,
        y_true=y_test,
    )

    test_row = {
        "best_model": best_model_name,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        f"test_top{TOP_K}_accuracy": test_metrics[f"top{TOP_K}_accuracy"],
    }

    test_results_df = pd.DataFrame([test_row])

    test_results_df.to_csv(
        OUTPUT_DIR / "test_results_best_model.csv",
        index=False,
    )

    print(test_results_df.to_string(index=False))

    y_test_pred = best_model.predict(X_test)

    save_classification_report(
        model_name=best_model_name,
        y_true=y_test,
        y_pred=y_test_pred,
        output_path=OUTPUT_DIR / "best_model_test_classification_report.txt",
    )

    labels = sorted(train_df[LABEL_COL].unique())

    save_confusion_outputs(
        y_true=y_test,
        y_pred=y_test_pred,
        labels=labels,
    )

    test_predictions = build_predictions_df(
        model=best_model,
        X=X_test,
        df=test_df,
        split_name="test",
    )

    test_predictions.to_csv(
        OUTPUT_DIR / "test_predictions_best_model.csv",
        index=False,
    )

    plot_validation_comparison(results_df)

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": best_model,
            "best_model_name": best_model_name,
            "text_column": TEXT_COL,
            "label_column": LABEL_COL,
            "top_k": TOP_K,
        },
        OUTPUT_DIR / "best_tfidf_baseline_model.pkl",
    )

    config = {
        "train_path": str(TRAIN_PATH),
        "valid_path": str(VALID_PATH),
        "test_path": str(TEST_PATH),
        "text_column": TEXT_COL,
        "label_column": LABEL_COL,
        "random_state": RANDOM_STATE,
        "top_k": TOP_K,
        "tfidf": {
            "lowercase": True,
            "stop_words": "english",
            "max_features": 30000,
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
        },
        "best_model": best_model_name,
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "test_rows": len(test_df),
        "number_of_classes": int(train_df[LABEL_COL].nunique()),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
    }

    with open(OUTPUT_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("TEXT TF-IDF BASELINE SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Train rows: {len(train_df)}\n")
        f.write(f"Valid rows: {len(valid_df)}\n")
        f.write(f"Test rows: {len(test_df)}\n")
        f.write(f"Number of classes: {train_df[LABEL_COL].nunique()}\n")
        f.write(f"Vocabulary size: {len(vectorizer.vocabulary_)}\n\n")

        f.write("Validation results:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")

        f.write("Best model:\n")
        f.write(str(best_model_name))
        f.write("\n\n")

        f.write("Final test results:\n")
        f.write(test_results_df.to_string(index=False))
        f.write("\n")

    print_section("SAVED OUTPUTS")
    print(OUTPUT_DIR / "validation_model_results.csv")
    print(OUTPUT_DIR / "test_results_best_model.csv")
    print(OUTPUT_DIR / "best_model_test_classification_report.txt")
    print(OUTPUT_DIR / "best_model_test_confusion_matrix.csv")
    print(OUTPUT_DIR / "test_predictions_best_model.csv")
    print(OUTPUT_DIR / "best_tfidf_baseline_model.pkl")
    print(OUTPUT_DIR / "summary.txt")

    print("\nDONE.")


if __name__ == "__main__":
    main()