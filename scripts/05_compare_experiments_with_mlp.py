"""
Compare Experiments With MLP
============================

Compară experimentele finale:
1. Metadata-only Age + Gender
2. TF-IDF + MLP Neural Network
3. TF-IDF + Linear SVM Calibrated

Scop:
- citește rezultatele salvate automat de scripturile anterioare;
- construiește un tabel comparativ;
- calculează diferențele față de experimentul metadata-only;
- generează grafice pentru lucrare;
- scrie un fișier text cu interpretarea rezultatelor.

Output:
outputs/05_experiment_comparison_with_mlp/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

OUTPUT_DIR = Path("outputs/05_experiment_comparison_with_mlp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5

EXPERIMENTS = [
    {
        "experiment": "Metadata-only Age + Gender",
        "experiment_type": "control",
        "test_path": Path("outputs/02_metadata_age_gender/test_results_best_model.csv"),
        "valid_path": Path("outputs/02_metadata_age_gender/validation_model_results.csv"),
        "valid_candidate_col": "model",
    },
    {
        "experiment": "TF-IDF + MLP",
        "experiment_type": "neural_network",
        "test_path": Path("outputs/04_tfidf_mlp_neural_network/test_results_best_model.csv"),
        "valid_path": Path("outputs/04_tfidf_mlp_neural_network/validation_config_results.csv"),
        "valid_candidate_col": "config",
    },
    {
        "experiment": "TF-IDF + Linear SVM",
        "experiment_type": "classical_ml",
        "test_path": Path("outputs/01_text_tfidf_baseline/test_results_best_model.csv"),
        "valid_path": Path("outputs/01_text_tfidf_baseline/validation_model_results.csv"),
        "valid_candidate_col": "model",
    },
]


# =========================================================
# HELPERS
# =========================================================

def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def find_topk_column(df: pd.DataFrame, prefix: str) -> str:
    candidates = [
        col for col in df.columns
        if col.startswith(prefix) and "top" in col and "accuracy" in col
    ]

    if not candidates:
        raise ValueError(
            f"No Top-K column found with prefix '{prefix}'. "
            f"Available columns: {list(df.columns)}"
        )

    return candidates[0]


def load_experiment_result(exp: dict) -> dict:
    test_path = exp["test_path"]
    valid_path = exp["valid_path"]
    valid_candidate_col = exp["valid_candidate_col"]

    if not test_path.exists():
        raise FileNotFoundError(f"Missing test results: {test_path}")

    if not valid_path.exists():
        raise FileNotFoundError(f"Missing validation results: {valid_path}")

    test_df = pd.read_csv(test_path)
    valid_df = pd.read_csv(valid_path)

    if test_df.empty:
        raise ValueError(f"Empty test results file: {test_path}")

    if valid_df.empty:
        raise ValueError(f"Empty validation results file: {valid_path}")

    if "best_model" not in test_df.columns:
        raise ValueError(f"'best_model' column missing in {test_path}")

    if valid_candidate_col not in valid_df.columns:
        raise ValueError(
            f"'{valid_candidate_col}' column missing in {valid_path}"
        )

    test_row = test_df.iloc[0].to_dict()
    best_model = str(test_row["best_model"])

    valid_match = valid_df[
        valid_df[valid_candidate_col].astype(str) == best_model
    ]

    if valid_match.empty:
        raise ValueError(
            f"Best model/config '{best_model}' not found in validation file: {valid_path}"
        )

    valid_row = valid_match.iloc[0].to_dict()

    test_top_col = find_topk_column(test_df, "test_")
    valid_top_col = find_topk_column(valid_df, "valid_")

    result = {
        "experiment": exp["experiment"],
        "experiment_type": exp["experiment_type"],
        "best_model": best_model,

        "valid_accuracy": float(valid_row["valid_accuracy"]),
        "valid_macro_f1": float(valid_row["valid_macro_f1"]),
        "valid_weighted_f1": float(valid_row["valid_weighted_f1"]),
        f"valid_top{TOP_K}_accuracy": float(valid_row[valid_top_col]),

        "test_accuracy": float(test_row["test_accuracy"]),
        "test_macro_f1": float(test_row["test_macro_f1"]),
        "test_weighted_f1": float(test_row["test_weighted_f1"]),
        f"test_top{TOP_K}_accuracy": float(test_row[test_top_col]),
    }

    if "best_epoch" in valid_row:
        result["best_epoch"] = int(valid_row["best_epoch"])
    else:
        result["best_epoch"] = None

    if "valid_loss" in valid_row:
        result["valid_loss"] = float(valid_row["valid_loss"])
    else:
        result["valid_loss"] = None

    return result


def build_delta_table(results_df: pd.DataFrame) -> pd.DataFrame:
    baseline_name = "Metadata-only Age + Gender"

    baseline_rows = results_df[results_df["experiment"] == baseline_name]

    if baseline_rows.empty:
        raise ValueError(f"Baseline experiment not found: {baseline_name}")

    baseline = baseline_rows.iloc[0]

    metrics = [
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        f"test_top{TOP_K}_accuracy",
    ]

    rows = []

    for _, row in results_df.iterrows():
        item = {
            "experiment": row["experiment"],
            "best_model": row["best_model"],
        }

        for metric in metrics:
            delta = row[metric] - baseline[metric]

            item[f"{metric}_delta"] = delta
            item[f"{metric}_delta_percentage_points"] = delta * 100

        rows.append(item)

    return pd.DataFrame(rows)


def build_pairwise_differences(results_df: pd.DataFrame) -> pd.DataFrame:
    tfidf_rows = results_df[results_df["experiment"] == "TF-IDF + Linear SVM"]
    mlp_rows = results_df[results_df["experiment"] == "TF-IDF + MLP"]

    if tfidf_rows.empty or mlp_rows.empty:
        raise ValueError("TF-IDF + Linear SVM or TF-IDF + MLP result is missing.")

    tfidf = tfidf_rows.iloc[0]
    mlp = mlp_rows.iloc[0]

    metrics = [
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        f"test_top{TOP_K}_accuracy",
    ]

    rows = []

    for metric in metrics:
        diff = tfidf[metric] - mlp[metric]

        rows.append({
            "comparison": "TF-IDF + Linear SVM minus TF-IDF + MLP",
            "metric": metric,
            "difference": diff,
            "difference_percentage_points": diff * 100,
        })

    return pd.DataFrame(rows)


def plot_all_test_metrics(results_df: pd.DataFrame) -> None:
    metrics = [
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        f"test_top{TOP_K}_accuracy",
    ]

    x = np.arange(len(results_df))
    width = 0.2

    plt.figure(figsize=(14, 7))

    for idx, metric in enumerate(metrics):
        offset = (idx - 1.5) * width

        plt.bar(
            x + offset,
            results_df[metric],
            width,
            label=metric,
        )

    plt.xticks(
        x,
        results_df["experiment"],
        rotation=15,
        ha="right",
    )

    plt.ylabel("Score")
    plt.title("Final Test Results - Experiment Comparison")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "01_all_test_metrics_comparison.png",
        dpi=300,
    )

    plt.close()


def plot_single_metric(
    results_df: pd.DataFrame,
    metric: str,
    title: str,
    file_name: str,
) -> None:
    plot_df = results_df.sort_values(
        by=metric,
        ascending=True,
    )

    plt.figure(figsize=(10, 5))

    plt.barh(
        plot_df["experiment"],
        plot_df[metric],
    )

    plt.xlabel(metric)
    plt.title(title)
    plt.xlim(0, 1.0)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / file_name,
        dpi=300,
    )

    plt.close()


def write_markdown_table(results_df: pd.DataFrame) -> None:
    cols = [
        "experiment",
        "best_model",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        f"test_top{TOP_K}_accuracy",
    ]

    table_df = results_df[cols].copy()

    for col in [
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        f"test_top{TOP_K}_accuracy",
    ]:
        table_df[col] = table_df[col].map(lambda x: f"{x:.4f}")

    headers = list(table_df.columns)

    markdown_lines = []

    markdown_lines.append(
        "| " + " | ".join(headers) + " |"
    )

    markdown_lines.append(
        "| " + " | ".join(["---"] * len(headers)) + " |"
    )

    for _, row in table_df.iterrows():
        markdown_lines.append(
            "| " + " | ".join(str(row[col]) for col in headers) + " |"
        )

    markdown = "\n".join(markdown_lines)

    with open(
        OUTPUT_DIR / "comparison_table_for_thesis.md",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(markdown)
        f.write("\n")

def write_interpretation(
    results_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
) -> None:
    metadata = results_df[
        results_df["experiment"] == "Metadata-only Age + Gender"
    ].iloc[0]

    mlp = results_df[
        results_df["experiment"] == "TF-IDF + MLP"
    ].iloc[0]

    svm = results_df[
        results_df["experiment"] == "TF-IDF + Linear SVM"
    ].iloc[0]

    mlp_delta = delta_df[
        delta_df["experiment"] == "TF-IDF + MLP"
    ].iloc[0]

    svm_delta = delta_df[
        delta_df["experiment"] == "TF-IDF + Linear SVM"
    ].iloc[0]

    svm_vs_mlp_macro = pairwise_df[
        pairwise_df["metric"] == "test_macro_f1"
    ].iloc[0]

    text = f"""EXPERIMENT COMPARISON WITH MLP
==================================================

Experimente comparate:
1. Metadata-only Age + Gender
2. TF-IDF + MLP
3. TF-IDF + Linear SVM

Rezultate pe test:

Metadata-only Age + Gender:
- Best model: {metadata["best_model"]}
- Accuracy: {metadata["test_accuracy"]:.4f}
- Macro F1: {metadata["test_macro_f1"]:.4f}
- Weighted F1: {metadata["test_weighted_f1"]:.4f}
- Top-{TOP_K} Accuracy: {metadata[f"test_top{TOP_K}_accuracy"]:.4f}

TF-IDF + MLP:
- Best model: {mlp["best_model"]}
- Accuracy: {mlp["test_accuracy"]:.4f}
- Macro F1: {mlp["test_macro_f1"]:.4f}
- Weighted F1: {mlp["test_weighted_f1"]:.4f}
- Top-{TOP_K} Accuracy: {mlp[f"test_top{TOP_K}_accuracy"]:.4f}
- Best epoch: {mlp["best_epoch"]}

TF-IDF + Linear SVM:
- Best model: {svm["best_model"]}
- Accuracy: {svm["test_accuracy"]:.4f}
- Macro F1: {svm["test_macro_f1"]:.4f}
- Weighted F1: {svm["test_weighted_f1"]:.4f}
- Top-{TOP_K} Accuracy: {svm[f"test_top{TOP_K}_accuracy"]:.4f}

Diferență față de Metadata-only:

TF-IDF + MLP:
- Accuracy: +{mlp_delta["test_accuracy_delta_percentage_points"]:.2f} puncte procentuale
- Macro F1: +{mlp_delta["test_macro_f1_delta_percentage_points"]:.2f} puncte procentuale
- Weighted F1: +{mlp_delta["test_weighted_f1_delta_percentage_points"]:.2f} puncte procentuale
- Top-{TOP_K} Accuracy: +{mlp_delta[f"test_top{TOP_K}_accuracy_delta_percentage_points"]:.2f} puncte procentuale

TF-IDF + Linear SVM:
- Accuracy: +{svm_delta["test_accuracy_delta_percentage_points"]:.2f} puncte procentuale
- Macro F1: +{svm_delta["test_macro_f1_delta_percentage_points"]:.2f} puncte procentuale
- Weighted F1: +{svm_delta["test_weighted_f1_delta_percentage_points"]:.2f} puncte procentuale
- Top-{TOP_K} Accuracy: +{svm_delta[f"test_top{TOP_K}_accuracy_delta_percentage_points"]:.2f} puncte procentuale

Comparație TF-IDF + Linear SVM vs TF-IDF + MLP:
- Diferență Macro F1 pe test: {svm_vs_mlp_macro["difference_percentage_points"]:.2f} puncte procentuale în favoarea modelului Linear SVM.

Interpretare:
Rezultatele arată că metadatele simple, reprezentate prin vârstă și gen, nu sunt suficiente pentru clasificarea diagnosticului. Performanța foarte scăzută a experimentului metadata-only confirmă că informația relevantă nu provine din aceste variabile.

Atât modelul TF-IDF + MLP, cât și modelul TF-IDF + Linear SVM obțin rezultate mult mai bune, ceea ce arată că descrierea textuală a cazului clinic este sursa principală de informație pentru predicție.

Modelul MLP peste TF-IDF obține o performanță competitivă, dar nu depășește modelul Linear SVM calibrat. Acest rezultat sugerează că, pentru acest set de date și pentru reprezentarea TF-IDF, un model liniar bine regularizat poate fi mai eficient decât o rețea neuronală simplă.

Graficul de loss al modelului MLP este util pentru analiza procesului de antrenare. Training loss scade aproape de zero, în timp ce validation loss scade inițial și apoi se stabilizează. Din acest motiv, monitorizarea validation loss și folosirea early stopping sunt necesare pentru a evita alegerea unui model supraantrenat pe setul de train.
"""

    with open(OUTPUT_DIR / "interpretation.txt", "w", encoding="utf-8") as f:
        f.write(text)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    print_section("LOAD EXPERIMENT RESULTS")

    rows = []

    for exp in EXPERIMENTS:
        print(f"Loading: {exp['experiment']}")
        rows.append(load_experiment_result(exp))

    results_df = pd.DataFrame(rows)

    results_df = results_df.sort_values(
        by=[
            "test_macro_f1",
            "test_accuracy",
            f"test_top{TOP_K}_accuracy",
        ],
        ascending=False,
    ).reset_index(drop=True)

    print_section("COMPARISON RESULTS")
    print(results_df.to_string(index=False))

    results_df.to_csv(
        OUTPUT_DIR / "experiment_comparison_with_mlp_results.csv",
        index=False,
    )

    delta_df = build_delta_table(results_df)

    delta_df.to_csv(
        OUTPUT_DIR / "experiment_comparison_with_mlp_deltas_vs_metadata.csv",
        index=False,
    )

    print_section("DELTAS VS METADATA-ONLY")
    print(delta_df.to_string(index=False))

    pairwise_df = build_pairwise_differences(results_df)

    pairwise_df.to_csv(
        OUTPUT_DIR / "svm_vs_mlp_differences.csv",
        index=False,
    )

    print_section("SVM VS MLP DIFFERENCES")
    print(pairwise_df.to_string(index=False))

    plot_all_test_metrics(results_df)

    plot_single_metric(
        results_df=results_df,
        metric="test_accuracy",
        title="Test Accuracy by Experiment",
        file_name="02_test_accuracy_comparison.png",
    )

    plot_single_metric(
        results_df=results_df,
        metric="test_macro_f1",
        title="Test Macro F1 by Experiment",
        file_name="03_test_macro_f1_comparison.png",
    )

    plot_single_metric(
        results_df=results_df,
        metric="test_weighted_f1",
        title="Test Weighted F1 by Experiment",
        file_name="04_test_weighted_f1_comparison.png",
    )

    plot_single_metric(
        results_df=results_df,
        metric=f"test_top{TOP_K}_accuracy",
        title=f"Test Top-{TOP_K} Accuracy by Experiment",
        file_name=f"05_test_top{TOP_K}_comparison.png",
    )

    write_markdown_table(results_df)

    write_interpretation(
        results_df=results_df,
        delta_df=delta_df,
        pairwise_df=pairwise_df,
    )

    config = {
        "experiments": [
            {
                "experiment": exp["experiment"],
                "experiment_type": exp["experiment_type"],
                "test_path": str(exp["test_path"]),
                "valid_path": str(exp["valid_path"]),
                "valid_candidate_col": exp["valid_candidate_col"],
            }
            for exp in EXPERIMENTS
        ],
        "top_k": TOP_K,
        "selection_rule": "Models/configurations are selected on validation Macro F1 and evaluated once on test.",
        "baseline_for_deltas": "Metadata-only Age + Gender",
        "pairwise_comparison": "TF-IDF + Linear SVM minus TF-IDF + MLP",
    }

    with open(OUTPUT_DIR / "comparison_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print_section("SAVED OUTPUTS")
    print(OUTPUT_DIR / "experiment_comparison_with_mlp_results.csv")
    print(OUTPUT_DIR / "experiment_comparison_with_mlp_deltas_vs_metadata.csv")
    print(OUTPUT_DIR / "svm_vs_mlp_differences.csv")
    print(OUTPUT_DIR / "comparison_table_for_thesis.md")
    print(OUTPUT_DIR / "01_all_test_metrics_comparison.png")
    print(OUTPUT_DIR / "02_test_accuracy_comparison.png")
    print(OUTPUT_DIR / "03_test_macro_f1_comparison.png")
    print(OUTPUT_DIR / "04_test_weighted_f1_comparison.png")
    print(OUTPUT_DIR / f"05_test_top{TOP_K}_comparison.png")
    print(OUTPUT_DIR / "interpretation.txt")
    print(OUTPUT_DIR / "comparison_config.json")

    print("\nDONE.")


if __name__ == "__main__":
    main()