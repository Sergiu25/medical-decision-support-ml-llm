"""
Compare Experiments
===================

Scop:
- compară rezultatele finale dintre experimente;
- citește automat output-urile salvate de scripturile anterioare;
- generează tabel comparativ, diferențe față de metadata-only și grafice.

Input:
outputs/01_text_tfidf_baseline/test_results_best_model.csv
outputs/01_text_tfidf_baseline/validation_model_results.csv

outputs/02_metadata_age_gender/test_results_best_model.csv
outputs/02_metadata_age_gender/validation_model_results.csv

Output:
outputs/03_experiment_comparison/
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

OUTPUT_DIR = Path("outputs/03_experiment_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5

EXPERIMENTS = [
    {
        "experiment": "Metadata-only Age + Gender",
        "type": "control",
        "output_dir": Path("outputs/02_metadata_age_gender"),
    },
    {
        "experiment": "TF-IDF + Text",
        "type": "classical_ml",
        "output_dir": Path("outputs/01_text_tfidf_baseline"),
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
    output_dir = exp["output_dir"]

    test_path = output_dir / "test_results_best_model.csv"
    valid_path = output_dir / "validation_model_results.csv"

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

    test_row = test_df.iloc[0].to_dict()

    if "best_model" not in test_row:
        raise ValueError(f"'best_model' column missing in {test_path}")

    best_model = test_row["best_model"]

    valid_match = valid_df[valid_df["model"] == best_model]

    if valid_match.empty:
        raise ValueError(
            f"Best model '{best_model}' not found in validation results: {valid_path}"
        )

    valid_row = valid_match.iloc[0].to_dict()

    test_top_col = find_topk_column(test_df, "test_")
    valid_top_col = find_topk_column(valid_df, "valid_")

    return {
        "experiment": exp["experiment"],
        "experiment_type": exp["type"],
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


def plot_test_metrics(results_df: pd.DataFrame) -> None:
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
        OUTPUT_DIR / "01_test_metrics_comparison.png",
        dpi=300,
    )

    plt.close()


def plot_macro_f1_comparison(results_df: pd.DataFrame) -> None:
    plot_df = results_df.sort_values(
        by="test_macro_f1",
        ascending=True,
    )

    plt.figure(figsize=(10, 5))

    plt.barh(
        plot_df["experiment"],
        plot_df["test_macro_f1"],
    )

    plt.xlabel("Test Macro F1")
    plt.title("Test Macro F1 by Experiment")
    plt.xlim(0, 1.0)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "02_test_macro_f1_comparison.png",
        dpi=300,
    )

    plt.close()


def plot_top5_comparison(results_df: pd.DataFrame) -> None:
    metric = f"test_top{TOP_K}_accuracy"

    plot_df = results_df.sort_values(
        by=metric,
        ascending=True,
    )

    plt.figure(figsize=(10, 5))

    plt.barh(
        plot_df["experiment"],
        plot_df[metric],
    )

    plt.xlabel(f"Test Top-{TOP_K} Accuracy")
    plt.title(f"Test Top-{TOP_K} Accuracy by Experiment")
    plt.xlim(0, 1.0)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"03_test_top{TOP_K}_comparison.png",
        dpi=300,
    )

    plt.close()


def write_interpretation(
    results_df: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> None:
    metadata = results_df[
        results_df["experiment"] == "Metadata-only Age + Gender"
    ].iloc[0]

    tfidf = results_df[
        results_df["experiment"] == "TF-IDF + Text"
    ].iloc[0]

    tfidf_delta = delta_df[
        delta_df["experiment"] == "TF-IDF + Text"
    ].iloc[0]

    text = f"""EXPERIMENT COMPARISON SUMMARY
==================================================

Experimente comparate:
1. Metadata-only Age + Gender
2. TF-IDF + Text

Rezultate pe test:

Metadata-only Age + Gender:
- Best model: {metadata["best_model"]}
- Accuracy: {metadata["test_accuracy"]:.4f}
- Macro F1: {metadata["test_macro_f1"]:.4f}
- Weighted F1: {metadata["test_weighted_f1"]:.4f}
- Top-{TOP_K} Accuracy: {metadata[f"test_top{TOP_K}_accuracy"]:.4f}

TF-IDF + Text:
- Best model: {tfidf["best_model"]}
- Accuracy: {tfidf["test_accuracy"]:.4f}
- Macro F1: {tfidf["test_macro_f1"]:.4f}
- Weighted F1: {tfidf["test_weighted_f1"]:.4f}
- Top-{TOP_K} Accuracy: {tfidf[f"test_top{TOP_K}_accuracy"]:.4f}

Diferență TF-IDF față de Metadata-only:
- Accuracy: +{tfidf_delta["test_accuracy_delta_percentage_points"]:.2f} puncte procentuale
- Macro F1: +{tfidf_delta["test_macro_f1_delta_percentage_points"]:.2f} puncte procentuale
- Weighted F1: +{tfidf_delta["test_weighted_f1_delta_percentage_points"]:.2f} puncte procentuale
- Top-{TOP_K} Accuracy: +{tfidf_delta[f"test_top{TOP_K}_accuracy_delta_percentage_points"]:.2f} puncte procentuale

Interpretare:
Rezultatele arată că metadatele simple, reprezentate prin vârstă și gen, nu sunt suficiente pentru predicția diagnosticului. Performanța foarte scăzută a experimentului metadata-only indică faptul că aceste variabile nu conțin informație discriminativă suficientă pentru cele 43 de clase.

În schimb, modelul bazat pe TF-IDF și textul clinic al cazului obține rezultate mult mai bune pe toate metricile evaluate. Diferența majoră dintre cele două experimente arată că informația relevantă pentru clasificarea diagnosticului provine în principal din descrierea textuală a cazului, nu din metadate simple.

Top-{TOP_K} Accuracy este deosebit de relevantă pentru scopul aplicației, deoarece sistemul propus are rol de suport decizional și poate returna o listă scurtă de diagnostice probabile, fără a înlocui decizia medicului.
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
        OUTPUT_DIR / "experiment_comparison_results.csv",
        index=False,
    )

    delta_df = build_delta_table(results_df)

    delta_df.to_csv(
        OUTPUT_DIR / "experiment_comparison_deltas_vs_metadata.csv",
        index=False,
    )

    print_section("DELTAS VS METADATA-ONLY")
    print(delta_df.to_string(index=False))

    plot_test_metrics(results_df)
    plot_macro_f1_comparison(results_df)
    plot_top5_comparison(results_df)

    write_interpretation(
        results_df=results_df,
        delta_df=delta_df,
    )

    config = {
        "experiments": [
            {
                "experiment": exp["experiment"],
                "type": exp["type"],
                "output_dir": str(exp["output_dir"]),
            }
            for exp in EXPERIMENTS
        ],
        "top_k": TOP_K,
        "selection_rule": "Models are selected on validation Macro F1 and evaluated once on test.",
        "baseline_for_deltas": "Metadata-only Age + Gender",
    }

    with open(OUTPUT_DIR / "comparison_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print_section("SAVED OUTPUTS")
    print(OUTPUT_DIR / "experiment_comparison_results.csv")
    print(OUTPUT_DIR / "experiment_comparison_deltas_vs_metadata.csv")
    print(OUTPUT_DIR / "01_test_metrics_comparison.png")
    print(OUTPUT_DIR / "02_test_macro_f1_comparison.png")
    print(OUTPUT_DIR / f"03_test_top{TOP_K}_comparison.png")
    print(OUTPUT_DIR / "interpretation.txt")
    print(OUTPUT_DIR / "comparison_config.json")

    print("\nDONE.")


if __name__ == "__main__":
    main()