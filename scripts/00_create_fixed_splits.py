"""
Create fixed train / validation / test splits
============================================

Scop:
- creează splituri fixe pentru toate experimentele;
- folosește stratificare după eticheta de diagnostic;
- verifică distribuția claselor;
- verifică overlap exact de texte între splituri;
- salvează fișiere utile pentru documentație și reproducibilitate.

Output:
data/splits/train.csv
data/splits/valid.csv
data/splits/test.csv
data/splits/split_summary.csv
data/splits/label_distribution.csv
data/splits/split_config.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = "multicare_clean_for_ml_noleak.csv"

OUTPUT_DIR = Path("data/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "case_text"
LABEL_COL = "label"

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALID_SIZE = 0.15
TEST_SIZE = 0.15


# =========================================================
# HELPERS
# =========================================================

def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def validate_split_ratios() -> None:
    total = TRAIN_SIZE + VALID_SIZE + TEST_SIZE

    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(
            f"Split ratios must sum to 1.0. Current sum: {total}"
        )

    if TRAIN_SIZE <= 0 or VALID_SIZE <= 0 or TEST_SIZE <= 0:
        raise ValueError(
            "TRAIN_SIZE, VALID_SIZE and TEST_SIZE must be positive."
        )


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)

    required_cols = [TEXT_COL, LABEL_COL]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Important: drop missing values before converting to string.
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()

    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower()

    df = df[df[TEXT_COL].str.len() > 0].copy()
    df = df[df[LABEL_COL].str.len() > 0].copy()

    return df


def validate_class_counts(df: pd.DataFrame) -> pd.Series:
    class_counts = df[LABEL_COL].value_counts()

    min_required = math.ceil(1 / min(VALID_SIZE, TEST_SIZE)) + 1

    too_small = class_counts[class_counts < min_required]

    print_section("CLASS COUNT CHECK")
    print(f"Minimum required examples per class: {min_required}")
    print(f"Smallest class size: {class_counts.min()}")

    if not too_small.empty:
        raise ValueError(
            "Some classes have too few examples for a stable stratified "
            "train / validation / test split.\n\n"
            f"{too_small}"
        )

    return class_counts


def compute_exact_split_sizes(total_rows: int) -> tuple[int, int, int]:
    """
    Calculează dimensiuni întregi pentru splituri.

    Train primește aproximativ TRAIN_SIZE.
    Restul este împărțit cât mai echilibrat între validation și test,
    deoarece VALID_SIZE și TEST_SIZE sunt egale în configurația curentă.
    """

    n_train = int(total_rows * TRAIN_SIZE)

    n_temp = total_rows - n_train

    n_valid = n_temp // 2
    n_test = n_temp - n_valid

    if n_train <= 0 or n_valid <= 0 or n_test <= 0:
        raise ValueError(
            "Invalid split sizes. Check TRAIN_SIZE, VALID_SIZE and TEST_SIZE."
        )

    return n_train, n_valid, n_test


def check_exact_text_overlap(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    train_texts = set(train_df[TEXT_COL])
    valid_texts = set(valid_df[TEXT_COL])
    test_texts = set(test_df[TEXT_COL])

    overlap_pairs = [
        ("train", "valid", train_texts & valid_texts),
        ("train", "test", train_texts & test_texts),
        ("valid", "test", valid_texts & test_texts),
    ]

    rows = []

    print_section("EXACT TEXT OVERLAP CHECK")

    for split_a, split_b, overlap_set in overlap_pairs:
        print(f"{split_a} ∩ {split_b}: {len(overlap_set)}")

        for text in overlap_set:
            rows.append({
                "split_a": split_a,
                "split_b": split_b,
                "case_text": text,
            })

    overlap_df = pd.DataFrame(rows)

    if not overlap_df.empty:
        overlap_path = OUTPUT_DIR / "exact_text_overlap_between_splits.csv"
        overlap_df.to_csv(overlap_path, index=False)

        raise ValueError(
            "Exact text overlap detected between splits. "
            "This may cause data leakage.\n"
            f"Details saved to: {overlap_path}"
        )


def build_label_distribution(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    splits = {
        "full": full_df,
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
    }

    base = pd.DataFrame({
        "label": sorted(full_df[LABEL_COL].unique())
    })

    for split_name, split_df in splits.items():
        counts = split_df[LABEL_COL].value_counts()

        count_col = f"{split_name}_count"
        ratio_col = f"{split_name}_ratio"

        temp = pd.DataFrame({
            "label": counts.index,
            count_col: counts.values,
            ratio_col: counts.values / len(split_df),
        })

        base = base.merge(temp, on="label", how="left")

        base[count_col] = base[count_col].fillna(0).astype(int)
        base[ratio_col] = base[ratio_col].fillna(0.0)

    base = base.sort_values(
        by="full_count",
        ascending=False,
    ).reset_index(drop=True)

    return base


def check_class_coverage(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    all_labels = set(full_df[LABEL_COL].unique())

    split_labels = {
        "train": set(train_df[LABEL_COL].unique()),
        "valid": set(valid_df[LABEL_COL].unique()),
        "test": set(test_df[LABEL_COL].unique()),
    }

    print_section("CLASS COVERAGE")

    missing_any = False

    for split_name, labels in split_labels.items():
        missing = all_labels - labels

        print(f"Missing labels in {split_name}: {len(missing)}")

        if missing:
            missing_any = True
            print(f"Missing in {split_name}: {sorted(missing)}")

    if missing_any:
        raise ValueError(
            "At least one split is missing one or more labels. "
            "The split is not suitable for final experiments."
        )


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    validate_split_ratios()

    df = load_dataset()

    print_section("DATASET LOADED")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Classes: {df[LABEL_COL].nunique()}")

    class_counts = validate_class_counts(df)

    # Duplicatele sunt raportate pentru transparență.
    # Nu blocăm execuția aici, deoarece leakage-ul relevant apare doar dacă
    # același text este prezent în splituri diferite. Acest lucru este verificat
    # separat în check_exact_text_overlap().
    duplicate_text_label = df.duplicated(
        subset=[TEXT_COL, LABEL_COL]
    ).sum()

    duplicate_text_only = df.duplicated(
        subset=[TEXT_COL]
    ).sum()

    print_section("DUPLICATE CHECK")
    print(f"Exact duplicate case_text + label rows: {duplicate_text_label}")
    print(f"Exact duplicate case_text rows: {duplicate_text_only}")

    n_train, n_valid, n_test = compute_exact_split_sizes(len(df))
    n_temp = n_valid + n_test

    print_section("REQUESTED SPLIT SIZES")
    print(f"Train: {n_train}")
    print(f"Valid: {n_valid}")
    print(f"Test:  {n_test}")

    train_df, temp_df = train_test_split(
        df,
        test_size=n_temp,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COL],
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=n_test,
        random_state=RANDOM_STATE,
        stratify=temp_df[LABEL_COL],
    )

    print_section("FINAL SPLIT SIZES")
    print(f"Train: {len(train_df)}")
    print(f"Valid: {len(valid_df)}")
    print(f"Test:  {len(test_df)}")

    print_section("FINAL SPLIT RATIOS")
    print(f"Train: {len(train_df) / len(df):.4f}")
    print(f"Valid: {len(valid_df) / len(df):.4f}")
    print(f"Test:  {len(test_df) / len(df):.4f}")

    check_class_coverage(
        full_df=df,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )

    check_exact_text_overlap(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )

    label_distribution = build_label_distribution(
        full_df=df,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )

    split_summary = pd.DataFrame([
        {
            "split": "full",
            "rows": len(df),
            "classes": df[LABEL_COL].nunique(),
        },
        {
            "split": "train",
            "rows": len(train_df),
            "classes": train_df[LABEL_COL].nunique(),
        },
        {
            "split": "valid",
            "rows": len(valid_df),
            "classes": valid_df[LABEL_COL].nunique(),
        },
        {
            "split": "test",
            "rows": len(test_df),
            "classes": test_df[LABEL_COL].nunique(),
        },
    ])

    config = {
        "dataset_path": DATASET_PATH,
        "text_column": TEXT_COL,
        "label_column": LABEL_COL,
        "random_state": RANDOM_STATE,
        "train_size_ratio": TRAIN_SIZE,
        "valid_size_ratio": VALID_SIZE,
        "test_size_ratio": TEST_SIZE,
        "total_rows": len(df),
        "number_of_classes": int(df[LABEL_COL].nunique()),
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "test_rows": len(test_df),
        "minimum_examples_per_class_required": math.ceil(
            1 / min(VALID_SIZE, TEST_SIZE)
        ) + 1,
        "smallest_class_size": int(class_counts.min()),
        "duplicate_case_text_label_rows": int(duplicate_text_label),
        "duplicate_case_text_rows": int(duplicate_text_only),
        "exact_text_overlap_blocking_enabled": True,
    }

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    valid_df.to_csv(OUTPUT_DIR / "valid.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    split_summary.to_csv(
        OUTPUT_DIR / "split_summary.csv",
        index=False,
    )

    label_distribution.to_csv(
        OUTPUT_DIR / "label_distribution.csv",
        index=False,
    )

    with open(OUTPUT_DIR / "split_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print_section("SAVED OUTPUTS")
    print(OUTPUT_DIR / "train.csv")
    print(OUTPUT_DIR / "valid.csv")
    print(OUTPUT_DIR / "test.csv")
    print(OUTPUT_DIR / "split_summary.csv")
    print(OUTPUT_DIR / "label_distribution.csv")
    print(OUTPUT_DIR / "split_config.json")

    print("\nDONE.")


if __name__ == "__main__":
    main()