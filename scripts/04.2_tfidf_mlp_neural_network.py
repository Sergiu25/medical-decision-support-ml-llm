"""
TF-IDF + MLP Neural Network on Fixed Splits
==========================================

Scop:
- separă TF-IDF de modelul neural;
- antrenează rețele neuronale simple peste reprezentarea TF-IDF;
- include o configurație fără strat ascuns;
- salvează model summary pentru fiecare configurație;
- salvează training loss și validation loss pe epoci;
- aplică early stopping pe validation loss;
- selectează cea mai bună configurație pe validation;
- evaluează o singură dată pe test.

Input:
data/splits/train.csv
data/splits/valid.csv
data/splits/test.csv

Output:
outputs/04.2_tfidf_mlp_neural_network/
"""

from __future__ import annotations

import copy
import json
import random
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================

TRAIN_PATH = Path("data/splits/train.csv")
VALID_PATH = Path("data/splits/valid.csv")
TEST_PATH = Path("data/splits/test.csv")

OUTPUT_DIR = Path("outputs/04.2_tfidf_mlp_neural_network")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "case_text"
LABEL_COL = "label"

RANDOM_STATE = 42
TOP_K = 5

MAX_FEATURES = 30000
MAX_EPOCHS = 300
PATIENCE = 20
MIN_DELTA = 1e-4
BATCH_SIZE = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class MLPConfig:
    name: str
    hidden_layers: tuple[int, ...]
    dropout: float
    learning_rate: float
    weight_decay: float


MLP_CONFIGS = [
    MLPConfig(
        name="mlp_256_dropout_03",
        hidden_layers=(256,),
        dropout=0.3,
        learning_rate=1e-3,
        weight_decay=1e-4,
    ),
    MLPConfig(
        name="mlp_512_128_dropout_04",
        hidden_layers=(512, 128),
        dropout=0.4,
        learning_rate=1e-3,
        weight_decay=1e-4,
    ),
    MLPConfig(
        name="mlp_256_128_dropout_05",
        hidden_layers=(256, 128),
        dropout=0.5,
        learning_rate=7e-4,
        weight_decay=5e-4,
    ),
    MLPConfig(
        name="mlp_no_hidden_linear",
        hidden_layers=(),
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=1e-4,
    ),
]


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    return name.strip("_")


def format_seconds(seconds: float) -> str:
    minutes = seconds / 60
    return f"{seconds:.2f} seconds ({minutes:.2f} minutes)"


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
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def sparse_to_tensor(X: Any) -> torch.Tensor:
    return torch.tensor(
        X.astype(np.float32).toarray(),
        dtype=torch.float32,
    )


def build_loader(
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


# =========================================================
# MODEL
# =========================================================

class TfidfMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        previous_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total_params, trainable_params


def save_model_summary(
    model: nn.Module,
    config: MLPConfig,
    input_dim: int,
    output_dim: int,
    output_dir: Path,
) -> None:
    total_params, trainable_params = count_parameters(model)

    lines = []
    lines.append(f"Model summary for: {config.name}")
    lines.append("=" * 70)
    lines.append(f"Input dimension: {input_dim}")
    lines.append(f"Output dimension: {output_dim}")
    lines.append(f"Hidden layers: {config.hidden_layers}")
    lines.append(f"Dropout: {config.dropout}")
    lines.append(f"Learning rate: {config.learning_rate}")
    lines.append(f"Weight decay: {config.weight_decay}")
    lines.append(f"Total parameters: {total_params}")
    lines.append(f"Trainable parameters: {trainable_params}")
    lines.append("")
    lines.append("Architecture:")
    lines.append(str(model))
    lines.append("")
    lines.append("Layer details:")

    for name, module in model.named_modules():
        if name and not isinstance(module, nn.Sequential):
            lines.append(f"{name}: {module}")

    summary_path = output_dir / f"model_summary_{safe_name(config.name)}.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================================================
# TRAINING + EVALUATION
# =========================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_examples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        if is_training:
            optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = X_batch.size(0)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def predict_proba(
    model: nn.Module,
    X_tensor: torch.Tensor,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()

    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    all_probs = []

    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(DEVICE)

            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)


def compute_metrics_from_probs(
    y_true: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    y_pred = probs.argmax(axis=1)

    k = min(TOP_K, len(labels))

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
            probs,
            k=k,
            labels=np.arange(len(labels)),
        ),
    }


def train_one_config(
    config: MLPConfig,
    input_dim: int,
    output_dim: int,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    X_valid_tensor: torch.Tensor,
    y_valid_np: np.ndarray,
    class_weights_tensor: torch.Tensor,
    labels: np.ndarray,
) -> tuple[nn.Module, pd.DataFrame, dict[str, float], int, float]:
    print_section(f"TRAIN CONFIG: {config.name}")

    model = TfidfMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
    ).to(DEVICE)

    save_model_summary(
        model=model,
        config=config,
        input_dim=input_dim,
        output_dim=output_dim,
        output_dir=OUTPUT_DIR,
    )

    print(model)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor.to(DEVICE),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    start_time = time.time()

    best_valid_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    history_rows = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        valid_loss = run_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            optimizer=None,
        )

        valid_probs = predict_proba(
            model=model,
            X_tensor=X_valid_tensor,
        )

        valid_metrics = compute_metrics_from_probs(
            y_true=y_valid_np,
            probs=valid_probs,
            labels=labels,
        )

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_weighted_f1": valid_metrics["weighted_f1"],
            f"valid_top{TOP_K}_accuracy": valid_metrics[f"top{TOP_K}_accuracy"],
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"valid_loss={valid_loss:.4f} | "
            f"valid_macro_f1={valid_metrics['macro_f1']:.4f}"
        )

        if valid_loss < best_valid_loss - MIN_DELTA:
            best_valid_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}"
            )
            break

    training_time_seconds = time.time() - start_time

    print(
        f"Training time for {config.name}: "
        f"{format_seconds(training_time_seconds)}"
    )

    if best_state is None:
        raise RuntimeError(f"No best state saved for config: {config.name}")

    model.load_state_dict(best_state)

    history_df = pd.DataFrame(history_rows)

    best_valid_probs = predict_proba(
        model=model,
        X_tensor=X_valid_tensor,
    )

    best_valid_metrics = compute_metrics_from_probs(
        y_true=y_valid_np,
        probs=best_valid_probs,
        labels=labels,
    )

    best_valid_metrics["best_valid_loss"] = best_valid_loss

    return model, history_df, best_valid_metrics, best_epoch, training_time_seconds


# =========================================================
# PLOTS
# =========================================================

def plot_loss_curves(
    history_df: pd.DataFrame,
    config_name: str,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Training loss",
    )

    plt.plot(
        history_df["epoch"],
        history_df["valid_loss"],
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss - {config_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / f"{safe_name(config_name)}_train_valid_loss.png",
        dpi=300,
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Training loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - {config_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / f"{safe_name(config_name)}_training_loss_only.png",
        dpi=300,
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    plt.plot(
        history_df["epoch"],
        history_df["valid_loss"],
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Validation Loss - {config_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / f"{safe_name(config_name)}_validation_loss_only.png",
        dpi=300,
    )

    plt.close()


def plot_config_comparison(results_df: pd.DataFrame) -> None:
    plot_df = results_df.sort_values(
        by="valid_macro_f1",
        ascending=True,
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        plot_df["config"],
        plot_df["valid_macro_f1"],
    )

    plt.xlabel("Validation Macro F1")
    plt.title("TF-IDF + MLP Config Comparison")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "mlp_config_comparison_valid_macro_f1.png",
        dpi=300,
    )

    plt.close()


def plot_best_loss_curve(
    best_history_df: pd.DataFrame,
    best_config_name: str,
) -> None:
    plt.figure(figsize=(10, 6))

    plt.plot(
        best_history_df["epoch"],
        best_history_df["train_loss"],
        label="Training loss",
    )

    plt.plot(
        best_history_df["epoch"],
        best_history_df["valid_loss"],
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Best MLP Loss Curve - {best_config_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "best_model_train_valid_loss.png",
        dpi=300,
    )

    plt.close()


# =========================================================
# OUTPUTS
# =========================================================

def build_predictions_df(
    df: pd.DataFrame,
    y_true_encoded: np.ndarray,
    probs: np.ndarray,
    label_encoder: LabelEncoder,
    split_name: str,
) -> pd.DataFrame:
    y_pred_encoded = probs.argmax(axis=1)

    true_labels = label_encoder.inverse_transform(y_true_encoded)
    pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    classes = label_encoder.classes_
    k = min(TOP_K, len(classes))

    top_idx = np.argsort(probs, axis=1)[:, ::-1][:, :k]
    top_labels = classes[top_idx]
    top_scores = np.take_along_axis(probs, top_idx, axis=1)

    result = pd.DataFrame({
        "split": split_name,
        "case_text": df[TEXT_COL].values,
        "true_label": true_labels,
        "predicted_label": pred_labels,
        "correct": true_labels == pred_labels,
    })

    for i in range(k):
        result[f"top_{i + 1}_label"] = top_labels[:, i]
        result[f"top_{i + 1}_score"] = top_scores[:, i]

    return result


def save_confusion_outputs(
    y_true_encoded: np.ndarray,
    probs: np.ndarray,
    label_encoder: LabelEncoder,
) -> None:
    y_pred_encoded = probs.argmax(axis=1)

    labels_encoded = np.arange(len(label_encoder.classes_))
    labels_text = list(label_encoder.classes_)

    cm = confusion_matrix(
        y_true_encoded,
        y_pred_encoded,
        labels=labels_encoded,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels_text,
        columns=labels_text,
    )

    cm_df.to_csv(
        OUTPUT_DIR / "best_model_test_confusion_matrix.csv",
        index=True,
    )

    rows = []

    for true_idx, true_label in enumerate(labels_text):
        for pred_idx, pred_label in enumerate(labels_text):
            value = cm[true_idx, pred_idx]

            if true_idx != pred_idx and value > 0:
                rows.append({
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": int(value),
                })

    confusion_pairs = pd.DataFrame(rows)

    if not confusion_pairs.empty:
        confusion_pairs = confusion_pairs.sort_values(
            by="count",
            ascending=False,
        )

        confusion_pairs.to_csv(
            OUTPUT_DIR / "best_model_test_top_confusions.csv",
            index=False,
        )


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    set_seed(RANDOM_STATE)

    print_section("ENVIRONMENT")
    print(f"Device: {DEVICE}")
    print(f"Torch version: {torch.__version__}")

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

    print_section("TF-IDF VECTORIZATION")

    vectorizer = build_vectorizer()

    X_train_sparse = vectorizer.fit_transform(train_df[TEXT_COL])
    X_valid_sparse = vectorizer.transform(valid_df[TEXT_COL])
    X_test_sparse = vectorizer.transform(test_df[TEXT_COL])

    print(f"Train TF-IDF matrix: {X_train_sparse.shape}")
    print(f"Valid TF-IDF matrix: {X_valid_sparse.shape}")
    print(f"Test TF-IDF matrix:  {X_test_sparse.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    print_section("LABEL ENCODING")

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(train_df[LABEL_COL])
    y_valid = label_encoder.transform(valid_df[LABEL_COL])
    y_test = label_encoder.transform(test_df[LABEL_COL])

    print(f"Number of classes: {len(label_encoder.classes_)}")

    print_section("CONVERT TO TENSORS")

    X_train_tensor = sparse_to_tensor(X_train_sparse)
    X_valid_tensor = sparse_to_tensor(X_valid_sparse)
    X_test_tensor = sparse_to_tensor(X_test_sparse)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = build_loader(
        X_tensor=X_train_tensor,
        y_tensor=y_train_tensor,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    valid_loader = build_loader(
        X_tensor=X_valid_tensor,
        y_tensor=y_valid_tensor,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(label_encoder.classes_)),
        y=y_train,
    )

    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
    )

    print_section("TRAIN MLP CONFIGURATIONS")

    config_results = []
    trained_models = {}
    histories = {}
    best_epochs = {}
    training_times = {}

    input_dim = X_train_tensor.shape[1]
    output_dim = len(label_encoder.classes_)

    for config in MLP_CONFIGS:
        model, history_df, valid_metrics, best_epoch, training_time_seconds = train_one_config(
            config=config,
            input_dim=input_dim,
            output_dim=output_dim,
            train_loader=train_loader,
            valid_loader=valid_loader,
            X_valid_tensor=X_valid_tensor,
            y_valid_np=y_valid,
            class_weights_tensor=class_weights_tensor,
            labels=label_encoder.classes_,
        )

        trained_models[config.name] = model
        histories[config.name] = history_df
        best_epochs[config.name] = best_epoch
        training_times[config.name] = training_time_seconds

        history_path = OUTPUT_DIR / f"history_{safe_name(config.name)}.csv"
        history_df.to_csv(history_path, index=False)

        plot_loss_curves(
            history_df=history_df,
            config_name=config.name,
            output_dir=OUTPUT_DIR,
        )

        config_results.append({
            "config": config.name,
            "hidden_layers": str(config.hidden_layers),
            "dropout": config.dropout,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "best_epoch": best_epoch,
            "training_time_seconds": training_time_seconds,
            "training_time_minutes": training_time_seconds / 60,
            "valid_loss": valid_metrics["best_valid_loss"],
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_weighted_f1": valid_metrics["weighted_f1"],
            f"valid_top{TOP_K}_accuracy": valid_metrics[f"top{TOP_K}_accuracy"],
        })

    results_df = pd.DataFrame(config_results)

    results_df = results_df.sort_values(
        by=[
            "valid_macro_f1",
            "valid_accuracy",
            f"valid_top{TOP_K}_accuracy",
        ],
        ascending=False,
    ).reset_index(drop=True)

    results_df.to_csv(
        OUTPUT_DIR / "validation_config_results.csv",
        index=False,
    )

    print_section("VALIDATION CONFIG RESULTS")
    print(results_df.to_string(index=False))

    plot_config_comparison(results_df)

    best_config_name = results_df.iloc[0]["config"]
    best_model = trained_models[best_config_name]
    best_history = histories[best_config_name]

    best_config = next(
        config
        for config in MLP_CONFIGS
        if config.name == best_config_name
    )

    plot_best_loss_curve(
        best_history_df=best_history,
        best_config_name=best_config_name,
    )

    print_section("BEST CONFIG SELECTED")
    print(f"Best config: {best_config_name}")
    print(f"Best epoch: {best_epochs[best_config_name]}")
    print(f"Training time: {format_seconds(training_times[best_config_name])}")

    print_section("FINAL TEST EVALUATION")

    test_probs = predict_proba(
        model=best_model,
        X_tensor=X_test_tensor,
    )

    test_metrics = compute_metrics_from_probs(
        y_true=y_test,
        probs=test_probs,
        labels=label_encoder.classes_,
    )

    test_results = {
        "best_model": best_config_name,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        f"test_top{TOP_K}_accuracy": test_metrics[f"top{TOP_K}_accuracy"],
    }

    test_results_df = pd.DataFrame([test_results])

    test_results_df.to_csv(
        OUTPUT_DIR / "test_results_best_model.csv",
        index=False,
    )

    print(test_results_df.to_string(index=False))

    test_predictions_df = build_predictions_df(
        df=test_df,
        y_true_encoded=y_test,
        probs=test_probs,
        label_encoder=label_encoder,
        split_name="test",
    )

    test_predictions_df.to_csv(
        OUTPUT_DIR / "test_predictions_best_model.csv",
        index=False,
    )

    y_test_pred = test_probs.argmax(axis=1)

    report = classification_report(
        y_test,
        y_test_pred,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0,
    )

    with open(
        OUTPUT_DIR / "best_model_test_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(f"Best model: {best_config_name}\n\n")
        f.write(report)

    save_confusion_outputs(
        y_true_encoded=y_test,
        probs=test_probs,
        label_encoder=label_encoder,
    )

    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "best_config": best_config_name,
            "hidden_layers": best_config.hidden_layers,
            "dropout": best_config.dropout,
            "learning_rate": best_config.learning_rate,
            "weight_decay": best_config.weight_decay,
            "label_classes": list(label_encoder.classes_),
        },
        OUTPUT_DIR / "best_tfidf_mlp_model.pt",
    )

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "label_encoder": label_encoder,
            "best_config": best_config_name,
            "mlp_configs": [config.__dict__ for config in MLP_CONFIGS],
        },
        OUTPUT_DIR / "tfidf_mlp_preprocessing.pkl",
    )

    experiment_config = {
        "train_path": str(TRAIN_PATH),
        "valid_path": str(VALID_PATH),
        "test_path": str(TEST_PATH),
        "text_column": TEXT_COL,
        "label_column": LABEL_COL,
        "random_state": RANDOM_STATE,
        "device": DEVICE,
        "torch_version": torch.__version__,
        "top_k": TOP_K,
        "max_features": MAX_FEATURES,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "batch_size": BATCH_SIZE,
        "early_stopping_metric": "validation_loss",
        "selection_metric": "validation_macro_f1",
        "best_config": best_config_name,
        "best_epoch": int(best_epochs[best_config_name]),
        "best_training_time_seconds": float(training_times[best_config_name]),
        "best_training_time_minutes": float(training_times[best_config_name] / 60),
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "test_rows": len(test_df),
        "number_of_classes": int(len(label_encoder.classes_)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
    }

    with open(
        OUTPUT_DIR / "experiment_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            experiment_config,
            f,
            indent=4,
            ensure_ascii=False,
        )

    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("TF-IDF + MLP NEURAL NETWORK SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Train rows: {len(train_df)}\n")
        f.write(f"Valid rows: {len(valid_df)}\n")
        f.write(f"Test rows: {len(test_df)}\n")
        f.write(f"Number of classes: {len(label_encoder.classes_)}\n")
        f.write(f"Vocabulary size: {len(vectorizer.vocabulary_)}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Torch version: {torch.__version__}\n\n")

        f.write("Validation config results:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")

        f.write("Best config:\n")
        f.write(str(best_config_name))
        f.write("\n\n")

        f.write("Best config training time:\n")
        f.write(format_seconds(training_times[best_config_name]))
        f.write("\n\n")

        f.write("Final test results:\n")
        f.write(test_results_df.to_string(index=False))
        f.write("\n")

    print_section("SAVED OUTPUTS")
    print(OUTPUT_DIR / "validation_config_results.csv")
    print(OUTPUT_DIR / "test_results_best_model.csv")
    print(OUTPUT_DIR / "best_model_train_valid_loss.png")
    print(OUTPUT_DIR / "best_model_test_classification_report.txt")
    print(OUTPUT_DIR / "test_predictions_best_model.csv")
    print(OUTPUT_DIR / "best_tfidf_mlp_model.pt")
    print(OUTPUT_DIR / "tfidf_mlp_preprocessing.pkl")
    print(OUTPUT_DIR / "experiment_config.json")
    print(OUTPUT_DIR / "summary.txt")
    print("Model summaries:")
    for config in MLP_CONFIGS:
        print(OUTPUT_DIR / f"model_summary_{safe_name(config.name)}.txt")

    print("\nDONE.")


if __name__ == "__main__":
    main()