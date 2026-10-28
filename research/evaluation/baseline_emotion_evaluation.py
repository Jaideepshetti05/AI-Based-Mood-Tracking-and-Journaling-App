"""Reproducible baseline evaluation for the existing offline artifacts.

This script does not train or modify a model. It evaluates:

1. Existing transformer probabilities against the seven-label Ekman-style
   test split as a multi-label task.
2. A clearly marked deterministic first-label proxy for confusion matrices and
   accuracy. This is not a replacement for the original multi-label task.
3. Existing VADER predictions against the three-class sentiment test split.

Ground truth is always read from the reference TSV files. Predictions are
never used to choose or rewrite ground-truth labels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    precision_recall_fscore_support,
)


EKMAN_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
SENTIMENT_LABELS = ["positive", "negative", "ambiguous"]


def parse_multilabel(value: object, size: int) -> list[int]:
    result: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if token.isdigit():
            index = int(token)
            if 0 <= index < size:
                result.append(index)
            else:
                raise ValueError(f"Label index {index} is outside 0..{size - 1}")
        elif token:
            raise ValueError(f"Invalid label token: {token!r}")
    if not result:
        raise ValueError("A reference row contains no valid labels")
    return sorted(set(result))


def binary_matrix(values: Iterable[object], width: int) -> np.ndarray:
    matrix = np.zeros((len(values), width), dtype=int)
    for row_index, value in enumerate(values):
        for label_index in parse_multilabel(value, width):
            matrix[row_index, label_index] = 1
    return matrix


def report_multilabel(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    y_pred = (probabilities >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(y_true.shape[1])),
        average=None,
        zero_division=0,
    )
    return {
        "task": "seven-label multi-label classification",
        "threshold": threshold,
        "samples": int(y_true.shape[0]),
        "exact_match_accuracy": float(np.all(y_true == y_pred, axis=1).mean()),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "micro_precision": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(EKMAN_LABELS)
        },
        "predicted_positive_counts": {
            label: int(y_pred[:, index].sum())
            for index, label in enumerate(EKMAN_LABELS)
        },
    }


def report_first_label_proxy(
    y_true: np.ndarray, probabilities: np.ndarray, labels: list[str]
) -> dict:
    """Report a deterministic proxy, not a single-label ground truth."""
    true_primary = np.argmax(y_true, axis=1)
    predicted_primary = np.argmax(probabilities, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_primary,
        predicted_primary,
        labels=list(range(len(labels))),
        average=None,
        zero_division=0,
    )
    return {
        "task": "single-label proxy using the first encoded reference label",
        "warning": (
            "The source task is multi-label. This proxy is deterministic for "
            "summary accuracy and confusion matrix only; it is not the original "
            "single-label annotation task."
        ),
        "samples": int(len(true_primary)),
        "accuracy": float(accuracy_score(true_primary, predicted_primary)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(
            f1_score(true_primary, predicted_primary, average="weighted", zero_division=0)
        ),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(
            true_primary, predicted_primary, labels=list(range(len(labels)))
        ).tolist(),
        "confusion_matrix_labels": labels,
    }


def evaluate_transformer(test_path: Path, prediction_path: Path, threshold: float) -> dict:
    started = time.perf_counter()
    test = pd.read_csv(
        test_path,
        sep="\t",
        header=None,
        names=["text", "labels", "id"],
    )
    predictions = pd.read_csv(prediction_path, sep="\t")
    if len(test) != len(predictions):
        raise ValueError(f"Test rows ({len(test)}) != prediction rows ({len(predictions)})")
    missing = [label for label in EKMAN_LABELS if label not in predictions.columns]
    if missing:
        raise ValueError(f"Transformer predictions are missing columns: {missing}")

    y_true = binary_matrix(test["labels"].tolist(), len(EKMAN_LABELS))
    probabilities = predictions[EKMAN_LABELS].to_numpy(dtype=float)
    elapsed = time.perf_counter() - started
    result = {
        "model": "j-hartmann/emotion-english-distilroberta-base",
        "reference_file": str(test_path),
        "prediction_file": str(prediction_path),
        "labels": EKMAN_LABELS,
        "reference_records": int(len(test)),
        "duplicate_prediction_rows": int(predictions.duplicated().sum()),
        "evaluation_time_seconds": round(elapsed, 6),
        "evaluation_time_ms_per_sample": round(elapsed * 1000 / len(test), 6),
        "multilabel": report_multilabel(y_true, probabilities, threshold),
        "single_label_proxy": report_first_label_proxy(
            y_true, probabilities, EKMAN_LABELS
        ),
    }
    return result


def evaluate_vader(reference_path: Path, prediction_path: Path) -> dict:
    reference = pd.read_csv(
        reference_path,
        sep="\t",
        header=None,
        names=["text", "labels", "id"],
    )
    predictions = pd.read_csv(prediction_path, sep="\t")
    if len(reference) != len(predictions):
        raise ValueError(
            f"Sentiment reference rows ({len(reference)}) != prediction rows ({len(predictions)})"
        )
    if "vader_sentiment" not in predictions:
        raise ValueError("VADER predictions do not contain vader_sentiment")

    y_true = binary_matrix(reference["labels"].tolist(), len(SENTIMENT_LABELS))
    y_pred_labels = predictions["vader_sentiment"].tolist()
    label_to_index = {label: index for index, label in enumerate(SENTIMENT_LABELS)}
    unknown = sorted(set(y_pred_labels) - set(label_to_index))
    if unknown:
        raise ValueError(f"Unexpected VADER labels: {unknown}")
    y_pred = np.zeros_like(y_true)
    for row_index, label in enumerate(y_pred_labels):
        y_pred[row_index, label_to_index[label]] = 1

    true_primary = np.argmax(y_true, axis=1)
    predicted_primary = np.argmax(y_pred, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_primary,
        predicted_primary,
        labels=list(range(len(SENTIMENT_LABELS))),
        average=None,
        zero_division=0,
    )
    return {
        "model": "VADER",
        "reference_file": str(reference_path),
        "prediction_file": str(prediction_path),
        "labels": SENTIMENT_LABELS,
        "reference_records": int(len(reference)),
        "multilabel_reference_note": (
            "VADER emits one sentiment label. Accuracy and per-class scores use "
            "the first encoded reference label as a deterministic proxy. "
            "This is not a fully multilabel VADER evaluation."
        ),
        "accuracy": float(accuracy_score(true_primary, predicted_primary)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(
            f1_score(true_primary, predicted_primary, average="weighted", zero_division=0)
        ),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(SENTIMENT_LABELS)
        },
        "confusion_matrix": confusion_matrix(
            true_primary, predicted_primary, labels=list(range(len(SENTIMENT_LABELS)))
        ).tolist(),
        "confusion_matrix_labels": SENTIMENT_LABELS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).parents[2] / "dataset 1",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[2]
        / "reports"
        / "baseline_emotion_evaluation_results.json",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    result = {
        "evaluation_scope": (
            "Existing held-out prediction artifacts; no model training or "
            "application changes were performed."
        ),
        "transformer": evaluate_transformer(
            dataset_root / "preprocessed" / "test_ekman.tsv",
            dataset_root / "preprocessed" / "hf_model_test_predictions.tsv",
            args.threshold,
        ),
        "vader": evaluate_vader(
            dataset_root / "preprocessed" / "test_sentiment.tsv",
            dataset_root / "preprocessed" / "vader_test_predictions.tsv",
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
