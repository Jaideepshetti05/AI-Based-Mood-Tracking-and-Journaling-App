"""Reproduce metrics from the existing held-out GoEmotions-derived predictions.

This script never downloads data or trains a model. It uses test_ekman.tsv as
the held-out labels and hf_model_test_predictions.tsv as predictions generated
by the existing evaluation pipeline.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def evaluate(test_path: Path, predictions_path: Path, output_path: Path) -> dict:
    labels = [line.strip() for line in (test_path.parent / "ekman_labels.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    test = pd.read_csv(test_path, sep="\t", header=None, names=["text", "labels", "id"])
    predictions = pd.read_csv(predictions_path, sep="\t")
    if len(test) != len(predictions):
        raise ValueError(f"Split size mismatch: {len(test)} labels vs {len(predictions)} predictions")

    y_true = []
    for raw in test["labels"].fillna(""):
        ids = {int(value) for value in str(raw).split(",") if value.strip().isdigit()}
        y_true.append(max((labels[index] for index in ids), key=lambda label: float(predictions.iloc[len(y_true)][label])) if ids else "neutral")
    y_pred = predictions[labels].idxmax(axis=1).tolist()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    result = {
        "evaluation_type": "single-label primary emotion selected from multi-label ground truth",
        "samples": len(test),
        "labels": labels,
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "per_label": {
            label: {"precision": float(p), "recall": float(r), "f1": float(score), "support": int(count)}
            for label, p, r, score, count in zip(labels, precision, recall, f1, support)
        },
        "confusion_matrix": matrix,
        "confusion_matrix_labels": labels,
        "limitations": [
            "Ground truth can contain multiple emotions; this report selects the highest-scoring true label for a comparable primary-label summary.",
            "The held-out data is Reddit-derived GoEmotions text and does not establish performance on private journal entries.",
        ],
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=Path("preprocessed/test_ekman.tsv"))
    parser.add_argument("--predictions", type=Path, default=Path("preprocessed/hf_model_test_predictions.tsv"))
    parser.add_argument("--output", type=Path, default=Path("preprocessed/reproducible_evaluation.json"))
    args = parser.parse_args()
    report = evaluate(args.test, args.predictions, args.output)
    print(json.dumps({key: report[key] for key in ("samples", "macro_precision", "macro_recall", "macro_f1")}, indent=2))
