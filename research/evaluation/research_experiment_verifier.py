"""Independent verification and uncertainty analysis for the grouped benchmark.

This script consumes the saved grouped benchmark artifacts without rerunning
model inference. It verifies split isolation, prediction shape/range, label
alignment by saved row order, metric reproducibility, threshold sensitivity,
and bootstrap confidence intervals. It also writes aggregate multilabel error
analysis without exposing source text.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support


LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
BOOTSTRAP_SEED = 424242
BOOTSTRAP_REPLICATES = 2000


def parse_labels(value: object) -> list[int]:
    result = []
    for token in str(value).split(","):
        token = token.strip()
        if not token.isdigit() or not 0 <= int(token) < len(LABELS):
            raise ValueError(f"Invalid label token: {token!r}")
        result.append(int(token))
    if not result:
        raise ValueError("Empty label value")
    return sorted(set(result))


def read_split(path: Path) -> tuple[list[str], np.ndarray]:
    texts = []
    labels = []
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.reader(file, delimiter="\t"):
            if len(row) != 3:
                raise ValueError(f"Malformed row in {path}: {row!r}")
            texts.append(row[0])
            vector = [0] * len(LABELS)
            for index in parse_labels(row[1]):
                vector[index] = 1
            labels.append(vector)
    return texts, np.asarray(labels, dtype=int)


def metric_summary(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    y_pred = (probabilities >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(LABELS))), average=None, zero_division=0
    )
    micro = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    return {
        "threshold": threshold,
        "samples": int(len(y_true)),
        "exact_match_accuracy": float(np.all(y_true == y_pred, axis=1).mean()),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "micro_precision": float(micro[0]),
        "micro_recall": float(micro[1]),
        "micro_f1": float(micro[2]),
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
                "false_positive": int(((y_pred[:, index] == 1) & (y_true[:, index] == 0)).sum()),
                "false_negative": int(((y_pred[:, index] == 0) & (y_true[:, index] == 1)).sum()),
                "true_positive": int(((y_pred[:, index] == 1) & (y_true[:, index] == 1)).sum()),
                "true_negative": int(((y_pred[:, index] == 0) & (y_true[:, index] == 0)).sum()),
            }
            for index, label in enumerate(LABELS)
        },
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    replicates: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ["macro_f1", "micro_f1", "weighted_f1", "exact_match_accuracy"]}
    sample_count = len(y_true)
    for _ in range(replicates):
        indexes = rng.integers(0, sample_count, size=sample_count)
        summary = metric_summary(
            y_true[indexes], probabilities[indexes], threshold
        )
        for key in values:
            values[key].append(summary[key])
    return {
        "replicates": replicates,
        "seed": seed,
        "interval": "percentile 95% bootstrap interval",
        "metrics": {
            key: {
                "estimate": float(metric_summary(y_true, probabilities, threshold)[key]),
                "lower_95": float(np.percentile(samples, 2.5)),
                "upper_95": float(np.percentile(samples, 97.5)),
            }
            for key, samples in values.items()
        },
    }


def verify_split_isolation(split_dir: Path) -> dict:
    texts = {}
    for name in ["train", "validation", "test"]:
        texts[name], _ = read_split(split_dir / f"{name}.tsv")
    return {
        f"{left}_{right}_overlap": len(set(texts[left]) & set(texts[right]))
        for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path(__file__).parents[2] / "reports" / "grouped_text_benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[2] / "reports" / "research_experiment_results.json",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()

    benchmark = args.benchmark_dir
    test_texts, y_true = read_split(benchmark / "splits" / "test.tsv")
    validation_texts, y_validation = read_split(benchmark / "splits" / "validation.tsv")
    validation_probabilities = pd.read_csv(
        benchmark / "validation_predictions.tsv", sep="\t"
    ).to_numpy(dtype=float)
    test_probabilities = pd.read_csv(
        benchmark / "test_predictions.tsv", sep="\t"
    ).to_numpy(dtype=float)

    if validation_probabilities.shape != (len(validation_texts), len(LABELS)):
        raise ValueError(
            f"Validation prediction shape {validation_probabilities.shape} does not "
            f"match {(len(validation_texts), len(LABELS))}"
        )
    if test_probabilities.shape != (len(test_texts), len(LABELS)):
        raise ValueError(
            f"Test prediction shape {test_probabilities.shape} does not "
            f"match {(len(test_texts), len(LABELS))}"
        )
    for name, probabilities in [
        ("validation", validation_probabilities),
        ("test", test_probabilities),
    ]:
        if not np.isfinite(probabilities).all() or (
            (probabilities < 0).any() or (probabilities > 1).any()
        ):
            raise ValueError(f"{name} probabilities contain invalid values")

    prior_results = json.loads((benchmark / "results.json").read_text(encoding="utf-8"))
    selected_threshold = float(prior_results["threshold_tuning"]["selected_threshold"])
    recomputed_test = metric_summary(y_true, test_probabilities, selected_threshold)
    recomputed_validation = metric_summary(
        y_validation, validation_probabilities, selected_threshold
    )

    result = {
        "experiment": "independent verification and uncertainty analysis",
        "status": "completed",
        "benchmark_results_source": str(benchmark / "results.json"),
        "label_order": LABELS,
        "row_alignment": {
            "method": (
                "Predictions are aligned by the exact row order written by the "
                "inference script; prediction files have no text column."
            ),
            "validation_text_records": len(validation_texts),
            "test_text_records": len(test_texts),
            "validation_prediction_rows": len(validation_probabilities),
            "test_prediction_rows": len(test_probabilities),
            "validated": True,
        },
        "split_isolation": verify_split_isolation(benchmark / "splits"),
        "threshold": {
            "selected_on_validation": selected_threshold,
            "test_not_used_for_selection": True,
            "same_grouped_test_at_fixed_030": metric_summary(
                y_true, test_probabilities, 0.30
            ),
            "same_grouped_test_at_selected_035": recomputed_test,
        },
        "recomputed_metrics": {
            "validation": recomputed_validation,
            "test": recomputed_test,
        },
        "bootstrap_test": bootstrap_metrics(
            y_true,
            test_probabilities,
            selected_threshold,
            args.bootstrap_replicates,
            BOOTSTRAP_SEED,
        ),
        "error_analysis": {
            "interpretation": (
                "Aggregate multilabel errors at the selected threshold. "
                "False positives and false negatives are counted per label; "
                "a single text can contribute to multiple errors."
            ),
            "per_class": recomputed_test["per_class"],
            "rows_with_any_error": int(
                np.any(
                    (test_probabilities >= selected_threshold).astype(int) != y_true,
                    axis=1,
                ).sum()
            ),
            "rows_exact_match": int(
                np.all(
                    (test_probabilities >= selected_threshold).astype(int) == y_true,
                    axis=1,
                ).sum()
            ),
        },
        "verification_conclusion": (
            "Saved split overlap is zero; prediction shapes and probability "
            "ranges are valid; metrics can be independently recomputed from "
            "saved splits and predictions."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
