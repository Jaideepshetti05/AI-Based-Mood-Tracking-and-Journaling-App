"""Leakage-controlled benchmark for the existing pretrained emotion model.

The script:

1. Reads the existing processed Ekman TSV splits.
2. Groups identical text values before splitting.
3. Unions all valid labels observed for each identical text group.
4. Creates deterministic 80/10/10 train/validation/test group splits.
5. Runs the existing pretrained transformer on validation and test text.
6. Tunes the multi-label threshold on validation only.
7. Evaluates the selected threshold once on the frozen test split.

It does not modify the application, original datasets, model files, or
existing evaluation artifacts. Generated files are written below the output
directory supplied on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
SEED = 20261027
SPLIT_RATIOS = {"train": 0.80, "validation": 0.10, "test": 0.10}
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"


def parse_labels(value: object) -> list[int]:
    result = []
    for token in str(value).split(","):
        token = token.strip()
        if not token.isdigit() or not 0 <= int(token) < len(LABELS):
            raise ValueError(f"Invalid Ekman label value: {token!r}")
        result.append(int(token))
    if not result:
        raise ValueError("A row has no labels")
    return sorted(set(result))


def label_vector(labels: Iterable[int]) -> list[int]:
    result = [0] * len(LABELS)
    for index in labels:
        result[index] = 1
    return result


def binary_matrix(values: Iterable[object]) -> np.ndarray:
    matrix = np.zeros((len(values), len(LABELS)), dtype=int)
    for row_index, value in enumerate(values):
        for label in parse_labels(value):
            matrix[row_index, label] = 1
    return matrix


def load_and_group(dataset_root: Path) -> tuple[pd.DataFrame, dict]:
    records = []
    source_files = [
        dataset_root / "preprocessed" / "train_ekman.tsv",
        dataset_root / "preprocessed" / "dev_ekman.tsv",
        dataset_root / "preprocessed" / "test_ekman.tsv",
    ]
    for source in source_files:
        frame = pd.read_csv(
            source,
            sep="\t",
            header=None,
            names=["text", "labels", "id"],
            dtype=str,
        )
        frame["source_file"] = str(source)
        records.append(frame)

    combined = pd.concat(records, ignore_index=True)
    grouped = []
    for text, frame in combined.groupby("text", sort=False, dropna=False):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Blank text encountered while grouping")
        labels = sorted(
            {
                label
                for value in frame["labels"]
                for label in parse_labels(value)
            }
        )
        grouped.append(
            {
                "text": text,
                "labels": ",".join(str(label) for label in labels),
                "id": str(frame.iloc[0]["id"]),
                "group_size": int(len(frame)),
                "source_record_count": int(len(frame)),
                "primary_label": int(labels[0]),
            }
        )

    result = pd.DataFrame(grouped)
    metadata = {
        "source_files": [str(path) for path in source_files],
        "source_records": int(len(combined)),
        "unique_text_groups": int(len(result)),
        "duplicate_records_collapsed": int(len(combined) - len(result)),
        "groups_with_multiple_source_records": int((result["group_size"] > 1).sum()),
        "label_union_policy": (
            "Identical text groups retain the union of every valid Ekman label "
            "observed in their source records."
        ),
    }
    return result, metadata


def stable_group_split(frame: pd.DataFrame, seed: int) -> tuple[dict[str, pd.DataFrame], dict]:
    """Assign complete text groups, stratified by deterministic primary label."""
    rng = random.Random(seed)
    assignments = {}
    target_names = list(SPLIT_RATIOS)
    for primary, subset in frame.groupby("primary_label", sort=True):
        indices = list(subset.index)
        rng.shuffle(indices)
        total = len(indices)
        train_end = round(total * SPLIT_RATIOS["train"])
        validation_end = train_end + round(total * SPLIT_RATIOS["validation"])
        for position, index in enumerate(indices):
            if position < train_end:
                assignments[index] = "train"
            elif position < validation_end:
                assignments[index] = "validation"
            else:
                assignments[index] = "test"

    split_frames = {
        name: frame.loc[
            [index for index, split in assignments.items() if split == name]
        ].copy()
        for name in target_names
    }
    for split in split_frames.values():
        split.sort_values("text", inplace=True, kind="stable")
        split.reset_index(drop=True, inplace=True)

    metadata = {
        "seed": seed,
        "assignment_method": (
            "Deterministic shuffled assignment within primary-label strata; "
            "identical text groups are indivisible."
        ),
        "split_ratios_requested": SPLIT_RATIOS,
        "split_group_counts": {
            name: int(len(split)) for name, split in split_frames.items()
        },
        "split_record_counts": {
            name: int(split["group_size"].sum()) for name, split in split_frames.items()
        },
    }
    return split_frames, metadata


def write_split_files(split_frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in split_frames.items():
        frame[["text", "labels", "id"]].to_csv(
            split_dir / f"{name}.tsv",
            sep="\t",
            header=False,
            index=False,
            encoding="utf-8",
        )


def split_label_summary(split_frames: dict[str, pd.DataFrame]) -> dict:
    result = {}
    for name, frame in split_frames.items():
        matrix = binary_matrix(frame["labels"].tolist())
        result[name] = {
            "groups": int(len(frame)),
            "records_before_grouping": int(frame["group_size"].sum()),
            "label_presence": {
                label: int(matrix[:, index].sum())
                for index, label in enumerate(LABELS)
            },
            "primary_label_counts": {
                LABELS[index]: int(count)
                for index, count in Counter(frame["primary_label"]).items()
            },
        }
    return result


def run_inference(
    texts: list[str], model_name: str, batch_size: int, max_length: int
) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, local_files_only=True
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    id_to_label = {int(key): str(value).lower() for key, value in model.config.id2label.items()}
    if set(id_to_label.values()) != set(LABELS):
        raise ValueError(
            f"Model labels {sorted(id_to_label.values())} do not match {LABELS}"
        )
    label_indices = [next(index for index, label in id_to_label.items() if label == name) for name in LABELS]

    probabilities = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            batch = torch.softmax(logits, dim=-1).cpu().numpy()
            probabilities.append(batch[:, label_indices])

    elapsed = time.perf_counter() - started
    return np.vstack(probabilities), {
        "model": model_name,
        "device": str(device),
        "batch_size": batch_size,
        "max_length": max_length,
        "samples": len(texts),
        "elapsed_seconds": round(elapsed, 6),
        "milliseconds_per_sample": round(elapsed * 1000 / len(texts), 6),
        "model_id2label": id_to_label,
    }


def metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    y_pred = (probabilities >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        average=None,
        zero_division=0,
    )
    return {
        "threshold": threshold,
        "samples": int(len(y_true)),
        "exact_match_accuracy": float(np.all(y_true == y_pred, axis=1).mean()),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "micro_precision": float(
            precision_recall_fscore_support(
                y_true, y_pred, average="micro", zero_division=0
            )[0]
        ),
        "micro_recall": float(
            precision_recall_fscore_support(
                y_true, y_pred, average="micro", zero_division=0
            )[1]
        ),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
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
            for index, label in enumerate(LABELS)
        },
        "predicted_positive_counts": {
            label: int(y_pred[:, index].sum())
            for index, label in enumerate(LABELS)
        },
    }


def tune_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    candidates = [round(value, 2) for value in np.arange(0.10, 0.91, 0.01)]
    scored = [
        {
            "threshold": threshold,
            "macro_f1": metrics(y_true, probabilities, threshold)["macro_f1"],
            "micro_f1": metrics(y_true, probabilities, threshold)["micro_f1"],
        }
        for threshold in candidates
    ]
    selected = max(
        scored,
        key=lambda item: (item["macro_f1"], item["micro_f1"], -abs(item["threshold"] - 0.30)),
    )
    return {
        "selection_metric": "validation macro-F1",
        "candidate_thresholds": scored,
        "selected_threshold": selected["threshold"],
        "selected_validation_metrics": metrics(
            y_true, probabilities, selected["threshold"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).parents[2] / "dataset 1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[2] / "reports" / "grouped_text_benchmark",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped, grouping_metadata = load_and_group(args.dataset_root)
    split_frames, split_metadata = stable_group_split(grouped, args.seed)
    write_split_files(split_frames, output_dir)

    texts_by_split = {
        name: frame["text"].tolist() for name, frame in split_frames.items()
    }
    validation_probabilities, validation_inference = run_inference(
        texts_by_split["validation"], MODEL_NAME, args.batch_size, args.max_length
    )
    test_probabilities, test_inference = run_inference(
        texts_by_split["test"], MODEL_NAME, args.batch_size, args.max_length
    )

    y_validation = binary_matrix(split_frames["validation"]["labels"].tolist())
    y_test = binary_matrix(split_frames["test"]["labels"].tolist())
    tuning = tune_threshold(y_validation, validation_probabilities)
    selected_threshold = tuning["selected_threshold"]

    pd.DataFrame(validation_probabilities, columns=LABELS).to_csv(
        output_dir / "validation_predictions.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(test_probabilities, columns=LABELS).to_csv(
        output_dir / "test_predictions.tsv",
        sep="\t",
        index=False,
    )

    # The test result is computed once with the threshold selected on validation.
    result = {
        "experiment": "grouped-identical-text leakage-controlled benchmark",
        "status": "completed",
        "seed": args.seed,
        "model": MODEL_NAME,
        "labels": LABELS,
        "threshold_policy": (
            "Threshold selected by validation macro-F1 over 0.10..0.90 in "
            "0.01 increments; the selected value is applied once to frozen test."
        ),
        "grouping": grouping_metadata,
        "splitting": split_metadata,
        "label_summary": split_label_summary(split_frames),
        "overlap_check": {
            "train_validation_text_overlap": int(
                len(set(split_frames["train"]["text"]) & set(split_frames["validation"]["text"]))
            ),
            "train_test_text_overlap": int(
                len(set(split_frames["train"]["text"]) & set(split_frames["test"]["text"]))
            ),
            "validation_test_text_overlap": int(
                len(set(split_frames["validation"]["text"]) & set(split_frames["test"]["text"]))
            ),
        },
        "validation_inference": validation_inference,
        "test_inference": test_inference,
        "threshold_tuning": tuning,
        "frozen_test_metrics": metrics(
            y_test, test_probabilities, selected_threshold
        ),
        "limitations": [
            "This is benchmark-domain evaluation on Reddit-derived GoEmotions text, not journal-domain validation.",
            "Identical text is grouped exactly as stored; semantic near-duplicates are not removed.",
            "Label union within duplicate-text groups preserves observed labels but changes the row-level representation.",
            "The model is pretrained and not retrained on the newly created training split.",
            "Threshold tuning uses validation only; test labels are not used for selection.",
        ],
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
