"""Experiment 3: fine-tune and compare a multi-label emotion classifier.

This script uses the frozen grouped-text splits from the previous benchmark.
It fine-tunes a separate copy of the cached transformer with BCE-with-logits
loss, selects the best epoch and threshold using validation macro-F1, and
evaluates the selected checkpoint once on the frozen test set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
SEED = 20261027
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_split(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    texts, ids, labels = [], [], []
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.reader(file, delimiter="\t"):
            if len(row) != 3:
                raise ValueError(f"Malformed row in {path}: {row!r}")
            texts.append(row[0])
            labels.append(row[1])
            ids.append(row[2])
    matrix = np.zeros((len(labels), len(PROJECT_LABELS)), dtype=np.float32)
    for row_index, value in enumerate(labels):
        indexes = [int(token) for token in value.split(",") if token.strip().isdigit()]
        if not indexes or any(index < 0 or index >= len(PROJECT_LABELS) for index in indexes):
            raise ValueError(f"Invalid labels in {path}: {value!r}")
        matrix[row_index, sorted(set(indexes))] = 1.0
    return texts, ids, matrix


class EmotionDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray) -> None:
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        return self.texts[index], self.labels[index]


def collate_batch(batch: list[tuple[str, torch.Tensor]]) -> tuple[list[str], torch.Tensor]:
    texts, labels = zip(*batch)
    return list(texts), torch.stack(labels)


def model_order_labels(model) -> list[str]:
    mapping = {int(key): str(value).lower() for key, value in model.config.id2label.items()}
    labels = [mapping[index] for index in sorted(mapping)]
    if set(labels) != set(PROJECT_LABELS):
        raise ValueError(f"Model labels {labels} are incompatible with {PROJECT_LABELS}")
    return labels


def reorder_to_model(labels: np.ndarray, model_labels: list[str]) -> np.ndarray:
    return labels[:, [PROJECT_LABELS.index(label) for label in model_labels]]


def reorder_to_project(values: np.ndarray, model_labels: list[str]) -> np.ndarray:
    indexes = [model_labels.index(label) for label in PROJECT_LABELS]
    return values[:, indexes]


def metric_summary(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    y_pred = (probabilities >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(PROJECT_LABELS))), average=None, zero_division=0
    )
    micro = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
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
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "true_positive": int(((y_pred[:, index] == 1) & (y_true[:, index] == 1)).sum()),
                "false_positive": int(((y_pred[:, index] == 1) & (y_true[:, index] == 0)).sum()),
                "false_negative": int(((y_pred[:, index] == 0) & (y_true[:, index] == 1)).sum()),
            }
            for index, label in enumerate(PROJECT_LABELS)
        },
    }


def tune_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    candidates = [round(value, 2) for value in np.arange(0.10, 0.91, 0.01)]
    scores = [
        {
            "threshold": threshold,
            "macro_f1": metric_summary(y_true, probabilities, threshold)["macro_f1"],
            "micro_f1": metric_summary(y_true, probabilities, threshold)["micro_f1"],
        }
        for threshold in candidates
    ]
    selected = max(
        scores,
        key=lambda item: (item["macro_f1"], item["micro_f1"], -abs(item["threshold"] - 0.35)),
    )
    return {
        "selection_metric": "validation macro-F1",
        "candidate_scores": scores,
        "selected_threshold": selected["threshold"],
        "selected_metrics": metric_summary(y_true, probabilities, selected["threshold"]),
    }


def predict(
    model,
    loader: DataLoader,
    device: torch.device,
    model_labels: list[str],
    tokenizer,
    max_length: int,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in loader:
            texts, _ = batch
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return reorder_to_project(np.vstack(outputs), model_labels)


def write_predictions(
    path: Path,
    ids: list[str],
    texts: list[str],
    probabilities: np.ndarray,
) -> None:
    frame = pd.DataFrame(
        {
            "id": ids,
            "text_sha256": [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts],
        }
    )
    for index, label in enumerate(PROJECT_LABELS):
        frame[label] = probabilities[:, index]
    frame.to_csv(path, sep="\t", index=False)


def bootstrap_difference(
    y_true: np.ndarray,
    baseline_probabilities: np.ndarray,
    experimental_probabilities: np.ndarray,
    baseline_threshold: float,
    experimental_threshold: float,
    replicates: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    differences = {key: [] for key in ["macro_f1", "micro_f1", "weighted_f1", "exact_match_accuracy"]}
    baseline = metric_summary(y_true, baseline_probabilities, baseline_threshold)
    experimental = metric_summary(y_true, experimental_probabilities, experimental_threshold)
    for _ in range(replicates):
        indexes = rng.integers(0, len(y_true), size=len(y_true))
        old = metric_summary(y_true[indexes], baseline_probabilities[indexes], baseline_threshold)
        new = metric_summary(y_true[indexes], experimental_probabilities[indexes], experimental_threshold)
        for key in differences:
            differences[key].append(new[key] - old[key])
    return {
        "replicates": replicates,
        "seed": seed,
        "interval": "paired sample-level percentile 95% bootstrap interval",
        "differences": {
            key: {
                "point_estimate": float(experimental[key] - baseline[key]),
                "lower_95": float(np.percentile(values, 2.5)),
                "upper_95": float(np.percentile(values, 97.5)),
            }
            for key, values in differences.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).parents[2]
    parser.add_argument("--benchmark-dir", type=Path, default=root / "reports" / "grouped_text_benchmark")
    parser.add_argument("--output-dir", type=Path, default=root / "reports" / "experiment_3")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    set_seed(SEED)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "best_model"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    splits = {}
    for name in ("train", "validation", "test"):
        splits[name] = read_split(args.benchmark_dir / "splits" / f"{name}.tsv")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, local_files_only=True)
    model_labels = model_order_labels(model)
    if len(model_labels) != len(PROJECT_LABELS):
        raise ValueError("Unexpected number of model labels")
    model.config.problem_type = "multi_label_classification"

    train_dataset = EmotionDataset(
        splits["train"][0], reorder_to_model(splits["train"][2], model_labels)
    )
    validation_dataset = EmotionDataset(
        splits["validation"][0], reorder_to_model(splits["validation"][2], model_labels)
    )
    test_dataset = EmotionDataset(
        splits["test"][0], reorder_to_model(splits["test"][2], model_labels)
    )
    loader_args = {"batch_size": args.batch_size, "collate_fn": collate_batch, "num_workers": 0}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    model.to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup_steps)
        if step < warmup_steps
        else max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)),
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    training_log = []
    best = None
    started = time.perf_counter()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for texts, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            encoded = tokenizer(
                texts, padding=True, truncation=True, max_length=args.max_length,
                return_tensors="pt",
            )
            logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += float(loss.item())

        validation_probabilities = predict(
            model, validation_loader, device, model_labels, tokenizer, args.max_length
        )
        validation_labels = splits["validation"][2]
        tuning = tune_threshold(validation_labels, validation_probabilities)
        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "mean_training_loss": epoch_loss / len(train_loader),
            "validation": tuning["selected_metrics"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        training_log.append(epoch_record)
        if best is None or epoch_record["validation"]["macro_f1"] > best["validation"]["macro_f1"]:
            best = epoch_record
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

    if best is None:
        raise RuntimeError("Training produced no checkpoint")
    best_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, local_files_only=True)
    best_model.config.problem_type = "multi_label_classification"
    best_model.to(device)
    validation_probabilities = predict(
        best_model, validation_loader, device, model_labels, tokenizer, args.max_length
    )
    test_started = time.perf_counter()
    test_probabilities = predict(
        best_model, test_loader, device, model_labels, tokenizer, args.max_length
    )
    test_elapsed = time.perf_counter() - test_started
    validation_tuning = tune_threshold(splits["validation"][2], validation_probabilities)
    threshold = float(validation_tuning["selected_threshold"])

    baseline_validation = pd.read_csv(args.benchmark_dir / "validation_predictions.tsv", sep="\t").to_numpy(dtype=float)
    baseline_test = pd.read_csv(args.benchmark_dir / "test_predictions.tsv", sep="\t").to_numpy(dtype=float)
    baseline_threshold = 0.35
    baseline_test_metrics = metric_summary(splits["test"][2], baseline_test, baseline_threshold)
    experimental_test_metrics = metric_summary(splits["test"][2], test_probabilities, threshold)
    write_predictions(output / "baseline_validation_predictions.tsv", splits["validation"][1], splits["validation"][0], baseline_validation)
    write_predictions(output / "baseline_test_predictions.tsv", splits["test"][1], splits["test"][0], baseline_test)
    write_predictions(output / "experimental_validation_predictions.tsv", splits["validation"][1], splits["validation"][0], validation_probabilities)
    write_predictions(output / "experimental_test_predictions.tsv", splits["test"][1], splits["test"][0], test_probabilities)

    paired = bootstrap_difference(
        splits["test"][2], baseline_test, test_probabilities, baseline_threshold, threshold,
        args.bootstrap_replicates, 424242,
    )
    metadata = {
        "status": "completed",
        "experiment": "experiment 3 fine-tuned multi-label emotion model comparison",
        "model_name": args.model_name,
        "model_revision": "local Hugging Face cache revision not pinned in prior benchmark",
        "seed": SEED,
        "device": str(device),
        "dataset": {
            "benchmark_dir": str(args.benchmark_dir),
            "train_groups": len(splits["train"][0]),
            "validation_groups": len(splits["validation"][0]),
            "test_groups": len(splits["test"][0]),
            "frozen_splits_reused": True,
            "label_order": PROJECT_LABELS,
        },
        "label_mapping": {
            "model_head_order": model_labels,
            "research_order": PROJECT_LABELS,
            "mapping_applied": True,
        },
        "training": {
            "loss": "BCEWithLogitsLoss",
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
            "encoder_frozen": True,
            "training_mode": (
                "classification-head transfer learning; the encoder was frozen "
                "because available host memory could not support full AdamW state"
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "best_epoch": best["epoch"],
            "selection_metric": "validation macro-F1",
        },
        "baseline": {
            "threshold": baseline_threshold,
            "metrics": baseline_test_metrics,
            "source": str(args.benchmark_dir / "test_predictions.tsv"),
        },
        "experimental": {
            "threshold_selected_on_validation": threshold,
            "validation_metrics": validation_tuning["selected_metrics"],
            "test_metrics": experimental_test_metrics,
            "test_inference_seconds": test_elapsed,
            "test_milliseconds_per_sample": test_elapsed * 1000 / len(splits["test"][0]),
        },
        "paired_bootstrap": paired,
        "training_log": training_log,
        "artifacts": {
            "best_model": str(checkpoint_dir),
            "training_log": str(output / "training_log.json"),
        },
    }
    (output / "training_log.json").write_text(json.dumps(training_log, indent=2), encoding="utf-8")
    (output / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
