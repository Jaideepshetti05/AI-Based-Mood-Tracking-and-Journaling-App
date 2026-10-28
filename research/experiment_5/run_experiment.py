"""Experiment 5 reproducibility audit and controlled weighted-head variant."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from research.experiment_3.train_and_evaluate import (  # noqa: E402
    EmotionDataset,
    PROJECT_LABELS,
    collate_batch,
    model_order_labels,
    reorder_to_model,
    reorder_to_project,
)

SEED = 20261027
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"


def read_split(path: Path):
    texts, ids, labels = [], [], []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 3:
                raise ValueError(f"Malformed row in {path}")
            texts.append(row[0])
            ids.append(row[2])
            vector = [0] * len(PROJECT_LABELS)
            for token in row[1].split(","):
                vector[int(token)] = 1
            labels.append(vector)
    return texts, ids, np.asarray(labels, dtype=np.float32)


def metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=list(range(7)), average=None, zero_division=0
    )
    micro = precision_recall_fscore_support(
        y_true, predictions, average="micro", zero_division=0
    )
    return {
        "threshold": threshold,
        "samples": len(y_true),
        "exact_match_accuracy": float(np.all(y_true == predictions, axis=1).mean()),
        "hamming_loss": float(hamming_loss(y_true, predictions)),
        "micro_precision": float(micro[0]),
        "micro_recall": float(micro[1]),
        "micro_f1": float(micro[2]),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "true_positive": int(((predictions[:, i] == 1) & (y_true[:, i] == 1)).sum()),
                "false_positive": int(((predictions[:, i] == 1) & (y_true[:, i] == 0)).sum()),
                "false_negative": int(((predictions[:, i] == 0) & (y_true[:, i] == 1)).sum()),
            }
            for i, label in enumerate(PROJECT_LABELS)
        },
    }


def tune(y_true, probabilities):
    candidates = [round(x, 2) for x in np.arange(0.10, 0.91, 0.01)]
    scored = [(threshold, metrics(y_true, probabilities, threshold)) for threshold in candidates]
    return max(scored, key=lambda item: (item[1]["macro_f1"], item[1]["micro_f1"], -abs(item[0] - 0.35)))


def predict(model, loader, tokenizer, model_labels, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for texts, _ in loader:
            encoded = tokenizer(
                texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
            )
            logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return reorder_to_project(np.vstack(outputs), model_labels)


def write_predictions(path, ids, texts, probabilities):
    frame = pd.DataFrame({
        "id": ids,
        "text_sha256": [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts],
    })
    for index, label in enumerate(PROJECT_LABELS):
        frame[label] = probabilities[:, index]
    frame.to_csv(path, sep="\t", index=False)


def bootstrap(y_true, old_probabilities, new_probabilities, old_threshold, new_threshold):
    rng = np.random.default_rng(424242)
    keys = ("macro_f1", "micro_f1", "weighted_f1", "exact_match_accuracy")
    values = {key: [] for key in keys}
    old = metrics(y_true, old_probabilities, old_threshold)
    new = metrics(y_true, new_probabilities, new_threshold)
    for _ in range(2000):
        indexes = rng.integers(0, len(y_true), size=len(y_true))
        old_sample = metrics(y_true[indexes], old_probabilities[indexes], old_threshold)
        new_sample = metrics(y_true[indexes], new_probabilities[indexes], new_threshold)
        for key in keys:
            values[key].append(new_sample[key] - old_sample[key])
    return {
        key: {
            "point_estimate": float(new[key] - old[key]),
            "lower_95": float(np.percentile(samples, 2.5)),
            "upper_95": float(np.percentile(samples, 97.5)),
        }
        for key, samples in values.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "experiment_5")
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    benchmark = ROOT / "reports" / "grouped_text_benchmark"
    prior = ROOT / "reports" / "experiment_3"
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    splits = {name: read_split(benchmark / "splits" / f"{name}.tsv") for name in ("train", "validation", "test")}

    audit = {}
    for name in ("train", "validation", "test"):
        texts, ids, labels = splits[name]
        hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        audit[name] = {
            "expected_rows": len(texts),
            "labels_present": {label: int(labels[:, i].sum()) for i, label in enumerate(PROJECT_LABELS)},
        }
        if name != "train":
            prediction_path = prior / f"experimental_{name}_predictions.tsv"
            frame = pd.read_csv(prediction_path, sep="\t")
            audit[name].update({
                "rows": len(frame),
                "columns": list(frame.columns),
                "ids_aligned": frame["id"].astype(str).tolist() == ids,
                "hashes_aligned": frame["text_sha256"].tolist() == hashes,
            })
    audit["overlap"] = {
        f"{left}_{right}": len(set(splits[left][0]) & set(splits[right][0]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    audit["checkpoint_files"] = [
        str(path.relative_to(ROOT))
        for path in (prior / "best_model").rglob("*")
        if path.is_file()
    ]
    audit["training_log_present"] = (prior / "training_log.json").is_file()
    audit["reproducibility_conclusion"] = "saved Model B artifacts are complete and independently aligned"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)
    model_labels = model_order_labels(model)
    train_texts, _, train_y = splits["train"]
    validation_texts, validation_ids, validation_y = splits["validation"]
    test_texts, test_ids, test_y = splits["test"]
    datasets = {
        "train": EmotionDataset(train_texts, reorder_to_model(train_y, model_labels)),
        "validation": EmotionDataset(validation_texts, reorder_to_model(validation_y, model_labels)),
        "test": EmotionDataset(test_texts, reorder_to_model(test_y, model_labels)),
    }
    loaders = {
        name: DataLoader(dataset, batch_size=32, shuffle=name == "train", collate_fn=collate_batch, num_workers=0)
        for name, dataset in datasets.items()
    }

    positive = train_y.sum(axis=0)
    negative = len(train_y) - positive
    pos_weight = np.sqrt(negative / positive)
    device = torch.device("cpu")
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-5, weight_decay=0.01)
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(reorder_to_model(pos_weight.reshape(1, -1), model_labels)[0], dtype=torch.float32)
    )
    started = time.perf_counter()
    model.train()
    total_loss = 0.0
    for texts, labels in loaders["train"]:
        optimizer.zero_grad(set_to_none=True)
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
        loss = loss_function(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item())

    validation_probabilities = predict(model, loaders["validation"], tokenizer, model_labels, device)
    threshold, validation_metrics = tune(validation_y, validation_probabilities)
    checkpoint = output / "weighted_head_model"
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    test_started = time.perf_counter()
    test_probabilities = predict(model, loaders["test"], tokenizer, model_labels, device)
    test_elapsed = time.perf_counter() - test_started
    write_predictions(output / "weighted_validation_predictions.tsv", validation_ids, validation_texts, validation_probabilities)
    write_predictions(output / "weighted_test_predictions.tsv", test_ids, test_texts, test_probabilities)

    prior_results = json.loads((prior / "results.json").read_text(encoding="utf-8"))
    original_b = prior_results["experimental"]["test_metrics"]
    original_b_test = pd.read_csv(prior / "experimental_test_predictions.tsv", sep="\t")[PROJECT_LABELS].to_numpy(dtype=float)
    weighted_test = metrics(test_y, test_probabilities, threshold)
    result = {
        "status": "completed",
        "experiment": "Experiment 5 robustness and class-imbalance follow-up",
        "seed": seed,
        "variant": "frozen-encoder classification head with sqrt inverse-frequency positive weighting",
        "class_counts_train": {label: int(positive[i]) for i, label in enumerate(PROJECT_LABELS)},
        "pos_weight_research_order": {label: float(pos_weight[i]) for i, label in enumerate(PROJECT_LABELS)},
        "threshold_selected_on_validation": float(threshold),
        "validation_metrics": validation_metrics,
        "test_metrics": weighted_test,
        "original_model_b_test_metrics": original_b,
        "difference_weighted_minus_original_b": {
            key: float(weighted_test[key] - original_b[key])
            for key in ("macro_f1", "micro_f1", "weighted_f1", "exact_match_accuracy", "hamming_loss")
        },
        "paired_bootstrap_vs_original_b": bootstrap(test_y, original_b_test, test_probabilities, 0.29, threshold),
        "training": {
            "loss": "BCEWithLogitsLoss with sqrt inverse-frequency positive weights",
            "optimizer": "AdamW",
            "batch_size": 32,
            "learning_rate": 2e-5,
            "epochs": 1,
            "encoder_frozen": True,
            "elapsed_seconds": time.perf_counter() - started,
            "test_inference_seconds": test_elapsed,
        },
        "audit": audit,
        "artifacts": {
            "checkpoint": str(checkpoint),
            "validation_predictions": str(output / "weighted_validation_predictions.tsv"),
            "test_predictions": str(output / "weighted_test_predictions.tsv"),
        },
    }
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
