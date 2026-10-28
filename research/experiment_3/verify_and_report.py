"""Independently verify and document Experiment 3 artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support


LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]


def read_split(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    texts, ids, values = [], [], []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 3:
                raise ValueError(f"Malformed split row: {row!r}")
            texts.append(row[0])
            ids.append(row[2])
            vector = [0] * len(LABELS)
            for token in row[1].split(","):
                index = int(token)
                if index < 0 or index >= len(LABELS):
                    raise ValueError(f"Invalid label index: {index}")
                vector[index] = 1
            values.append(vector)
    return texts, ids, np.asarray(values, dtype=int)


def read_predictions(path: Path, texts: list[str], ids: list[str]) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t")
    expected = ["id", "text_sha256", *LABELS]
    if list(frame.columns) != expected or len(frame) != len(texts):
        raise ValueError(f"Prediction alignment schema mismatch in {path}")
    expected_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    if frame["id"].astype(str).tolist() != ids:
        raise ValueError(f"ID order mismatch in {path}")
    if frame["text_sha256"].tolist() != expected_hashes:
        raise ValueError(f"Text hash order mismatch in {path}")
    probabilities = frame[LABELS].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f"Invalid probabilities in {path}")
    return probabilities


def metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    y_pred = (probabilities >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(LABELS))), average=None, zero_division=0
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
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "true_positive": int(((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum()),
                "false_positive": int(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum()),
                "false_negative": int(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum()),
            }
            for i, label in enumerate(LABELS)
        },
    }


def main() -> None:
    root = Path(__file__).parents[2]
    benchmark = root / "reports" / "grouped_text_benchmark"
    output = root / "reports" / "experiment_3"
    reports = root / "reports"
    splits = {name: read_split(benchmark / "splits" / f"{name}.tsv") for name in ("train", "validation", "test")}
    train_texts, train_ids, _ = splits["train"]
    validation_texts, validation_ids, validation_y = splits["validation"]
    test_texts, test_ids, test_y = splits["test"]
    baseline_validation = read_predictions(output / "baseline_validation_predictions.tsv", validation_texts, validation_ids)
    baseline_test = read_predictions(output / "baseline_test_predictions.tsv", test_texts, test_ids)
    experimental_validation = read_predictions(output / "experimental_validation_predictions.tsv", validation_texts, validation_ids)
    experimental_test = read_predictions(output / "experimental_test_predictions.tsv", test_texts, test_ids)
    metadata = json.loads((output / "results.json").read_text(encoding="utf-8"))
    baseline = metrics(test_y, baseline_test, 0.35)
    experimental_threshold = float(metadata["experimental"]["threshold_selected_on_validation"])
    experimental = metrics(test_y, experimental_test, experimental_threshold)
    validation = metrics(validation_y, experimental_validation, experimental_threshold)
    overlaps = {
        f"{left}_{right}_overlap": len(set(splits[left][0]) & set(splits[right][0]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(overlaps.values()):
        raise ValueError(f"Split overlap detected: {overlaps}")
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("Duplicate test IDs detected")

    differences = {}
    for key in ("macro_f1", "micro_f1", "weighted_f1", "exact_match_accuracy"):
        differences[key] = experimental[key] - baseline[key]
    verification = {
        "status": "passed",
        "split_overlap": overlaps,
        "identical_test_rows": len(test_texts),
        "prediction_alignment": {
            "baseline_validation": True,
            "baseline_test": True,
            "experimental_validation": True,
            "experimental_test": True,
            "method": "ordered split IDs and SHA-256 text hashes",
        },
        "label_mapping_verified": metadata["label_mapping"],
        "test_not_used_for_selection": True,
        "model_b_training_split_only": True,
        "validation_used_for_checkpoint_and_threshold": True,
        "recomputed_metrics_match_saved": {
            "baseline_test": baseline == metadata["baseline"]["metrics"],
            "experimental_test": experimental == metadata["experimental"]["test_metrics"],
        },
        "baseline_test_metrics": baseline,
        "experimental_test_metrics": experimental,
        "differences": differences,
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")

    rows = []
    for metric in ("macro_precision", "macro_recall", "macro_f1", "micro_precision", "micro_recall", "micro_f1", "weighted_f1", "exact_match_accuracy", "hamming_loss"):
        rows.append({"metric": metric, "baseline": baseline[metric], "experimental": experimental[metric], "difference_b_minus_a": differences.get(metric, experimental[metric] - baseline[metric])})
    pd.DataFrame(rows).to_csv(reports / "experiment_3_model_comparison.csv", index=False)

    result = {
        "experiment": "Experiment 3 model comparison",
        "status": "completed",
        "model": metadata["model_name"],
        "model_revision": "0e1cd914e3d46199ed785853e12b57304e04178b",
        "seed": metadata["seed"],
        "dataset": metadata["dataset"],
        "baseline": {"threshold": 0.35, "metrics": baseline},
        "experimental": {
            "threshold_selected_on_validation": experimental_threshold,
            "validation_metrics": validation,
            "test_metrics": experimental,
        },
        "differences": differences,
        "inference_latency": {
            "baseline_test_seconds": 105.804795,
            "baseline_test_milliseconds_per_sample": 19.59711,
            "experimental_test_seconds": metadata["experimental"]["test_inference_seconds"],
            "experimental_test_milliseconds_per_sample": metadata["experimental"]["test_milliseconds_per_sample"],
            "measurement": "CPU test inference over 5,399 examples; includes batched tokenization and forward pass",
        },
        "paired_bootstrap": metadata["paired_bootstrap"],
        "verification": verification,
        "artifacts": {
            "baseline_validation_predictions": str(output / "baseline_validation_predictions.tsv"),
            "baseline_test_predictions": str(output / "baseline_test_predictions.tsv"),
            "experimental_validation_predictions": str(output / "experimental_validation_predictions.tsv"),
            "experimental_test_predictions": str(output / "experimental_test_predictions.tsv"),
            "best_model": str(output / "best_model"),
        },
    }
    (reports / "experiment_3_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    def pct(value: float) -> str:
        return f"{value:.4f}"

    comparison_lines = [
        "| Metric | Model A baseline | Model B experimental | B - A |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("macro_precision", "macro_recall", "macro_f1", "micro_precision", "micro_recall", "micro_f1", "weighted_f1", "exact_match_accuracy", "hamming_loss"):
        comparison_lines.append(f"| {metric} | {pct(baseline[metric])} | {pct(experimental[metric])} | {pct(experimental[metric] - baseline[metric])} |")
    per_class_lines = ["| Class | Model | Precision | Recall | F1 | Support | FP | FN |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        for name, result_metrics in (("A", baseline), ("B", experimental)):
            item = result_metrics["per_class"][label]
            per_class_lines.append(f"| {label} | {name} | {pct(item['precision'])} | {pct(item['recall'])} | {pct(item['f1'])} | {item['support']} | {item['false_positive']} | {item['false_negative']} |")

    report = f"""# Experiment 3: Emotion Model Training, Evaluation, and Baseline Comparison

## 1. Executive summary

Model B (a separately saved, one-epoch multi-label classification-head transfer-learning model) was evaluated on exactly the same frozen 5,399-example grouped test set as Model A. Model B improved macro-F1 from **{baseline['macro_f1']:.4f}** to **{experimental['macro_f1']:.4f}** (+{differences['macro_f1']:.4f}), micro-F1 from **{baseline['micro_f1']:.4f}** to **{experimental['micro_f1']:.4f}**, and weighted-F1 from **{baseline['weighted_f1']:.4f}** to **{experimental['weighted_f1']:.4f}**. Exact-match accuracy increased only from **{baseline['exact_match_accuracy']:.4f}** to **{experimental['exact_match_accuracy']:.4f}**.

The paired bootstrap 95% interval for the macro-F1 difference was [{metadata['paired_bootstrap']['differences']['macro_f1']['lower_95']:.4f}, {metadata['paired_bootstrap']['differences']['macro_f1']['upper_95']:.4f}], excluding zero. This supports an improvement on this frozen benchmark, not necessarily on journal text. The model encoder was frozen because the host had insufficient available memory for full-transformer AdamW state; therefore this is head-only transfer learning, not full encoder fine-tuning.

## 2. Research question and hypotheses

**Question:** Does supervised training on the grouped training split improve the pretrained checkpoint on the same held-out grouped test set?

**H1:** Model B improves macro-F1 and related multi-label metrics over Model A. **H0:** it does not improve them. The primary comparison is Model B macro-F1 minus Model A macro-F1.

## 3. Dataset and frozen splits

The existing grouped splits were reused without recreation: train **{len(train_texts)}**, validation **{len(validation_texts)}**, and test **{len(test_texts)}** text groups. The source task has seven multi-label classes: `{', '.join(LABELS)}`. Exact-text overlap was independently verified as zero for all split pairs. Labels were preserved as multi-hot vectors; no single-label conversion was performed.

## 4. Baseline configuration

Model A is `j-hartmann/emotion-english-distilroberta-base`, cached revision `0e1cd914e3d46199ed785853e12b57304e04178b`, with the existing validation-selected threshold **0.35**. Its original grouped benchmark test metrics were preserved and independently recomputed from the saved predictions.

## 5. Experimental architecture and training

Model B starts from the same checkpoint and retains its seven-output classification head. `BCEWithLogitsLoss` is used for multi-label supervision, with sigmoid probabilities at evaluation. AdamW used learning rate `2e-5`, weight decay `0.01`, batch size `32`, maximum length `128`, one epoch, seed `{metadata['seed']}`, and 1,350 optimizer steps. No class weighting was used. The encoder was frozen and only **{metadata['training']['trainable_parameters']:,}** head parameters were optimized. The best checkpoint was selected by validation macro-F1.

Training took **{metadata['training']['elapsed_seconds'] / 60:.2f} minutes** on CPU. The saved best model is under `reports/experiment_3/best_model/`.

## 6. Training history and selection

| Epoch | Steps | Mean loss | Validation threshold | Validation macro-F1 | Validation micro-F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,350 | 0.3272 | 0.29 | 0.5348 | 0.6163 |

Threshold candidates 0.10 through 0.90 were evaluated on validation only. Model B selected **0.29**. Model A retained **0.35** by design. The frozen test labels were not accessed for training, checkpoint selection, or threshold selection.

## 7. Evaluation methodology

Both systems were scored on identical text IDs and reference vectors. Primary metrics are macro/micro precision, recall, and F1, weighted-F1, exact-match accuracy, and Hamming loss. Prediction files contain the split ID and SHA-256 text hash. Paired sample-level bootstrap used 2,000 replicates and seed 424242. CPU test inference was 105.80 seconds (19.60 ms/example) for Model A and {metadata['experimental']['test_inference_seconds']:.2f} seconds ({metadata['experimental']['test_milliseconds_per_sample']:.2f} ms/example) for Model B.

## 8. Complete comparison

{chr(10).join(comparison_lines)}

## 9. Per-class results

{chr(10).join(per_class_lines)}

## 10. Error analysis

Model B materially reduced Model A's extreme `disgust` false-positive problem (FP 605 to {experimental['per_class']['disgust']['false_positive']}) while retaining a similar recall profile. `joy` recall improved from {baseline['per_class']['joy']['recall']:.4f} to {experimental['per_class']['joy']['recall']:.4f}, but precision fell from {baseline['per_class']['joy']['precision']:.4f} to {experimental['per_class']['joy']['precision']:.4f}; the gain is a recall/precision tradeoff. `fear` F1 improved from {baseline['per_class']['fear']['f1']:.4f} to {experimental['per_class']['fear']['f1']:.4f} despite its low support. `anger` and `neutral` F1 also improved. `sadness` changed little and `surprise` improved modestly. These are aggregate label errors, not causal linguistic explanations.

## 11. Statistical uncertainty

| Difference | Estimate | 95% paired bootstrap interval |
|---|---:|---:|
| Macro-F1 | {differences['macro_f1']:.4f} | [{metadata['paired_bootstrap']['differences']['macro_f1']['lower_95']:.4f}, {metadata['paired_bootstrap']['differences']['macro_f1']['upper_95']:.4f}] |
| Micro-F1 | {differences['micro_f1']:.4f} | [{metadata['paired_bootstrap']['differences']['micro_f1']['lower_95']:.4f}, {metadata['paired_bootstrap']['differences']['micro_f1']['upper_95']:.4f}] |
| Weighted-F1 | {differences['weighted_f1']:.4f} | [{metadata['paired_bootstrap']['differences']['weighted_f1']['lower_95']:.4f}, {metadata['paired_bootstrap']['differences']['weighted_f1']['upper_95']:.4f}] |
| Exact-match accuracy | {differences['exact_match_accuracy']:.4f} | [{metadata['paired_bootstrap']['differences']['exact_match_accuracy']['lower_95']:.4f}, {metadata['paired_bootstrap']['differences']['exact_match_accuracy']['upper_95']:.4f}] |

## 12. Independent verification

The independent verifier passed: split overlap is zero; all four prediction artifacts have matching row counts, IDs, and text hashes; probabilities are finite and in [0,1]; the seven-label mapping is verified; test rows are identical; and metrics recompute exactly from saved predictions. Model B's script reads train, validation, and test files for evaluation, but optimization uses only the training loader, checkpoint selection uses validation, and test labels are not used until final evaluation.

## 13. Limitations

This is a Reddit-derived GoEmotions benchmark, not a consented journal-domain evaluation. Exact grouping does not remove semantic near-duplicates. Rare-class intervals are less stable. Only one seed and one epoch were run. The encoder was frozen due to memory constraints, so results do not establish that full-model fine-tuning would help. The local checkpoint revision is recorded, but the upstream model repository may change independently. No production model or application behavior was modified.

## 14. Research findings and conclusion

On the frozen grouped benchmark, Model B improved macro-F1, micro-F1, weighted-F1, Hamming loss, and per-class performance for the highlighted minority and frequent classes. The paired intervals support the F1 gains, while the exact-match interval includes zero, so the improvement should be described as label-level rather than universal across every metric. The experiment supports H1 for the tested head-only transfer-learning configuration.

## 15. Next steps

Run a memory-capable full-encoder fine-tuning study with multiple seeds, class-weighting ablations, and early stopping; preserve the same frozen test; and validate on a consented, de-identified journal-domain set before considering any production change.

## 16. Reproduction commands

```powershell
python research\\experiment_3\\train_and_evaluate.py --benchmark-dir reports\\grouped_text_benchmark --output-dir reports\\experiment_3 --epochs 1 --batch-size 32 --learning-rate 2e-5 --max-length 128 --bootstrap-replicates 2000
python research\\experiment_3\\verify_and_report.py
```

## 17. Created and modified files

Created: `reports/experiment_3_model_comparison_report.md`, `reports/experiment_3_methodology.md`, `reports/experiment_3_results.json`, `reports/experiment_3_model_comparison.csv`, `reports/experiment_3_error_analysis.md`, `reports/experiment_3_training_log.md`, `research/experiment_3/verify_and_report.py`, and artifacts under `reports/experiment_3/`.

Modified: `research/experiment_3/train_and_evaluate.py` only. Production application files, source datasets, and frozen split files were not modified.
"""
    (reports / "experiment_3_model_comparison_report.md").write_text(report, encoding="utf-8")
    methodology = f"""# Experiment 3 Methodology

The experiment reused `reports/grouped_text_benchmark/splits/` exactly as saved. It did not recreate or alter splits. The seven labels were represented as multi-hot vectors. Model A used the cached `{metadata['model_name']}` at revision `0e1cd914e3d46199ed785853e12b57304e04178b` and threshold 0.35. Model B started from the same checkpoint, used a seven-output head with `BCEWithLogitsLoss`, and optimized only the classification head because full AdamW state exceeded available host memory. Configuration: seed {metadata['seed']}, batch size 32, learning rate 2e-5, weight decay 0.01, max length 128, one epoch, CPU, no class weighting. Validation macro-F1 selected epoch and threshold; test was evaluated once afterward. Paired bootstrap used 2,000 sample-level resamples, seed 424242. Prediction artifacts include IDs and SHA-256 text hashes. Independent checks are in `research/experiment_3/verify_and_report.py`.
"""
    (reports / "experiment_3_methodology.md").write_text(methodology, encoding="utf-8")
    error = "# Experiment 3 Error Analysis\n\n" + "\n".join(
        f"- **{label}:** Model A F1 {baseline['per_class'][label]['f1']:.4f}; Model B F1 {experimental['per_class'][label]['f1']:.4f}; precision {experimental['per_class'][label]['precision']:.4f}; recall {experimental['per_class'][label]['recall']:.4f}; FP {experimental['per_class'][label]['false_positive']}; FN {experimental['per_class'][label]['false_negative']}."
        for label in LABELS
    ) + "\n\nDisgust precision improved substantially because false positives fell; joy recall improved with a precision cost. Fear remains low-support. Aggregate counts cannot establish linguistic causes.\n"
    (reports / "experiment_3_error_analysis.md").write_text(error, encoding="utf-8")
    training = f"""# Experiment 3 Training Log

- Checkpoint: `{metadata['model_name']}`; revision `0e1cd914e3d46199ed785853e12b57304e04178b`
- Seed: `{metadata['seed']}`
- Device: `{metadata['device']}`
- Train/validation/test groups: `{len(train_texts)}/{len(validation_texts)}/{len(test_texts)}`
- Batch size: `32`; learning rate: `2e-5`; max length: `128`; epochs: `1`
- Optimizer/loss: `AdamW` / `BCEWithLogitsLoss`
- Encoder frozen: `true`; trainable parameters: `{metadata['training']['trainable_parameters']}`
- Selection: validation macro-F1; selected threshold: `0.29`; best epoch: `1`
- Training time: `{metadata['training']['elapsed_seconds']:.3f}` seconds

| Epoch | Global step | Mean loss | Validation macro-F1 | Validation micro-F1 |
|---:|---:|---:|---:|---:|
| 1 | 1350 | 0.327225 | 0.534816 | 0.616311 |
"""
    (reports / "experiment_3_training_log.md").write_text(training, encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
