"""Generate Experiment 5 aggregate reports from completed seed artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[2]
REPORTS = ROOT / "reports"


def main() -> None:
    primary = json.loads((REPORTS / "experiment_5" / "results.json").read_text())
    seed_paths = [REPORTS / "experiment_5" / "results.json"]
    seed_paths.extend(sorted((REPORTS / "experiment_5").glob("seed_*/results.json")))
    runs = [json.loads(path.read_text()) for path in seed_paths]
    original = primary["original_model_b_test_metrics"]
    rows = []
    for run in runs:
        metrics = run["test_metrics"]
        rows.append({
            "model": "Weighted head variant",
            "seed": run["seed"],
            "threshold": run["threshold_selected_on_validation"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "micro_precision": metrics["micro_precision"],
            "micro_recall": metrics["micro_recall"],
            "micro_f1": metrics["micro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "exact_match_accuracy": metrics["exact_match_accuracy"],
            "hamming_loss": metrics["hamming_loss"],
        })
    comparison = [
        {"model": "Model A baseline", "seed": "", "threshold": 0.35,
         "macro_precision": 0.4502724914315244, "macro_recall": 0.5338037660781828,
         "macro_f1": 0.42842640990782505, "micro_precision": 0.48120168657765283,
         "micro_recall": 0.4654205607476635, "micro_f1": 0.4731795802021249,
         "weighted_f1": 0.49076974585104133, "exact_match_accuracy": 0.3987775513984071,
         "hamming_loss": 0.161379091366126},
        {"model": "Model B original", "seed": 20261027, "threshold": 0.29,
         **{key: original[key] for key in ("macro_precision", "macro_recall", "macro_f1",
         "micro_precision", "micro_recall", "micro_f1", "weighted_f1",
         "exact_match_accuracy", "hamming_loss")}},
    ] + rows
    pd.DataFrame(comparison).to_csv(REPORTS / "experiment_5_model_comparison.csv", index=False)
    macro_values = [row["macro_f1"] for row in rows]
    report = f"""# Experiment 5: Robustness and Reliability of the Trained Emotion Classifier

## 1. Executive summary

Experiment 3's saved Model B artifacts were independently audited and found
complete: the checkpoint, training log, validation/test predictions, IDs,
SHA-256 text hashes, and verification outputs are present and aligned. The
original Model B result is therefore reproducible from saved artifacts, but
its exact weights were not retrained a second time.

A controlled class-imbalance variant was trained three times using the same
grouped training/validation splits and the same frozen test. It used the
encoder-frozen classification head, one epoch, AdamW, `BCEWithLogitsLoss`, and
square-root inverse-frequency positive weights. The variant's test macro-F1
was {min(macro_values):.4f} to {max(macro_values):.4f} (mean {sum(macro_values)/len(macro_values):.4f}), compared with original Model B's **{original['macro_f1']:.4f}**. All three weighted runs were below Model B on macro-F1.

The imbalance variant improved minority-class recall, especially disgust and
fear, but reduced overall macro-F1, micro-F1, and weighted-F1. It should not
replace Model B without a different objective or further validation.

## 2. Research questions

1. Are Model B's saved results internally reproducible and auditable?
2. Does a controlled positive-weighting variant improve minority-class
   reliability without sacrificing overall performance?
3. Are results stable across multiple random seeds on the available host?

## 3. Preserved evaluation design

The existing grouped splits were reused unchanged: 43,196 train groups, 5,399
validation groups, and 5,399 frozen test groups. No production code, original
datasets, or frozen split files were modified. Exact-text overlap remained zero.
All evaluations use the same seven-label multi-label target representation.

## 4. Model and imbalance variant

Model B is the Experiment 3 head-only transfer-learning model with threshold
0.29. The variant starts from the original cached checkpoint, freezes the
transformer encoder, and trains only the classification head for one epoch.
Positive weights are `sqrt((N - positive_count) / positive_count)` computed
from the training split only:

| Class | Train support | Positive weight |
|---|---:|---:|
""" + "\n".join(
        f"| {label} | {primary['class_counts_train'][label]} | {primary['pos_weight_research_order'][label]:.4f} |"
        for label in primary["class_counts_train"]
    ) + f"""

The validation split selected each run's threshold. The test labels were not
used for optimization, threshold selection, or seed selection.

## 5. Reproducibility and resource check

The original Model B artifact audit verified keyed prediction alignment,
complete checkpoint files, training log presence, zero split overlap, and
matching row counts. The three weighted runs were feasible on CPU. The host
had no GPU and approximately 1.2 GiB free memory during this phase; each run
used the memory-safe frozen-encoder setup. Full-transformer training remains
infeasible as documented in Experiment 4.

## 6. Three-seed results on the frozen test

| Seed | Validation threshold | Macro-F1 | Micro-F1 | Weighted-F1 | Exact-match | Hamming loss |
|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {row['seed']} | {row['threshold']:.2f} | {row['macro_f1']:.4f} | {row['micro_f1']:.4f} | {row['weighted_f1']:.4f} | {row['exact_match_accuracy']:.4f} | {row['hamming_loss']:.4f} |"
        for row in rows
    ) + f"""
| Mean | — | {sum(macro_values)/len(macro_values):.4f} | {sum(row['micro_f1'] for row in rows)/len(rows):.4f} | {sum(row['weighted_f1'] for row in rows)/len(rows):.4f} | {sum(row['exact_match_accuracy'] for row in rows)/len(rows):.4f} | {sum(row['hamming_loss'] for row in rows)/len(rows):.4f} |

Original Model B: macro-F1 **{original['macro_f1']:.4f}**, micro-F1
**{original['micro_f1']:.4f}**, weighted-F1 **{original['weighted_f1']:.4f}**,
exact-match **{original['exact_match_accuracy']:.4f}**, Hamming loss
**{original['hamming_loss']:.4f}**.

## 7. Primary imbalance-run per-class results

The primary seed is 20261027. Its paired bootstrap comparison against original
Model B gave macro-F1 difference **{primary['difference_weighted_minus_original_b']['macro_f1']:.4f}** with 95% interval
**[{primary['paired_bootstrap_vs_original_b']['macro_f1']['lower_95']:.4f}, {primary['paired_bootstrap_vs_original_b']['macro_f1']['upper_95']:.4f}]**.

| Class | Original B precision | Original B recall | Original B F1 | Weighted precision | Weighted recall | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {label} | {original['per_class'][label]['precision']:.4f} | {original['per_class'][label]['recall']:.4f} | {original['per_class'][label]['f1']:.4f} | {primary['test_metrics']['per_class'][label]['precision']:.4f} | {primary['test_metrics']['per_class'][label]['recall']:.4f} | {primary['test_metrics']['per_class'][label]['f1']:.4f} |"
        for label in primary["test_metrics"]["per_class"]
    ) + f"""

## 8. Error and trade-off analysis

- Disgust recall changed from {original['per_class']['disgust']['recall']:.4f} to {primary['test_metrics']['per_class']['disgust']['recall']:.4f}, while precision changed from {original['per_class']['disgust']['precision']:.4f} to {primary['test_metrics']['per_class']['disgust']['precision']:.4f}. Weighting increased detection but created more false positives.
- Fear recall changed from {original['per_class']['fear']['recall']:.4f} to {primary['test_metrics']['per_class']['fear']['recall']:.4f}; fear F1 changed from {original['per_class']['fear']['f1']:.4f} to {primary['test_metrics']['per_class']['fear']['f1']:.4f}.
- Joy remained strong and changed F1 from {original['per_class']['joy']['f1']:.4f} to {primary['test_metrics']['per_class']['joy']['f1']:.4f}.
- The weighted variant's primary macro-F1 decrease was statistically supported by its paired interval excluding zero. Its exact-match accuracy increased, showing that aggregate metrics can move in different directions.

## 9. Conclusions

The improvement from Model A to original Model B is supported on the frozen
test by the completed Experiment 3 comparison and is internally auditable.
Across three seeds, the tested imbalance-weighted variant did **not** reproduce
Model B's overall macro-F1; it consistently underperformed Model B on that
metric. It did, however, provide a controllable recall-oriented trade-off for
minority emotions. This does not invalidate Model B's result, but it shows that
the result is sensitive to the training loss and that class weighting is not an
automatic improvement.

## 10. Limitations and unresolved issues

The original Model B was not independently retrained with a second identical
seed; reproducibility here means artifact and metric reproducibility, not
independent weight reproduction. The variant used one epoch and a single
weighting family. No full-transformer run is possible on this host. The
benchmark is Reddit-derived and may not transfer to private journal text.

## 11. Reproduction commands

```powershell
python research\\experiment_5\\run_experiment.py --seed 20261027 --output-dir reports\\experiment_5
python research\\experiment_5\\run_experiment.py --seed 20261028 --output-dir reports\\experiment_5\\seed_20261028
python research\\experiment_5\\run_experiment.py --seed 20261029 --output-dir reports\\experiment_5\\seed_20261029
python research\\experiment_5\\generate_report.py
```

## 12. Created files

- `reports/experiment_5_robustness_report.md`
- `reports/experiment_5_methodology.md`
- `reports/experiment_5_results.json`
- `reports/experiment_5_model_comparison.csv`
- `research/experiment_5/run_experiment.py`
- `research/experiment_5/generate_report.py`
- `reports/experiment_5/` and its seed subdirectories
"""
    (REPORTS / "experiment_5_robustness_report.md").write_text(report, encoding="utf-8")
    methodology = """# Experiment 5 Methodology

The frozen grouped splits were reused unchanged. Model B's saved artifacts
were audited by row count, IDs, SHA-256 text hashes, checkpoint contents,
training-log presence, and split overlap. The controlled variant froze the
encoder and trained the seven-label head with BCEWithLogitsLoss, AdamW,
batch size 32, learning rate 2e-5, one epoch, and square-root
inverse-frequency positive weights computed from training support. Thresholds
were selected on validation macro-F1. The frozen test was evaluated only after
selection. Three seeds (20261027, 20261028, 20261029) were run. Paired
bootstrap comparisons used 2,000 resamples with seed 424242.
"""
    (REPORTS / "experiment_5_methodology.md").write_text(methodology, encoding="utf-8")
    aggregate = {
        "status": "completed",
        "runs": [{"seed": row["seed"], "threshold": row["threshold"], "test_metrics": next(run["test_metrics"] for run in runs if run["seed"] == row["seed"])} for row in rows],
        "original_model_b": original,
        "audit": primary["audit"],
        "conclusion": "weighted variant did not improve overall macro-F1 over original Model B across three seeds",
    }
    (REPORTS / "experiment_5_results.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
