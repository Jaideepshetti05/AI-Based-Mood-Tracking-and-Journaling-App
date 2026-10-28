# Experiment 5: Robustness and Reliability of the Trained Emotion Classifier

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
was 0.5189 to 0.5320 (mean 0.5268), compared with original Model B's **0.5405**. All three weighted runs were below Model B on macro-F1.

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
| anger | 5607 | 2.5892 |
| disgust | 811 | 7.2293 |
| fear | 738 | 7.5849 |
| joy | 17235 | 1.2273 |
| sadness | 3210 | 3.5294 |
| surprise | 5353 | 2.6589 |
| neutral | 14194 | 1.4294 |

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
| 20261027 | 0.48 | 0.5189 | 0.5942 | 0.6060 | 0.4558 | 0.1329 |
| 20261028 | 0.46 | 0.5320 | 0.6015 | 0.6104 | 0.4545 | 0.1316 |
| 20261029 | 0.52 | 0.5296 | 0.5909 | 0.5984 | 0.4729 | 0.1266 |
| Mean | — | 0.5268 | 0.5955 | 0.6049 | 0.4611 | 0.1304 |

Original Model B: macro-F1 **0.5405**, micro-F1
**0.6171**, weighted-F1 **0.6148**,
exact-match **0.4058**, Hamming loss
**0.1332**.

## 7. Primary imbalance-run per-class results

The primary seed is 20261027. Its paired bootstrap comparison against original
Model B gave macro-F1 difference **-0.0217** with 95% interval
**[-0.0362, -0.0068]**.

| Class | Original B precision | Original B recall | Original B F1 | Weighted precision | Weighted recall | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| anger | 0.4410 | 0.5435 | 0.4869 | 0.4205 | 0.6148 | 0.4994 |
| disgust | 0.4414 | 0.4757 | 0.4579 | 0.2153 | 0.7670 | 0.3362 |
| fear | 0.4746 | 0.6087 | 0.5333 | 0.3429 | 0.7826 | 0.4768 |
| joy | 0.6742 | 0.8121 | 0.7368 | 0.7627 | 0.7302 | 0.7461 |
| sadness | 0.4823 | 0.4728 | 0.4775 | 0.4221 | 0.5767 | 0.4874 |
| surprise | 0.4875 | 0.4734 | 0.4803 | 0.4929 | 0.5784 | 0.5322 |
| neutral | 0.5185 | 0.7436 | 0.6110 | 0.6044 | 0.5110 | 0.5538 |

## 8. Error and trade-off analysis

- Disgust recall changed from 0.4757 to 0.7670, while precision changed from 0.4414 to 0.2153. Weighting increased detection but created more false positives.
- Fear recall changed from 0.6087 to 0.7826; fear F1 changed from 0.5333 to 0.4768.
- Joy remained strong and changed F1 from 0.7368 to 0.7461.
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
python research\experiment_5\run_experiment.py --seed 20261027 --output-dir reports\experiment_5
python research\experiment_5\run_experiment.py --seed 20261028 --output-dir reports\experiment_5\seed_20261028
python research\experiment_5\run_experiment.py --seed 20261029 --output-dir reports\experiment_5\seed_20261029
python research\experiment_5\generate_report.py
```

## 12. Created files

- `reports/experiment_5_robustness_report.md`
- `reports/experiment_5_methodology.md`
- `reports/experiment_5_results.json`
- `reports/experiment_5_model_comparison.csv`
- `research/experiment_5/run_experiment.py`
- `research/experiment_5/generate_report.py`
- `reports/experiment_5/` and its seed subdirectories
