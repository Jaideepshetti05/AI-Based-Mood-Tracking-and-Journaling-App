# Final Research Experiment Report

## Personalized AI-Based Mood Tracking and Journaling Application Using NLP

**Experiment:** Leakage-Controlled Grouped-Text Benchmark with Validation
Threshold Calibration and Independent Verification  
**Report date:** 27 October 2026  
**Status:** Completed successfully  
**Production application modified:** No  
**Original datasets modified:** No  
**Existing model files modified:** No  
**Transformer retrained or fine-tuned:** No

---

## 1. Executive Summary

This experiment established a stronger benchmark for the existing pretrained
emotion classifier before model improvement.

The existing processed GoEmotions-derived data was combined and grouped by
exact text before new train, validation, and test splits were created. All
labels observed for an identical-text group were retained through label union.
The generated splits have zero exact-text overlap.

The model was evaluated without retraining:

```text
j-hartmann/emotion-english-distilroberta-base
```

The prediction threshold was selected exclusively on validation macro-F1. The
selected threshold was **0.35**. The frozen test set was then evaluated once.

### Frozen grouped-test results

| Metric | Result |
|---|---:|
| Exact-match accuracy | 0.3988 |
| Hamming loss | 0.1614 |
| Micro precision | 0.4812 |
| Micro recall | 0.4654 |
| Micro-F1 | 0.4732 |
| Macro precision | 0.4503 |
| Macro recall | 0.5338 |
| Macro-F1 | 0.4284 |
| Weighted-F1 | 0.4908 |

Bootstrap 95% percentile intervals over 2,000 frozen-test resamples:

| Metric | Estimate | Lower 95% | Upper 95% |
|---|---:|---:|---:|
| Macro-F1 | 0.4284 | 0.4120 | 0.4447 |
| Micro-F1 | 0.4732 | 0.4612 | 0.4846 |
| Weighted-F1 | 0.4908 | 0.4781 | 0.5030 |
| Exact-match accuracy | 0.3988 | 0.3858 | 0.4108 |

The benchmark is better controlled for exact-text leakage than the original
provided split. However, the new result must not be described as a model
improvement over the prior baseline because the test population, duplicate-text
handling, and threshold procedure changed.

The most important limitation remains domain mismatch: GoEmotions contains
Reddit comments, not private journal entries.

---

## 2. Research Question and Objectives

### Research question

> How does the existing pretrained emotion model perform on a frozen
> multi-label GoEmotions-derived test set when identical text entries are
> grouped before splitting and the prediction threshold is tuned using
> validation data only?

### Objectives

1. Audit the previous grouped benchmark methodology.
2. Preserve multi-label annotations wherever possible.
3. Ensure identical text cannot cross generated splits.
4. Keep validation and test roles separate.
5. Select the threshold using validation data only.
6. Evaluate the selected configuration once on a frozen test set.
7. Report imbalance-aware multi-label metrics.
8. Independently verify saved predictions and metrics.
9. Quantify uncertainty with bootstrap intervals.
10. Document remaining leakage risks and domain limitations.

---

## 3. Previous Experiment Findings

The previous grouped benchmark established:

- 53,994 unique exact-text groups from 54,080 processed records.
- Zero exact-text overlap across its generated train, validation, and test
  splits.
- A validation-selected threshold of `0.35`.
- Frozen grouped-test macro-F1 of `0.4284`.

The previous baseline used the original supplied test split and fixed threshold
`0.30`:

| Metric | Previous baseline |
|---|---:|
| Exact-match accuracy | 0.3687 |
| Macro-F1 | 0.4339 |
| Weighted-F1 | 0.4885 |

The prior audit also identified:

- Exact text overlap in the original supplied splits.
- Class imbalance.
- Reddit-to-journal domain mismatch.
- A historical evaluator that selected a ground-truth label using model
  probabilities, which is not an unbiased primary-label evaluation.
- Five duplicate complete rows in the old transformer prediction artifact.

The current experiment does not overwrite or silently alter those historical
artifacts.

### Prior-file verification

Inspected:

- `reports\baseline_emotion_evaluation_audit.md`
- `reports\baseline_emotion_evaluation_results.json`
- `reports\grouped_text_benchmark_report.md`
- `reports\grouped_text_benchmark\results.json`
- `research\evaluation\baseline_emotion_evaluation.py`
- `research\evaluation\grouped_text_benchmark.py`
- `research\evaluation\grouped_text_benchmark_methodology.md`

The requested path
`reports\grouped_text_benchmark\methodology.md` was not present. The actual
methodology file is:

```text
research\evaluation\grouped_text_benchmark_methodology.md
```

---

## 4. Dataset Details

### 4.1 Source files used

The experiment read:

```text
dataset 1\preprocessed\train_ekman.tsv
dataset 1\preprocessed\dev_ekman.tsv
dataset 1\preprocessed\test_ekman.tsv
```

Each is a headerless TSV with:

```text
text<TAB>labels<TAB>id
```

### 4.2 Source sizes

| Source split | Records | Columns |
|---|---:|---:|
| Processed train | 43,227 | 3 |
| Processed dev | 5,426 | 3 |
| Processed test | 5,427 | 3 |
| **Total** | **54,080** | **3** |

The processed training split had already removed 183 duplicate training text
records from the original 43,410-row train source.

### 4.3 Labels

The seven processed labels are:

```text
anger
disgust
fear
joy
sadness
surprise
neutral
```

They are derived from the original GoEmotions 27-emotion-plus-neutral label
space using:

```text
dataset 1\data\ekman_mapping.json
```

The original task is multi-label. A text can contain more than one valid
emotion label.

### 4.4 Generated group counts

| Quantity | Count |
|---|---:|
| Source records read | 54,080 |
| Unique exact-text groups | 53,994 |
| Duplicate records collapsed | 86 |
| Groups with multiple source records | 72 |

For each exact-text group, all valid labels observed in its source rows were
unioned. This avoids discarding annotation information when duplicate text rows
are collapsed.

### 4.5 Class distribution

Counts below are label-presence counts after grouping and label union:

| Label | Train | Validation | Frozen test |
|---|---:|---:|---:|
| anger | 5,607 | 701 | 701 |
| disgust | 811 | 96 | 103 |
| fear | 738 | 95 | 92 |
| joy | 17,235 | 2,169 | 2,161 |
| sadness | 3,210 | 407 | 404 |
| surprise | 5,353 | 640 | 657 |
| neutral | 14,194 | 1,757 | 1,767 |

The dataset is imbalanced. `joy` and `neutral` are frequent; `disgust` and
`fear` are rare. Macro metrics and per-class support are therefore essential.

---

## 5. Experimental Methodology

### 5.1 Grouping

All three processed files were combined before splitting. Exact string equality
on the stored `text` field defined a group.

For every group:

1. One text value was retained.
2. All valid labels were unioned.
3. One source ID was retained for traceability.
4. Source-row count was stored as metadata.

This prevents exact duplicate text from crossing splits. It does not prevent
semantic near-duplicates or paraphrases.

### 5.2 Splitting

Configuration:

| Setting | Value |
|---|---|
| Random seed | 20261027 |
| Requested ratio | 80% train / 10% validation / 10% test |
| Assignment unit | Complete exact-text group |
| Stratification key | First encoded label |
| Sort after assignment | Text, stable sort |

Generated splits:

| Split | Groups | Source records represented |
|---|---:|---:|
| Train | 43,196 | 43,261 |
| Validation | 5,399 | 5,411 |
| Frozen test | 5,399 | 5,408 |

### 5.3 Text isolation

| Split pair | Shared exact texts |
|---|---:|
| Train / validation | 0 |
| Train / test | 0 |
| Validation / test | 0 |

### 5.4 Model inference

The pretrained checkpoint was loaded from the local cache with
`local_files_only=True`.

| Setting | Value |
|---|---|
| Model | `j-hartmann/emotion-english-distilroberta-base` |
| Device | CPU |
| Batch size | 32 |
| Maximum token length | 128 |
| Fine-tuning | None |

The model output mapping was verified from `model.config.id2label` and reordered
to the reference label order before scoring.

### 5.5 Threshold selection

Candidate thresholds:

```text
0.10, 0.11, ..., 0.89, 0.90
```

For every candidate, a label was predicted when probability was at least the
candidate threshold. The threshold with the highest validation macro-F1 was
selected.

Selected threshold:

```text
0.35
```

The test labels were not used for threshold selection.

---

## 6. Evaluation Metrics

The primary task is multi-label classification.

- **Precision:** fraction of predicted positive labels that are correct.
- **Recall:** fraction of reference positive labels detected.
- **F1:** harmonic mean of precision and recall.
- **Macro-F1:** average F1 across labels; treats classes equally.
- **Micro-F1:** aggregates all label decisions; influenced by frequent labels.
- **Weighted-F1:** averages class F1 weighted by support.
- **Exact-match accuracy:** fraction of groups for which every label decision is
  correct.
- **Hamming loss:** fraction of individual label decisions that are incorrect.

No single-label confusion matrix is used as the primary result because the
source task is multi-label.

---

## 7. Validation Results

At the selected threshold `0.35`:

| Metric | Validation |
|---|---:|
| Exact-match accuracy | 0.4025 |
| Hamming loss | 0.1601 |
| Micro precision | 0.4838 |
| Micro recall | 0.4706 |
| Micro-F1 | 0.4771 |
| Macro precision | 0.4585 |
| Macro recall | 0.5524 |
| Macro-F1 | 0.4346 |
| Weighted-F1 | 0.4901 |

This validation result was used to choose the threshold and is not the final
held-out estimate.

---

## 8. Frozen Test Results

The selected threshold was applied once to the frozen test predictions.

| Metric | Frozen test |
|---|---:|
| Exact-match accuracy | 0.3988 |
| Hamming loss | 0.1614 |
| Micro precision | 0.4812 |
| Micro recall | 0.4654 |
| Micro-F1 | 0.4732 |
| Macro precision | 0.4503 |
| Macro recall | 0.5338 |
| Macro-F1 | 0.4284 |
| Weighted-F1 | 0.4908 |

### 8.1 Per-class test results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| anger | 0.4374 | 0.3538 | 0.3912 | 701 |
| disgust | 0.1282 | 0.8641 | 0.2233 | 103 |
| fear | 0.3333 | 0.6739 | 0.4460 | 92 |
| joy | 0.9026 | 0.3901 | 0.5447 | 2,161 |
| sadness | 0.4986 | 0.4554 | 0.4761 | 404 |
| surprise | 0.3873 | 0.4079 | 0.3973 | 657 |
| neutral | 0.4644 | 0.5914 | 0.5203 | 1,767 |

### 8.2 Plain-English interpretation

- The model is conservative when predicting `joy`: high precision, lower
  recall.
- `disgust` is detected often but over-predicted: high recall, very low
  precision.
- `neutral` has the highest F1 among the classes.
- Minority classes have wider practical uncertainty because their support is
  small.
- Exact-match accuracy is stricter than label-level F1 because every label in a
  group must be correct.

---

## 9. Bootstrap Confidence Intervals

Bootstrap method:

- Unit: frozen test text group.
- Resampling: independent sampling with replacement.
- Replicates: 2,000.
- Seed: 424242.
- Interval: percentile 95% interval.

| Metric | Estimate | Lower 95% | Upper 95% |
|---|---:|---:|---:|
| Macro-F1 | 0.4284 | 0.4120 | 0.4447 |
| Micro-F1 | 0.4732 | 0.4612 | 0.4846 |
| Weighted-F1 | 0.4908 | 0.4781 | 0.5030 |
| Exact-match accuracy | 0.3988 | 0.3858 | 0.4108 |

These intervals quantify sampling variability within this grouped Reddit test
population. They do not quantify uncertainty from model-version changes,
dataset construction choices, annotation quality, or journal-domain shift.

---

## 10. Error Analysis

At threshold `0.35`:

- Exact-match correct groups: **2,153**
- Groups with at least one label error: **3,246**

| Class | Support | True positive | False positive | False negative | F1 |
|---|---:|---:|---:|---:|---:|
| anger | 701 | 248 | 319 | 453 | 0.3912 |
| disgust | 103 | 89 | 605 | 14 | 0.2233 |
| fear | 92 | 62 | 124 | 30 | 0.4460 |
| joy | 2,161 | 843 | 91 | 1,318 | 0.5447 |
| sadness | 404 | 184 | 185 | 220 | 0.4761 |
| surprise | 657 | 268 | 424 | 389 | 0.3973 |
| neutral | 1,767 | 1,045 | 1,205 | 722 | 0.5203 |

The largest observed error patterns are:

1. False-negative `joy` decisions, consistent with the model's high joy
   precision and low joy recall.
2. False-positive `disgust` decisions, consistent with its low precision.
3. False-negative `anger` decisions.
4. False-positive and false-negative `neutral` decisions due to its broad and
   frequent class.

This is aggregate label-level analysis. It does not claim a linguistic cause
for individual examples. A future qualitative study should sample errors and
classify causes such as sarcasm, ambiguity, mixed emotion, short context, and
annotation disagreement.

The full aggregate analysis is saved in:

```text
reports\research_experiment_error_analysis.md
```

---

## 11. Threshold Sensitivity on the Same Frozen Test

The `0.30` row is included only as a sensitivity analysis. It was not selected
using test labels.

| Threshold | Macro-F1 | Micro-F1 | Weighted-F1 | Exact-match | Hamming loss |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.4318 | 0.4807 | 0.5031 | 0.3836 | 0.1662 |
| 0.35 selected | 0.4284 | 0.4732 | 0.4908 | 0.3988 | 0.1614 |

The selected threshold improves exact-match accuracy and Hamming loss on this
test, while the unselected `0.30` threshold has higher F1 values on this test.
This illustrates why threshold selection must use validation data rather than
test performance.

---

## 12. Comparison With Previous Baseline

| Experiment | Test population | Threshold | Macro-F1 | Weighted-F1 | Exact-match |
|---|---|---:|---:|---:|---:|
| Previous baseline | Original supplied test | 0.30 fixed | 0.4339 | 0.4885 | 0.3687 |
| Grouped benchmark | New grouped frozen test | 0.35 validation-selected | 0.4284 | 0.4908 | 0.3988 |

These are not a valid causal improvement comparison because:

- The test populations differ.
- Duplicate-text grouping and label union differ.
- The threshold procedure differs.
- The new benchmark has stronger exact-text isolation.

The defensible conclusion is:

> The new experiment improves evaluation control, not necessarily model
> performance. It does not demonstrate that the model improved or degraded.

The same grouped frozen test at thresholds `0.30` and `0.35` is a valid
threshold-sensitivity comparison, but it does not replace a model comparison.

---

## 13. Independent Verification

The verifier:

```text
research\evaluation\research_experiment_verifier.py
```

was run with:

```powershell
python research\evaluation\research_experiment_verifier.py
```

It independently verified:

- Train/validation overlap: 0.
- Train/test overlap: 0.
- Validation/test overlap: 0.
- Validation split rows: 5,399.
- Validation prediction rows: 5,399.
- Test split rows: 5,399.
- Test prediction rows: 5,399.
- Probability values are finite and within `[0, 1]`.
- Recomputed metrics match the saved grouped benchmark results.
- Test threshold selection flag is false.
- Bootstrap result generation completed.

### Alignment limitation

The saved probability TSV files contain only probability columns and no text or
ID column. Alignment is therefore verified by exact row count and row order,
matching the inference script's deterministic write order. Future experiments
should include `text_hash` and `id` columns in prediction artifacts to make
alignment independently auditable without relying on row order.

### Independent results file

```text
reports\research_experiment_results.json
```

---

## 14. Computational Requirements

The experiment ran on CPU using the locally cached model.

Measured inference:

- Validation: 164.020943 seconds.
- Test: 105.804795 seconds.
- Total inference: approximately 269.83 seconds.

The verification and bootstrap analysis ran locally without model inference.
No additional packages were installed. The environment already provided
PyTorch, Transformers, pandas, NumPy, scikit-learn, and SciPy.

The experiment is practical for offline research, but repeated CPU inference
is not an appropriate assumption for interactive application latency.

---

## 15. Limitations and Threats to Validity

1. **Domain mismatch:** GoEmotions is Reddit-derived, not journal-derived.
2. **Exact-only grouping:** Semantic near-duplicates remain possible.
3. **Label-union effect:** Duplicate groups are represented by the union of
   their labels.
4. **Class imbalance:** Rare labels have limited support.
5. **Model revision:** The exact immutable Hugging Face model revision is not
   recorded locally beyond the model name.
6. **No retraining:** This experiment evaluates the existing pretrained model;
   it does not test whether grouped training improves performance.
7. **No confidence intervals for all design choices:** Bootstrap intervals
   cover sample resampling, not all sources of uncertainty.
8. **Prediction alignment:** Existing prediction files lack text/ID keys.
9. **No clinical evidence:** Classification metrics do not establish diagnosis,
   treatment value, or mental-health benefit.
10. **No emotional-consistency validation:** Longitudinal consistency remains
    untested on real users.
11. **Potential annotation disagreement:** Unioning duplicate labels can
    preserve disagreements rather than resolve them.
12. **Frozen-test governance:** The generated test must not be repeatedly used
    for future tuning.

---

## 16. Research Gap Supported by the Evidence

The repository evidence supports the following narrowly stated research gap:

> The current system has benchmark-domain evaluation on Reddit-derived
> multi-label emotion data, but it does not yet have a leakage-controlled,
> journal-domain evaluation with real consented journal entries, nor a
> longitudinal evaluation of whether personalization improves user-specific
> emotion agreement over time.

This is a repository-grounded gap, not a claim about the complete published
literature. A broader literature-supported gap requires a separate systematic
literature review.

The current experiment also shows that evaluation design matters: threshold
choice, duplicate-text handling, and multi-label reference construction affect
reported results.

---

## 17. Conclusions

The complete controlled experiment ran successfully.

The main verified result is:

```text
Grouped frozen-test macro-F1: 0.4284
Grouped frozen-test micro-F1: 0.4732
Grouped frozen-test weighted-F1: 0.4908
Grouped frozen-test exact-match accuracy: 0.3988
Selected validation threshold: 0.35
```

The generated train, validation, and test splits have zero exact-text overlap.
Threshold selection was performed using validation data only. The frozen test
was then evaluated once, and the results were independently recomputed from
saved artifacts.

The results do not prove model improvement over the previous baseline because
the populations and protocols differ. They do establish a better-controlled
benchmark for future model comparisons.

The model appears strongest for precise `joy` predictions and weakest for
`disgust` precision and `joy` recall. Class imbalance and domain mismatch limit
the conclusions.

---

## 18. Recommended Next Steps

1. Keep the generated validation and test files frozen.
2. Record an immutable Hugging Face model revision or commit hash.
3. Add text hashes and IDs to future prediction artifacts.
4. Train a separate experimental model using only the grouped training split.
5. Tune model and threshold choices only on validation.
6. Compare the experimental model and current checkpoint on the same frozen
   grouped test set.
7. Use bootstrap confidence intervals for both models.
8. Investigate duplicate-text label disagreement and semantic near-duplicates.
9. Create a consented, de-identified journal-domain evaluation set.
10. Evaluate personalization using real longitudinal feedback where ethically
    and legally appropriate.
11. Do not compare VADER directly with seven-label emotion results unless a
    shared, pre-specified label mapping is created.

---

## 19. Complete Artifact List

### New research scripts

```text
research\evaluation\research_experiment_verifier.py
```

The grouped benchmark script used:

```text
research\evaluation\grouped_text_benchmark.py
```

### New reports and results

```text
reports\research_experiment_final_report.md
reports\research_experiment_methodology.md
reports\research_experiment_results.json
reports\research_experiment_comparison.csv
reports\research_experiment_error_analysis.md
```

### Existing grouped benchmark artifacts used without overwriting

```text
reports\grouped_text_benchmark\results.json
reports\grouped_text_benchmark\splits\train.tsv
reports\grouped_text_benchmark\splits\validation.tsv
reports\grouped_text_benchmark\splits\test.tsv
reports\grouped_text_benchmark\validation_predictions.tsv
reports\grouped_text_benchmark\test_predictions.tsv
```

No production application file, original dataset, database, or model file was
modified by this experiment.

---

## 20. Final Reproduction Checklist

- [x] Previous reports and scripts inspected.
- [x] Existing grouped benchmark methodology audited.
- [x] Exact-text grouping performed before split assignment.
- [x] Multi-label annotations preserved through label union.
- [x] Train, validation, and test groups saved.
- [x] Zero exact-text overlap independently verified.
- [x] Pretrained model output labels verified.
- [x] Validation-only threshold tuning completed.
- [x] Frozen test evaluated after selection.
- [x] Macro, micro, weighted, per-class, exact-match, and Hamming metrics
  calculated.
- [x] Bootstrap confidence intervals calculated.
- [x] Aggregate error analysis completed.
- [x] Saved predictions independently shape-checked and range-checked.
- [x] Important metrics independently recomputed.
- [x] Original application and datasets preserved.
- [x] Limitations and unresolved risks documented.
- [x] Complete report saved in `reports\`.

### Commands

```powershell
python research\evaluation\grouped_text_benchmark.py
python research\evaluation\research_experiment_verifier.py
```
