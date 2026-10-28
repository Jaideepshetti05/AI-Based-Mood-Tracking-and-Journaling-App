# Leakage-Controlled Grouped-Text Emotion Benchmark

## Personalized AI-Based Mood Tracking and Journaling Application Using NLP

**Report date:** 27 October 2026  
**Experiment status:** Completed successfully  
**Application changes:** None  
**Original dataset changes:** None  
**Transformer training or fine-tuning:** None

---

## 1. Purpose and Research Question

This experiment establishes a more scientifically controlled benchmark for the
existing emotion-analysis system before any model improvement is attempted.

The research question is:

> How does the existing pretrained emotion model perform on a frozen
> multi-label GoEmotions-derived test set when identical text entries are
> grouped before splitting, and the prediction threshold is selected using
> validation data only?

This experiment is separate from the original GoEmotions train/development/test
split and from the previous fixed-threshold baseline. It is intended to reduce
exact-text leakage and prevent test-set threshold selection.

---

## 2. Reproduction Command

Run this command from the project root:

```powershell
python research\evaluation\grouped_text_benchmark.py
```

The experiment generated:

```text
reports\grouped_text_benchmark_report.md
reports\grouped_text_benchmark\results.json
reports\grouped_text_benchmark\splits\train.tsv
reports\grouped_text_benchmark\splits\validation.tsv
reports\grouped_text_benchmark\splits\test.tsv
reports\grouped_text_benchmark\validation_predictions.tsv
reports\grouped_text_benchmark\test_predictions.tsv
```

The evaluator used the locally cached Hugging Face model. No package
installation or model download was required.

---

## 3. Existing Dataset and Model Context

### 3.1 Source files

The grouped experiment read the existing processed Ekman files:

```text
dataset 1\preprocessed\train_ekman.tsv
dataset 1\preprocessed\dev_ekman.tsv
dataset 1\preprocessed\test_ekman.tsv
```

Each file is a headerless TSV with three fields:

```text
text<TAB>labels<TAB>id
```

### 3.2 Source split sizes

| Original split | Records | Columns |
|---|---:|---:|
| Training | 43,227 | 3 |
| Development | 5,426 | 3 |
| Test | 5,427 | 3 |
| **Total** | **54,080** | **3** |

These are the processed Ekman files, not the original 54,263-row source split.
The training file had already been deduplicated by the existing preprocessing
pipeline.

### 3.3 Reference labels

The seven reference labels are:

```text
anger
disgust
fear
joy
sadness
surprise
neutral
```

These labels are produced from the original GoEmotions categories using
`dataset 1\data\ekman_mapping.json`. The source task remains multi-label:
one text may have multiple valid labels.

### 3.4 Existing application model

The evaluated model is:

```text
j-hartmann/emotion-english-distilroberta-base
```

The project loads this model as a pretrained Hugging Face text-classification
checkpoint. It does not fine-tune or retrain the model in this experiment.

The model's verified output mapping is:

| Model index | Label |
|---:|---|
| 0 | anger |
| 1 | disgust |
| 2 | fear |
| 3 | joy |
| 4 | neutral |
| 5 | sadness |
| 6 | surprise |

The evaluator reorders the model probabilities into the reference order:

```text
anger, disgust, fear, joy, sadness, surprise, neutral
```

---

## 4. Grouping and Leakage-Controlled Splitting

### 4.1 Grouping procedure

All records from the three processed files were combined before splitting.
Rows were grouped by the exact stored `text` value.

For each identical-text group:

1. One text value was retained.
2. All valid labels observed in the group were combined using set union.
3. One source ID was retained only as a trace identifier.
4. The number of source records represented by the group was recorded.

This preserves all observed labels for identical text while ensuring that an
exact text value cannot occur in more than one generated split.

### 4.2 Grouping results

| Quantity | Count |
|---|---:|
| Source records read | 54,080 |
| Unique text groups | 53,994 |
| Duplicate records collapsed | 86 |
| Groups with multiple source records | 72 |

The experiment removes exact-text overlap only. It does not detect semantic
near-duplicates, paraphrases, shared topics, or related Reddit discussions.

### 4.3 Split procedure

The generated split uses:

- Random seed: **20261027**
- Requested split ratio: **80% train / 10% validation / 10% test**
- Stratification key: deterministic first encoded label
- Assignment unit: complete text group
- Group shuffling: seeded and reproducible

Generated split sizes:

| Generated split | Text groups | Source records represented |
|---|---:|---:|
| Train | 43,196 | 43,261 |
| Validation | 5,399 | 5,411 |
| Frozen test | 5,399 | 5,408 |

The source-record totals are slightly larger than the group counts because some
identical-text groups represented multiple original rows.

### 4.4 Exact-text isolation verification

| Split pair | Shared exact texts |
|---|---:|
| Train / validation | 0 |
| Train / test | 0 |
| Validation / test | 0 |

This verifies the primary objective of the experiment: identical text entries
cannot appear across different generated splits.

---

## 5. Class Distribution and Imbalance

The label counts below count label presence after identical-text grouping and
label union.

| Label | Train | Validation | Frozen test |
|---|---:|---:|---:|
| anger | 5,607 | 701 | 701 |
| disgust | 811 | 96 | 103 |
| fear | 738 | 95 | 92 |
| joy | 17,235 | 2,169 | 2,161 |
| sadness | 3,210 | 407 | 404 |
| surprise | 5,353 | 640 | 657 |
| neutral | 14,194 | 1,757 | 1,767 |

The dataset is imbalanced:

- `joy` and `neutral` are the most frequent classes.
- `disgust` and `fear` are the least frequent classes.
- Macro metrics are therefore more informative than accuracy alone.
- Per-class support must be reported with every result.

---

## 6. Inference Configuration

| Setting | Value |
|---|---|
| Model | `j-hartmann/emotion-english-distilroberta-base` |
| Device | CPU |
| Batch size | 32 |
| Maximum token length | 128 |
| Random seed | 20261027 |
| Validation samples | 5,399 |
| Test samples | 5,399 |

Measured local inference time:

| Split | Total time | Milliseconds per sample |
|---|---:|---:|
| Validation | 164.020943 seconds | 30.379875 ms |
| Test | 105.804795 seconds | 19.597110 ms |

These are local CPU timings and are not production latency guarantees.

---

## 7. Threshold Tuning Methodology

The model produces a probability for each of the seven emotion labels. A label
is predicted when its probability is greater than or equal to the selected
threshold.

Candidate thresholds:

```text
0.10, 0.11, 0.12, ..., 0.89, 0.90
```

Selection rule:

```text
Select the threshold with the highest validation macro-F1.
```

The frozen test labels were not used during threshold selection.

### Selected threshold

```text
Selected threshold: 0.35
Validation macro-F1: 0.4346
```

The selected threshold was applied once to the frozen test predictions.

---

## 8. Validation Results

At the selected threshold of `0.35`, validation performance was:

| Metric | Validation result |
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

Validation results were used for threshold selection and are not the final
held-out result.

---

## 9. Frozen Test Results

The following results were calculated once on the frozen grouped test set using
the validation-selected threshold `0.35`.

| Metric | Frozen test result |
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

### 9.1 Per-class frozen test metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| anger | 0.4374 | 0.3538 | 0.3912 | 701 |
| disgust | 0.1282 | 0.8641 | 0.2233 | 103 |
| fear | 0.3333 | 0.6739 | 0.4460 | 92 |
| joy | 0.9026 | 0.3901 | 0.5447 | 2,161 |
| sadness | 0.4986 | 0.4554 | 0.4761 | 404 |
| surprise | 0.3873 | 0.4079 | 0.3973 | 657 |
| neutral | 0.4644 | 0.5914 | 0.5203 | 1,767 |

### 9.2 Interpretation

- The model has high precision for `joy` but relatively low recall, meaning it
  predicts joy conservatively.
- `disgust` has high recall but very low precision, indicating many false
  positive disgust predictions.
- `neutral` has the highest per-class F1 at 0.5203.
- Macro-F1 is lower than weighted-F1 because minority classes perform worse.
- Exact-match accuracy is lower than ordinary single-label accuracy because all
  labels for a group must be correct simultaneously.

---

## 10. Comparison With the Previous Baseline

The previous baseline used:

- The original supplied test split.
- A fixed threshold of `0.30`.
- No cross-split exact-text regrouping.

Previous baseline results:

| Metric | Previous baseline |
|---|---:|
| Exact-match accuracy | 0.3687 |
| Macro-F1 | 0.4339 |
| Weighted-F1 | 0.4885 |

Grouped benchmark results:

| Metric | Grouped benchmark |
|---|---:|
| Exact-match accuracy | 0.3988 |
| Macro-F1 | 0.4284 |
| Weighted-F1 | 0.4908 |

These results must not be presented as evidence that the model improved or
declined. The experiments use different test populations, different threshold
selection procedures, and different duplicate-text handling:

- Previous test: original supplied test split, fixed `0.30` threshold.
- New test: grouped text split, validation-selected `0.35` threshold, label
  union for duplicate-text groups.

The scientifically defensible conclusion is:

> The new benchmark provides stronger exact-text isolation. It is a better
> controlled evaluation design, but it does not establish model improvement.

---

## 11. Why This Is a Multi-Label Evaluation

The original records may contain multiple emotion labels. Therefore, the
primary evaluation reports:

- Macro precision, recall, and F1.
- Micro precision, recall, and F1.
- Weighted-F1.
- Exact-match accuracy.
- Hamming loss.
- Per-class metrics and support.

No conventional single-label confusion matrix is used as the primary result.
Forcing one label from a multi-label reference would discard information.

The generated results file contains no model-informed rewriting of ground
truth. The evaluator reads reference labels directly, converts them to binary
multi-label vectors, and applies the threshold only to model probabilities.

---

## 12. Leakage and Validity Assessment

### Verified improvements in experimental control

- Exact duplicate text cannot cross train, validation, or test.
- Threshold selection uses validation only.
- Test predictions are evaluated after threshold selection.
- Original datasets and application code are unchanged.
- The model is not retrained on test data.
- The random seed and split files are saved.

### Remaining risks

1. Semantic near-duplicates and paraphrases may remain.
2. Related Reddit discussions may share vocabulary or context.
3. Label union changes duplicate-group representation.
4. The model revision is recorded by name but not by an immutable commit hash.
5. The benchmark is based on Reddit comments, not private journals.
6. Class imbalance affects minority-class reliability.
7. No confidence intervals were calculated in this first run.
8. Repeated decisions based on the frozen test set would create test-set
   overfitting.

### Suitability

The generated test set is suitable for a controlled benchmark of the existing
checkpoint under this protocol. It is not evidence of:

- Clinical validity.
- Mental-health diagnosis.
- Reliability on personal journal entries.
- Emotional consistency in real users.
- Generalization to all populations or writing styles.

---

## 13. Files Created

### Experimental script

```text
research\evaluation\grouped_text_benchmark.py
```

Responsibilities:

- Load existing processed files.
- Group identical text.
- Union labels.
- Create seeded train/validation/test groups.
- Verify split isolation.
- Run the cached pretrained transformer.
- Tune threshold on validation.
- Evaluate frozen test.
- Save JSON results and prediction files.

### Methodology

```text
research\evaluation\grouped_text_benchmark_methodology.md
```

### Machine-readable results

```text
reports\grouped_text_benchmark\results.json
```

### Generated split and prediction artifacts

```text
reports\grouped_text_benchmark\splits\train.tsv
reports\grouped_text_benchmark\splits\validation.tsv
reports\grouped_text_benchmark\splits\test.tsv
reports\grouped_text_benchmark\validation_predictions.tsv
reports\grouped_text_benchmark\test_predictions.tsv
```

---

## 14. Conclusions

The grouped-text experiment successfully establishes a more defensible baseline
for the existing pretrained emotion model.

The key final result is:

```text
Frozen grouped-test macro-F1: 0.4284
Frozen grouped-test micro-F1: 0.4732
Frozen grouped-test weighted-F1: 0.4908
Selected threshold: 0.35
```

The benchmark has zero exact-text overlap across its generated train,
validation, and test splits. This improves the experimental design compared
with the original split, which had observed exact-text overlap.

The result should be treated as a benchmark-domain measurement, not a claim of
real-world journal reliability. The next research experiment should train or
calibrate a model using only the grouped training split, keep validation and
test frozen, and report bootstrap confidence intervals. A separate
consented, de-identified journal-domain dataset should then be used to measure
domain transfer.

---

## 15. Final Verification Checklist

- [x] Existing evaluation script and audit report inspected.
- [x] Original dataset files preserved.
- [x] Production application preserved.
- [x] No model retraining or fine-tuning performed.
- [x] Identical text grouped before splitting.
- [x] Labels unioned within duplicate-text groups.
- [x] Seed recorded.
- [x] Train, validation, and test split files saved.
- [x] Exact-text overlap verified as zero across generated splits.
- [x] Class distributions recorded.
- [x] Threshold tuned using validation only.
- [x] Frozen test evaluated after threshold selection.
- [x] Macro, micro, weighted, per-class, exact-match, and Hamming metrics
  reported.
- [x] Results saved as JSON.
- [x] Methodology saved separately.
- [x] Complete report saved in the project `reports\` folder.

