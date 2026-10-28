# Grouped-Text Benchmark Methodology

## Purpose

This experiment creates a leakage-controlled benchmark for the existing
pretrained emotion model used by the project. It is separate from the original
GoEmotions split and from the previous baseline evaluation.

The experiment does not retrain or fine-tune the transformer. The generated
training split is retained for a future training experiment, while the current
experiment evaluates the existing pretrained checkpoint on newly created
validation and test groups.

## Reproducibility

Run from the repository root:

```powershell
python research\evaluation\grouped_text_benchmark.py
```

The exact run used:

```text
Seed: 20261027
Model: j-hartmann/emotion-english-distilroberta-base
Device: CPU
Batch size: 32
Maximum token length: 128
Threshold candidates: 0.10 through 0.90 in increments of 0.01
Threshold selection metric: validation macro-F1
```

The local Hugging Face model cache was used with `local_files_only=True`.
No model download or dependency installation was performed.

## Source data

The experiment reads only these existing processed files:

```text
dataset 1\preprocessed\train_ekman.tsv
dataset 1\preprocessed\dev_ekman.tsv
dataset 1\preprocessed\test_ekman.tsv
```

Each source row has:

```text
text<TAB>labels<TAB>id
```

The seven reference labels are:

```text
anger, disgust, fear, joy, sadness, surprise, neutral
```

The original source files and existing evaluation artifacts are not overwritten.

## Grouping identical text

All 54,080 rows from the three existing processed Ekman files are combined.
Rows are grouped by the exact stored `text` value before splitting.

For each text group:

1. The group is represented by one text value.
2. All valid labels observed in the group are unioned.
3. The first source ID is retained only as a trace identifier.
4. The number of source rows represented by the group is retained as metadata.

This label-union rule preserves observed annotations as far as possible without
allowing the same exact text to occur in multiple splits.

Observed grouping results:

| Quantity | Value |
|---|---:|
| Source rows | 54,080 |
| Unique text groups | 53,994 |
| Duplicate rows collapsed | 86 |
| Groups with multiple source rows | 72 |

This experiment only removes exact-text overlap. It does not remove semantic
near-duplicates, paraphrases, or related Reddit discussions.

## Split construction

Groups are assigned to approximately 80% train, 10% validation, and 10% test.
Assignment is deterministic:

1. Groups are bucketed by their first encoded label.
2. Each bucket is shuffled with seed `20261027`.
3. Groups are assigned within each bucket to train, validation, and test.
4. A group is indivisible and can belong to only one split.
5. Each split is sorted by text before being written.

Generated group counts:

| Split | Text groups | Source rows represented |
|---|---:|---:|
| Train | 43,196 | 43,261 |
| Validation | 5,399 | 5,411 |
| Test | 5,399 | 5,408 |

The generated files are:

```text
reports\grouped_text_benchmark\splits\train.tsv
reports\grouped_text_benchmark\splits\validation.tsv
reports\grouped_text_benchmark\splits\test.tsv
```

Verified exact-text overlap:

| Split pair | Shared texts |
|---|---:|
| Train / validation | 0 |
| Train / test | 0 |
| Validation / test | 0 |

## Model inference

The existing pretrained checkpoint is loaded without fine-tuning:

```text
j-hartmann/emotion-english-distilroberta-base
```

The model's actual `id2label` mapping was verified:

```text
0 anger
1 disgust
2 fear
3 joy
4 neutral
5 sadness
6 surprise
```

The evaluator reorders model probabilities to the reference label order:

```text
anger, disgust, fear, joy, sadness, surprise, neutral
```

The model was run on:

- 5,399 validation groups
- 5,399 frozen test groups

Inference time:

- Validation: 164.020943 seconds
- Test: 105.804795 seconds

These are local CPU timings and should not be treated as production latency
measurements.

## Threshold tuning

The evaluator tests thresholds from `0.10` through `0.90` in steps of `0.01`.
For each threshold, a label is predicted when its probability is at least the
threshold.

The threshold is selected only from validation macro-F1:

```text
Selected threshold: 0.35
Validation macro-F1: 0.4346215378
```

The test labels are not used to select the threshold. The selected threshold is
then applied once to the frozen test probabilities.

## Metrics

The primary task is multi-label classification. The reported metrics are:

- Macro precision
- Macro recall
- Macro-F1
- Micro precision
- Micro recall
- Micro-F1
- Weighted-F1
- Exact-match accuracy
- Hamming loss
- Per-class precision, recall, F1, and support

Exact-match accuracy requires every label for a text group to be correct.
Hamming loss measures individual label decisions and is more forgiving of
partially correct multi-label predictions.

No single-label confusion matrix is used as the primary result because the
source task is multi-label.

## Frozen test results

The selected validation threshold was `0.35`.

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

### Frozen test per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| anger | 0.4374 | 0.3538 | 0.3912 | 701 |
| disgust | 0.1282 | 0.8641 | 0.2233 | 103 |
| fear | 0.3333 | 0.6739 | 0.4460 | 92 |
| joy | 0.9026 | 0.3901 | 0.5447 | 2,161 |
| sadness | 0.4986 | 0.4554 | 0.4761 | 404 |
| surprise | 0.3873 | 0.4079 | 0.3973 | 657 |
| neutral | 0.4644 | 0.5914 | 0.5203 | 1,767 |

## Comparison with previous baseline

The previous baseline used the supplied original test split and a fixed
threshold of `0.30`:

```text
Previous test macro-F1: 0.4339
Previous test weighted-F1: 0.4885
Previous test exact-match accuracy: 0.3687
```

The grouped-text benchmark produced:

```text
Grouped test macro-F1: 0.4284
Grouped test weighted-F1: 0.4908
Grouped test exact-match accuracy: 0.3988
```

These numbers must not be presented as a causal improvement or degradation.
The test populations differ, the grouped benchmark uses a validation-selected
threshold of `0.35`, and the grouped procedure unions annotations for
duplicate-text groups. The correct conclusion is that the new benchmark is
better controlled for exact-text leakage, not that the model improved.

## Remaining limitations

1. Only exact text is grouped. Semantic or paraphrase leakage remains possible.
2. Label union changes duplicate-group representation from multiple rows to one
   multi-label group.
3. The pretrained model was not trained on the newly generated training split.
4. The benchmark remains Reddit-domain evaluation, not journal-domain
   evaluation.
5. The source data is class-imbalanced.
6. The model revision is not recorded beyond its Hugging Face model name.
7. Threshold selection is global, not class-specific.
8. No confidence intervals were calculated in this first controlled run.
9. The generated test is frozen by the saved files and seed, but should not be
   repeatedly used for future threshold or model decisions.
10. The unrelated worktree currently contains a pre-existing modification to
    `dataset 1\data\full_dataset\goemotions_1.csv`; this experiment did not
    write to that file.

## Recommended next experiment

Use the saved grouped training split for a separately tracked model-training
experiment, while keeping the generated validation and test files frozen.
Tune any thresholds or calibration only on validation, evaluate once on the
frozen test, and report bootstrap confidence intervals. After that, obtain a
small consented and de-identified journal-domain evaluation set to measure
domain transfer separately from GoEmotions benchmark performance.
