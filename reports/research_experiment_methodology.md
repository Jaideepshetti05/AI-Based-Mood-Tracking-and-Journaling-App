# Research Experiment Methodology

## Experiment

Leakage-controlled grouped-text benchmark for the existing pretrained
seven-label emotion classifier.

## Inputs

```text
dataset 1\preprocessed\train_ekman.tsv
dataset 1\preprocessed\dev_ekman.tsv
dataset 1\preprocessed\test_ekman.tsv
```

The source task is multi-label with labels:

```text
anger, disgust, fear, joy, sadness, surprise, neutral
```

## Grouping

All source rows were combined and grouped by exact text before any split was
created. Identical-text group labels were combined using set union. This
produced 53,994 groups from 54,080 source rows and collapsed 86 duplicate
records.

## Splitting

- Seed: `20261027`
- Requested ratio: 80% train, 10% validation, 10% test
- Assignment: deterministic shuffled assignment within first-label strata
- Unit of assignment: complete exact-text group
- Text overlap across generated splits: zero

The saved split files are under:

```text
reports\grouped_text_benchmark\splits\
```

## Model

```text
j-hartmann/emotion-english-distilroberta-base
```

The model was loaded from the local Hugging Face cache with
`local_files_only=True`. It was not retrained or fine-tuned. Inference used
CPU, batch size 32, and maximum token length 128.

## Threshold tuning

Threshold candidates were `0.10` through `0.90` in steps of `0.01`. A label
was predicted when its probability was at least the candidate threshold.

The threshold maximizing validation macro-F1 was selected. The selected
threshold was `0.35`. The test labels were not used during selection.

## Metrics

Primary multi-label metrics:

- Macro precision, recall, and F1.
- Micro precision, recall, and F1.
- Weighted-F1.
- Exact-match accuracy.
- Hamming loss.
- Per-class support and metrics.

Bootstrap intervals use 2,000 iid sample-level resamples of the frozen test
groups, percentile intervals, and seed `424242`. These intervals quantify
sample uncertainty under the grouped test population; they do not account for
dataset construction, model revision, or domain-shift uncertainty.

## Verification

`research\evaluation\research_experiment_verifier.py` independently:

- Reads the saved split and prediction files.
- Verifies row counts and prediction shape.
- Verifies probabilities are finite and in `[0, 1]`.
- Verifies zero exact-text overlap.
- Recomputes validation and test metrics.
- Computes same-test threshold sensitivity.
- Computes bootstrap confidence intervals.
- Produces aggregate per-class error counts.

## Reproduction

```powershell
python research\evaluation\grouped_text_benchmark.py
python research\evaluation\research_experiment_verifier.py
```

The first command performs inference and creates grouped artifacts. The second
command independently verifies the saved artifacts without rerunning inference.
