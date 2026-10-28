# Grouped Benchmark Error Analysis

## Scope

This analysis uses the grouped benchmark frozen test predictions at the
validation-selected threshold of `0.35`.

The task is multi-label. Counts below are per-label decisions, so one test
group can contribute multiple false positives or false negatives.

## Overall error profile

| Quantity | Count |
|---|---:|
| Frozen test groups | 5,399 |
| Exact-match correct groups | 2,153 |
| Groups with at least one label error | 3,246 |
| Exact-match accuracy | 0.3988 |
| Hamming loss | 0.1614 |

## Per-class errors

| Class | Support | True positive | False positive | False negative | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| anger | 701 | 248 | 319 | 453 | 0.4374 | 0.3538 | 0.3912 |
| disgust | 103 | 89 | 605 | 14 | 0.1282 | 0.8641 | 0.2233 |
| fear | 92 | 62 | 124 | 30 | 0.3333 | 0.6739 | 0.4460 |
| joy | 2,161 | 843 | 91 | 1,318 | 0.9026 | 0.3901 | 0.5447 |
| sadness | 404 | 184 | 185 | 220 | 0.4986 | 0.4554 | 0.4761 |
| surprise | 657 | 268 | 424 | 389 | 0.3873 | 0.4079 | 0.3973 |
| neutral | 1,767 | 1,045 | 1,205 | 722 | 0.4644 | 0.5914 | 0.5203 |

## Interpretation

### Strongest behavior

- `joy` has the highest precision at **0.9026**. When the model predicts joy,
  it is usually correct.
- `neutral` has the highest F1 at **0.5203** among the seven classes.
- `disgust` and `fear` have high recall, so the model detects many true
  examples of these labels.

### Weakest behavior

- `disgust` has very low precision (**0.1282**) despite high recall
  (**0.8641**). The model predicts disgust too often relative to its support.
- `joy` has low recall (**0.3901**) despite high precision. The threshold
  suppresses many true joy labels.
- `anger` and `surprise` have substantial false-negative counts.
- `neutral` has many false positives because it is a frequent class and its
  language is broad.

### Threshold sensitivity on the same frozen test

The threshold of `0.30` was not selected for this test; it is included only as
a sensitivity comparison. The threshold of `0.35` was selected from validation
macro-F1.

| Threshold | Macro-F1 | Micro-F1 | Weighted-F1 | Exact-match | Hamming loss |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.4318 | 0.4807 | 0.5031 | 0.3836 | 0.1662 |
| 0.35 | 0.4284 | 0.4732 | 0.4908 | 0.3988 | 0.1614 |

The selected threshold improves exact-match accuracy and Hamming loss on this
test, but the fixed 0.30 threshold has higher macro-F1, micro-F1, and
weighted-F1 on this same test. This is expected threshold variance and is not
evidence that the test set was used for tuning.

## Error-analysis limitations

- This report intentionally does not reproduce Reddit text entries.
- Semantic near-duplicate leakage is not measured.
- The error categories are label-level, not linguistic or topic-level.
- A future qualitative analysis should sample errors using a documented,
  privacy-conscious protocol and classify causes such as sarcasm, ambiguity,
  mixed emotion, short context, and annotation disagreement.
