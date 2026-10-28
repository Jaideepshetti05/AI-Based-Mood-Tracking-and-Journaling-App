# Baseline Emotion Evaluation Audit

## Personalized AI-Based Mood Tracking and Journaling Application Using NLP

**Audit date:** 27 October 2026  
**Evaluation scope:** Existing repository, dataset artifacts, preprocessing
scripts, backend model integration, and existing held-out predictions  
**Application changes:** None  
**Dataset changes:** None  
**Model training/fine-tuning:** None

---

## 1. Executive Summary

This audit establishes a reproducible baseline for the existing emotion-analysis
implementation. The audit inspected the local GoEmotions data, preprocessed
splits, label mappings, model configuration, backend code, research scripts,
previous audit material, and generated prediction artifacts.

The main verified findings are:

1. The source GoEmotions split contains **43,410 training**, **5,426
   development**, and **5,427 test** records.
2. The original task is **multi-label**, not single-label.
3. The processed evaluation representation has seven labels:
   `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, and `neutral`.
4. The transformer prediction artifact contains the same seven output columns,
   so a valid seven-label multi-label evaluation is possible.
5. The held-out test labels contain no observed ID overlap with train or dev,
   but exact text overlap exists:
   - 41 train/dev text matches
   - 32 train/test text matches
   - 10 dev/test text matches
6. The existing prediction artifact was evaluated successfully using a new
   evaluator that never uses predictions to define ground truth.
7. At threshold `0.30`, the transformer achieved:
   - Exact-match accuracy: **0.3687**
   - Macro precision: **0.4408**
   - Macro recall: **0.5538**
   - Macro-F1: **0.4339**
   - Weighted-F1: **0.4885**
   - Hamming loss: **0.1696**
8. VADER was evaluated separately on the three-class sentiment task. Using a
   clearly marked deterministic first-label proxy for the multi-label reference,
   it achieved:
   - Accuracy: **0.6191**
   - Macro precision: **0.6218**
   - Macro recall: **0.6274**
   - Macro-F1: **0.6015**
   - Weighted-F1: **0.6064**
9. VADER and the transformer should **not** be claimed to compete directly:
   VADER predicts sentiment polarity, while the transformer predicts emotions.
10. The existing `reproducible_evaluation.py` report is not an unbiased
    single-label evaluation because it chooses the ground-truth label among
    multiple true labels using the model's own probabilities. The new evaluator
    avoids that issue and labels its deterministic first-label results as a
    proxy only.

The results establish a useful benchmark for the current offline artifacts.
They do not establish reliability on private journal entries, clinical validity,
or real-world longitudinal emotional consistency.

---

## 2. Files and Components Inspected

### Dataset and research files

- `dataset 1\data\train.tsv`
- `dataset 1\data\dev.tsv`
- `dataset 1\data\test.tsv`
- `dataset 1\data\emotions.txt`
- `dataset 1\data\ekman_mapping.json`
- `dataset 1\data\sentiment_mapping.json`
- `dataset 1\data\sentiment_dict.json`
- `dataset 1\data\full_dataset\goemotions_1.csv`
- `dataset 1\data\full_dataset\goemotions_2.csv`
- `dataset 1\data\full_dataset\goemotions_3.csv`
- `dataset 1\preprocessed\train_dedup.tsv`
- `dataset 1\preprocessed\train_ekman.tsv`
- `dataset 1\preprocessed\dev_ekman.tsv`
- `dataset 1\preprocessed\test_ekman.tsv`
- `dataset 1\preprocessed\train_sentiment.tsv`
- `dataset 1\preprocessed\dev_sentiment.tsv`
- `dataset 1\preprocessed\test_sentiment.tsv`
- `dataset 1\preprocessed\hf_model_test_predictions.tsv`
- `dataset 1\preprocessed\vader_test_predictions.tsv`
- `dataset 1\preprocessed\evaluation_results.json`
- `dataset 1\preprocessed\reproducible_evaluation.json`
- `dataset 1\preprocessed\vader_evaluation_results.json`
- `dataset 1\preprocessed\personalization_evaluation_results.json`

### Application files

- `backend\app\config.py`
- `backend\app\main.py`
- `backend\app\schemas.py`
- `backend\app\services.py`
- `backend\app\personalization.py`
- `backend\app\models.py`
- `backend\requirements.txt`
- `backend\tests\test_api.py`

### Research scripts

- `dataset 1\run_pipeline_steps1_5.py`
- `dataset 1\run_hf_model_eval.py`
- `dataset 1\evaluate_existing_predictions.py`
- `dataset 1\evaluate_personalization.py`
- `dataset 1\replace_emotions.py`

### Previous reports

- `reports\antigravity_application_audit.md`
- `DATASET_AUDIT_REPORT.md`

### New evaluation script and output

- `research\evaluation\baseline_emotion_evaluation.py`
- `reports\baseline_emotion_evaluation_results.json`

---

## 3. Dataset Audit Findings

### 3.1 Source split sizes

The source files are headerless TSV files with three fields:

```text
text<TAB>labels<TAB>id
```

| Split | File | Records | Columns | Size |
|---|---|---:|---:|---:|
| Training | `dataset 1\data\train.tsv` | 43,410 | 3 | 3,519,053 bytes |
| Development | `dataset 1\data\dev.tsv` | 5,426 | 3 | 439,059 bytes |
| Test | `dataset 1\data\test.tsv` | 5,427 | 3 | 436,706 bytes |
| **Total** |  | **54,263** |  |  |

The split is approximately:

- Training: **80.00%**
- Development: **10.00%**
- Test: **10.00%**

There is a separate development split. It should be described as development
or validation data, not combined with the test set.

### 3.2 Source fields

| Field | Meaning |
|---|---|
| `text` | Reddit comment |
| `labels` | Comma-separated original GoEmotions numeric IDs |
| `id` | Comment identifier |

The original label order comes from `dataset 1\data\emotions.txt`:

```text
admiration, amusement, anger, annoyance, approval, caring, confusion,
curiosity, desire, disappointment, disapproval, disgust, embarrassment,
excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride,
realization, relief, remorse, sadness, surprise, neutral
```

There are 28 original labels: 27 emotion categories plus neutral.

### 3.3 Multi-label structure

| Split | One label | Two labels | Three labels | Four labels | Five labels |
|---|---:|---:|---:|---:|---:|
| Train | 36,308 | 6,541 | 532 | 28 | 1 |
| Dev | 4,548 | 809 | 62 | 7 | 0 |
| Test | 4,590 | 774 | 61 | 2 | 0 |

The source task is therefore multi-label. A record can contain multiple valid
emotions, and a single categorical accuracy score cannot represent the whole
task without an explicit reduction rule.

### 3.4 Preprocessed seven-label representation

The preprocessing script maps the original labels into:

```text
anger
disgust
fear
joy
sadness
surprise
neutral
```

The mapping is defined in `dataset 1\data\ekman_mapping.json`; neutral is added
separately by the preprocessing code.

| File | Records | Columns | Purpose |
|---|---:|---:|---|
| `preprocessed\train_dedup.tsv` | 43,227 | 3 | Deduplicated training source |
| `preprocessed\train_ekman.tsv` | 43,227 | 3 | Seven-label train data |
| `preprocessed\dev_ekman.tsv` | 5,426 | 3 | Seven-label development data |
| `preprocessed\test_ekman.tsv` | 5,427 | 3 | Seven-label test data |

The training pipeline removed 183 duplicate training text records:

```text
43,410 - 43,227 = 183
```

### 3.5 Three-class sentiment representation

The preprocessing script also maps labels to:

```text
positive
negative
ambiguous
```

| File | Records | Columns |
|---|---:|---:|
| `preprocessed\train_sentiment.tsv` | 43,227 | 3 |
| `preprocessed\dev_sentiment.tsv` | 5,426 | 3 |
| `preprocessed\test_sentiment.tsv` | 5,427 | 3 |

### 3.6 Missing values and duplicates

Verified results:

- No missing fields in source train/dev/test files.
- No empty text entries in source train/dev/test files.
- No malformed row widths in source train/dev/test files.
- No duplicate IDs within train, dev, or test.
- No observed ID overlap between train, dev, and test.
- No exact duplicate complete rows in the raw GoEmotions CSV files.

### 3.7 Cross-split text overlap

Exact text overlap was found even though ID overlap was not:

| Split pair | Shared IDs | Shared exact texts |
|---|---:|---:|
| Train / Dev | 0 | 41 |
| Train / Test | 0 | 32 |
| Dev / Test | 0 | 10 |

This is a data-leakage risk. The source test set remains useful as a benchmark,
but it should not be described as completely leakage-free.

The existing preprocessing removes duplicate text from the training split only.
It does not remove matching texts across all three splits.

### 3.8 Class imbalance

The seven-label test supports used in the multi-label evaluation were:

| Label | Test support |
|---|---:|
| anger | 726 |
| disgust | 123 |
| fear | 98 |
| joy | 2,104 |
| sadness | 379 |
| surprise | 677 |
| neutral | 1,787 |

The smallest class is fear with 98 positive labels, while joy has 2,104. Macro
metrics are therefore important because weighted metrics are strongly influenced
by common classes.

---

## 4. Model Implementation Audit

### 4.1 VADER

The backend initializes VADER in `backend\app\services.py`:

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
```

For each journal text, it computes:

```python
compound = analyzer.polarity_scores(text)["compound"]
```

The runtime sentiment thresholds are:

```text
compound >= 0.05  -> positive
compound <= -0.05 -> negative
otherwise         -> neutral
```

The offline preprocessing script uses the same numerical thresholds but calls
the third category `ambiguous` when evaluating the GoEmotions-derived
sentiment labels.

VADER is a rule-based sentiment analyzer. It does not use:

- `train.tsv`
- `dev.tsv`
- `test.tsv`
- GoEmotions CSV files
- `ekman_mapping.json`
- `sentiment_mapping.json`

at application runtime.

### 4.2 Transformer

The configured model name is:

```text
j-hartmann/emotion-english-distilroberta-base
```

The backend loads it through a Hugging Face text-classification pipeline:

```python
pipeline(
    "text-classification",
    model=self.settings.emotion_model_name,
    top_k=None,
    device=-1,
)
```

The backend:

- Uses the model as a pretrained checkpoint.
- Does not train or fine-tune it.
- Requests the full score distribution.
- Selects the highest-scoring emotion as the primary prediction.
- Retains a secondary emotion when the score is sufficiently high or close.
- Truncates input text to 10,000 characters.

The existing prediction artifact has the following seven output columns:

```text
anger, disgust, fear, joy, neutral, sadness, surprise
```

These match the processed seven-label reference representation used by the
offline evaluation scripts.

### 4.3 Fallback

When the transformer is disabled or unavailable, the backend uses a heuristic
fallback:

- Positive sentiment → `joy`
- Negative sentiment → `sadness`
- Neutral sentiment → `neutral`

The fallback includes a fixed local distribution and marks the result as a
fallback. It is not a trained classifier.

### 4.4 Personalization

The application stores user corrections in SQLite and applies an Empirical
Bayes-style bounded calibration after sufficient feedback. The configured
minimum feedback threshold is five events.

This personalization layer is not trained on the GoEmotions files at runtime.
The existing personalization benchmark uses simulated personas and the
held-out GoEmotions-derived test data. It does not demonstrate performance on
real users or real journal entries.

### 4.5 Application mood labels

User-selected moods are:

```text
happy, sad, stressed, anxious, angry, calm, excited, neutral
```

AI emotion outputs are:

```text
anger, disgust, fear, joy, sadness, surprise, neutral
```

These are not identical taxonomies. The application compares selected moods
with broad sentiment alignment:

- `happy`, `excited`, `calm` → positive
- `sad`, `stressed`, `anxious`, `angry` → negative
- `neutral` → neutral

Therefore, a user mood and an AI emotion should not be treated as the same
ground-truth label system.

---

## 5. Evaluation Design

### 5.1 Transformer evaluation task

The compatible evaluation task is:

```text
Input: test_ekman.tsv
Output: hf_model_test_predictions.tsv
Labels: anger, disgust, fear, joy, sadness, surprise, neutral
Task: multi-label classification
```

For each test record:

- The reference labels are converted into a seven-element binary vector.
- The model probabilities are read from the existing prediction TSV.
- A predicted label is positive when its probability is at least `0.30`.
- No prediction is used to alter or select the reference labels.

This is the primary baseline protocol.

### 5.2 Why multi-label metrics are primary

The source data contains multiple true labels for many records. Appropriate
primary metrics include:

- Macro precision
- Macro recall
- Macro-F1
- Weighted-F1
- Micro precision/recall/F1
- Hamming loss
- Exact-match accuracy
- Per-class scores

Exact-match accuracy is strict: every label for a record must be correct.
Hamming loss measures the fraction of individual label decisions that are
incorrect.

### 5.3 Single-label confusion matrix

A confusion matrix requires one reference class per record. The evaluator
therefore reports a separate **first-label proxy**:

- It selects the first encoded reference label deterministically.
- It does not use model probabilities.
- It is clearly labeled as a proxy.
- It must not be presented as the original single-label ground truth.

This provides a reproducible accuracy and confusion matrix without the more
serious bias of selecting the reference class using the model's own scores.

### 5.4 Invalid and unmapped labels

The evaluator:

- Rejects non-numeric label tokens.
- Rejects numeric labels outside the expected range.
- Rejects rows with no valid labels.
- Rejects missing transformer output columns.
- Rejects unexpected VADER sentiment labels.

It does not silently convert invalid data to neutral.

### 5.5 VADER comparison

VADER and the transformer are not directly comparable at their raw outputs:

| Model | Output task |
|---|---|
| VADER | Positive/negative/ambiguous sentiment |
| Transformer | Seven-label emotion probabilities |

VADER is evaluated separately against `test_sentiment.tsv`. Its prediction is
single-label while the reference can still be multi-label, so the reported
accuracy and confusion matrix use the same deterministic first-label proxy and
are explicitly labeled as such.

The VADER score must not be described as directly outperforming the
transformer, because the label spaces and tasks differ.

---

## 6. Baseline Experiment Execution

### Command executed

From the repository root:

```powershell
python research\evaluation\baseline_emotion_evaluation.py
```

### Output generated

```text
reports\baseline_emotion_evaluation_results.json
```

The evaluator used existing prediction artifacts. It did not download a model,
train a model, modify the application, overwrite source datasets, or overwrite
existing evaluation files.

### Experiment status

**Completed successfully.**

This was an artifact-based baseline evaluation. It recalculated metrics from
the existing held-out reference and prediction files. It was not a fresh
transformer inference run.

The existing fresh-inference script is:

```text
dataset 1\run_hf_model_eval.py
```

It loads the Hugging Face checkpoint and writes
`preprocessed\hf_model_test_predictions.tsv`. A separate fresh inference run
was not required because the existing prediction artifact was present and
already matched the seven-label evaluation schema.

---

## 7. Actual Transformer Results

### 7.1 Primary multi-label result

Reference:

```text
dataset 1\preprocessed\test_ekman.tsv
```

Predictions:

```text
dataset 1\preprocessed\hf_model_test_predictions.tsv
```

Samples: **5,427**  
Threshold: **0.30**  
Output classes: **7**

| Metric | Result |
|---|---:|
| Exact-match accuracy | 0.3687 |
| Hamming loss | 0.1696 |
| Micro precision | 0.4562 |
| Macro precision | 0.4408 |
| Macro recall | 0.5538 |
| Macro-F1 | 0.4339 |
| Weighted-F1 | 0.4885 |

### 7.2 Per-class transformer results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| anger | 0.4387 | 0.3650 | 0.3985 | 726 |
| disgust | 0.1294 | 0.8455 | 0.2244 | 123 |
| fear | 0.3614 | 0.7449 | 0.4867 | 98 |
| joy | 0.8802 | 0.3945 | 0.5448 | 2,104 |
| sadness | 0.4668 | 0.4644 | 0.4656 | 379 |
| surprise | 0.3690 | 0.4431 | 0.4027 | 677 |
| neutral | 0.4401 | 0.6189 | 0.5144 | 1,787 |

The low disgust precision and low joy recall should be discussed together with
class imbalance and the multi-label nature of the task. Macro-F1 is more
informative than accuracy alone for this dataset.

### 7.3 Prediction artifact quality

The prediction file contains **5 excess duplicate complete rows**. This does
not change the row count or alignment check because the file still contains
5,427 rows aligned positionally with the test file, but it should be
investigated before treating the artifact as a final publication result.

### 7.4 Deterministic first-label proxy

This is not the primary result. It is provided only for a conventional
single-label summary and confusion matrix:

| Metric | Result |
|---|---:|
| Accuracy | 0.4406 |
| Macro precision | 0.4231 |
| Macro recall | 0.5233 |
| Macro-F1 | 0.4064 |
| Weighted-F1 | 0.4568 |

The proxy uses the first encoded reference label and does not resolve the
underlying multi-label evaluation problem.

### 7.5 Existing report caveat

The existing `dataset 1\evaluate_existing_predictions.py` chooses the true
label among multiple labels by selecting the label with the highest model
probability. That means model predictions influence the reference label
selection. Its output should not be used as an unbiased primary-label
benchmark.

The new evaluator avoids that procedure. The existing result may still be
reported as a historical artifact, but it must be labeled as
model-informed-primary-label evaluation.

---

## 8. Actual VADER Results

Reference:

```text
dataset 1\preprocessed\test_sentiment.tsv
```

Predictions:

```text
dataset 1\preprocessed\vader_test_predictions.tsv
```

Samples: **5,427**  
Reference classes: **positive, negative, ambiguous**

| Metric | Result |
|---|---:|
| Accuracy | 0.6191 |
| Macro precision | 0.6218 |
| Macro recall | 0.6274 |
| Macro-F1 | 0.6015 |
| Weighted-F1 | 0.6064 |

### VADER per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| positive | 0.6470 | 0.8170 | 0.7221 | 2,104 |
| negative | 0.4965 | 0.6672 | 0.5694 | 1,184 |
| ambiguous | 0.7218 | 0.3978 | 0.5130 | 2,139 |

These metrics use the first encoded sentiment label as a deterministic
single-label proxy because VADER emits one label while the source reference
can contain multiple mapped sentiment labels. They are not a full multilabel
VADER evaluation.

### Fair-comparison conclusion

The VADER results cannot be directly ranked against the transformer's emotion
results. The models solve different tasks with different label spaces:

- VADER: three-class sentiment polarity.
- Transformer: seven-class emotion prediction.

A future common-label experiment could map both outputs to positive, negative,
and ambiguous, but that would require an explicit, pre-registered mapping from
the seven emotion outputs to sentiment. It would still measure sentiment
agreement, not emotion classification quality.

---

## 9. Data-Leakage Assessment

### Verified leakage risks

1. **Exact text overlap across splits**
   - Train/dev: 41
   - Train/test: 32
   - Dev/test: 10
2. **Training-only deduplication**
   - The preprocessing removes duplicate train texts but does not remove
     cross-split text overlap.
3. **Model-informed historical evaluation**
   - The old reproducible evaluator uses prediction probabilities to select a
     true label from multi-label references.
4. **Prediction artifact duplicates**
   - Five duplicate complete rows exist in the transformer prediction file.

### What was not observed

- No duplicate IDs across train, dev, and test.
- No missing text in the source splits.
- No malformed source split row widths.
- No evidence that the application loads benchmark files at runtime.

### Suitability of current test set

The current test set is suitable for a **conditional baseline** because:

- It is held out by the provided split.
- It contains reference labels.
- It aligns with the transformer output schema.
- It supports a valid multi-label evaluation.

It is not suitable for claiming a completely leakage-free or journal-specific
benchmark because exact text overlap exists and the domain is Reddit comments,
not personal journal entries.

---

## 10. Verified Findings, Proposed Experiments, and Unknowns

### Verified findings

- The application uses VADER and an optional pretrained transformer.
- The transformer output artifact has seven Ekman-style labels.
- The source GoEmotions-derived task is multi-label.
- The baseline evaluator completed successfully.
- Transformer and VADER metrics were calculated from actual local files.
- Cross-split text overlap exists.
- No ID overlap was observed across source splits.
- The application does not load the GoEmotions files at runtime.
- The application mood labels differ from the AI emotion labels.

### Proposed future experiments

- Rebuild a leakage-controlled split at the unique-text level.
- Regenerate transformer predictions after fixing or investigating duplicate
  prediction rows.
- Tune the multi-label threshold on development data only, then freeze it for
  test evaluation.
- Add confidence intervals using bootstrap resampling.
- Evaluate on a consented, de-identified journal dataset.
- Compare VADER and transformer outputs after defining a shared sentiment task.
- Measure personalization using a real longitudinal protocol rather than only
  simulated personas.

### Unresolved questions

- The exact revision/version of the locally stored GoEmotions files is not
  recorded.
- The exact source/version of the external Dataset 2 Kaggle files is not
  present locally.
- It is not yet confirmed whether Dataset 2 contributed to the final
  application.
- The five duplicate transformer prediction rows need provenance review.
- No unbiased benchmark on real journal entries exists in this repository.

---

## 11. Reproduction Instructions

From the repository root:

```powershell
python research\evaluation\baseline_emotion_evaluation.py
```

Optional custom threshold:

```powershell
python research\evaluation\baseline_emotion_evaluation.py --threshold 0.30
```

Optional custom output:

```powershell
python research\evaluation\baseline_emotion_evaluation.py `
  --output reports\baseline_emotion_evaluation_results.json
```

The evaluator requires the existing Python packages used by the repository,
including pandas, NumPy, and scikit-learn. It reads:

```text
dataset 1\preprocessed\test_ekman.tsv
dataset 1\preprocessed\hf_model_test_predictions.tsv
dataset 1\preprocessed\test_sentiment.tsv
dataset 1\preprocessed\vader_test_predictions.tsv
```

It writes only:

```text
reports\baseline_emotion_evaluation_results.json
```

The evaluator does not overwrite the source datasets or existing prediction
files.

---

## 12. Recommendations for the Next Research Experiment

### Immediate next step

Create a new leakage-controlled benchmark split by grouping exact duplicate
texts before splitting. Keep the current test set as a historical benchmark,
but do not use it as the only evidence.

Recommended protocol:

1. Combine the original labeled split records.
2. Group identical normalized texts.
3. Assign groups, not individual rows, to train/dev/test.
4. Preserve label distribution as closely as possible.
5. Fit any threshold or calibration rule on development data only.
6. Evaluate once on the frozen test set.
7. Report multilabel metrics as the primary result.
8. Report per-class support and confidence intervals.

### Second priority

Investigate the five duplicate rows in
`hf_model_test_predictions.tsv`. If they are an artifact of generation rather
than legitimate duplicate examples, regenerate predictions from the exact
frozen test file and record the model revision and inference environment.

### Third priority

Collect or obtain a small, consented, de-identified journal-style evaluation
set. This should be annotated by clear guidelines and evaluated separately from
GoEmotions. Do not claim that Reddit benchmark results prove journal-domain
reliability.

### Personalization experiment

The personalization benchmark should be extended with:

- Real user feedback sequences where ethically permissible.
- Separate users for adaptation and evaluation.
- A no-personalization control.
- Pre-registered feedback thresholds.
- Time-based splits.
- Calibration metrics and confidence intervals.

---

## 13. Limitations and Threats to Validity

1. The benchmark domain is Reddit, not private journaling.
2. The source task is multi-label and some reported proxy metrics reduce it to
   one label.
3. Exact text overlap exists across the provided splits.
4. The current transformer predictions were generated previously; this audit
   recalculated metrics but did not perform a fresh model download/inference.
5. The exact model revision is not recorded in the local evaluation artifact.
6. The threshold `0.30` is inherited from the existing evaluation pipeline and
   was not tuned in this audit.
7. The test set is imbalanced, so accuracy alone is insufficient.
8. VADER and transformer scores are not directly comparable without a shared
   task.
9. Personalization results from simulated personas do not prove improvement for
   real users.
10. No classification metric establishes emotional consistency, clinical
    usefulness, or mental-health benefit.

---

## 14. Final Conclusion

The existing implementation has a reproducible, technically valid baseline for
the transformer when evaluated as a seven-label multi-label classifier against
the processed GoEmotions test split. The primary verified transformer result at
threshold `0.30` is macro-F1 **0.4339**, with weighted-F1 **0.4885** and
exact-match accuracy **0.3687**.

VADER has a separate three-class sentiment baseline with first-label-proxy
macro-F1 **0.6015**, but this must not be compared directly to the transformer's
emotion score.

The most important research concern is that the provided splits contain exact
text overlap, and the older single-label report uses model probabilities to
select a reference label. The next experiment should rebuild a
grouped-by-text, leakage-controlled split and report multi-label metrics as the
primary evaluation.

The results support the claim that the current system can be benchmarked
offline. They do not support claims that it is reliable for personal journal
entries, clinically valid, or validated for emotional consistency.
