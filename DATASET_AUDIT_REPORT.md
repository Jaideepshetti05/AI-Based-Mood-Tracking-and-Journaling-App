# Dataset Audit Report

## AI-Based Mood Tracking and Journaling App

**Audit date:** 27 October 2026  
**Audit scope:** `dataset 1\`, `dataset 2\`, backend source, mobile source,
research scripts, and generated evaluation files  
**Method:** Read-only inspection. No project or dataset files were modified or
deleted.

---

## 1. Executive Summary

The project contains two dataset areas.

### Dataset 1: GoEmotions-based research data

`dataset 1\` is the main dataset and research area. It contains:

- Raw GoEmotions CSV annotation files.
- Processed train, development, and test TSV files.
- Emotion and sentiment label mappings.
- VADER prediction outputs.
- Transformer prediction outputs.
- Evaluation results and preprocessing scripts.

The three raw CSV files contain:

- **211,225 annotation records**
- **37 columns**
- Approximately **58,011 unique comment IDs**
- Approximately **57,732 unique text values**
- **28 emotion categories**: 27 emotions plus neutral

The processed source split contains:

- Training: **43,410 records**
- Development: **5,426 records**
- Test: **5,427 records**
- Total: **54,263 records**
- Split: approximately **80% / 10% / 10%**

### Dataset 2: Stress-analysis notebook

`dataset 2\` contains only:

```text
dataset 2\stress-analysis-in-social-media.ipynb
```

The notebook references external Kaggle files named
`dreaddit-train.csv` and `dreaddit-test.csv`, but those CSV files are not
stored in this repository. Their statistics could not be verified locally.

### Runtime application conclusion

The running FastAPI application does **not** load the GoEmotions CSV or TSV
files at startup. It uses:

- VADER for sentiment scoring.
- An optional pretrained Hugging Face model:
  `j-hartmann/emotion-english-distilroberta-base`
- A local heuristic fallback when the transformer is disabled or unavailable.
- SQLite for journal entries, analyses, corrections, and personalization.

The GoEmotions files are used by offline research and evaluation scripts, not
directly by the live application.

---

## 2. Dataset Inventory

### 2.1 Raw GoEmotions CSV files

| Dataset/resource | Relative path | Format | Size | Data records | Columns |
|---|---|---:|---:|---:|---:|
| GoEmotions raw part 1 | `dataset 1\data\full_dataset\goemotions_1.csv` | CSV | 14,174,600 bytes / 13,842.38 KB / 13.518 MB | 70,000 | 37 |
| GoEmotions raw part 2 | `dataset 1\data\full_dataset\goemotions_2.csv` | CSV | 14,173,154 bytes / 13,840.97 KB / 13.517 MB | 70,000 | 37 |
| GoEmotions raw part 3 | `dataset 1\data\full_dataset\goemotions_3.csv` | CSV | 14,395,164 bytes / 14,057.78 KB / 13.728 MB | 71,225 | 37 |

The raw files have a header row. The record counts above exclude the header.

Combined raw-file results:

- Total data records: **211,225**
- Exact duplicate complete rows: **0**
- Empty text entries: **0**
- Unique comment IDs: **58,011**
- Unique text values: **57,732**
- `example_very_unclear=True`: **3,411**
- `example_very_unclear=False`: **207,814**

Because these are annotation-level files, multiple rows can refer to the same
Reddit comment. Therefore, 211,225 rows does not mean 211,225 unique comments.

### Raw CSV columns and meanings

All three raw files have the same 37-column schema:

| Column | Meaning |
|---|---|
| `text` | Reddit comment text |
| `id` | Comment identifier |
| `author` | Reddit author name |
| `subreddit` | Subreddit associated with the comment |
| `link_id` | Reddit link identifier |
| `parent_id` | Parent comment identifier |
| `created_utc` | Original UTC timestamp |
| `rater_id` | Annotator identifier |
| `example_very_unclear` | Whether the annotator marked the example unclear |
| `admiration` through `neutral` | Binary emotion annotation columns |

The emotion columns are:

```text
admiration, amusement, anger, annoyance, approval, caring, confusion,
curiosity, desire, disappointment, disapproval, disgust, embarrassment,
excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride,
realization, relief, remorse, sadness, surprise, neutral
```

### Raw annotation label frequencies

These are annotation counts, not unique-comment counts:

| Emotion | Count | Percentage |
|---|---:|---:|
| admiration | 17,131 | 8.11% |
| amusement | 9,245 | 4.38% |
| anger | 8,084 | 3.83% |
| annoyance | 13,618 | 6.45% |
| approval | 17,620 | 8.34% |
| caring | 5,999 | 2.84% |
| confusion | 7,359 | 3.48% |
| curiosity | 9,692 | 4.59% |
| desire | 3,817 | 1.81% |
| disappointment | 8,469 | 4.01% |
| disapproval | 11,424 | 5.41% |
| disgust | 5,301 | 2.51% |
| embarrassment | 2,476 | 1.17% |
| excitement | 5,629 | 2.66% |
| fear | 3,197 | 1.51% |
| gratitude | 11,625 | 5.50% |
| grief | 673 | 0.32% |
| joy | 7,983 | 3.78% |
| love | 8,191 | 3.88% |
| nervousness | 1,810 | 0.86% |
| optimism | 8,715 | 4.13% |
| pride | 1,302 | 0.62% |
| realization | 8,785 | 4.16% |
| relief | 1,289 | 0.61% |
| remorse | 2,525 | 1.20% |
| sadness | 6,758 | 3.20% |
| surprise | 5,514 | 2.61% |
| neutral | 55,298 | 26.18% |

The raw data is imbalanced. Neutral, approval, and admiration are relatively
frequent. Grief, pride, relief, and nervousness are comparatively rare.

---

## 3. Processed GoEmotions Split Files

The source split files are headerless TSV files with three fields:

```text
text<TAB>labels<TAB>id
```

| File | Format | Size | Data records | Columns |
|---|---|---:|---:|---:|
| `dataset 1\data\train.tsv` | TSV | 3,519,053 bytes / 3,436.58 KB / 3.356 MB | 43,410 | 3 |
| `dataset 1\data\dev.tsv` | TSV | 439,059 bytes / 428.77 KB / 0.419 MB | 5,426 | 3 |
| `dataset 1\data\test.tsv` | TSV | 436,706 bytes / 426.47 KB / 0.416 MB | 5,427 | 3 |

There is **no header row** in these files. All rows are data records.

### Processed TSV fields

| Field | Meaning |
|---|---|
| Field 1: `text` | Reddit comment text |
| Field 2: `labels` | Comma-separated numeric emotion IDs |
| Field 3: `id` | Comment identifier |

Numeric label IDs refer to the order in
`dataset 1\data\emotions.txt`.

### Original 28 labels

```text
0  admiration
1  amusement
2  anger
3  annoyance
4  approval
5  caring
6  confusion
7  curiosity
8  desire
9  disappointment
10 disapproval
11 disgust
12 embarrassment
13 excitement
14 fear
15 gratitude
16 grief
17 joy
18 love
19 nervousness
20 optimism
21 pride
22 realization
23 relief
24 remorse
25 sadness
26 surprise
27 neutral
```

### Split sizes

| Split | Records | Percentage |
|---|---:|---:|
| Training | 43,410 | 80.00% |
| Development | 5,426 | 10.00% |
| Test | 5,427 | 10.00% |
| **Total** | **54,263** | **100.00%** |

There is a separate development split, so the project does have a
development/validation set.

### Multi-label characteristics

| Split | One label | Two labels | Three labels | Four labels | Five labels |
|---|---:|---:|---:|---:|---:|
| Train | 36,308 | 6,541 | 532 | 28 | 1 |
| Dev | 4,548 | 809 | 62 | 7 | 0 |
| Test | 4,590 | 774 | 61 | 2 | 0 |

The original task is multi-label. Selecting one primary label for evaluation is
a reduction of the original task.

### Text length

| Split | Minimum characters | Maximum characters | Average characters |
|---|---:|---:|---:|
| Train | 2 | 703 | 68.40 |
| Dev | 5 | 187 | 68.24 |
| Test | 5 | 184 | 67.82 |

No empty text entries were detected.

---

## 4. Processed Ekman-Label Files

The project maps the original fine-grained emotions to seven classes:

```text
anger
disgust
fear
joy
sadness
surprise
neutral
```

| File | Size | Records | Columns |
|---|---:|---:|---:|
| `dataset 1\preprocessed\train_dedup.tsv` | 3,557,479 bytes / 3,474.10 KB / 3.393 MB | 43,227 | 3 |
| `dataset 1\preprocessed\train_ekman.tsv` | 3,519,053 bytes / 3,436.58 KB / 3.356 MB | 43,227 | 3 |
| `dataset 1\preprocessed\dev_ekman.tsv` | 439,660 bytes / 429.36 KB / 0.419 MB | 5,426 | 3 |
| `dataset 1\preprocessed\test_ekman.tsv` | 437,377 bytes / 427.13 KB / 0.417 MB | 5,427 | 3 |

The training deduplication removed **183 duplicate training text records**:

```text
43,410 original training records - 43,227 deduplicated records = 183 removed
```

### Ekman primary-label counts

| Label | Train | Dev | Test |
|---|---:|---:|---:|
| anger | 5,567 | 717 | 726 |
| disgust | 636 | 82 | 96 |
| fear | 689 | 97 | 95 |
| joy | 16,811 | 2,145 | 2,054 |
| sadness | 2,590 | 296 | 317 |
| surprise | 4,145 | 497 | 533 |
| neutral | 12,789 | 1,592 | 1,606 |

The Ekman representation is imbalanced, particularly for joy, neutral, and
disgust.

---

## 5. Processed Sentiment-Label Files

The project maps emotions to:

```text
positive
negative
ambiguous
```

| File | Size | Records | Columns |
|---|---:|---:|---:|
| `dataset 1\preprocessed\train_sentiment.tsv` | 3,517,395 bytes / 3,434.96 KB / 3.354 MB | 43,227 | 3 |
| `dataset 1\preprocessed\dev_sentiment.tsv` | 439,448 bytes / 429.15 KB / 0.419 MB | 5,426 | 3 |
| `dataset 1\preprocessed\test_sentiment.tsv` | 437,155 bytes / 426.91 KB / 0.417 MB | 5,427 | 3 |

### Primary sentiment counts

| Sentiment | Train | Dev | Test |
|---|---:|---:|---:|
| positive | 17,298 | 2,219 | 2,104 |
| negative | 8,995 | 1,118 | 1,184 |
| ambiguous | 16,934 | 2,089 | 2,139 |

Negative examples are less frequent than positive and ambiguous examples.

---

## 6. Detailed CSV and TSV Audit

### `dataset 1\data\ekman_labels.csv`

- Size: **55 bytes**
- Format: CSV
- Data records: **6**
- Columns: **1**
- Header: No
- Values displayed: anger, disgust, fear, joy, sadness, surprise, neutral

This is a label resource rather than a text dataset. The canonical seven-label
file used by the preprocessing pipeline is
`dataset 1\preprocessed\ekman_labels.txt`.

### `dataset 1\tables\emotion_words.csv`

- Size: **32,464 bytes**
- Format: CSV
- Records: **1,178**
- Columns: **4**
- Header: `emotion,word,odds,freq`
- Missing values: None detected
- Duplicate complete rows: None detected
- Empty words: None detected

Column meanings:

| Column | Meaning |
|---|---|
| `emotion` | Associated emotion |
| `word` | Word associated with that emotion |
| `odds` | Association-strength statistic |
| `freq` | Frequency or relative-frequency value |

This is an analysis output, not the main training dataset.

### `dataset 1\plots\colors.tsv`

- Size: **898 bytes**
- Format: TSV
- Records: **48**
- Columns: **3**
- Purpose: Plotting/color configuration
- Missing values: None detected
- Duplicate complete rows: None detected

This is a plotting support file, not a machine-learning dataset.

### `vader_test_predictions.tsv`

Path:

```text
dataset 1\preprocessed\vader_test_predictions.tsv
```

- Size: **527,836 bytes**
- Records: **5,427**
- Columns: **5**
- Header: `text, labels, id, vader_sentiment, vader_compound`
- Missing values: None detected
- Empty text: None
- Duplicate complete rows: None detected
- Score range: **-0.9735 to 0.9856**
- Average score: **0.12996**

Prediction distribution:

| Sentiment | Count |
|---|---:|
| positive | 2,657 |
| negative | 1,591 |
| ambiguous | 1,179 |

This is a generated evaluation output, not an independent dataset.

### `hf_model_test_predictions.tsv`

Path:

```text
dataset 1\preprocessed\hf_model_test_predictions.tsv
```

- Size: **827,934 bytes**
- Records: **5,427**
- Columns: **8**
- Header: `text, anger, disgust, fear, joy, neutral, sadness, surprise`
- Missing values: None detected
- Duplicate excess rows: **5**
- Purpose: Transformer prediction probabilities for the test split

### Quality summary

| File category | Missing values | Empty text | Duplicate complete rows |
|---|---:|---:|---:|
| Raw GoEmotions CSV | 0 | 0 | 0 |
| Original train/dev/test TSV | 0 | 0 | 0 |
| Processed Ekman TSV | 0 | 0 | 0 |
| Processed sentiment TSV | 0 | 0 | 0 |
| VADER prediction TSV | 0 | 0 | 0 |
| Transformer prediction TSV | 0 | 0 | 5 |
| `emotion_words.csv` | 0 | N/A | 0 |
| `colors.tsv` | 0 | N/A | 0 |

---

## 7. JSON, TXT, and Mapping File Audit

### `dataset 1\data\emotions.txt`

- Size: **248 bytes**
- Format: TXT
- Non-empty lines: **28**
- Purpose: Defines the original GoEmotions label order.
- Used by: preprocessing and analysis scripts.

This is a label-index configuration file, not a dataset.

### `dataset 1\data\ekman_mapping.json`

- Size: **396 bytes**
- Format: JSON object
- Top-level keys: **6**
- Purpose: Maps fine-grained emotions into broad Ekman-style groups.

Mappings:

- `anger`: anger, annoyance, disapproval
- `disgust`: disgust
- `fear`: fear, nervousness
- `joy`: joy, amusement, approval, excitement, gratitude, love, optimism,
  relief, pride, admiration, desire, caring
- `sadness`: sadness, disappointment, embarrassment, grief, remorse
- `surprise`: surprise, realization, confusion, curiosity

Neutral is added separately by the preprocessing code.

### `dataset 1\data\sentiment_mapping.json`

- Size: **367 bytes**
- Format: JSON object
- Top-level keys: **3**
- Classes: positive, negative, ambiguous

It maps the original emotion categories to three broad sentiment groups.

### `dataset 1\data\sentiment_dict.json`

- Size: **367 bytes**
- Format: JSON object
- Top-level keys: **3**
- Purpose: Sentiment dictionary/configuration resource.

It has the same positive, negative, and ambiguous structure as the sentiment
mapping file. It is used by analysis scripts, not by the live FastAPI runtime.

### Preprocessed result JSON files

| File | Size | Purpose |
|---|---:|---|
| `preprocessed\ekman_class_weights.json` | 149 bytes | Training class weights |
| `preprocessed\evaluation_results.json` | 3,846 bytes | Transformer evaluation |
| `preprocessed\personalization_evaluation_results.json` | 43,881 bytes | Personalization benchmark |
| `preprocessed\reproducible_evaluation.json` | 2,583 bytes | Held-out evaluation |
| `preprocessed\vader_evaluation_results.json` | 1,085 bytes | VADER evaluation |

These are generated result files, not raw datasets.

### Label TXT files

`dataset 1\preprocessed\ekman_labels.txt` contains seven labels:

```text
anger
disgust
fear
joy
sadness
surprise
neutral
```

`dataset 1\preprocessed\sentiment_labels.txt` contains three labels:

```text
positive
negative
ambiguous
```

---

## 8. Dataset Splitting and Leakage Analysis

### Split totals

| Split | Records | Percentage |
|---|---:|---:|
| Train | 43,410 | 80.00% |
| Dev | 5,426 | 10.00% |
| Test | 5,427 | 10.00% |
| **Total** | **54,263** | **100.00%** |

All splits use the same three-field schema:

```text
text, labels, id
```

No missing fields, malformed row widths, duplicate IDs within a split, or empty
texts were detected.

### Cross-split duplicate checks

No duplicate IDs were found across train, dev, and test. However, exact text
overlap was found:

| Split pair | Shared IDs | Shared text values |
|---|---:|---:|
| Train / Dev | 0 | 41 |
| Train / Test | 0 | 32 |
| Dev / Test | 0 | 10 |

This is a verified potential leakage risk. The preprocessing code deduplicates
the training split only; it does not remove matching texts across all splits.

Recommended wording:

> There is no observed ID overlap between the splits, but exact-text overlap
> exists across splits. This may slightly inflate evaluation estimates if
> identical text occurs in both training and evaluation data.

---

## 9. Dataset 2 Audit

The only local file under `dataset 2\` is:

```text
dataset 2\stress-analysis-in-social-media.ipynb
```

- Size: **129,686 bytes**
- Format: Jupyter Notebook JSON
- Local CSV files: **None**
- Local TSV files: **None**

The notebook references:

```python
/kaggle/input/stress-analysis-in-social-media/dreaddit-train.csv
/kaggle/input/stress-analysis-in-social-media/dreaddit-test.csv
```

It appears to use:

- `text`
- `label`
- `No Stress`
- `Stress`

Because these CSV files are not in the repository, their row counts, columns,
sizes, missing values, duplicates, class distribution, version, and leakage
could not be verified locally.

Do not present Dataset 2 statistics as verified project statistics unless the
original Kaggle files are obtained and audited separately.

---

## 10. Source and Publisher Information

### GoEmotions

The dataset documentation identifies GoEmotions as a Google Research dataset
based on Reddit comments.

Official repository:

<https://github.com/google-research/google-research/tree/master/goemotions>

Paper:

<https://arxiv.org/abs/2005.00547>

The exact downloaded-file revision was not available in the repository, so the
dataset version should be reported as **not specified locally**.

### VADER

The backend uses:

```text
vaderSentiment==3.3.2
```

VADER is a rule-based sentiment analyzer. It does not use any of the local
GoEmotions CSV, TSV, JSON, or TXT files at runtime.

### Transformer model

The optional model is:

```text
j-hartmann/emotion-english-distilroberta-base
```

The project loads this pretrained model through Hugging Face Transformers. The
project does not train this transformer. It evaluates the pretrained checkpoint
against the processed GoEmotions test split.

---

## 11. AI Model and Dataset Integration

### Datasets directly used at application runtime

The FastAPI application does not load these files when it starts:

- `goemotions_1.csv`
- `goemotions_2.csv`
- `goemotions_3.csv`
- `train.tsv`
- `dev.tsv`
- `test.tsv`
- `emotions.txt`
- `ekman_mapping.json`
- `sentiment_mapping.json`
- `sentiment_dict.json`

The live application uses:

1. VADER package code and its own lexicon.
2. The optional Hugging Face transformer model.
3. Local fallback logic.
4. SQLite database records.

### Research and experimentation files

The GoEmotions resources are used by:

- `dataset 1\run_pipeline_steps1_5.py`
- `dataset 1\run_hf_model_eval.py`
- `dataset 1\evaluate_existing_predictions.py`
- `dataset 1\evaluate_personalization.py`
- `dataset 1\analyze_data.py`
- `dataset 1\calculate_metrics.py`
- `dataset 1\extract_words.py`
- `dataset 1\replace_emotions.py`

### Application mood labels versus AI emotion labels

User-selected moods:

```text
happy
sad
stressed
anxious
angry
calm
excited
neutral
```

AI emotion labels:

```text
anger
disgust
fear
joy
sadness
surprise
neutral
```

These are not the same label system. The application compares the user's mood
with broader sentiment groups:

- Positive moods: `happy`, `excited`, `calm`
- Negative moods: `sad`, `stressed`, `anxious`, `angry`
- Neutral mood: `neutral`

Therefore:

- `happy` is compared mainly with positive sentiment.
- `stressed` is compared mainly with negative sentiment.
- `joy` is related to positive sentiment but is not identical to `happy`.
- `fear` is related to negative sentiment but is not identical to `anxious`.

---

## 12. Data Quality and Research Limitations

### Verified problems

1. **Class imbalance:** Classes are not evenly distributed.
2. **Multi-label structure:** Many records have multiple emotion labels.
3. **Cross-split exact-text overlap:** 41 train/dev, 32 train/test, and 10
   dev/test text overlaps were found.
4. **Transformer prediction duplicates:** Five excess duplicate rows were found
   in `hf_model_test_predictions.tsv`.
5. **Domain mismatch:** GoEmotions contains Reddit comments, while the
   application processes private journal entries.
6. **Dataset 2 incomplete locally:** The notebook is present, but its source
   CSV files are not.

### Potential risks

1. Annotation-level counts can be confused with unique-comment counts.
2. Primary-label evaluation loses multi-label information.
3. Exact text overlap can make evaluation optimistic.
4. English Reddit data may not represent all cultures, ages, or writing styles.
5. Raw GoEmotions metadata includes Reddit author, subreddit, link, parent, and
   timestamp fields.
6. Neither GoEmotions nor VADER is a clinical diagnostic dataset.
7. Automated predictions can be wrong for sarcasm, slang, short text, mixed
   emotion, or context-dependent writing.

The application must not be presented as a mental-health diagnosis system.

---

## 13. Evidence and Verification Commands

The following read-only Python pattern was used to inspect CSV and TSV files:

```python
import csv
from pathlib import Path

root = Path(r"C:\Users\Jaideep\Desktop\inhouse project")

for path in (root / "dataset 1").rglob("*"):
    if path.suffix.lower() not in {".csv", ".tsv"}:
        continue

    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

    with path.open(
        encoding="utf-8-sig",
        newline="",
        errors="replace",
    ) as file:
        rows = list(csv.reader(file, delimiter=delimiter))

    print(path)
    print("Physical rows:", len(rows))
    print("First row:", rows[0] if rows else [])
```

The source code confirms that the split files are headerless:

```python
pd.read_csv(
    os.path.join(DATA_DIR, "train.tsv"),
    sep="\t",
    header=None,
    names=["text", "labels", "id"],
)
```

Run the research pipeline with:

```powershell
python "dataset 1\run_pipeline_steps1_5.py"
```

Run transformer evaluation with:

```powershell
python "dataset 1\run_hf_model_eval.py"
```

Reproduce the existing held-out evaluation with:

```powershell
python "dataset 1\evaluate_existing_predictions.py"
```

Run backend tests with:

```powershell
python -m pytest backend\tests -q
```

---

## 14. Presentation-Ready Summary Table

| Dataset/resource | File | Format | File size | Records | Fields | Classes/labels | Purpose |
|---|---|---:|---:|---:|---:|---:|---|
| GoEmotions raw part 1 | `data\full_dataset\goemotions_1.csv` | CSV | 13.518 MB | 70,000 | 37 | 28 | Raw Reddit annotations |
| GoEmotions raw part 2 | `data\full_dataset\goemotions_2.csv` | CSV | 13.517 MB | 70,000 | 37 | 28 | Raw Reddit annotations |
| GoEmotions raw part 3 | `data\full_dataset\goemotions_3.csv` | CSV | 13.728 MB | 71,225 | 37 | 28 | Raw Reddit annotations |
| Training split | `data\train.tsv` | TSV | 3.356 MB | 43,410 | 3 | 28 | Original training data |
| Development split | `data\dev.tsv` | TSV | 0.419 MB | 5,426 | 3 | 28 | Development/validation |
| Test split | `data\test.tsv` | TSV | 0.416 MB | 5,427 | 3 | 28 | Held-out test data |
| Deduplicated training | `preprocessed\train_dedup.tsv` | TSV | 3.393 MB | 43,227 | 3 | 28 | Deduplicated training |
| Ekman train | `preprocessed\train_ekman.tsv` | TSV | 3.356 MB | 43,227 | 3 | 7 | Broad emotion classes |
| Ekman dev | `preprocessed\dev_ekman.tsv` | TSV | 0.419 MB | 5,426 | 3 | 7 | Broad emotion classes |
| Ekman test | `preprocessed\test_ekman.tsv` | TSV | 0.417 MB | 5,427 | 3 | 7 | Broad emotion classes |
| Sentiment train | `preprocessed\train_sentiment.tsv` | TSV | 3.354 MB | 43,227 | 3 | 3 | Sentiment classes |
| Sentiment dev | `preprocessed\dev_sentiment.tsv` | TSV | 0.419 MB | 5,426 | 3 | 3 | Sentiment development |
| Sentiment test | `preprocessed\test_sentiment.tsv` | TSV | 0.417 MB | 5,427 | 3 | 3 | Sentiment test |
| VADER predictions | `preprocessed\vader_test_predictions.tsv` | TSV | 0.503 MB | 5,427 | 5 | 3 predictions | Evaluation output |
| Transformer predictions | `preprocessed\hf_model_test_predictions.tsv` | TSV | 0.790 MB | 5,427 | 8 | 7 predictions | Evaluation output |
| Emotion word table | `tables\emotion_words.csv` | CSV | 0.031 MB | 1,178 | 4 | 28 | Word association analysis |
| Ekman label resource | `data\ekman_labels.csv` | CSV | 55 bytes | 6 | 1 | 6 listed | Label resource |
| Emotion list | `data\emotions.txt` | TXT | 248 bytes | 28 lines | N/A | 28 | Label index |
| Ekman mapping | `data\ekman_mapping.json` | JSON | 396 bytes | N/A | 6 keys | 7 with neutral | Label mapping |
| Sentiment mapping | `data\sentiment_mapping.json` | JSON | 367 bytes | N/A | 3 keys | 3 | Sentiment mapping |
| Sentiment dictionary | `data\sentiment_dict.json` | JSON | 367 bytes | N/A | 3 keys | 3 | Analysis support |
| Plot colors | `plots\colors.tsv` | TSV | 898 bytes | 48 | 3 | N/A | Plotting support |
| Stress notebook | `dataset 2\stress-analysis-in-social-media.ipynb` | Notebook JSON | 0.124 MB | N/A | N/A | 2 referenced | External experiment |

---

## 15. Technical Explanation for Review

### Sentiment path

The application sends journal text to VADER and receives a compound score.
The backend applies:

```text
compound >= 0.05   -> positive
compound <= -0.05  -> negative
otherwise          -> neutral
```

The research pipeline uses a related three-class representation:

```text
positive
negative
ambiguous
```

The runtime `neutral` label and research `ambiguous` label are conceptually
related but not exactly identical.

### Emotion path

When enabled, the backend loads:

```text
j-hartmann/emotion-english-distilroberta-base
```

This is a pretrained transformer. The project does not train it.

When unavailable, the fallback maps:

- Positive sentiment to `joy`.
- Negative sentiment to `sadness`.
- Neutral sentiment to `neutral`.

### Research path

GoEmotions is used offline to:

1. Deduplicate training text.
2. Map 28 labels into seven broad emotion classes.
3. Map emotions into positive, negative, or ambiguous sentiment.
4. Evaluate the pretrained transformer.
5. Evaluate VADER against the processed test split.
6. Evaluate local personalization.

---

## 16. Fifteen Viva Questions and Answers

### 1. What is the primary dataset?

GoEmotions, a Reddit-based dataset with 27 emotion classes plus neutral.

### 2. How many records are in the raw CSV data?

211,225 annotation records across three raw CSV files, representing about
58,011 unique comment IDs.

### 3. Why are there more rows than comments?

The raw files contain multiple annotator records for the same comment.

### 4. What is the train/dev/test split?

43,410 train, 5,426 development, and 5,427 test records: approximately
80%/10%/10%.

### 5. How many columns are in the raw files?

37 columns: metadata plus 28 emotion indicator columns.

### 6. How many columns are in processed TSV files?

Three fields: text, labels, and ID.

### 7. Is the task single-label or multi-label?

The original GoEmotions task is multi-label. Several records contain multiple
emotion IDs.

### 8. What are the original classes?

Twenty-seven fine-grained emotions plus neutral.

### 9. Why create Ekman labels?

To simplify the emotion space into seven broader categories for analysis and
evaluation.

### 10. What sentiment classes are used?

Positive, negative, and ambiguous in the research pipeline.

### 11. Is VADER trained on GoEmotions?

No. VADER is a rule-based baseline and does not use these files at runtime.

### 12. Is the transformer trained by this project?

No. The project loads and evaluates a pretrained Hugging Face checkpoint.

### 13. Does the application load CSV data at startup?

No. It uses VADER, the optional transformer, local fallback logic, and SQLite.

### 14. Are user moods identical to AI emotion labels?

No. User moods include happy, sad, stressed, anxious, angry, calm, excited,
and neutral. AI emotions include anger, disgust, fear, joy, sadness, surprise,
and neutral.

### 15. What are the main limitations?

Class imbalance, multi-label annotations, exact text overlap across splits,
Reddit-to-journal domain mismatch, and lack of clinical validation.

---

## 17. Five-Minute Presentation Script

> My project is an AI-based mood tracking and journaling application. The main
> research dataset used during experimentation is GoEmotions, which contains
> Reddit comments annotated with 27 emotion categories and a neutral category.
>
> The raw data is stored in three CSV files. Together, they contain 211,225
> annotation records, but approximately 58,000 unique comments because multiple
> annotators can label the same comment.
>
> For experimentation, the project uses processed TSV files containing text,
> numeric labels, and comment IDs. The data is divided into training,
> development, and test sets using an approximately 80%, 10%, and 10% split.
>
> The original dataset is multi-label, meaning that one comment can have more
> than one emotion. To simplify evaluation, the project maps the original
> emotions into seven broader categories: anger, disgust, fear, joy, sadness,
> surprise, and neutral. It also creates a three-class sentiment
> representation: positive, negative, and ambiguous.
>
> The application has two analysis components. VADER calculates sentiment
> polarity using a compound score. An optional pretrained DistilRoBERTa model
> predicts emotion probabilities. This transformer was not trained by my
> project; it was loaded as a pretrained model and evaluated using the
> GoEmotions test split.
>
> If the transformer is disabled or unavailable, the application uses a local
> heuristic fallback based on sentiment. This allows the application to work
> without downloading a large model.
>
> The GoEmotions files are not loaded whenever the application starts. They are
> used offline by preprocessing and evaluation scripts. At runtime, the
> application stores journal entries and analysis results in a local SQLite
> database.
>
> The user's mood labels are different from the AI emotion labels. For
> example, the user can select happy, stressed, anxious, or calm, while the AI
> may predict joy, fear, sadness, or neutral. The application compares these
> through broader sentiment categories rather than treating them as exactly
> identical.
>
> The main limitations are class imbalance, multi-label emotion annotations,
> some exact text overlap across splits, and the difference between Reddit text
> and private journal writing. Therefore, the results demonstrate technical
> feasibility, but they should not be described as clinical validation or
> guaranteed mental-health prediction.

---

## 18. Final Verification Checklist

### Verified

- [x] Dataset folders were inspected recursively.
- [x] All three raw GoEmotions CSV files exist.
- [x] Train, development, and test TSV files exist.
- [x] Raw CSV schemas contain 37 columns.
- [x] Processed split schemas contain three fields.
- [x] Raw row counts were calculated.
- [x] Train/dev/test record counts were calculated.
- [x] File sizes were calculated in bytes, KB, and MB.
- [x] Missing values were checked.
- [x] Empty text entries were checked.
- [x] Duplicate rows were checked.
- [x] Duplicate IDs within splits were checked.
- [x] Cross-split ID overlap was checked.
- [x] Cross-split exact-text overlap was checked.
- [x] Original emotion categories were verified.
- [x] Ekman mapping was verified.
- [x] Sentiment mapping was verified.
- [x] VADER prediction distribution was verified.
- [x] Transformer prediction schema was verified.
- [x] Backend runtime dataset usage was traced to source code.
- [x] Transformer loading behavior was traced to source code.
- [x] VADER integration was traced to source code.
- [x] User mood labels were compared with AI emotion labels.
- [x] Dataset 2 was inspected and found to contain only a notebook.
- [x] Dataset 2 external Kaggle paths were identified.
- [x] GoEmotions source documentation and official URLs were identified.

### Still requiring external confirmation

- [ ] Exact revision/version of the downloaded GoEmotions files.
- [ ] Exact Kaggle publisher/version for Dataset 2.
- [ ] Whether Dataset 2 was used in the final application or only explored.
- [ ] Whether duplicate transformer prediction rows should be regenerated.
- [ ] A leakage-controlled re-split after removing exact text overlap.
- [ ] Performance on real private journal entries.
- [ ] Clinical validity, which is outside the scope of this dataset and app.

## Final conclusion

The project has a complete local GoEmotions-based research pipeline and a
separate local-first journaling application. The GoEmotions dataset supports
offline preprocessing and evaluation, but it is not loaded by the application
at runtime. The live application uses VADER, an optional pretrained emotion
model, local fallback logic, and SQLite persistence.

The dataset is Reddit-based and should not be described as a clinical,
journal-specific, or universally representative dataset.
