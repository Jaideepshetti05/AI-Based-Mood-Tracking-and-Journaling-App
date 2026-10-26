import os
import json
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = r"c:\Users\Jaideep\Desktop\inhouse project\dataset 1"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "preprocessed")
os.makedirs(OUT_DIR, exist_ok=True)

print("--- Step 1: Deduplicate train.tsv ---")
train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.tsv"), sep="\t", header=None, names=["text", "labels", "id"])
before_cnt = len(train_raw)
train_dedup = train_raw.drop_duplicates(subset=["text"], keep="first").copy()
after_cnt = len(train_dedup)
print(f"Train raw count: {before_cnt}, Deduplicated count: {after_cnt}, Removed: {before_cnt - after_cnt}")
train_dedup.to_csv(os.path.join(OUT_DIR, "train_dedup.tsv"), sep="\t", header=False, index=False, encoding="utf-8")

print("\n--- Step 2: Remap to Ekman-6 + neutral ---")
# Load original emotions (28 labels, 0-27)
with open(os.path.join(DATA_DIR, "emotions.txt"), "r", encoding="utf-8") as f:
    raw_emotions = [line.strip() for line in f if line.strip()]
raw_idx2emotion = {i: e for i, e in enumerate(raw_emotions)}

# Load ekman mapping
with open(os.path.join(DATA_DIR, "ekman_mapping.json"), "r", encoding="utf-8") as f:
    ekman_mapping = json.load(f)

# Define canonical ekman list: anger, disgust, fear, joy, sadness, surprise, neutral
ekman_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
ekman_label2idx = {label: i for i, label in enumerate(ekman_labels)}

# Write ekman_labels.txt
with open(os.path.join(OUT_DIR, "ekman_labels.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(ekman_labels) + "\n")

# Map raw emotion string to ekman emotion string
raw2ekman = {}
for ek, raw_list in ekman_mapping.items():
    for r in raw_list:
        raw2ekman[r] = ek
raw2ekman["neutral"] = "neutral"

def remap_to_ekman(labels_str):
    raw_indices = [int(x.strip()) for x in str(labels_str).split(",") if x.strip().isdigit()]
    mapped_indices = set()
    for idx in raw_indices:
        raw_emo = raw_idx2emotion.get(idx)
        if raw_emo and raw_emo in raw2ekman:
            ek = raw2ekman[raw_emo]
            mapped_indices.add(ekman_label2idx[ek])
    if not mapped_indices:
        mapped_indices.add(ekman_label2idx["neutral"])
    return ",".join(str(i) for i in sorted(mapped_indices))

splits = [
    ("train_dedup", train_dedup, "train_ekman.tsv"),
    ("dev", pd.read_csv(os.path.join(DATA_DIR, "dev.tsv"), sep="\t", header=None, names=["text", "labels", "id"]), "dev_ekman.tsv"),
    ("test", pd.read_csv(os.path.join(DATA_DIR, "test.tsv"), sep="\t", header=None, names=["text", "labels", "id"]), "test_ekman.tsv")
]

for name, df, filename in splits:
    mapped_df = df.copy()
    mapped_df["labels"] = mapped_df["labels"].apply(remap_to_ekman)
    mapped_df.to_csv(os.path.join(OUT_DIR, filename), sep="\t", header=False, index=False, encoding="utf-8")
    print(f"Wrote {filename}: {len(mapped_df)} rows")

print("\n--- Step 3: Remap to 3-Class Sentiment ---")
# Positive, negative, ambiguous (neutral -> ambiguous)
with open(os.path.join(DATA_DIR, "sentiment_mapping.json"), "r", encoding="utf-8") as f:
    sentiment_mapping = json.load(f)

sentiment_labels = ["positive", "negative", "ambiguous"]
sentiment_label2idx = {label: i for i, label in enumerate(sentiment_labels)}

# Write sentiment_labels.txt
with open(os.path.join(OUT_DIR, "sentiment_labels.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(sentiment_labels) + "\n")

raw2sent = {}
for s, raw_list in sentiment_mapping.items():
    for r in raw_list:
        raw2sent[r] = s
raw2sent["neutral"] = "ambiguous"

def remap_to_sentiment(labels_str):
    raw_indices = [int(x.strip()) for x in str(labels_str).split(",") if x.strip().isdigit()]
    mapped_indices = set()
    for idx in raw_indices:
        raw_emo = raw_idx2emotion.get(idx)
        if raw_emo and raw_emo in raw2sent:
            s = raw2sent[raw_emo]
            mapped_indices.add(sentiment_label2idx[s])
    if not mapped_indices:
        mapped_indices.add(sentiment_label2idx["ambiguous"])
    return ",".join(str(i) for i in sorted(mapped_indices))

sentiment_files = [
    (train_dedup, "train_sentiment.tsv"),
    (splits[1][1], "dev_sentiment.tsv"),
    (splits[2][1], "test_sentiment.tsv")
]

for df, filename in sentiment_files:
    mapped_df = df.copy()
    mapped_df["labels"] = mapped_df["labels"].apply(remap_to_sentiment)
    mapped_df.to_csv(os.path.join(OUT_DIR, filename), sep="\t", header=False, index=False, encoding="utf-8")
    print(f"Wrote {filename}: {len(mapped_df)} rows")

print("\n--- Step 4: Compute Ekman Class Weights ---")
train_ekman = pd.read_csv(os.path.join(OUT_DIR, "train_ekman.tsv"), sep="\t", header=None, names=["text", "labels", "id"])
primary_labels = train_ekman["labels"].apply(lambda x: int(str(x).split(",")[0])).values
weights = compute_class_weight("balanced", classes=np.arange(len(ekman_labels)), y=primary_labels)
weight_dict = {ekman_labels[i]: round(float(weights[i]), 4) for i in range(len(ekman_labels))}
print("Computed Ekman Class Weights:", json.dumps(weight_dict, indent=2))
with open(os.path.join(OUT_DIR, "ekman_class_weights.json"), "w", encoding="utf-8") as f:
    json.dump(weight_dict, f, indent=2)

print("\n--- Step 5: VADER Baseline on Test Split ---")
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.tsv"), sep="\t", header=None, names=["text", "labels", "id"])
analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "positive", score
    elif score <= -0.05:
        return "negative", score
    else:
        return "ambiguous", score

res = [get_vader_sentiment(t) for t in test_df["text"].tolist()]
vader_pred_df = test_df.copy()
vader_pred_df["vader_sentiment"] = [r[0] for r in res]
vader_pred_df["vader_compound"] = [r[1] for r in res]
vader_pred_df.to_csv(os.path.join(OUT_DIR, "vader_test_predictions.tsv"), sep="\t", index=False, encoding="utf-8")
print("VADER predictions saved. Sentiment distribution:")
print(vader_pred_df["vader_sentiment"].value_counts())

# Evaluate VADER against test_sentiment.tsv ground truth
test_sent_df = pd.read_csv(os.path.join(OUT_DIR, "test_sentiment.tsv"), sep="\t", header=None, names=["text", "labels", "id"])
vader_y_pred = [sentiment_label2idx[s] for s in vader_pred_df["vader_sentiment"]]

# Build binary ground truth matrix for multi-label sentiment
y_true_binary = np.zeros((len(test_sent_df), len(sentiment_labels)), dtype=int)
for i, l_str in enumerate(test_sent_df["labels"]):
    for idx_s in str(l_str).split(","):
        if idx_s.strip().isdigit():
            y_true_binary[i, int(idx_s)] = 1

# VADER predicted binary matrix (one-hot)
vader_y_pred_binary = np.zeros((len(test_sent_df), len(sentiment_labels)), dtype=int)
for i, pred_idx in enumerate(vader_y_pred):
    vader_y_pred_binary[i, pred_idx] = 1

vader_report = classification_report(y_true_binary, vader_y_pred_binary, target_names=sentiment_labels, output_dict=True, zero_division=0)
print("\nVADER Classification Report against GoEmotions Test Split:")
print(classification_report(y_true_binary, vader_y_pred_binary, target_names=sentiment_labels, zero_division=0))

with open(os.path.join(OUT_DIR, "vader_evaluation_results.json"), "w", encoding="utf-8") as f:
    json.dump(vader_report, f, indent=2)

print("\n=== Steps 1-5 Completed Successfully! ===")
