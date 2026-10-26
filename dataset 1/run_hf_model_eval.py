import os
import json
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, precision_recall_fscore_support

BASE_DIR = r"c:\Users\Jaideep\Desktop\inhouse project\dataset 1"
PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessed")
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

def main():
    print("=== Loading Model and Tokenizer ===")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device} in {time.time() - start_time:.2f}s")

    # Load Ekman labels
    with open(os.path.join(PREPROCESSED_DIR, "ekman_labels.txt"), "r", encoding="utf-8") as f:
        ekman_labels = [line.strip() for line in f if line.strip()]
    print(f"Ekman labels: {ekman_labels}")

    # Inspect model's id2label
    id2label = model.config.id2label
    print(f"Model id2label mapping: {id2label}")

    # Load test split
    test_file = os.path.join(PREPROCESSED_DIR, "test_ekman.tsv")
    test_df = pd.read_csv(test_file, sep="\t", header=None, names=["text", "labels", "id"])
    texts = test_df["text"].fillna("").astype(str).tolist()
    total_samples = len(texts)
    print(f"Evaluating {total_samples} test samples...")

    # Batch inference
    batch_size = 64
    all_probs = []

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            if (i // batch_size) % 15 == 0 or (i + batch_size) >= total_samples:
                processed = min(i + batch_size, total_samples)
                elapsed = time.time() - start_time
                print(f"Processed {processed}/{total_samples} samples ({processed/total_samples*100:.1f}%) in {elapsed:.1f}s")

    all_probs = np.vstack(all_probs)

    # Save predictions dataframe
    pred_data = {"text": texts}
    for model_idx, label_name in id2label.items():
        pred_data[label_name] = all_probs[:, int(model_idx)]
    
    pred_df = pd.DataFrame(pred_data)
    pred_output_path = os.path.join(PREPROCESSED_DIR, "hf_model_test_predictions.tsv")
    pred_df.to_csv(pred_output_path, sep="\t", index=False, encoding="utf-8")
    print(f"Saved model predictions to {pred_output_path}")

    # Step 7: Evaluation metrics against test_ekman.tsv ground truth
    # Ground truth binary matrix
    y_true = np.zeros((total_samples, len(ekman_labels)), dtype=int)
    for i, lbl_str in enumerate(test_df["labels"]):
        for idx in str(lbl_str).split(","):
            if idx.strip().isdigit():
                y_true[i, int(idx)] = 1

    # Pred binary matrix at threshold 0.3 (standard multi-label)
    THRESHOLD = 0.3
    y_pred_thresh = np.zeros_like(y_true)
    for j, label in enumerate(ekman_labels):
        if label in pred_df.columns:
            y_pred_thresh[:, j] = (pred_df[label].values >= THRESHOLD).astype(int)

    # Also evaluate argmax (single primary emotion prediction)
    y_pred_argmax = np.zeros_like(y_true)
    ordered_cols = [pred_df[label].values for label in ekman_labels]
    matrix_probs = np.column_stack(ordered_cols)
    primary_preds = np.argmax(matrix_probs, axis=1)
    for i, p_idx in enumerate(primary_preds):
        y_pred_argmax[i, p_idx] = 1

    report_thresh = classification_report(y_true, y_pred_thresh, target_names=ekman_labels, output_dict=True, zero_division=0)
    report_argmax = classification_report(y_true, y_pred_argmax, target_names=ekman_labels, output_dict=True, zero_division=0)

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT (Multi-label threshold = 0.3):")
    print("="*50)
    print(classification_report(y_true, y_pred_thresh, target_names=ekman_labels, zero_division=0))

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT (Argmax Primary Emotion):")
    print("="*50)
    print(classification_report(y_true, y_pred_argmax, target_names=ekman_labels, zero_division=0))

    eval_results = {
        "model_name": MODEL_NAME,
        "evaluation_samples": total_samples,
        "threshold_0_3_report": report_thresh,
        "argmax_primary_report": report_argmax
    }

    eval_output_path = os.path.join(PREPROCESSED_DIR, "evaluation_results.json")
    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved evaluation results to {eval_output_path}")

if __name__ == "__main__":
    main()
