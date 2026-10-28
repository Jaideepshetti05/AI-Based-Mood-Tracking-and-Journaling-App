# Experiment 3: Emotion Model Training, Evaluation, and Baseline Comparison

## 1. Executive summary

Model B (a separately saved, one-epoch multi-label classification-head transfer-learning model) was evaluated on exactly the same frozen 5,399-example grouped test set as Model A. Model B improved macro-F1 from **0.4284** to **0.5405** (+0.1121), micro-F1 from **0.4732** to **0.6171**, and weighted-F1 from **0.4908** to **0.6148**. Exact-match accuracy increased only from **0.3988** to **0.4058**.

The paired bootstrap 95% interval for the macro-F1 difference was [0.0965, 0.1281], excluding zero. This supports an improvement on this frozen benchmark, not necessarily on journal text. The model encoder was frozen because the host had insufficient available memory for full-transformer AdamW state; therefore this is head-only transfer learning, not full encoder fine-tuning.

## 2. Research question and hypotheses

**Question:** Does supervised training on the grouped training split improve the pretrained checkpoint on the same held-out grouped test set?

**H1:** Model B improves macro-F1 and related multi-label metrics over Model A. **H0:** it does not improve them. The primary comparison is Model B macro-F1 minus Model A macro-F1.

## 3. Dataset and frozen splits

The existing grouped splits were reused without recreation: train **43196**, validation **5399**, and test **5399** text groups. The source task has seven multi-label classes: `anger, disgust, fear, joy, sadness, surprise, neutral`. Exact-text overlap was independently verified as zero for all split pairs. Labels were preserved as multi-hot vectors; no single-label conversion was performed.

## 4. Baseline configuration

Model A is `j-hartmann/emotion-english-distilroberta-base`, cached revision `0e1cd914e3d46199ed785853e12b57304e04178b`, with the existing validation-selected threshold **0.35**. Its original grouped benchmark test metrics were preserved and independently recomputed from the saved predictions.

## 5. Experimental architecture and training

Model B starts from the same checkpoint and retains its seven-output classification head. `BCEWithLogitsLoss` is used for multi-label supervision, with sigmoid probabilities at evaluation. AdamW used learning rate `2e-5`, weight decay `0.01`, batch size `32`, maximum length `128`, one epoch, seed `20261027`, and 1,350 optimizer steps. No class weighting was used. The encoder was frozen and only **595,975** head parameters were optimized. The best checkpoint was selected by validation macro-F1.

Training took **55.08 minutes** on CPU. The saved best model is under `reports/experiment_3/best_model/`.

## 6. Training history and selection

| Epoch | Steps | Mean loss | Validation threshold | Validation macro-F1 | Validation micro-F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,350 | 0.3272 | 0.29 | 0.5348 | 0.6163 |

Threshold candidates 0.10 through 0.90 were evaluated on validation only. Model B selected **0.29**. Model A retained **0.35** by design. The frozen test labels were not accessed for training, checkpoint selection, or threshold selection.

## 7. Evaluation methodology

Both systems were scored on identical text IDs and reference vectors. Primary metrics are macro/micro precision, recall, and F1, weighted-F1, exact-match accuracy, and Hamming loss. Prediction files contain the split ID and SHA-256 text hash. Paired sample-level bootstrap used 2,000 replicates and seed 424242. CPU test inference was 105.80 seconds (19.60 ms/example) for Model A and 186.29 seconds (34.50 ms/example) for Model B.

## 8. Complete comparison

| Metric | Model A baseline | Model B experimental | B - A |
|---|---:|---:|---:|
| macro_precision | 0.4503 | 0.5028 | 0.0525 |
| macro_recall | 0.5338 | 0.5900 | 0.0562 |
| macro_f1 | 0.4284 | 0.5405 | 0.1121 |
| micro_precision | 0.4812 | 0.5585 | 0.0773 |
| micro_recall | 0.4654 | 0.6894 | 0.2240 |
| micro_f1 | 0.4732 | 0.6171 | 0.1439 |
| weighted_f1 | 0.4908 | 0.6148 | 0.1240 |
| exact_match_accuracy | 0.3988 | 0.4058 | 0.0070 |
| hamming_loss | 0.1614 | 0.1332 | -0.0282 |

## 9. Per-class results

| Class | Model | Precision | Recall | F1 | Support | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| anger | A | 0.4374 | 0.3538 | 0.3912 | 701 | 319 | 453 |
| anger | B | 0.4410 | 0.5435 | 0.4869 | 701 | 483 | 320 |
| disgust | A | 0.1282 | 0.8641 | 0.2233 | 103 | 605 | 14 |
| disgust | B | 0.4414 | 0.4757 | 0.4579 | 103 | 62 | 54 |
| fear | A | 0.3333 | 0.6739 | 0.4460 | 92 | 124 | 30 |
| fear | B | 0.4746 | 0.6087 | 0.5333 | 92 | 62 | 36 |
| joy | A | 0.9026 | 0.3901 | 0.5447 | 2161 | 91 | 1318 |
| joy | B | 0.6742 | 0.8121 | 0.7368 | 2161 | 848 | 406 |
| sadness | A | 0.4986 | 0.4554 | 0.4761 | 404 | 185 | 220 |
| sadness | B | 0.4823 | 0.4728 | 0.4775 | 404 | 205 | 213 |
| surprise | A | 0.3873 | 0.4079 | 0.3973 | 657 | 424 | 389 |
| surprise | B | 0.4875 | 0.4734 | 0.4803 | 657 | 327 | 346 |
| neutral | A | 0.4644 | 0.5914 | 0.5203 | 1767 | 1205 | 722 |
| neutral | B | 0.5185 | 0.7436 | 0.6110 | 1767 | 1220 | 453 |

## 10. Error analysis

Model B materially reduced Model A's extreme `disgust` false-positive problem (FP 605 to 62) while retaining a similar recall profile. `joy` recall improved from 0.3901 to 0.8121, but precision fell from 0.9026 to 0.6742; the gain is a recall/precision tradeoff. `fear` F1 improved from 0.4460 to 0.5333 despite its low support. `anger` and `neutral` F1 also improved. `sadness` changed little and `surprise` improved modestly. These are aggregate label errors, not causal linguistic explanations.

## 11. Statistical uncertainty

| Difference | Estimate | 95% paired bootstrap interval |
|---|---:|---:|
| Macro-F1 | 0.1121 | [0.0965, 0.1281] |
| Micro-F1 | 0.1439 | [0.1333, 0.1554] |
| Weighted-F1 | 0.1240 | [0.1126, 0.1356] |
| Exact-match accuracy | 0.0070 | [-0.0069, 0.0219] |

## 12. Independent verification

The independent verifier passed: split overlap is zero; all four prediction artifacts have matching row counts, IDs, and text hashes; probabilities are finite and in [0,1]; the seven-label mapping is verified; test rows are identical; and metrics recompute exactly from saved predictions. Model B's script reads train, validation, and test files for evaluation, but optimization uses only the training loader, checkpoint selection uses validation, and test labels are not used until final evaluation.

## 13. Limitations

This is a Reddit-derived GoEmotions benchmark, not a consented journal-domain evaluation. Exact grouping does not remove semantic near-duplicates. Rare-class intervals are less stable. Only one seed and one epoch were run. The encoder was frozen due to memory constraints, so results do not establish that full-model fine-tuning would help. The local checkpoint revision is recorded, but the upstream model repository may change independently. No production model or application behavior was modified.

## 14. Research findings and conclusion

On the frozen grouped benchmark, Model B improved macro-F1, micro-F1, weighted-F1, Hamming loss, and per-class performance for the highlighted minority and frequent classes. The paired intervals support the F1 gains, while the exact-match interval includes zero, so the improvement should be described as label-level rather than universal across every metric. The experiment supports H1 for the tested head-only transfer-learning configuration.

## 15. Next steps

Run a memory-capable full-encoder fine-tuning study with multiple seeds, class-weighting ablations, and early stopping; preserve the same frozen test; and validate on a consented, de-identified journal-domain set before considering any production change.

## 16. Reproduction commands

```powershell
python research\experiment_3\train_and_evaluate.py --benchmark-dir reports\grouped_text_benchmark --output-dir reports\experiment_3 --epochs 1 --batch-size 32 --learning-rate 2e-5 --max-length 128 --bootstrap-replicates 2000
python research\experiment_3\verify_and_report.py
```

## 17. Created and modified files

Created: `reports/experiment_3_model_comparison_report.md`, `reports/experiment_3_methodology.md`, `reports/experiment_3_results.json`, `reports/experiment_3_model_comparison.csv`, `reports/experiment_3_error_analysis.md`, `reports/experiment_3_training_log.md`, `research/experiment_3/verify_and_report.py`, and artifacts under `reports/experiment_3/`.

Modified: `research/experiment_3/train_and_evaluate.py` only. Production application files, source datasets, and frozen split files were not modified.
