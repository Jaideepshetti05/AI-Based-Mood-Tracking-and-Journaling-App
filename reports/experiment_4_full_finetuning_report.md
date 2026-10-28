# Experiment 4: Full Transformer Fine-Tuning

## 1. Executive summary

The planned full-transformer fine-tuning experiment could not be run
responsibly on the measured host. Model C was **not trained**, no test
predictions were generated for it, and no Model C performance claim is made.

The preflight measured 82,123,783 parameters, no GPU, and approximately 0.82
GiB available RAM at the actual preflight point. A conservative fp32 lower bound for parameters, gradients,
and AdamW's two state tensors is approximately 0.918 GiB, before activations,
temporary tensors, Python overhead, and dataloaders. The prior head-only
experiment already used approximately 1 GiB working memory. Starting a full
run would risk memory exhaustion and would not constitute a reliable
experiment.

The valid results available for the preserved frozen test are:

| Model | Status | Threshold | Macro-F1 | Micro-F1 | Weighted-F1 | Exact-match | Hamming loss |
|---|---|---:|---:|---:|---:|---:|---:|
| A: pretrained baseline | Completed | 0.35 | 0.4284 | 0.4732 | 0.4908 | 0.3988 | 0.1614 |
| B: classification-head transfer learning | Completed | 0.29 | **0.5405** | **0.6171** | **0.6148** | 0.4058 | **0.1332** |
| C: full transformer fine-tuning | **Not trained** | — | — | — | — | — | — |

Model B's improvement over Model A remains the Experiment 3 result. It cannot
be attributed to full-transformer fine-tuning.

## 2. Research question and hypotheses

The question was whether unfreezing the complete transformer encoder improves
multi-label emotion classification over Models A and B. The planned primary
hypothesis was that Model C would improve macro-F1 on the same frozen test.
Because Model C was not trained, this hypothesis remains untested.

## 3. Dataset and evaluation design

The existing grouped splits were reused conceptually and were not recreated:
43,196 train groups, 5,399 validation groups, and 5,399 frozen test groups.
The seven labels are multi-label and were preserved as multi-hot targets.
Experiment 3 independently verified zero exact-text overlap and keyed
predictions with IDs and SHA-256 text hashes.

The frozen test was not touched for Model C training or selection. Existing
Model A and B artifacts were not overwritten.

## 4. Baseline and Model B

Model A is `j-hartmann/emotion-english-distilroberta-base` with threshold
0.35. Model B uses the same checkpoint and a separately trained
multi-label classification head with `BCEWithLogitsLoss`, threshold 0.29,
selected using validation macro-F1. Their complete per-class metrics remain in
`reports/experiment_3_results.json` and the Experiment 3 report.

## 5. Full fine-tuning design

The intended Model C design was full encoder plus classifier unfreezing,
`BCEWithLogitsLoss`, AdamW, validation-based checkpoint selection, and
validation-only threshold selection. No class weighting, threshold tuning, or
hyperparameter decision would have used test labels.

The design was not executed because the resource preflight failed. This is an
explicit infeasibility result, not a substitute Model C experiment.

## 6. Resource assessment

| Resource | Measurement |
|---|---:|
| Total parameters | 82,123,783 |
| Encoder parameters | 81,527,808 |
| Classifier parameters | 595,975 |
| fp32 parameter memory estimate | 0.306 GiB |
| AdamW state estimate | 0.612 GiB |
| Parameters + gradients + AdamW lower bound | 0.918 GiB |
| Available RAM at preflight | approximately 0.82 GiB |
| GPU/CUDA | unavailable |
| Checkpoint size | 328,507,660 bytes |

The lower bound excludes activations, forward/backward workspaces, tokenizer
and dataloader allocations, operating-system memory, and checkpoint copies.
Consequently, a full training run was not feasible with acceptable reliability.

## 7. Verification

The existing Experiment 3 verification passed: all frozen split overlaps were
zero, Model A and B prediction IDs and text hashes aligned, metrics recomputed
exactly, and the test set contained identical examples for both models.
Experiment 4 did not create Model C artifacts, so there are no Model C
predictions to align or metrics to recompute.

## 8. Limitations and unresolved issues

The full-transformer question remains unresolved. The result is limited to
the Reddit-derived GoEmotions benchmark and does not establish journal-domain
performance. Only one head-only training configuration and one seed were
completed. Model C needs a separate GPU or high-memory run with multiple
seeds, early stopping, checkpoint retention, and the frozen test evaluated
once after all decisions are finalized.

## 9. Recommended alternative

Use a GPU or a host with substantially more RAM. Prefer mixed precision,
gradient accumulation, conservative batch size, and activation checkpointing.
Train only on the existing training split, select checkpoints and threshold on
validation, then evaluate Model C once on the frozen test. Record the exact
checkpoint revision, hardware, peak memory, optimizer, loss, and all
prediction hashes.

## 10. Reproduction commands

```powershell
python research\experiment_4\preflight_full_finetuning.py
```

The command records the measured feasibility result in
`reports/experiment_4_preflight.json`. A full training command is intentionally
not provided as executable because the preflight demonstrates that it is not
safe on this host.

## 11. Created files

- `reports/experiment_4_full_finetuning_report.md`
- `reports/experiment_4_methodology.md`
- `reports/experiment_4_results.json`
- `reports/experiment_4_model_comparison.csv`
- `reports/experiment_4_training_log.md`
- `reports/experiment_4_preflight.json`
- `research/experiment_4/preflight_full_finetuning.py`
