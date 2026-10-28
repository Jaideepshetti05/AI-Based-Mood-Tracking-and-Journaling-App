# Experiment 4 Methodology

Experiment 4 was designed to compare the existing pretrained Model A, the
Experiment 3 classification-head Model B, and a fully fine-tuned Model C on
the same grouped train/validation/frozen-test splits.

The existing splits under `reports/grouped_text_benchmark/splits/` were reused
without modification. The task remains seven-label multi-label classification
with sigmoid outputs and `BCEWithLogitsLoss`. Model C would have started from
the cached `j-hartmann/emotion-english-distilroberta-base` checkpoint, with
all encoder and classifier parameters trainable, validation macro-F1 used for
checkpoint and threshold selection, and the test set evaluated only after
selection.

The required preflight measured 82,123,783 parameters. A conservative fp32
lower bound is approximately 0.306 GiB for parameters plus 0.612 GiB for
AdamW's two state tensors, or 0.918 GiB before activations, temporary tensors,
Python processes, and dataloaders. The host had approximately 0.82 GiB
available RAM at the actual preflight point, no CUDA device, and a 328,507,660-byte checkpoint. The prior
head-only experiment already reached approximately 1 GiB working set and took
55 minutes for one epoch. Full training was therefore not started; claiming a
Model C result would be scientifically invalid.

Model A and Model B results are retained from Experiment 3, not rerun or
changed. A valid future run requires a GPU or substantially more RAM, mixed
precision where supported, gradient accumulation, and the same frozen test
governance.
