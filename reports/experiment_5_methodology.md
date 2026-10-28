# Experiment 5 Methodology

The frozen grouped splits were reused unchanged. Model B's saved artifacts
were audited by row count, IDs, SHA-256 text hashes, checkpoint contents,
training-log presence, and split overlap. The controlled variant froze the
encoder and trained the seven-label head with BCEWithLogitsLoss, AdamW,
batch size 32, learning rate 2e-5, one epoch, and square-root
inverse-frequency positive weights computed from training support. Thresholds
were selected on validation macro-F1. The frozen test was evaluated only after
selection. Three seeds (20261027, 20261028, 20261029) were run. Paired
bootstrap comparisons used 2,000 resamples with seed 424242.
