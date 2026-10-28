# Experiment 3 Training Log

- Checkpoint: `j-hartmann/emotion-english-distilroberta-base`; revision `0e1cd914e3d46199ed785853e12b57304e04178b`
- Seed: `20261027`
- Device: `cpu`
- Train/validation/test groups: `43196/5399/5399`
- Batch size: `32`; learning rate: `2e-5`; max length: `128`; epochs: `1`
- Optimizer/loss: `AdamW` / `BCEWithLogitsLoss`
- Encoder frozen: `true`; trainable parameters: `595975`
- Selection: validation macro-F1; selected threshold: `0.29`; best epoch: `1`
- Training time: `3304.787` seconds

| Epoch | Global step | Mean loss | Validation macro-F1 | Validation micro-F1 |
|---:|---:|---:|---:|---:|
| 1 | 1350 | 0.327225 | 0.534816 | 0.616311 |
