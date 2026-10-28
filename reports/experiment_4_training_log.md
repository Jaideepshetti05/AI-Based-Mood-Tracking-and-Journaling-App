# Experiment 4 Training Log

## Preflight

- Status: **not started; infeasible on measured host**
- Checkpoint: `j-hartmann/emotion-english-distilroberta-base`
- Total parameters: `82,123,783`
- Encoder parameters: `81,527,808`
- Classification-head parameters: `595,975`
- Estimated fp32 model memory: `0.306 GiB`
- Estimated AdamW states: `0.612 GiB`
- Estimated parameters + gradients + AdamW states: `0.918 GiB`
- Available RAM at preflight: approximately `1.27 GiB`
- CUDA: unavailable
- Checkpoint size: `328,507,660` bytes

No optimizer was created, no training batch was executed, and no Model C
checkpoint or test predictions were produced. This is an intentional
resource-safety stop, not a failed or successful training result.

## Reproduction

```powershell
python research\experiment_4\preflight_full_finetuning.py
```
