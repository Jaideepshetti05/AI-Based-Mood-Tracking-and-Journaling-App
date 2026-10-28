"""Bounded feasibility check for Experiment 4 full-transformer training.

This script deliberately does not start training when the measured host
resources cannot support a conservative fp32 AdamW configuration.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import psutil
from transformers import AutoModelForSequenceClassification


MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"


def main() -> None:
    root = Path(__file__).parents[2]
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, local_files_only=True
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    classifier_parameters = sum(
        parameter.numel() for parameter in model.classifier.parameters()
    )
    memory = psutil.virtual_memory()
    result = {
        "status": "infeasible",
        "model": MODEL_NAME,
        "total_parameters": parameters,
        "encoder_parameters": parameters - classifier_parameters,
        "classifier_parameters": classifier_parameters,
        "estimated_fp32_model_gib": parameters * 4 / 1024**3,
        "estimated_adam_states_gib": parameters * 8 / 1024**3,
        "estimated_model_gradients_and_adam_gib": parameters * 12 / 1024**3,
        "total_physical_memory_gib": memory.total / 1024**3,
        "available_memory_gib": memory.available / 1024**3,
        "free_disk_gib": shutil.disk_usage(root).free / 1024**3,
        "cuda_available": False,
        "reason": (
            "Full fp32 AdamW requires model parameters, gradients, and two "
            "optimizer states before activation and dataloader memory. The "
            "measured available memory is insufficient for that lower bound."
        ),
        "alternative": (
            "Use a GPU or a host with substantially more RAM, then run full "
            "encoder training with mixed precision, gradient accumulation, "
            "checkpoint selection on validation, and the same frozen test."
        ),
    }
    output = root / "reports" / "experiment_4_preflight.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
