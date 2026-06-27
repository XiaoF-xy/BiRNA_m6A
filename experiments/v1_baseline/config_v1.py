from __future__ import annotations


def get_overrides(dataset_name: str, seed: int) -> dict:
    return {
        "experiment": {
            "version_name": "v1_baseline",
            "description": "BiRNA-BERT NUC frozen baseline with mean + center pooling and MLP classifier.",
        },
        "model": {
            "use_lora": False,
            "freeze_backbone": True,
        },
        "data": {
            "dataset_name": dataset_name,
            "sequence_length": 41,
        },
        "training": {
            "seed": seed,
            "epochs": 20,
            "batch_size": 32,
            "lr": 1e-4,
            "folds": 5,
            "max_length": 64,
            "keep_best_model": False,
        },
    }
