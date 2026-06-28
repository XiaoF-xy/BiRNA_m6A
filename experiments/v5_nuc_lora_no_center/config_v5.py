from __future__ import annotations


def get_overrides(dataset_name: str, seed: int) -> dict:
    return {
        "experiment": {
            "version_name": "v5_nuc_lora_no_center",
            "description": "BiRNA-BERT NUC + LoRA ablation without explicit center-token pooling.",
        },
        "model": {
            "use_center_pooling": False,
            "use_bpe_view": False,
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ["Wqkv"],
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

