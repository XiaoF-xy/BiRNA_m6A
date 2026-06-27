from __future__ import annotations


def get_overrides(dataset_name: str, seed: int) -> dict:
    return {
        "experiment": {
            "version_name": "v4_moe_fusion",
            "output_suffix": "moe_fusion",
            "description": "Reserved MoE fusion version for later multi-view feature fusion experiments.",
            "changes_from_previous": "Planned extension from LoRA/feature versions to MoE fusion.",
            "notes": "Scaffold only; model implementation is intentionally not enabled yet.",
        },
        "model": {
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ["Wqkv"],
            "freeze_backbone": True,
            "use_multi_feature": True,
            "use_cnn": False,
            "use_bilstm": False,
            "use_moe": True,
            "use_attention": True,
            "use_bpe": True,
            "use_nuc": True,
        },
        "data": {
            "active_dataset": dataset_name,
            "dataset_name": dataset_name,
            "use_one_hot": True,
            "use_ncp": True,
            "use_eiip": True,
            "use_enac": True,
        },
        "training": {
            "seed": seed,
            "num_train_epochs": 20,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 32,
            "learning_rate": 1e-4,
            "folds": 5,
            "max_length": 64,
            "keep_best_model": False,
        },
    }

