from __future__ import annotations


def get_overrides(dataset_name: str, seed: int) -> dict:
    return {
        "experiment": {
            "version_name": "v3_birna_bert_lora",
            "output_suffix": "nuc_lora",
            "description": "BiRNA-BERT NUC + LoRA on Wqkv with mean + center pooling and MLP classifier.",
            "changes_from_previous": "Adds LoRA adapter training to the frozen NUC baseline.",
            "notes": "Backbone base weights stay frozen; LoRA adapter and classifier are trained.",
        },
        "model": {
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ["Wqkv"],
            "lora_save_adapter_only": True,
            "lora_merge_weights": False,
            "lora_resume_adapter": False,
            "freeze_backbone": True,
            "use_multi_feature": False,
            "use_cnn": False,
            "use_bilstm": False,
            "use_moe": False,
            "use_attention": False,
            "use_bpe": False,
            "use_nuc": True,
        },
        "data": {
            "active_dataset": dataset_name,
            "dataset_name": dataset_name,
            "sequence_length": 41,
            "sequence_window": 41,
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

