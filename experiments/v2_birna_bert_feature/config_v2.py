from __future__ import annotations


def get_overrides(dataset_name: str, seed: int) -> dict:
    return {
        "experiment": {
            "version_name": "v2_birna_bert_feature",
            "output_suffix": "birna_bert_feature",
            "description": "Reserved BiRNA-BERT feature extraction version for adding extra sequence features.",
            "changes_from_previous": "Planned extension from NUC-only baseline to multi-feature inputs.",
            "notes": "Scaffold only; implementation should be added before running as a formal result.",
        },
        "model": {
            "use_lora": False,
            "freeze_backbone": True,
            "use_multi_feature": True,
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
            "use_one_hot": True,
            "use_ncp": False,
            "use_eiip": False,
            "use_enac": False,
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

