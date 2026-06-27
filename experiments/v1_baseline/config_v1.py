from __future__ import annotations


def get_overrides(dataset_name: str, seed: int) -> dict:
    return {
        "experiment": {
            "version_name": "v1_baseline",
            "output_suffix": "nuc_frozen",
            "description": "BiRNA-BERT NUC frozen baseline with mean + center pooling and MLP classifier.",
            "changes_from_previous": "Initial reproducible baseline.",
            "notes": "Backbone frozen; only classifier is trained.",
        },
        "model": {
            "use_lora": False,
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

