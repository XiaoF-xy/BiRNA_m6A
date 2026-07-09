from __future__ import annotations


def make_v11_lora_overrides(
    version_name: str,
    dataset_name: str,
    seed: int,
    lora_r: int,
    lora_alpha: int,
    lr: float,
    weight_decay: float,
    purpose: str,
    lora_dropout: float = 0.05,
) -> dict:
    return {
        "experiment": {
            "version_name": version_name,
            "description": (
                "v11 LoRA sweep based on v9a: BiRNA-BERT v7b NUC FiLM center-window CNN "
                "plus handcrafted physicochemical multi-scale CNN branch. "
                f"{purpose}"
            ),
        },
        "model": {
            "use_center_pooling": False,
            "use_bpe_view": False,
            "use_film": True,
            "film_global_view": "nuc",
            "film_nuc_pooling": "center_cnn_mean",
            "local_window_radius": 3,
            "cnn_kernel_sizes": [3, 5, 7],
            "use_lora": True,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": ["Wqkv"],
            "freeze_backbone": True,
            "use_handcrafted_features": True,
            "handcrafted_feature_names": ["onehot", "ncp", "eiip", "enac"],
            "handcrafted_cnn_channels": 64,
            "handcrafted_output_dim": 128,
        },
        "data": {
            "dataset_name": dataset_name,
            "sequence_length": 41,
        },
        "training": {
            "seed": seed,
            "epochs": 20,
            "batch_size": 32,
            "lr": lr,
            "weight_decay": weight_decay,
            "folds": 1,
            "max_length": 64,
            "keep_best_model": False,
        },
    }
