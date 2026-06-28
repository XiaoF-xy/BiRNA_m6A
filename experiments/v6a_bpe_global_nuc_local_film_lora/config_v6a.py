from __future__ import annotations


def get_overrides(dataset_name: str, seed: int) -> dict:
    return {
        "experiment": {
            "version_name": "v6a_bpe_global_nuc_local_film_lora",
            "description": "BiRNA-BERT BPE global -> FiLM -> NUC local window with LoRA.",
        },
        "model": {
            "use_center_pooling": False,
            "use_bpe_view": True,
            "use_film": True,
            "film_global_view": "bpe",
            "local_window_radius": 3,
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
            "folds": 1,
            "max_length": 64,
            "keep_best_model": False,
        },
    }
