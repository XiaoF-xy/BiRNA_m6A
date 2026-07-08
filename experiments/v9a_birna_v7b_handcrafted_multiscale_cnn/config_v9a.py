from __future__ import annotations


def get_overrides(dataset_name: str, seed: int) -> dict:
    return {
        "experiment": {
            "version_name": "v9a_birna_v7b_handcrafted_multiscale_cnn",
            "description": (
                "BiRNA-BERT v7b NUC FiLM center-window CNN with a 12-channel "
                "handcrafted physicochemical multi-scale CNN branch."
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
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ["Wqkv"],
            "freeze_backbone": True,
            "use_handcrafted_features": True,
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
            "lr": 1e-4,
            "folds": 1,
            "max_length": 64,
            "keep_best_model": False,
        },
    }

