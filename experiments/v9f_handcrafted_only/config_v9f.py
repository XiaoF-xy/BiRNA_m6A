from __future__ import annotations


def get_overrides(dataset_name: str, seed: int) -> dict:
    return {
        "experiment": {
            "version_name": "v9f_handcrafted_only",
            "description": "Handcrafted-only baseline with ONEHOT/NCP/EIIP/ENAC multi-scale CNN.",
        },
        "model": {
            "use_center_pooling": False,
            "use_bpe_view": False,
            "use_film": False,
            "film_global_view": "nuc",
            "film_nuc_pooling": "center_cnn_mean",
            "local_window_radius": 3,
            "cnn_kernel_sizes": [3, 5, 7],
            "use_lora": False,
            "freeze_backbone": False,
            "use_handcrafted_features": True,
            "handcrafted_feature_names": ["onehot", "ncp", "eiip", "enac"],
            "handcrafted_only": True,
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

