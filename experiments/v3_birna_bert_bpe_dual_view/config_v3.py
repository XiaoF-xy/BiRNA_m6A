from __future__ import annotations


def get_overrides(dataset_name: str, seed: int) -> dict:
    return {
        "experiment": {
            "version_name": "v3_birna_bert_bpe_dual_view",
            "description": "BiRNA-BERT frozen dual view: NUC mean+center pooling plus BPE mask-aware mean pooling.",
        },
        "model": {
            "use_bpe_view": True,
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
