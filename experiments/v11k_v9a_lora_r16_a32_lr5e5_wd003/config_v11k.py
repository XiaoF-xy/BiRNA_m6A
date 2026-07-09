from __future__ import annotations

from experiments.v11_lora_sweep_common import make_v11_lora_overrides


def get_overrides(dataset_name: str, seed: int) -> dict:
    return make_v11_lora_overrides(
        version_name="v11k_v9a_lora_r16_a32_lr5e5_wd003",
        dataset_name=dataset_name,
        seed=seed,
        lora_r=16,
        lora_alpha=32,
        lr=5e-5,
        weight_decay=0.03,
        purpose="Medium rank, lower learning rate, and stronger weight decay.",
    )
