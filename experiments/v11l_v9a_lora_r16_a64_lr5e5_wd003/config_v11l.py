from __future__ import annotations

from experiments.v11_lora_sweep_common import make_v11_lora_overrides


def get_overrides(dataset_name: str, seed: int) -> dict:
    return make_v11_lora_overrides(
        version_name="v11l_v9a_lora_r16_a64_lr5e5_wd003",
        dataset_name=dataset_name,
        seed=seed,
        lora_r=16,
        lora_alpha=64,
        lr=5e-5,
        weight_decay=0.03,
        purpose="Medium-high capacity, lower learning rate, and stronger weight decay.",
    )
