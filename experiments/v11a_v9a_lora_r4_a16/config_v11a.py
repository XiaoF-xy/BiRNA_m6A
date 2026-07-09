from __future__ import annotations

from experiments.v11_lora_sweep_common import make_v11_lora_overrides


def get_overrides(dataset_name: str, seed: int) -> dict:
    return make_v11_lora_overrides(
        version_name="v11a_v9a_lora_r4_a16",
        dataset_name=dataset_name,
        seed=seed,
        lora_r=4,
        lora_alpha=16,
        lr=1e-4,
        weight_decay=0.01,
        purpose="Smaller LoRA capacity; test whether lower rank is more stable.",
    )
