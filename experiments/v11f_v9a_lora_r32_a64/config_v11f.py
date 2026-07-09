from __future__ import annotations

from experiments.v11_lora_sweep_common import make_v11_lora_overrides


def get_overrides(dataset_name: str, seed: int) -> dict:
    return make_v11_lora_overrides(
        version_name="v11f_v9a_lora_r32_a64",
        dataset_name=dataset_name,
        seed=seed,
        lora_r=32,
        lora_alpha=64,
        lr=1e-4,
        weight_decay=0.01,
        purpose="High-capacity LoRA upper-bound test.",
    )
