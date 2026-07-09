from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "m6A_41bp"
MODEL_ROOT = PROJECT_ROOT / "pretrained"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


DATASET_ALIASES = {
    "H_b": "Human_Brain",
    "H_k": "Human_Kidney",
    "H_l": "Human_Liver",
    "M_b": "Mouse_brain",
    "M_h": "Mouse_heart",
    "M_k": "Mouse_kidney",
    "M_l": "Mouse_liver",
    "M_t": "Mouse_test",
    "R_b": "rat_brain",
    "R_k": "rat_kidney",
    "R_l": "rat_liver",
}


BASE_VERSION_CONFIG_MODULES = {
    "v1_baseline": "experiments.v1_baseline.config_v1",
    "v2_birna_bert_lora": "experiments.v2_birna_bert_lora.config_v2",
    "v3_birna_bert_bpe_dual_view": "experiments.v3_birna_bert_bpe_dual_view.config_v3",
    "v4_birna_bert_bpe_dual_view_lora": "experiments.v4_birna_bert_bpe_dual_view_lora.config_v4",
    "v5_nuc_lora_no_center": "experiments.v5_nuc_lora_no_center.config_v5",
    "v6a_bpe_global_nuc_local_film_lora": "experiments.v6a_bpe_global_nuc_local_film_lora.config_v6a",
    "v6b_nuc_global_nuc_local_film_lora": "experiments.v6b_nuc_global_nuc_local_film_lora.config_v6b",
    "v7a_nuc_global_nuc_full_mean_film_lora": "experiments.v7a_nuc_global_nuc_full_mean_film_lora.config_v7a",
    "v7b_nuc_global_nuc_center_cnn_film_lora": "experiments.v7b_nuc_global_nuc_center_cnn_film_lora.config_v7b",
    "v7c_bpe_global_nuc_full_cnn_film_lora": "experiments.v7c_bpe_global_nuc_full_cnn_film_lora.config_v7c",
    "v7d_nuc_global_nuc_full_cnn_film_lora": "experiments.v7d_nuc_global_nuc_full_cnn_film_lora.config_v7d",
    "v8_nuc_full_mean_center_cnn_film_lora": "experiments.v8_nuc_full_mean_center_cnn_film_lora.config_v8",
    "v9a_birna_v7b_handcrafted_multiscale_cnn": "experiments.v9a_birna_v7b_handcrafted_multiscale_cnn.config_v9a",
    "v9b_no_enac_handcrafted_ablation": "experiments.v9b_no_enac_handcrafted_ablation.config_v9b",
    "v9c_onehot_handcrafted_ablation": "experiments.v9c_onehot_handcrafted_ablation.config_v9c",
    "v9d_ncp_eiip_handcrafted_ablation": "experiments.v9d_ncp_eiip_handcrafted_ablation.config_v9d",
    "v9e_enac_handcrafted_ablation": "experiments.v9e_enac_handcrafted_ablation.config_v9e",
    "v9f_handcrafted_only": "experiments.v9f_handcrafted_only.config_v9f",
    "v10a_gated_v9a": "experiments.v10a_gated_v9a.config_v10a",
    "v11a_v9a_lora_r4_a16": "experiments.v11a_v9a_lora_r4_a16.config_v11a",
    "v11b_v9a_lora_r8_a16": "experiments.v11b_v9a_lora_r8_a16.config_v11b",
    "v11c_v9a_lora_r8_a64": "experiments.v11c_v9a_lora_r8_a64.config_v11c",
    "v11d_v9a_lora_r16_a32": "experiments.v11d_v9a_lora_r16_a32.config_v11d",
    "v11e_v9a_lora_r16_a64": "experiments.v11e_v9a_lora_r16_a64.config_v11e",
    "v11f_v9a_lora_r32_a64": "experiments.v11f_v9a_lora_r32_a64.config_v11f",
    "v11g_v9a_lora_r8_a32_lr5e5": "experiments.v11g_v9a_lora_r8_a32_lr5e5.config_v11g",
    "v11h_v9a_lora_r16_a32_lr5e5": "experiments.v11h_v9a_lora_r16_a32_lr5e5.config_v11h",
    "v11i_v9a_lora_r16_a64_lr5e5": "experiments.v11i_v9a_lora_r16_a64_lr5e5.config_v11i",
    "v11j_v9a_lora_r32_a64_lr5e5": "experiments.v11j_v9a_lora_r32_a64_lr5e5.config_v11j",
    "v11k_v9a_lora_r16_a32_lr5e5_wd003": "experiments.v11k_v9a_lora_r16_a32_lr5e5_wd003.config_v11k",
    "v11l_v9a_lora_r16_a64_lr5e5_wd003": "experiments.v11l_v9a_lora_r16_a64_lr5e5_wd003.config_v11l",
}

TEST_AS_VAL_ONLY_VERSIONS = {
    "v6a_bpe_global_nuc_local_film_lora",
    "v6b_nuc_global_nuc_local_film_lora",
    "v7a_nuc_global_nuc_full_mean_film_lora",
    "v7b_nuc_global_nuc_center_cnn_film_lora",
    "v7c_bpe_global_nuc_full_cnn_film_lora",
    "v7d_nuc_global_nuc_full_cnn_film_lora",
    "v8_nuc_full_mean_center_cnn_film_lora",
    "v9a_birna_v7b_handcrafted_multiscale_cnn",
    "v9b_no_enac_handcrafted_ablation",
    "v9c_onehot_handcrafted_ablation",
    "v9d_ncp_eiip_handcrafted_ablation",
    "v9e_enac_handcrafted_ablation",
    "v9f_handcrafted_only",
    "v10a_gated_v9a",
    "v11a_v9a_lora_r4_a16",
    "v11b_v9a_lora_r8_a16",
    "v11c_v9a_lora_r8_a64",
    "v11d_v9a_lora_r16_a32",
    "v11e_v9a_lora_r16_a64",
    "v11f_v9a_lora_r32_a64",
    "v11g_v9a_lora_r8_a32_lr5e5",
    "v11h_v9a_lora_r16_a32_lr5e5",
    "v11i_v9a_lora_r16_a64_lr5e5",
    "v11j_v9a_lora_r32_a64_lr5e5",
    "v11k_v9a_lora_r16_a32_lr5e5_wd003",
    "v11l_v9a_lora_r16_a64_lr5e5_wd003",
}

TEST_AS_VAL_ALIASES = {
    "v1_baseline_test_as_val": "v1_baseline",
    "v2_birna_bert_lora_test_as_val": "v2_birna_bert_lora",
    "v3_birna_bert_bpe_dual_view_test_as_val": "v3_birna_bert_bpe_dual_view",
    "v4_birna_bert_bpe_dual_view_lora_test_as_val": "v4_birna_bert_bpe_dual_view_lora",
    "v5_nuc_lora_no_center_test_as_val": "v5_nuc_lora_no_center",
}

VERSION_CONFIG_MODULES = {
    **BASE_VERSION_CONFIG_MODULES,
    **{
        alias: BASE_VERSION_CONFIG_MODULES[base_version]
        for alias, base_version in TEST_AS_VAL_ALIASES.items()
    },
}


def get_eval_protocol(version_name: str) -> str:
    return "test_as_val" if version_name in TEST_AS_VAL_ALIASES or version_name in TEST_AS_VAL_ONLY_VERSIONS else "strict_cv"


def canonical_dataset_name(dataset: str) -> str:
    return DATASET_ALIASES.get(dataset, dataset)


def get_active_data_dir(dataset: str) -> Path:
    return DATA_ROOT / canonical_dataset_name(dataset)


def get_output_dir(version: str, dataset: str, seed: int) -> Path:
    return OUTPUT_ROOT / version / canonical_dataset_name(dataset) / f"seed_{seed}"


@dataclass
class ModelConfig:
    model_dir: Path = MODEL_ROOT / "birna-bert-model"
    tokenizer_dir: Path = MODEL_ROOT / "birna-bert-model"
    freeze_backbone: bool = True
    use_center_pooling: bool = True
    use_bpe_view: bool = False
    use_film: bool = False
    film_global_view: str = "bpe"
    local_window_radius: int = 3
    film_nuc_pooling: str = "center_mean"
    cnn_kernel_sizes: list[int] = field(default_factory=lambda: [3, 5, 7])
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["Wqkv"])
    use_handcrafted_features: bool = False
    handcrafted_feature_names: list[str] = field(default_factory=lambda: ["onehot", "ncp", "eiip", "enac"])
    handcrafted_only: bool = False
    handcrafted_cnn_channels: int = 64
    handcrafted_output_dim: int = 128
    use_gated_fusion: bool = False
    gated_fusion_dim: int = 256
    gated_hidden_dim: int = 128


@dataclass
class DataConfig:
    dataset_name: str = "Human_Brain"
    data_dir: Path = DATA_ROOT / "Human_Brain"
    sequence_length: int = 41


@dataclass
class TrainConfig:
    output_dir: Path = OUTPUT_ROOT / "v1_baseline" / "Human_Brain" / "seed_42"
    eval_protocol: str = "strict_cv"
    selection_metric: str = "ACC"
    folds: int = 5
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.01
    seed: int = 42
    max_length: int = 64
    keep_best_model: bool = False


@dataclass
class ExperimentConfig:
    version_name: str = "v1_baseline"
    description: str = "BiRNA-BERT NUC frozen baseline"


@dataclass
class ProjectConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainConfig = field(default_factory=TrainConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_overrides(config: ProjectConfig, overrides: dict[str, dict[str, Any]]) -> ProjectConfig:
    for section_name, section_values in overrides.items():
        current_section = getattr(config, section_name)
        setattr(config, section_name, replace(current_section, **section_values))
    return config


def load_experiment_config(version_name: str, dataset_name: str = "Human_Brain", seed: int = 42) -> ProjectConfig:
    if version_name not in VERSION_CONFIG_MODULES:
        supported = ", ".join(sorted(VERSION_CONFIG_MODULES))
        raise ValueError(f"Unknown version: {version_name}. Supported versions: {supported}")

    dataset = canonical_dataset_name(dataset_name)
    eval_protocol = get_eval_protocol(version_name)
    module = importlib.import_module(VERSION_CONFIG_MODULES[version_name])
    config = apply_overrides(ProjectConfig(), module.get_overrides(dataset_name=dataset, seed=seed))

    description = config.experiment.description
    if eval_protocol == "test_as_val":
        description = f"{description} Test-as-validation benchmark protocol."
    config.experiment = replace(config.experiment, version_name=version_name, description=description)
    config.data = replace(
        config.data,
        dataset_name=dataset,
        data_dir=get_active_data_dir(dataset),
    )
    training_updates = {
        "seed": seed,
        "output_dir": get_output_dir(version_name, dataset, seed),
        "eval_protocol": eval_protocol,
    }
    if eval_protocol == "test_as_val":
        training_updates["folds"] = 1
    config.training = replace(
        config.training,
        **training_updates,
    )
    return config


def ensure_output_dirs(config: ProjectConfig) -> None:
    Path(config.training.output_dir).mkdir(parents=True, exist_ok=True)
