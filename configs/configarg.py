from __future__ import annotations

import importlib
import inspect
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw"
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"
MODEL_ROOT = PROJECT_ROOT / "pretrained"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
LOG_ROOT = OUTPUT_ROOT / "logs"
RESULT_ROOT = OUTPUT_ROOT / "results"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
LORA_ADAPTER_ROOT = OUTPUT_ROOT / "lora_adapters"


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


VERSION_CONFIG_MODULES = {
    "v1_baseline": "experiments.v1_baseline.config_v1",
    "v2_birna_bert_feature": "experiments.v2_birna_bert_feature.config_v2",
    "v3_birna_bert_lora": "experiments.v3_birna_bert_lora.config_v3",
    "v4_moe_fusion": "experiments.v4_moe_fusion.config_v4",
}


def canonical_dataset_name(active_dataset: str) -> str:
    return DATASET_ALIASES.get(active_dataset, active_dataset)


def get_active_data_dir(active_dataset: str) -> Path:
    return DATA_ROOT / "m6A_41bp" / canonical_dataset_name(active_dataset)


def get_output_dir(version_name: str, dataset_name: str, seed: int) -> Path:
    dataset = canonical_dataset_name(dataset_name)
    return OUTPUT_ROOT / version_name / dataset / f"seed_{seed}"


def get_checkpoint_dir(version_name: str, dataset_name: str, seed: int) -> Path:
    dataset = canonical_dataset_name(dataset_name)
    return CHECKPOINT_ROOT / version_name / dataset / f"seed_{seed}"


def get_lora_adapter_dir(version_name: str, dataset_name: str, seed: int) -> Path:
    dataset = canonical_dataset_name(dataset_name)
    return LORA_ADAPTER_ROOT / version_name / dataset / f"seed_{seed}"


def get_log_dir(version_name: str, dataset_name: str, seed: int) -> Path:
    dataset = canonical_dataset_name(dataset_name)
    return LOG_ROOT / version_name / dataset / f"seed_{seed}"


def get_result_dir(version_name: str, dataset_name: str, seed: int) -> Path:
    dataset = canonical_dataset_name(dataset_name)
    return RESULT_ROOT / version_name / dataset / f"seed_{seed}"


@dataclass
class ModelArgs:
    pretrained_model_path: Path = MODEL_ROOT / "birna-bert-model"
    tokenizer_path: Path = MODEL_ROOT / "birna-bert-model"
    model_name: str = "birna_bert_nuc"
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["Wqkv"])
    lora_save_adapter_only: bool = True
    lora_merge_weights: bool = False
    lora_resume_adapter: bool = False
    lora_adapter_path: Path | None = None
    lora_log_dir: Path | None = None
    lora_result_dir: Path | None = None
    dropout: float = 0.2
    hidden_size: int = 768
    classifier_hidden_size: int = 256
    num_labels: int = 2
    freeze_backbone: bool = True
    use_multi_feature: bool = False
    use_cnn: bool = False
    use_bilstm: bool = False
    use_moe: bool = False
    use_attention: bool = False
    use_bpe: bool = False
    use_nuc: bool = True


@dataclass
class DataArgs:
    raw_data_root: Path = RAW_DATA_ROOT
    processed_data_root: Path = PROCESSED_DATA_ROOT
    data_root: Path = DATA_ROOT / "m6A_41bp"
    active_dataset: str = "Human_Brain"
    dataset_name: str = "Human_Brain"
    species_or_tissue: str = "Human_Brain"
    train_file: Path | None = None
    val_file: Path | None = None
    test_file: Path | None = None
    sequence_length: int = 41
    sequence_window: int = 41
    convert_u_to_t: bool = True
    use_one_hot: bool = False
    use_ncp: bool = False
    use_eiip: bool = False
    use_enac: bool = False
    label_column: str = "label"
    sequence_column: str = "sequence"


@dataclass
class TrainingArgs:
    # Mirrors the TrainingArguments fields used by this project. Use
    # to_hf_training_arguments() when switching to Hugging Face Trainer.
    output_dir: Path = OUTPUT_ROOT / "default"
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 0
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "MCC"
    greater_is_better: bool = True
    seed: int = 42
    fp16: bool = False
    dataloader_num_workers: int = 0
    save_total_limit: int = 1
    remove_unused_columns: bool = False
    folds: int = 5
    max_length: int = 64
    keep_best_model: bool = False

    def to_hf_training_arguments(self):
        from transformers import TrainingArguments as HFTrainingArguments

        kwargs = asdict(self)
        kwargs["output_dir"] = str(kwargs["output_dir"])
        evaluation_strategy = kwargs.pop("evaluation_strategy")
        signature = inspect.signature(HFTrainingArguments)
        if "eval_strategy" in signature.parameters:
            kwargs["eval_strategy"] = evaluation_strategy
        else:
            kwargs["evaluation_strategy"] = evaluation_strategy
        kwargs.pop("folds")
        kwargs.pop("max_length")
        kwargs.pop("keep_best_model")
        return HFTrainingArguments(**kwargs)


@dataclass
class ExperimentArgs:
    version_name: str = "v1_baseline"
    version_config_path: Path | None = None
    output_suffix: str = "nuc_frozen"
    description: str = "BiRNA-BERT NUC frozen baseline"
    changes_from_previous: str = "Initial NUC baseline."
    notes: str = ""


@dataclass
class ProjectConfig:
    experiment: ExperimentArgs = field(default_factory=ExperimentArgs)
    model: ModelArgs = field(default_factory=ModelArgs)
    data: DataArgs = field(default_factory=DataArgs)
    training: TrainingArgs = field(default_factory=TrainingArgs)

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
    config = ProjectConfig()
    module = importlib.import_module(VERSION_CONFIG_MODULES[version_name])
    overrides = module.get_overrides(dataset_name=dataset, seed=seed)
    config = apply_overrides(config, overrides)

    active_data_dir = get_active_data_dir(dataset)
    config.data = replace(
        config.data,
        active_dataset=dataset,
        dataset_name=dataset,
        species_or_tissue=dataset,
        train_file=active_data_dir / "train.csv",
        test_file=active_data_dir / "test.csv",
    )
    config.training = replace(
        config.training,
        seed=seed,
        output_dir=get_output_dir(version_name, dataset, seed),
    )
    config.model = replace(
        config.model,
        lora_adapter_path=get_lora_adapter_dir(version_name, dataset, seed),
        lora_log_dir=get_log_dir(version_name, dataset, seed),
        lora_result_dir=get_result_dir(version_name, dataset, seed),
    )
    config.experiment = replace(
        config.experiment,
        version_name=version_name,
        version_config_path=PROJECT_ROOT / (VERSION_CONFIG_MODULES[version_name].replace(".", "/") + ".py"),
    )
    return config


def ensure_output_dirs(config: ProjectConfig) -> None:
    paths = [
        config.training.output_dir,
        get_checkpoint_dir(config.experiment.version_name, config.data.dataset_name, config.training.seed),
        get_log_dir(config.experiment.version_name, config.data.dataset_name, config.training.seed),
        get_result_dir(config.experiment.version_name, config.data.dataset_name, config.training.seed),
    ]
    if config.model.use_lora:
        paths.append(get_lora_adapter_dir(config.experiment.version_name, config.data.dataset_name, config.training.seed))
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
