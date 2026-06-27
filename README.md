# BiRNA_m6A

BiRNA_m6A is a versioned research project for RNA m6A site prediction with BiRNA-BERT, NUC tokenization, LoRA fine-tuning, and later feature-fusion experiments.

## Current Runnable Versions

| Version | Status | Method |
|---|---|---|
| `v1_baseline` | runnable | BiRNA-BERT NUC frozen baseline: mean pooling + center pooling + MLP classifier |
| `v2_birna_bert_lora` | runnable | BiRNA-BERT NUC + LoRA on `Wqkv` + MLP classifier |

## Directory Structure

```text
BiRNA_m6A/
├── configs/
│   └── configarg.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── m6A_41bp/
├── docs/
├── env/
├── experiments/
│   ├── v1_baseline/
│   └── v2_birna_bert_lora/
├── pretrained/
│   └── birna-bert-model/
├── scripts/
├── src/
├── outputs/
│   ├── v1_baseline/
│   └── v2_birna_bert_lora/
├── train.py
├── requirements_birna.txt
└── README_run.md
```

## Configuration Design

The global config is Python-based:

```text
configs/configarg.py
```

Each version only stores the differences from the global config:

```text
experiments/v1_baseline/config_v1.py
experiments/v2_birna_bert_lora/config_v2.py
```

`configs/configarg.py` only keeps the parameters currently needed by v1/v2: model path, tokenizer path, dataset alias, output path, CV settings, and LoRA settings. Version configs only override the small differences between frozen baseline and LoRA.

## Run Experiments

Terminal commands should only specify version, dataset, and seed.

Frozen baseline:

```bash
python train.py --version v1_baseline --dataset H_b --seed 42
```

LoRA:

```bash
python train.py --version v2_birna_bert_lora --dataset H_b --seed 42
```

Dataset aliases:

```text
H_b -> Human_Brain
H_k -> Human_Kidney
H_l -> Human_Liver
M_b -> Mouse_brain
M_h -> Mouse_heart
M_k -> Mouse_kidney
M_l -> Mouse_liver
M_t -> Mouse_test
R_b -> rat_brain
R_k -> rat_kidney
R_l -> rat_liver
```

## Outputs

Versioned runs save outputs under:

```text
outputs/<version>/<dataset>/seed_<seed>/
```

The current CV training script writes all metrics, logs, predictions, and temporary checkpoints into that run directory:

```text
fold_x/metrics.json
fold_x/train_log.csv
fold_x/test_predictions.csv
cv_summary.csv
cv_summary.json
resolved_config.json
```

By default, `best_model.pt` is deleted after evaluation to save disk space. Add `--keep_best_model` if a run needs checkpoints.

## Environment

Recommended setup:

```bash
bash env/create_conda_env_cuda121.sh
conda activate birna_m6a
python env/check_runtime.py
```

## Local Model Weights

Large pretrained weights are intentionally not tracked by git. Before training, place the BiRNA-BERT weight file locally at:

```text
pretrained/birna-bert-model/pytorch_model.bin
```

The repository keeps lightweight model code, config, tokenizer files, and experiment code, but excludes binary checkpoints and training outputs.

## Notes

The duplicated legacy training entrypoints were removed to avoid multiple sources of truth. The current `train.py` entrypoint reads `configs/configarg.py`, applies the selected version config, and launches `src/train_cv.py`.
