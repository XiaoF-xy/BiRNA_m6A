# BiRNA_m6A

BiRNA_m6A is a versioned research project for RNA m6A site prediction with BiRNA-BERT, NUC tokenization, LoRA fine-tuning, and later feature-fusion experiments.

## Current Runnable Versions

| Version | Status | Method |
|---|---|---|
| `v1_baseline` | runnable | BiRNA-BERT NUC frozen baseline: mean pooling + center pooling + MLP classifier |
| `v2_birna_bert_lora` | runnable | BiRNA-BERT NUC + LoRA on `Wqkv` + MLP classifier |
| `v3_birna_bert_bpe_dual_view` | runnable | BiRNA-BERT NUC view + BPE view: concat([nuc_mean, nuc_center, bpe_mean]) + MLP classifier |
| `v4_birna_bert_bpe_dual_view_lora` | runnable | BiRNA-BERT NUC+BPE dual view + LoRA on `Wqkv` + MLP classifier |
| `v5_nuc_lora_no_center` | runnable | v2 ablation: BiRNA-BERT NUC + LoRA without explicit center-token pooling |
| `v6a_bpe_global_nuc_local_film_lora` | runnable | BPE global -> FiLM -> NUC local window + LoRA |
| `v6b_nuc_global_nuc_local_film_lora` | runnable | NUC global -> FiLM -> NUC local window + LoRA |

Versions v1-v5 use the strict protocol by default: `train.csv` is split into stratified train/val folds, and `test.csv` is used only for final evaluation. Versions v6a/v6b are test-as-validation-only experiments by default.

Benchmark aliases using test-as-validation are also runnable:

| Version Alias | Base Method | Protocol |
|---|---|---|
| `v1_baseline_test_as_val` | `v1_baseline` | train on full `train.csv`; select best epoch on `test.csv` |
| `v2_birna_bert_lora_test_as_val` | `v2_birna_bert_lora` | train on full `train.csv`; select best epoch on `test.csv` |
| `v3_birna_bert_bpe_dual_view_test_as_val` | `v3_birna_bert_bpe_dual_view` | train on full `train.csv`; select best epoch on `test.csv` |
| `v4_birna_bert_bpe_dual_view_lora_test_as_val` | `v4_birna_bert_bpe_dual_view_lora` | train on full `train.csv`; select best epoch on `test.csv` |
| `v5_nuc_lora_no_center_test_as_val` | `v5_nuc_lora_no_center` | train on full `train.csv`; select best epoch on `test.csv` |

Use the test-as-validation aliases only for benchmark-style comparison with methods that follow the same protocol. These outputs should not be described as strict independent-test results. The default best-epoch selection metric is `ACC`, matching the benchmark-style selection used by DFM code.

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
│   ├── v2_birna_bert_lora/
│   ├── v3_birna_bert_bpe_dual_view/
│   ├── v4_birna_bert_bpe_dual_view_lora/
│   ├── v5_nuc_lora_no_center/
│   ├── v6a_bpe_global_nuc_local_film_lora/
│   └── v6b_nuc_global_nuc_local_film_lora/
├── pretrained/
│   └── birna-bert-model/
├── scripts/
├── src/
├── outputs/
│   ├── v1_baseline/
│   ├── v2_birna_bert_lora/
│   ├── v3_birna_bert_bpe_dual_view/
│   ├── v4_birna_bert_bpe_dual_view_lora/
│   ├── v5_nuc_lora_no_center/
│   ├── v6a_bpe_global_nuc_local_film_lora/
│   └── v6b_nuc_global_nuc_local_film_lora/
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
experiments/v3_birna_bert_bpe_dual_view/config_v3.py
experiments/v4_birna_bert_bpe_dual_view_lora/config_v4.py
experiments/v5_nuc_lora_no_center/config_v5.py
experiments/v6a_bpe_global_nuc_local_film_lora/config_v6a.py
experiments/v6b_nuc_global_nuc_local_film_lora/config_v6b.py
```

`configs/configarg.py` keeps the shared parameters currently needed by v1-v6: model path, tokenizer path, dataset alias, output path, evaluation protocol, best-epoch selection metric, BPE-view switch, FiLM switch, local-window size, and LoRA settings. Version configs only override the small differences between methods.

## Run Experiments

Terminal commands should only specify version, dataset, and seed.

Strict protocol:

Frozen baseline:

```bash
python train.py --version v1_baseline --dataset H_b --seed 42
```

LoRA:

```bash
python train.py --version v2_birna_bert_lora --dataset H_b --seed 42
```

BPE dual view:

```bash
python train.py --version v3_birna_bert_bpe_dual_view --dataset H_b --seed 42
```

BPE dual view + LoRA:

```bash
python train.py --version v4_birna_bert_bpe_dual_view_lora --dataset H_b --seed 42
```

No-center pooling ablation:

```bash
python train.py --version v5_nuc_lora_no_center --dataset H_b --seed 42
```

FiLM global-local experiments use test-as-validation by default:

```bash
python train.py --version v6a_bpe_global_nuc_local_film_lora --dataset H_b --seed 42
python train.py --version v6b_nuc_global_nuc_local_film_lora --dataset H_b --seed 42
```

Test-as-validation benchmark protocol:

```bash
python train.py --version v1_baseline_test_as_val --dataset H_b --seed 42
python train.py --version v2_birna_bert_lora_test_as_val --dataset H_b --seed 42
python train.py --version v3_birna_bert_bpe_dual_view_test_as_val --dataset H_b --seed 42
python train.py --version v4_birna_bert_bpe_dual_view_lora_test_as_val --dataset H_b --seed 42
python train.py --version v5_nuc_lora_no_center_test_as_val --dataset H_b --seed 42
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

For example, the strict v4 and benchmark v4 outputs are separated:

```text
outputs/v4_birna_bert_bpe_dual_view_lora/Human_Brain/seed_42/
outputs/v4_birna_bert_bpe_dual_view_lora_test_as_val/Human_Brain/seed_42/
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
