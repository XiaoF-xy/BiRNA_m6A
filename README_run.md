# BiRNA_m6A Run Guide

当前正式保留四种可复现实验方案：

```text
v1_baseline                   = BiRNA-BERT NUC frozen baseline
v2_birna_bert_lora            = BiRNA-BERT NUC + LoRA
v3_birna_bert_bpe_dual_view   = BiRNA-BERT NUC + BPE dual view
v4_birna_bert_bpe_dual_view_lora = BiRNA-BERT NUC + BPE dual view + LoRA
```

旧的重复训练入口已经移除，避免后续维护时出现多个训练入口不一致的问题。

当前统一入口：

```text
train.py
src/train_cv.py
src/training_utils.py
```

## 环境

```bash
cd BiRNA_m6A
bash env/create_conda_env_cuda121.sh
conda activate birna_m6a
python env/check_runtime.py
```

看到：

```text
runtime_check: OK
```

再开始训练。

## 数据与模型

数据默认位于：

```text
data/m6A_41bp/<dataset>/train.csv
data/m6A_41bp/<dataset>/test.csv
```

模型默认位于：

```text
pretrained/birna-bert-model/
```

大模型权重 `pytorch_model.bin` 不进入 git。训练前需要本地存在：

```text
pretrained/birna-bert-model/pytorch_model.bin
```

## 数据集别名

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

## v1: NUC frozen baseline

方法：

```text
BiRNA-BERT + NUC tokenization + mean pooling + center pooling + MLP classifier
```

该版本冻结 BiRNA-BERT backbone，只训练 MLP classifier。

运行 Human_Brain：

```bash
python train.py --version v1_baseline --dataset H_b --seed 42
```

运行人类三个数据集：

```bash
python train.py --version v1_baseline --dataset H_b --seed 42
python train.py --version v1_baseline --dataset H_k --seed 42
python train.py --version v1_baseline --dataset H_l --seed 42
```

## v2: NUC + LoRA

方法：

```text
BiRNA-BERT + NUC tokenization + LoRA(Wqkv) + mean pooling + center pooling + MLP classifier
```

LoRA 默认参数：

```text
lora_r = 8
lora_alpha = 32
lora_dropout = 0.05
lora_target_modules = Wqkv
```

该版本不做 full fine-tune，训练 LoRA adapter 和 MLP classifier。

运行 Human_Brain：

```bash
python train.py --version v2_birna_bert_lora --dataset H_b --seed 42
```

运行人类三个数据集：

```bash
python train.py --version v2_birna_bert_lora --dataset H_b --seed 42
python train.py --version v2_birna_bert_lora --dataset H_k --seed 42
python train.py --version v2_birna_bert_lora --dataset H_l --seed 42
```

## v3: NUC + BPE dual view

方法：

```text
NUC view = BiRNA-BERT + NUC tokenization + mask-aware mean pooling + center pooling
BPE view = BiRNA-BERT + BPE tokenization + mask-aware mean pooling
Fusion   = concat([nuc_mean, nuc_center, bpe_mean]) + MLP classifier
```

该版本冻结 BiRNA-BERT backbone，不启用 LoRA，只训练 MLP classifier。相比 v1，只额外加入 BPE 视图，用于判断 BPE 片段级信息是否有独立贡献。

运行 Human_Brain：

```bash
python train.py --version v3_birna_bert_bpe_dual_view --dataset H_b --seed 42
```

运行人类三个数据集：

```bash
python train.py --version v3_birna_bert_bpe_dual_view --dataset H_b --seed 42
python train.py --version v3_birna_bert_bpe_dual_view --dataset H_k --seed 42
python train.py --version v3_birna_bert_bpe_dual_view --dataset H_l --seed 42
```

## v4: NUC + BPE dual view + LoRA

方法：

```text
NUC view = BiRNA-BERT + NUC tokenization + mask-aware mean pooling + center pooling
BPE view = BiRNA-BERT + BPE tokenization + mask-aware mean pooling
Fusion   = concat([nuc_mean, nuc_center, bpe_mean]) + MLP classifier
LoRA     = Wqkv
```

LoRA 默认参数：

```text
lora_r = 8
lora_alpha = 32
lora_dropout = 0.05
lora_target_modules = Wqkv
```

该版本不做 full fine-tune，训练 LoRA adapter 和 MLP classifier。相比 v3，只增加 LoRA；相比 v2，只增加 BPE 视图。

运行 Human_Brain：

```bash
python train.py --version v4_birna_bert_bpe_dual_view_lora --dataset H_b --seed 42
```

运行人类三个数据集：

```bash
python train.py --version v4_birna_bert_bpe_dual_view_lora --dataset H_b --seed 42
python train.py --version v4_birna_bert_bpe_dual_view_lora --dataset H_k --seed 42
python train.py --version v4_birna_bert_bpe_dual_view_lora --dataset H_l --seed 42
```

## 评估协议

每个数据集使用自己的：

```text
train.csv
test.csv
```

训练流程：

```text
train.csv -> 分层 5 折
每折 80% train / 20% val
test.csv 始终作为 independent test
最终汇总 5 折 independent test mean/std
```

## 输出文件

输出位置：

```text
outputs/<version>/<dataset>/seed_<seed>/
```

例如：

```text
outputs/v2_birna_bert_lora/Human_Brain/seed_42/
```

输出内容：

```text
fold_1/metrics.json
fold_1/train_log.csv
fold_1/test_predictions.csv
...
fold_5/metrics.json
fold_5/train_log.csv
fold_5/test_predictions.csv
cv_summary.csv
cv_summary.json
resolved_config.json
```

默认不长期保存 `best_model.pt`。脚本会在评估完成后删除每折最优权重，保留指标和预测结果。若确实需要保留模型权重：

```bash
python train.py --version v2_birna_bert_lora --dataset H_b --seed 42 --keep_best_model
```

## dry-run 检查

只查看解析后的命令，不启动训练：

```bash
python train.py --version v1_baseline --dataset H_b --seed 42 --dry_run
python train.py --version v2_birna_bert_lora --dataset H_b --seed 42 --dry_run
python train.py --version v3_birna_bert_bpe_dual_view --dataset H_b --seed 42 --dry_run
python train.py --version v4_birna_bert_bpe_dual_view_lora --dataset H_b --seed 42 --dry_run
```
