# BiRNA_m6A Run Guide

当前正式保留以下可复现实验方案：

```text
v1_baseline                   = BiRNA-BERT NUC frozen baseline
v2_birna_bert_lora            = BiRNA-BERT NUC + LoRA
v3_birna_bert_bpe_dual_view   = BiRNA-BERT NUC + BPE dual view
v4_birna_bert_bpe_dual_view_lora = BiRNA-BERT NUC + BPE dual view + LoRA
v5_nuc_lora_no_center         = v2 消融：NUC + LoRA，不使用显式中心位点 pooling
v6a_bpe_global_nuc_local_film_lora = BPE global -> FiLM -> NUC local + LoRA
v6b_nuc_global_nuc_local_film_lora = NUC global -> FiLM -> NUC local + LoRA
v7a_nuc_global_nuc_full_mean_film_lora = NUC global -> FiLM -> NUC full 41bp mean + LoRA
v7b_nuc_global_nuc_center_cnn_film_lora = NUC global -> FiLM -> NUC center-window CNN mean + LoRA
v7c_bpe_global_nuc_full_cnn_film_lora = BPE global -> FiLM -> NUC full 41bp CNN mean + LoRA
v7d_nuc_global_nuc_full_cnn_film_lora = NUC global -> FiLM -> NUC full 41bp CNN mean + LoRA
v8_nuc_full_mean_center_cnn_film_lora = NUC global -> FiLM -> NUC full mean + center-window CNN fusion + LoRA
v9a_birna_v7b_handcrafted_multiscale_cnn = v7b + handcrafted physicochemical multi-scale CNN branch
v9b_no_enac_handcrafted_ablation = v9a without ENAC
v9c_onehot_handcrafted_ablation = v9a with ONEHOT only
v9d_ncp_eiip_handcrafted_ablation = v9a with NCP+EIIP only
v9e_enac_handcrafted_ablation = v9a with ENAC only
v9f_handcrafted_only = handcrafted-only multi-scale CNN baseline
```

v1-v5 有两套评估协议；v6a/v6b/v7a/v7b/v7c/v7d/v8/v9a-v9f 默认只使用 test_as_val：

```text
strict_cv    = train.csv 内部分层 5 折验证，test.csv 只做最终评估
test_as_val  = 完整 train.csv 训练，每个 epoch 用 test.csv 选 best epoch
```

`test_as_val` 只用于和已有 test-as-validation 论文代码做 benchmark-style 对标，不能表述为严格 independent test。当前默认使用 `ACC` 选择 best epoch，以对齐 DFM 代码的 benchmark-style 选择方式；MCC/AUC/AUPRC/F1 仍会完整输出用于分析。

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

test-as-val 对标运行：

```bash
python train.py --version v1_baseline_test_as_val --dataset H_b --seed 42
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

test-as-val 对标运行：

```bash
python train.py --version v2_birna_bert_lora_test_as_val --dataset H_b --seed 42
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

test-as-val 对标运行：

```bash
python train.py --version v3_birna_bert_bpe_dual_view_test_as_val --dataset H_b --seed 42
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

test-as-val 对标运行：

```bash
python train.py --version v4_birna_bert_bpe_dual_view_lora_test_as_val --dataset H_b --seed 42
```

运行人类三个数据集：

```bash
python train.py --version v4_birna_bert_bpe_dual_view_lora --dataset H_b --seed 42
python train.py --version v4_birna_bert_bpe_dual_view_lora --dataset H_k --seed 42
python train.py --version v4_birna_bert_bpe_dual_view_lora --dataset H_l --seed 42
```

## v5: NUC + LoRA without center pooling

方法：

```text
BiRNA-BERT + NUC tokenization + LoRA(Wqkv) + mean pooling + MLP classifier
```

该版本是 v2 的 no-center 消融实验：

```text
v2 = concat([nuc_mean, nuc_center]) + LoRA
v5 = nuc_mean + LoRA
```

用途：判断显式中心位点 token pooling 是否真正带来性能增益。

运行 Human_Brain：

```bash
python train.py --version v5_nuc_lora_no_center --dataset H_b --seed 42
```

test-as-val 对标运行：

```bash
python train.py --version v5_nuc_lora_no_center_test_as_val --dataset H_b --seed 42
```

运行人类三个数据集：

```bash
python train.py --version v5_nuc_lora_no_center --dataset H_b --seed 42
python train.py --version v5_nuc_lora_no_center --dataset H_k --seed 42
python train.py --version v5_nuc_lora_no_center --dataset H_l --seed 42
```

## v6a: BPE global -> FiLM -> NUC local + LoRA

方法：

```text
BPE global = BiRNA-BERT BPE tokenization + mask-aware mean pooling
NUC local  = BiRNA-BERT NUC tokenization + center +/-3 nt local window mean pooling
FiLM       = gamma, beta = MLP(BPE global)
Fusion     = concat([BPE global, gamma * NUC local + beta])
LoRA       = Wqkv
```

局部窗口：

```text
center_index = 20
local_window_radius = 3
NUC local = nuc_emb[:, 17:24, :].mean(dim=1)
```

该版本默认就是 test-as-val 对标协议，不需要 `_test_as_val` 后缀。

运行 Human_Brain：

```bash
python train.py --version v6a_bpe_global_nuc_local_film_lora --dataset H_b --seed 42
```

运行人类三个数据集：

```bash
python train.py --version v6a_bpe_global_nuc_local_film_lora --dataset H_b --seed 42
python train.py --version v6a_bpe_global_nuc_local_film_lora --dataset H_k --seed 42
python train.py --version v6a_bpe_global_nuc_local_film_lora --dataset H_l --seed 42
```

## v6b: NUC global -> FiLM -> NUC local + LoRA

方法：

```text
NUC global = BiRNA-BERT NUC tokenization + mask-aware mean pooling
NUC local  = BiRNA-BERT NUC tokenization + center +/-3 nt local window mean pooling
FiLM       = gamma, beta = MLP(NUC global)
Fusion     = concat([NUC global, gamma * NUC local + beta])
LoRA       = Wqkv
```

该版本是 v6a 的关键对照，用于判断 BPE global 是否真的比 NUC global 有额外价值。v6b 默认也是 test-as-val 对标协议。

运行 Human_Brain：

```bash
python train.py --version v6b_nuc_global_nuc_local_film_lora --dataset H_b --seed 42
```

运行人类三个数据集：

```bash
python train.py --version v6b_nuc_global_nuc_local_film_lora --dataset H_b --seed 42
python train.py --version v6b_nuc_global_nuc_local_film_lora --dataset H_k --seed 42
python train.py --version v6b_nuc_global_nuc_local_film_lora --dataset H_l --seed 42
```

## 与 DFM 的关系

v6a/v6b 不是完整复刻 DFM。它们只复用 DFM 的核心思想：

```text
global view -> FiLM -> local view
```

当前暂时不加入 MoE experts 和 gating，目的是先验证 FiLM/global-local 机制是否有效。如果 v6a/v6b 有稳定提升，再考虑后续版本加入 MoE。

## v7: FiLM NUC 分支消融与 CNN 消融

v7 系列都默认使用 test-as-val 对标协议、LoRA(Wqkv)、ACC 选择 best epoch，并在训练结束后删除 `best_model.pt`。

| 版本 | global | 被调制分支 | CNN | 中心窗口 | 目的 |
|---|---|---|---|---|---|
| `v7a_nuc_global_nuc_full_mean_film_lora` | NUC mean | NUC full 41bp mean | 否 | 否 | 验证去掉中心窗口是否有用 |
| `v7b_nuc_global_nuc_center_cnn_film_lora` | NUC mean | NUC center-window CNN mean | 是 | 是 | 验证保留中心窗口时 CNN 是否优于普通 mean |
| `v7c_bpe_global_nuc_full_cnn_film_lora` | BPE mean | NUC full 41bp CNN mean | 是 | 否 | 验证 BPE global 调制可学习 NUC CNN 是否有用 |
| `v7d_nuc_global_nuc_full_cnn_film_lora` | NUC mean | NUC full 41bp CNN mean | 是 | 否 | 验证 NUC global 调制可学习 NUC CNN 是否有用 |

v7a 运行：

```bash
python train.py --version v7a_nuc_global_nuc_full_mean_film_lora --dataset H_b --seed 42
```

v7b 运行：

```bash
python train.py --version v7b_nuc_global_nuc_center_cnn_film_lora --dataset H_b --seed 42
```

v7c 运行：

```bash
python train.py --version v7c_bpe_global_nuc_full_cnn_film_lora --dataset H_b --seed 42
```

v7d 运行：

```bash
python train.py --version v7d_nuc_global_nuc_full_cnn_film_lora --dataset H_b --seed 42
```

两张卡并行示例：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v7c_bpe_global_nuc_full_cnn_film_lora --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v7d_nuc_global_nuc_full_cnn_film_lora --dataset H_b --seed 42
```

v7 主要对照逻辑：

```text
v6b vs v7a: 中心窗口是否必要
v6b vs v7b: 中心窗口内 CNN 是否优于 mean
v7a vs v7d: 去掉中心窗口后，全 41bp CNN 是否优于全 41bp mean
v7b vs v7d: CNN 使用中心窗口还是全 41bp 更好
v7c vs v7d: BPE global 和 NUC global 哪个更适合 FiLM 调制
```

## v8: v7a + v7b feature fusion

v8 用于验证 v7a 和 v7b 的互补性：

```text
NUC global      = BiRNA-BERT NUC tokenization + mask-aware mean pooling
NUC full mean   = full 41bp NUC token embedding mean
NUC center CNN  = multi-scale Conv1d(kernel=3,5,7) + center window 17:24 mean
FiLM full       = gamma_full, beta_full = MLP(NUC global)
FiLM center CNN = gamma_center, beta_center = MLP(NUC global)
Fusion          = concat([NUC global, FiLM(full mean), FiLM(center CNN)])
LoRA            = Wqkv
```

该版本默认使用 test-as-val 对标协议、LoRA(Wqkv)、ACC 选择 best epoch，并在训练结束后删除 `best_model.pt`。

运行 Human_Brain：

```bash
python train.py --version v8_nuc_full_mean_center_cnn_film_lora --dataset H_b --seed 42
```

运行人类三个数据集：

```bash
python train.py --version v8_nuc_full_mean_center_cnn_film_lora --dataset H_b --seed 42
python train.py --version v8_nuc_full_mean_center_cnn_film_lora --dataset H_k --seed 42
python train.py --version v8_nuc_full_mean_center_cnn_film_lora --dataset H_l --seed 42
```

三张卡并行示例：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v8_nuc_full_mean_center_cnn_film_lora --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v8_nuc_full_mean_center_cnn_film_lora --dataset H_k --seed 42
CUDA_VISIBLE_DEVICES=2 python train.py --version v8_nuc_full_mean_center_cnn_film_lora --dataset H_l --seed 42
```

v8 对照逻辑：

```text
v7a vs v8: 加入中心窗口 CNN 分支是否提升 MCC/F1/Recall
v7b vs v8: 保留全长 mean 分支是否提升 ACC/AUC/AUPRC 稳定性
v7d vs v8: 全长 CNN 和中心 CNN 哪个更适合与全长 mean 融合
```

## v9a: v7b + handcrafted physicochemical multi-scale CNN

v9a 用于验证显式手工物化特征是否能补充 BiRNA-BERT v7b 的上下文表示。

```text
BiRNA-BERT branch:
NUC global mean -> FiLM -> NUC center-window CNN mean

Handcrafted branch:
ONEHOT [4, 41]
NCP    [3, 41]
EIIP   [1, 41]
ENAC   [4, 41]
concat -> [12, 41]
multi-scale Conv1d(kernel=3,5,7)
mean pooling
MLP -> handcrafted feature

Fusion:
concat([birna_feat, handcrafted_feat]) -> MLP classifier
```

该版本默认使用 test-as-val 对标协议、LoRA(Wqkv)、ACC 选择 best epoch，并在训练结束后删除 `best_model.pt`。

运行 Human_Brain：

```bash
python train.py --version v9a_birna_v7b_handcrafted_multiscale_cnn --dataset H_b --seed 42
```

运行人类三个数据集：

```bash
python train.py --version v9a_birna_v7b_handcrafted_multiscale_cnn --dataset H_b --seed 42
python train.py --version v9a_birna_v7b_handcrafted_multiscale_cnn --dataset H_k --seed 42
python train.py --version v9a_birna_v7b_handcrafted_multiscale_cnn --dataset H_l --seed 42
```

三张卡并行示例：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v9a_birna_v7b_handcrafted_multiscale_cnn --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v9a_birna_v7b_handcrafted_multiscale_cnn --dataset H_k --seed 42
CUDA_VISIBLE_DEVICES=2 python train.py --version v9a_birna_v7b_handcrafted_multiscale_cnn --dataset H_l --seed 42
```

v9a 对照逻辑：

```text
v9a vs v7b: 手工物化 CNN 分支是否带来稳定增益
v9a vs v8: 显式物化特征是否比 BiRNA full-mean 融合更有效
```

## v9b-v9f: handcrafted branch ablations

这组实验用于解释 v9a 的提升到底来自哪一类手工特征。除 v9f 外，所有版本都保留 v7b BiRNA 分支：

```text
BiRNA branch:
NUC global mean -> FiLM -> NUC center-window CNN mean + LoRA

Handcrafted branch:
selected handcrafted features -> multi-scale Conv1d(kernel=3,5,7) -> mean pooling

Fusion:
concat([birna_feat, handcrafted_feat]) -> MLP classifier
```

消融矩阵：

| Version | BiRNA branch | Handcrafted branch | Channels | Purpose |
|---|---|---|---:|---|
| v9a | v7b | ONEHOT+NCP+EIIP+ENAC | 12 | Full handcrafted branch |
| v9b | v7b | ONEHOT+NCP+EIIP | 8 | Remove ENAC |
| v9c | v7b | ONEHOT | 4 | Base identity only |
| v9d | v7b | NCP+EIIP | 4 | Physicochemical attributes only |
| v9e | v7b | ENAC | 4 | Local composition only |
| v9f | none | ONEHOT+NCP+EIIP+ENAC | 12 | Handcrafted-only baseline |

运行 Human_Brain：

```bash
python train.py --version v9b_no_enac_handcrafted_ablation --dataset H_b --seed 42
python train.py --version v9c_onehot_handcrafted_ablation --dataset H_b --seed 42
python train.py --version v9d_ncp_eiip_handcrafted_ablation --dataset H_b --seed 42
python train.py --version v9e_enac_handcrafted_ablation --dataset H_b --seed 42
python train.py --version v9f_handcrafted_only --dataset H_b --seed 42
```

三个人类数据集建议先跑 H-b/H-k/H-l 的 v9b 和 v9f：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v9b_no_enac_handcrafted_ablation --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v9b_no_enac_handcrafted_ablation --dataset H_k --seed 42
CUDA_VISIBLE_DEVICES=2 python train.py --version v9b_no_enac_handcrafted_ablation --dataset H_l --seed 42

CUDA_VISIBLE_DEVICES=0 python train.py --version v9f_handcrafted_only --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v9f_handcrafted_only --dataset H_k --seed 42
CUDA_VISIBLE_DEVICES=2 python train.py --version v9f_handcrafted_only --dataset H_l --seed 42
```

判断逻辑：

```text
v9b ~= v9a: ENAC 不是关键贡献
v9b << v9a: ENAC 有价值
v9c 强: one-hot 碱基身份解释主要提升
v9d 强: NCP/EIIP 物化属性有真实贡献
v9e 强: 局部碱基组成有真实贡献
v9f 接近 v9a: handcrafted features 单独很强，BiRNA 分支贡献需要重新评估
v9a > v9f 且 v9a > v7b: BiRNA 表征和手工特征互补
```

## 评估协议

每个数据集使用自己的：

```text
train.csv
test.csv
```

训练流程：

```text
strict_cv:

train.csv -> 分层 5 折
每折 80% train / 20% val
test.csv 始终作为 independent test
最终汇总 5 折 independent test mean/std
```

test_as_val:

```text
train.csv -> 全部作为训练集
test.csv -> 每个 epoch 评估，并按 ACC 选择 best epoch
最终报告该 best epoch 在 test.csv 上的 benchmark 指标
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

test-as-val 版本会写入单独目录，例如：

```text
outputs/v2_birna_bert_lora_test_as_val/Human_Brain/seed_42/
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

strict 版本默认有 `fold_1` 到 `fold_5`。test-as-val 版本默认是单次 benchmark run，只会有 `fold_1`，因为它直接使用完整 `train.csv` 训练。

默认不长期保存 `best_model.pt`。脚本会在评估完成后删除每折最优权重，保留指标和预测结果。若确实需要保留模型权重：

```bash
python train.py --version v2_birna_bert_lora --dataset H_b --seed 42 --keep_best_model
```

## 概率平均 ensemble 分析

该步骤不训练新模型，只读取已有版本的：

```text
outputs/<version>/<dataset>/seed_<seed>/fold_1/test_predictions.csv
```

然后对 positive-class `prob` 做简单平均，重新计算 ACC、MCC、AUC、AUPRC、F1、Precision、Recall。

H-b 推荐先跑：

```bash
python scripts/ensemble_predictions.py \
  --dataset H_b \
  --seed 42 \
  --versions v7b_nuc_global_nuc_center_cnn_film_lora v9a_birna_v7b_handcrafted_multiscale_cnn \
  --name h_b_v7b_plus_v9a
```

H-k 推荐先跑两组：

```bash
python scripts/ensemble_predictions.py \
  --dataset H_k \
  --seed 42 \
  --versions v7b_nuc_global_nuc_center_cnn_film_lora v9a_birna_v7b_handcrafted_multiscale_cnn \
  --name h_k_v7b_plus_v9a

python scripts/ensemble_predictions.py \
  --dataset H_k \
  --seed 42 \
  --versions v7b_nuc_global_nuc_center_cnn_film_lora v9d_ncp_eiip_handcrafted_ablation \
  --name h_k_v7b_plus_v9d
```

H-l 推荐先跑：

```bash
python scripts/ensemble_predictions.py \
  --dataset H_l \
  --seed 42 \
  --versions v9a_birna_v7b_handcrafted_multiscale_cnn v9e_enac_handcrafted_ablation \
  --name h_l_v9a_plus_v9e
```

输出位置：

```text
outputs/ensembles/<ensemble_name>/<dataset>/seed_<seed>/
├── ensemble_predictions.csv
├── ensemble_metrics.json
└── ensemble_summary.csv
```

## dry-run 检查

只查看解析后的命令，不启动训练：

```bash
python train.py --version v1_baseline --dataset H_b --seed 42 --dry_run
python train.py --version v2_birna_bert_lora --dataset H_b --seed 42 --dry_run
python train.py --version v3_birna_bert_bpe_dual_view --dataset H_b --seed 42 --dry_run
python train.py --version v4_birna_bert_bpe_dual_view_lora --dataset H_b --seed 42 --dry_run
python train.py --version v4_birna_bert_bpe_dual_view_lora_test_as_val --dataset H_b --seed 42 --dry_run
python train.py --version v5_nuc_lora_no_center --dataset H_b --seed 42 --dry_run
python train.py --version v5_nuc_lora_no_center_test_as_val --dataset H_b --seed 42 --dry_run
python train.py --version v6a_bpe_global_nuc_local_film_lora --dataset H_b --seed 42 --dry_run
python train.py --version v6b_nuc_global_nuc_local_film_lora --dataset H_b --seed 42 --dry_run
python train.py --version v7a_nuc_global_nuc_full_mean_film_lora --dataset H_b --seed 42 --dry_run
python train.py --version v7b_nuc_global_nuc_center_cnn_film_lora --dataset H_b --seed 42 --dry_run
python train.py --version v7c_bpe_global_nuc_full_cnn_film_lora --dataset H_b --seed 42 --dry_run
python train.py --version v7d_nuc_global_nuc_full_cnn_film_lora --dataset H_b --seed 42 --dry_run
python train.py --version v8_nuc_full_mean_center_cnn_film_lora --dataset H_b --seed 42 --dry_run
python train.py --version v9a_birna_v7b_handcrafted_multiscale_cnn --dataset H_b --seed 42 --dry_run
python train.py --version v9b_no_enac_handcrafted_ablation --dataset H_b --seed 42 --dry_run
python train.py --version v9c_onehot_handcrafted_ablation --dataset H_b --seed 42 --dry_run
python train.py --version v9d_ncp_eiip_handcrafted_ablation --dataset H_b --seed 42 --dry_run
python train.py --version v9e_enac_handcrafted_ablation --dataset H_b --seed 42 --dry_run
python train.py --version v9f_handcrafted_only --dataset H_b --seed 42 --dry_run
```
