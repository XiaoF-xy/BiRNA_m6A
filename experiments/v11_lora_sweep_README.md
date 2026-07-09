# v11 LoRA Sweep

## 实验目的

v11 用于在当前主线 `v9a_birna_v7b_handcrafted_multiscale_cnn` 基础上做小范围 LoRA 参数微调。v9a 的结构不变，只调整 LoRA 容量、学习率和 weight decay。

固定不变的部分：

```text
BiRNA-BERT v7b branch:
NUC global mean -> FiLM -> NUC center-window multi-scale CNN mean + LoRA

Handcrafted branch:
ONEHOT+NCP+EIIP+ENAC -> multi-scale CNN(kernel=3,5,7) -> mean pooling

Fusion:
concat([birna_feat, handcrafted_feat]) -> MLP classifier
```

v11 不加入 gated fusion、MoE、attention、DNABERT 或新的手工特征。

## 评估协议

所有 v11 版本默认使用 `test_as_val`：

```text
train.csv -> 全部用于训练
test.csv  -> 每个 epoch 评估，并按 ACC 选择 best epoch
```

该协议用于与已有同协议论文代码做 benchmark-style 对标，不作为严格 independent test 表述。

## 参数矩阵

| 版本 | r | alpha | dropout | lr | weight_decay | 目的 |
|---|---:|---:|---:|---:|---:|---|
| `v11a_v9a_lora_r4_a16` | 4 | 16 | 0.05 | 1e-4 | 0.01 | 更小 LoRA，测试是否更稳 |
| `v11b_v9a_lora_r8_a16` | 8 | 16 | 0.05 | 1e-4 | 0.01 | 当前 r，降低 alpha |
| `v11c_v9a_lora_r8_a64` | 8 | 64 | 0.05 | 1e-4 | 0.01 | 当前 r，提高 alpha |
| `v11d_v9a_lora_r16_a32` | 16 | 32 | 0.05 | 1e-4 | 0.01 | 提高 rank，alpha 保守 |
| `v11e_v9a_lora_r16_a64` | 16 | 64 | 0.05 | 1e-4 | 0.01 | 提高 rank 和 alpha |
| `v11f_v9a_lora_r32_a64` | 32 | 64 | 0.05 | 1e-4 | 0.01 | 高容量 LoRA，测试上限 |
| `v11g_v9a_lora_r8_a32_lr5e5` | 8 | 32 | 0.05 | 5e-5 | 0.01 | 当前 LoRA 配置，只降低 lr |
| `v11h_v9a_lora_r16_a32_lr5e5` | 16 | 32 | 0.05 | 5e-5 | 0.01 | 中等 rank + 低 lr |
| `v11i_v9a_lora_r16_a64_lr5e5` | 16 | 64 | 0.05 | 5e-5 | 0.01 | 中高容量 + 低 lr |
| `v11j_v9a_lora_r32_a64_lr5e5` | 32 | 64 | 0.05 | 5e-5 | 0.01 | 高容量 + 低 lr |
| `v11k_v9a_lora_r16_a32_lr5e5_wd003` | 16 | 32 | 0.05 | 5e-5 | 0.03 | 中容量 + 更强正则 |
| `v11l_v9a_lora_r16_a64_lr5e5_wd003` | 16 | 64 | 0.05 | 5e-5 | 0.03 | 中高容量 + 更强正则 |

## 推荐运行顺序

先跑 6 个代表版本：

```text
v11d, v11e, v11g, v11h, v11i, v11k
```

如果这 6 个没有超过 v9a 的 ACC/MCC，剩余版本大概率只用于补充消融。

## 三卡并行示例

以 `v11d_v9a_lora_r16_a32` 为例：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v11d_v9a_lora_r16_a32 --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v11d_v9a_lora_r16_a32 --dataset H_k --seed 42
CUDA_VISIBLE_DEVICES=2 python train.py --version v11d_v9a_lora_r16_a32 --dataset H_l --seed 42
```

## 输出位置

```text
outputs/<v11_version>/<dataset>/seed_<seed>/
```

例如：

```text
outputs/v11d_v9a_lora_r16_a32/Human_Brain/seed_42/
```

## 结果判断

优先对比：

```text
v11x vs v9a
```

如果某个 v11 版本在至少两个 Human 数据集上超过 v9a 的 ACC/MCC，则保留为新的 LoRA 配置候选。若只提升 AUC/AUPRC 或 Recall，但 ACC/MCC 下降，不作为主线。
