# v9a_birna_v7b_handcrafted_multiscale_cnn

## 实验目的

v9a 用于验证显式手工物化特征是否能补充 BiRNA-BERT v7b 的上下文表示。v7b 已经证明 `NUC global -> FiLM -> NUC center-window CNN` 在 H-b/H-k 上对 MCC、F1 和 Recall 有价值；v9a 在此基础上增加一个轻量 handcrafted CNN 分支，避免一次性引入 MoE、BiLSTM 或四个独立手工特征分支。

核心问题：

```text
BiRNA-BERT v7b representation + handcrafted physicochemical CNN
是否优于 BiRNA-BERT v7b alone?
```

## 数据集与协议

默认使用 `data/m6A_41bp/<dataset>/train.csv` 和 `test.csv`。

当前版本默认采用 test-as-val 对标协议：

```text
train.csv -> 全部用于训练
test.csv  -> 每个 epoch 评估，并按 ACC 选择 best epoch
```

该协议只用于和已有同协议论文代码做 benchmark-style 对标，不作为严格 independent test 表述。

## 模型结构

v9a 包含两条分支：

```text
Branch 1: BiRNA-BERT v7b
NUC tokenization
-> BiRNA-BERT + LoRA(Wqkv)
-> NUC global mean
-> FiLM
-> NUC center-window multi-scale CNN mean
-> birna_feat

Branch 2: Handcrafted physicochemical CNN
ONEHOT [4, 41]
NCP    [3, 41]
EIIP   [1, 41]
ENAC   [4, 41]
-> concat [12, 41]
-> multi-scale CNN kernel 3/5/7
-> mean pooling
-> MLP
-> hand_feat

Fusion:
concat([birna_feat, hand_feat])
-> MLP classifier
```

## 手工特征

| 特征 | 维度 | 含义 |
|---|---:|---|
| ONEHOT | 4 x 41 | A/C/G/T 的 one-hot 碱基身份 |
| NCP | 3 x 41 | Nucleotide Chemical Property |
| EIIP | 1 x 41 | Electron-Ion Interaction Pseudopotential |
| ENAC | 4 x 41 | window size 5 的局部碱基组成 |

拼接后形成：

```text
[12, 41]
```

multi-scale CNN 使用 3、5、7 三种卷积核：

```text
kernel=3: 捕捉短局部 motif
kernel=5: 捕捉中等范围局部模式
kernel=7: 捕捉中心 A 附近更宽上下文
```

## 与上一版本相比的变化

| 版本 | 主分支 | 手工特征 | 目的 |
|---|---|---|---|
| v7b | NUC global -> FiLM -> NUC center-window CNN | 否 | 验证中心窗口 CNN 是否有用 |
| v8 | NUC full mean + center-window CNN fusion | 否 | 融合 v7a/v7b 的 BiRNA 分支 |
| v9a | v7b + handcrafted CNN branch | 是 | 验证显式物化特征是否补充 BiRNA-BERT |

v9a 不加入 BiLSTM、MoE、DNABERT 或四个独立手工 CNN 分支。这样可以把变量控制在“是否增加简洁手工物化 CNN 分支”这一点上。

## 运行命令

Human Brain：

```bash
python train.py --version v9a_birna_v7b_handcrafted_multiscale_cnn --dataset H_b --seed 42
```

三个人类数据集：

```bash
python train.py --version v9a_birna_v7b_handcrafted_multiscale_cnn --dataset H_b --seed 42
python train.py --version v9a_birna_v7b_handcrafted_multiscale_cnn --dataset H_k --seed 42
python train.py --version v9a_birna_v7b_handcrafted_multiscale_cnn --dataset H_l --seed 42
```

三张卡并行：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v9a_birna_v7b_handcrafted_multiscale_cnn --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v9a_birna_v7b_handcrafted_multiscale_cnn --dataset H_k --seed 42
CUDA_VISIBLE_DEVICES=2 python train.py --version v9a_birna_v7b_handcrafted_multiscale_cnn --dataset H_l --seed 42
```

## 输出位置

```text
outputs/v9a_birna_v7b_handcrafted_multiscale_cnn/<dataset>/seed_<seed>/
```

主要文件：

- `resolved_config.json`
- `cv_summary.csv`
- `cv_summary.json`
- `fold_1/metrics.json`
- `fold_1/train_log.csv`
- `fold_1/test_predictions.csv`

默认 `keep_best_model=False`，训练结束后删除 `best_model.pt` 以节省磁盘。

## 结果判断标准

优先对比：

```text
v9a vs v7b
```

如果 v9a 在 H-b/H-k/H-l 中至少两个数据集提升 ACC/MCC，说明手工物化 CNN 分支有稳定价值。若只提升 Precision 而 Recall/F1 下降，需要谨慎判断是否只是预测更保守。若 v9a 低于 v7b，则后续不应继续增加手工分支复杂度，应优先做 v7b/v8 ensemble 或更换融合策略。
