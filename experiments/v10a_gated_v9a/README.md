# v10a_gated_v9a

## 实验目的

v10a 用于验证 v9a 的简单拼接融合是否可以被更合理的可学习 gate 融合替代。

核心问题：

```text
BiRNA-BERT v7b 分支 + handcrafted 分支
使用 learnable gated fusion
是否优于 v9a 的 concat fusion?
```

该版本不加入 MoE、attention、DNABERT 或新的手工特征，只改变融合方式。

## 数据集与协议

默认使用：

```text
data/m6A_41bp/<dataset>/train.csv
data/m6A_41bp/<dataset>/test.csv
```

当前版本采用 test-as-validation 对标协议：

```text
train.csv -> 全部用于训练
test.csv  -> 每个 epoch 评估，并按 ACC 选择 best epoch
```

该协议用于和已有同协议论文代码做 benchmark-style 对标，不作为严格 independent test 表述。

## 模型结构

v10a 包含两条分支：

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
```

融合方式从 v9a 的直接拼接改为可学习向量门控：

```text
birna_proj = Linear(birna_feat)       -> [B, 256]
hand_proj  = Linear(hand_feat)        -> [B, 256]
gate       = sigmoid(MLP([birna_proj, hand_proj])) -> [B, 256]
fused      = gate * birna_proj + (1 - gate) * hand_proj
logits     = MLP(fused)
```

## 与上一版本相比的变化

| 版本 | BiRNA 分支 | 手工分支 | 融合方式 | 目的 |
|---|---|---|---|---|
| v9a | v7b | ONEHOT+NCP+EIIP+ENAC | concat + MLP | 验证手工分支是否补充 BiRNA |
| v10a | v7b | ONEHOT+NCP+EIIP+ENAC | gated fusion + MLP | 验证动态分支权重是否优于直接拼接 |

v10a 的变量只在融合层，便于和 v9a 做直接对比。

## 运行命令

Human Brain：

```bash
python train.py --version v10a_gated_v9a --dataset H_b --seed 42
```

三个人类数据集：

```bash
python train.py --version v10a_gated_v9a --dataset H_b --seed 42
python train.py --version v10a_gated_v9a --dataset H_k --seed 42
python train.py --version v10a_gated_v9a --dataset H_l --seed 42
```

三张卡并行：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v10a_gated_v9a --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v10a_gated_v9a --dataset H_k --seed 42
CUDA_VISIBLE_DEVICES=2 python train.py --version v10a_gated_v9a --dataset H_l --seed 42
```

## 输出位置

```text
outputs/v10a_gated_v9a/<dataset>/seed_<seed>/
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
v10a vs v9a
```

如果 v10a 在 H-b/H-k/H-l 中至少两个数据集提升 ACC/MCC，说明 gated fusion 比直接拼接更适合当前两分支结构。如果只提升 AUC/AUPRC 但 ACC/MCC 下降，则说明排序能力改善但 0.5 阈值下分类边界没有变好，后续更适合做校准或 ensemble，而不是继续加复杂结构。
