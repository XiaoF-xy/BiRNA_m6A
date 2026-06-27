# v3_birna_bert_bpe_dual_view

## 实验目的

验证在 v1 NUC baseline 的基础上加入 BiRNA-BERT BPE 视图，是否能补充局部 motif/片段级语义信息，提高 41bp RNA m6A 二分类性能。

## 使用的数据集

默认使用 `data/m6A_41bp/<dataset>/train.csv` 做分层 5 折交叉验证，并使用同目录下的 `test.csv` 作为 independent test。

## 使用的模型或方法

```text
NUC view:
BiRNA-BERT + nucleotide tokenization + mask-aware mean pooling + center pooling

BPE view:
BiRNA-BERT + BPE tokenization + mask-aware mean pooling

Fusion:
concat([nuc_mean, nuc_center, bpe_mean]) + MLP classifier
```

NUC 输入保持官方 nucleotide 格式：

```text
AGCTACGT -> A G C T A C G T
```

BPE 输入使用官方 BPE 格式：

```text
AGCTACGT -> AGCTACGT
```

该版本冻结 BiRNA-BERT backbone，不启用 LoRA，只训练 MLP classifier。

## 主要修改点

相对 v1，唯一结构变化是额外加入 BPE 视图：

```text
v1: concat([nuc_mean, nuc_center])
v3: concat([nuc_mean, nuc_center, bpe_mean])
```

BPE 视图不做 center pooling，因为 BPE token 是变长片段，无法保证某个 token 与第 21 位中心 m6A 位点一一对应。

## 训练命令

```bash
python train.py --version v3_birna_bert_bpe_dual_view --dataset H_b --seed 42
```

## 输出结果保存位置

```text
outputs/v3_birna_bert_bpe_dual_view/<dataset>/seed_<seed>/
```

## 与上一版本相比的变化

相对 v2，该版本不使用 LoRA；相对 v1，该版本加入 BPE 双视图，用于单独评估 BPE 视图本身是否有效。

## 当前版本的实验结论或备注

待运行 Human_Brain、Human_Kidney、Human_Liver 后，与 v1/v2 进行对照。
