# v2_birna_bert_feature

## 实验目的

预留给 BiRNA-BERT 表征与额外序列特征融合实验，例如 one-hot、NCP、EIIP、ENAC 等。

## 使用的数据集

默认沿用 `data/m6A_41bp/<dataset>/train.csv` 和 `test.csv`。

## 使用的模型或方法

计划在 v1 的 NUC 表征基础上加入额外特征输入。

## 主要修改点

相对 v1，计划启用 `use_multi_feature=True`，并逐步加入可解释的序列特征。

## 训练命令

当前为结构骨架，尚未作为正式可运行实验启用。

```bash
python train.py --version v2_birna_bert_feature --dataset H_b --seed 42
```

## 输出结果保存位置

```text
outputs/v2_birna_bert_feature/<dataset>/seed_<seed>/
```

## 与上一版本相比的变化

从 NUC-only baseline 扩展到多特征输入。

## 当前版本的实验结论或备注

当前仅保留配置和目录结构，正式实现前不作为论文结果。

