# v4_moe_fusion

## 实验目的

预留给后续 NUC、BPE、手工序列特征和 MoE/attention 融合实验。

## 使用的数据集

默认沿用 `data/m6A_41bp/<dataset>/train.csv` 和 `test.csv`，必要时扩展到 201bp 或其他窗口。

## 使用的模型或方法

计划方法：

```text
BiRNA-BERT NUC/BPE + sequence features + attention/MoE fusion + classifier
```

## 主要修改点

相对 LoRA 版本，计划加入多视角特征和 MoE 融合模块。

## 训练命令

当前为结构骨架，尚未作为正式可运行实验启用。

```bash
python train.py --version v4_moe_fusion --dataset H_b --seed 42
```

## 输出结果保存位置

```text
outputs/v4_moe_fusion/<dataset>/seed_<seed>/
```

## 与上一版本相比的变化

从单一 NUC/LoRA 表征扩展到多特征、多专家融合。

## 当前版本的实验结论或备注

当前仅保留配置和目录结构，正式实现前不作为论文结果。

