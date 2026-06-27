# v1_baseline

## 实验目的

建立第一版可复现的 BiRNA-BERT NUC frozen baseline，用于 41bp RNA m6A 二分类预测。

## 使用的数据集

默认使用 `data/m6A_41bp/<dataset>/train.csv` 做分层 5 折交叉验证，并使用同目录下的 `test.csv` 作为 independent test。

## 使用的模型或方法

```text
BiRNA-BERT + NUC tokenization + mean pooling + center pooling + MLP classifier
```

该版本冻结 BiRNA-BERT backbone，只训练 MLP classifier。

## 主要修改点

这是初始 baseline，不加入 LoRA、BPE、FiLM、MoE 或手工特征。

## 训练命令

```bash
python train.py --version v1_baseline --dataset H_b --seed 42
```

## 输出结果保存位置

```text
outputs/v1_baseline/<dataset>/seed_<seed>/
```

## 与上一版本相比的变化

无上一版本；这是第一个可复现版本。

## 当前版本的实验结论或备注

Human_Brain 上该方法能学习到有效信号，但相对 MKE-ResNet 仍有差距。该版本用于后续 LoRA 和融合方法的基础对照。

