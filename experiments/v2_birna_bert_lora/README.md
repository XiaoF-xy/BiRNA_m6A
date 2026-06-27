# v2_birna_bert_lora

## 实验目的

验证 BiRNA-BERT NUC baseline 加入 LoRA 后是否能提高 41bp RNA m6A 二分类性能。

## 使用的数据集

默认使用 `data/m6A_41bp/<dataset>/train.csv` 做分层 5 折交叉验证，并使用 `test.csv` 作为 independent test。

## 使用的模型或方法

```text
BiRNA-BERT + NUC tokenization + LoRA(Wqkv) + mean pooling + center pooling + MLP classifier
```

## 主要修改点

相对 v1，启用 LoRA：

```text
use_lora = True
lora_r = 8
lora_alpha = 32
lora_dropout = 0.05
lora_target_modules = ["Wqkv"]
```

原始 backbone 权重不做 full fine-tune，只训练 LoRA adapter 和 MLP classifier。

## 训练命令

```bash
python train.py --version v2_birna_bert_lora --dataset H_b --seed 42
```

## 输出结果保存位置

```text
outputs/v2_birna_bert_lora/<dataset>/seed_<seed>/
outputs/lora_adapters/v2_birna_bert_lora/<dataset>/seed_<seed>/
```

## 与上一版本相比的变化

在 v1 frozen baseline 上加入 LoRA 参数高效微调。

## 当前版本的实验结论或备注

Human_Brain 上 LoRA 相对 frozen baseline 已观察到 ACC、MCC、AUC、AUPRC、F1 和 Recall 提升。下一步优先跑 Human_Kidney 和 Human_Liver。
