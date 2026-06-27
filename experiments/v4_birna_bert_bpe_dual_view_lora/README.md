# v4_birna_bert_bpe_dual_view_lora

## 实验目的

验证在 v3 NUC+BPE 双视图基础上加入 LoRA 微调后，是否能进一步提升 41bp RNA m6A 二分类性能。

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

Fine-tuning:
LoRA(Wqkv)
```

该版本不做 full fine-tune，训练 LoRA adapter 和 MLP classifier。

## 主要修改点

相对 v3，启用 LoRA：

```text
use_lora = True
lora_r = 8
lora_alpha = 32
lora_dropout = 0.05
lora_target_modules = ["Wqkv"]
```

相对 v2，额外加入 BPE 视图：

```text
v2: concat([nuc_mean, nuc_center]) + LoRA
v4: concat([nuc_mean, nuc_center, bpe_mean]) + LoRA
```

## 训练命令

```bash
python train.py --version v4_birna_bert_bpe_dual_view_lora --dataset H_b --seed 42
```

## 输出结果保存位置

```text
outputs/v4_birna_bert_bpe_dual_view_lora/<dataset>/seed_<seed>/
```

## 与上一版本相比的变化

相对 v3，该版本加入 LoRA 参数高效微调；用于判断 BPE 双视图和 LoRA 是否有叠加收益。

## 当前版本的实验结论或备注

待运行 Human_Brain、Human_Kidney、Human_Liver 后，与 v1/v2/v3 进行对照。
