# v5_nuc_lora_no_center

## 实验目的

验证显式中心位点 pooling 对 41bp RNA m6A 二分类任务是否有贡献。

## 使用的数据集

默认使用 `data/m6A_41bp/<dataset>/train.csv` 做分层 5 折交叉验证，并使用同目录下的 `test.csv` 作为 independent test。

## 使用的模型或方法

```text
BiRNA-BERT + NUC tokenization + LoRA(Wqkv) + mean pooling + MLP classifier
```

该版本以 v2 为基础，只移除显式中心 token pooling：

```text
v2: concat([nuc_mean, nuc_center]) + LoRA
v5: nuc_mean + LoRA
```

## 主要修改点

相对 v2：

```text
use_center_pooling = False
use_lora = True
use_bpe_view = False
```

## 训练命令

```bash
python train.py --version v5_nuc_lora_no_center --dataset H_b --seed 42
```

test-as-val 对标协议：

```bash
python train.py --version v5_nuc_lora_no_center_test_as_val --dataset H_b --seed 42
```

## 输出结果保存位置

```text
outputs/v5_nuc_lora_no_center/<dataset>/seed_<seed>/
outputs/v5_nuc_lora_no_center_test_as_val/<dataset>/seed_<seed>/
```

## 与上一版本相比的变化

该版本不是主线结构升级，而是消融实验。目标是和 v2 直接比较，判断 `nuc_center` 是否真正提供增益。

## 当前版本的实验结论或备注

待运行 Human_Brain、Human_Kidney、Human_Liver 后判断是否继续保留显式中心位点 pooling。

