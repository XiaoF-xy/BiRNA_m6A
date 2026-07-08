# v8_nuc_full_mean_center_cnn_film_lora

## 实验目的

v8 用来验证一个更直接的融合假设：v7a 的全长 41bp NUC mean 分支在 H-b/H-k 上表现稳定，v7b 的中心窗口 CNN 分支在 MCC/F1/Recall 上更有优势，因此把两条分支同时保留，让模型自己在分类头中组合全局平均信息和中心窗口 CNN 信息。

该版本仍然只使用 BiRNA-BERT NUC 视图，不加入 BPE、MoE 或额外手工特征。

## 使用数据集

默认使用 `data/m6A_41bp/<dataset>/` 下的数据集。命令中只需要指定数据集别名：

- `H_b`: Human_Brain
- `H_k`: Human_Kidney
- `H_l`: Human_Liver

当前版本默认采用 test-as-validation 对标协议：使用完整 `train.csv` 训练，每个 epoch 在 `test.csv` 上选择 best epoch。该协议只用于和已有论文代码公平对标，不作为严格 independent test 表述。

## 方法结构

输入序列经过 BiRNA-BERT NUC tokenization：

```text
AGCTACGT -> A G C T A C G T
```

BiRNA-BERT 输出 NUC token embedding 后，v8 构造三部分特征：

```python
h_global = nuc_emb.mean(dim=1)

h_full = nuc_emb.mean(dim=1)

h_center_cnn = MultiScaleCNN(nuc_emb)
h_center_cnn = h_center_cnn[:, 17:24, :].mean(dim=1)

gamma_full, beta_full = FiLM_full(h_global)
h_full_mod = gamma_full * h_full + beta_full

gamma_center, beta_center = FiLM_center_cnn(h_global)
h_center_mod = gamma_center * h_center_cnn + beta_center

feat = concat([h_global, h_full_mod, h_center_mod])
logits = MLP(feat)
```

其中 MultiScaleCNN 使用 `kernel_size = 3, 5, 7`，中心窗口半径为 3，对应 41bp 序列中的 `17:24`。

## 与上一版本相比的变化

| 版本 | global | 被调制分支 | CNN | 中心窗口 | 作用 |
|---|---|---|---|---|---|
| v7a | NUC mean | NUC full 41bp mean | 否 | 否 | 验证去掉中心窗口是否有用 |
| v7b | NUC mean | NUC center-window CNN mean | 是 | 是 | 验证中心窗口 CNN 是否有用 |
| v8 | NUC mean | NUC full mean + center-window CNN mean | 是 | 部分保留 | 融合 v7a 的稳定全长信息和 v7b 的中心局部 CNN 信息 |

## 训练命令

在项目根目录运行：

```bash
python train.py --version v8_nuc_full_mean_center_cnn_film_lora --dataset H_b --seed 42
```

三个人类数据集：

```bash
python train.py --version v8_nuc_full_mean_center_cnn_film_lora --dataset H_b --seed 42
python train.py --version v8_nuc_full_mean_center_cnn_film_lora --dataset H_k --seed 42
python train.py --version v8_nuc_full_mean_center_cnn_film_lora --dataset H_l --seed 42
```

多卡并行时，每个终端指定一张卡：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v8_nuc_full_mean_center_cnn_film_lora --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v8_nuc_full_mean_center_cnn_film_lora --dataset H_k --seed 42
CUDA_VISIBLE_DEVICES=2 python train.py --version v8_nuc_full_mean_center_cnn_film_lora --dataset H_l --seed 42
```

## 输出位置

输出目录由 `configs/configarg.py` 自动生成：

```text
outputs/v8_nuc_full_mean_center_cnn_film_lora/<dataset>/seed_<seed>/
```

主要文件：

- `metrics.json`: 最终指标和配置摘要
- `train_log.csv`: 每个 epoch 的训练和选择集指标
- `test_predictions.csv`: `sequence,label,prob,pred`
- `fold_1/`: 单次 test-as-val run 的中间输出

默认 `keep_best_model=False`，训练完成后会删除 `best_model.pt`，避免大量实验占用磁盘。

## 当前结论备注

v8 是 v7a/v7b 之后的融合验证版本。重点看它是否同时保留 v7a 的 ACC/AUC/AUPRC 稳定性，以及 v7b 在 MCC/F1/Recall 上的优势。如果 v8 不能明显超过 v7a/v7b，说明简单拼接双局部分支不足以带来稳定收益，后续应优先考虑更强的专家选择机制或更贴近 DFM 的 global-vector/MoE 结构。
