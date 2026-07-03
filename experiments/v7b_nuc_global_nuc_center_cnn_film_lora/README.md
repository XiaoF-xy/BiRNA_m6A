# v7b_nuc_global_nuc_center_cnn_film_lora

## 实验目的

验证在保留中心窗口时，可学习的 NUC multi-scale CNN 是否优于普通中心窗口 mean。这个版本是 v6b 的直接升级对照。

## 使用的数据集

默认使用：

```text
data/m6A_41bp/<dataset>/train.csv
data/m6A_41bp/<dataset>/test.csv
```

v7b 默认采用 test-as-validation 对标协议：使用完整 `train.csv` 训练，使用 `test.csv` 选择 best epoch，并在同一个 `test.csv` 上报告结果。该协议用于和已有同协议论文代码对齐，不作为严格 independent test 表述。

## 使用的模型或方法

```text
BiRNA-BERT NUC mean global
  -> FiLM(gamma, beta)
  -> BiRNA-BERT NUC center-window multi-scale CNN mean
  -> LoRA(Wqkv)
  -> MLP classifier
```

被调制分支：

```text
nuc_emb [B, 41, H]
Conv1d kernels = 3, 5, 7
center window = 17:24
h_local = cnn_map[:, :, 17:24].mean(dim=2)
```

这个版本使用 CNN，保留中心窗口。

## 主要修改点

相对 v6b：

```text
film_global_view = nuc
film_nuc_pooling = center_cnn_mean
local_window_radius = 3
cnn_kernel_sizes = [3, 5, 7]
use_lora = True
```

## 训练命令

```bash
python train.py --version v7b_nuc_global_nuc_center_cnn_film_lora --dataset H_b --seed 42
```

多卡并行时可以手动指定单卡：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --version v7b_nuc_global_nuc_center_cnn_film_lora --dataset H_b --seed 42
```

## 输出结果保存位置

```text
outputs/v7b_nuc_global_nuc_center_cnn_film_lora/<dataset>/seed_<seed>/
```

默认训练结束后删除 `best_model.pt`，只保留指标、日志和预测结果。

## 与上一版本相比的变化

v6b 的被调制分支是不可学习的中心窗口 mean；v7b 在中心窗口之前加入多尺度 CNN，让局部 motif 表达可学习。它主要回答“中心附近的可学习局部模式是否有价值”。

## 当前版本的实验结论或备注

如果 v7b 高于 v6b，说明中心窗口里的 CNN 能提取比 mean 更有效的局部模式；如果没有提升，说明当前 41bp 任务中中心窗口 mean 已经足够，或者 CNN 引入了额外噪声。

