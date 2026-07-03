# v7d_nuc_global_nuc_full_cnn_film_lora

## 实验目的

验证 NUC global 调制 full 41bp 可学习 NUC CNN 是否有效。这个版本是 v7c 的 NUC global 对照，也是 v7a 的 CNN 对照。

## 使用的数据集

默认使用：

```text
data/m6A_41bp/<dataset>/train.csv
data/m6A_41bp/<dataset>/test.csv
```

v7d 默认采用 test-as-validation 对标协议：使用完整 `train.csv` 训练，使用 `test.csv` 选择 best epoch，并在同一个 `test.csv` 上报告结果。该协议用于和已有同协议论文代码对齐，不作为严格 independent test 表述。

## 使用的模型或方法

```text
BiRNA-BERT NUC mean global
  -> FiLM(gamma, beta)
  -> BiRNA-BERT NUC full 41bp multi-scale CNN mean
  -> LoRA(Wqkv)
  -> MLP classifier
```

被调制分支：

```text
nuc_emb [B, 41, H]
Conv1d kernels = 3, 5, 7
h_local = cnn_map.mean(dim=2)
```

这个版本使用 CNN，不使用中心窗口。

## 主要修改点

相对 v7a：

```text
film_global_view = nuc
film_nuc_pooling = full_cnn_mean
cnn_kernel_sizes = [3, 5, 7]
use_lora = True
```

## 训练命令

```bash
python train.py --version v7d_nuc_global_nuc_full_cnn_film_lora --dataset H_b --seed 42
```

多卡并行时可以手动指定单卡：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --version v7d_nuc_global_nuc_full_cnn_film_lora --dataset H_b --seed 42
```

## 输出结果保存位置

```text
outputs/v7d_nuc_global_nuc_full_cnn_film_lora/<dataset>/seed_<seed>/
```

默认训练结束后删除 `best_model.pt`，只保留指标、日志和预测结果。

## 与上一版本相比的变化

v7a 使用不可学习的 full 41bp mean；v7d 在 full 41bp NUC 分支加入多尺度 CNN。它主要回答“去掉中心窗口后，全序列 CNN 是否比全序列 mean 更好”。

## 当前版本的实验结论或备注

如果 v7d 高于 v7a，说明 CNN 的可学习 motif 表达有价值；如果 v7d 高于 v7c，说明 NUC global 作为调制信号比 BPE global 更适合当前 41bp 任务。

