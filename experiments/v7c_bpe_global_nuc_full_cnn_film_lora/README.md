# v7c_bpe_global_nuc_full_cnn_film_lora

## 实验目的

验证 BPE global 作为 FiLM 调制信号时，是否能提升可学习 NUC full 41bp CNN 分支。这个版本用于和 v7d 对比 BPE global 与 NUC global 的差异，也用于和 v6a 对比“中心窗口 mean”与“full 41bp CNN”的差异。

## 使用的数据集

默认使用：

```text
data/m6A_41bp/<dataset>/train.csv
data/m6A_41bp/<dataset>/test.csv
```

v7c 默认采用 test-as-validation 对标协议：使用完整 `train.csv` 训练，使用 `test.csv` 选择 best epoch，并在同一个 `test.csv` 上报告结果。该协议用于和已有同协议论文代码对齐，不作为严格 independent test 表述。

## 使用的模型或方法

```text
BiRNA-BERT BPE mean global
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

相对 v6a：

```text
film_global_view = bpe
film_nuc_pooling = full_cnn_mean
cnn_kernel_sizes = [3, 5, 7]
use_bpe_view = True
use_lora = True
```

## 训练命令

```bash
python train.py --version v7c_bpe_global_nuc_full_cnn_film_lora --dataset H_b --seed 42
```

多卡并行时可以手动指定单卡：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v7c_bpe_global_nuc_full_cnn_film_lora --dataset H_b --seed 42
```

## 输出结果保存位置

```text
outputs/v7c_bpe_global_nuc_full_cnn_film_lora/<dataset>/seed_<seed>/
```

默认训练结束后删除 `best_model.pt`，只保留指标、日志和预测结果。

## 与上一版本相比的变化

v6a 使用 BPE global 调制中心窗口 mean；v7c 使用 BPE global 调制 full 41bp CNN mean。该版本更接近“全局视图调制可学习局部/序列模式”的设计。

## 当前版本的实验结论或备注

需要重点和 v7d 比较。如果 v7c 明显优于 v7d，说明 BPE global 对 FiLM 有额外价值；如果不如 v7d，则说明当前 41bp m6A 任务里 NUC global 可能更稳定。

