# v7a_nuc_global_nuc_full_mean_film_lora

## 实验目的

验证在 FiLM 结构中去掉中心窗口后是否仍然有效。这个版本用于和 `v6b_nuc_global_nuc_local_film_lora` 对比，核心问题是：41bp m6A 任务中，显式中心窗口是否必要。

## 使用的数据集

默认使用：

```text
data/m6A_41bp/<dataset>/train.csv
data/m6A_41bp/<dataset>/test.csv
```

v7a 默认采用 test-as-validation 对标协议：使用完整 `train.csv` 训练，使用 `test.csv` 选择 best epoch，并在同一个 `test.csv` 上报告结果。该协议用于和已有同协议论文代码对齐，不作为严格 independent test 表述。

## 使用的模型或方法

```text
BiRNA-BERT NUC mean global
  -> FiLM(gamma, beta)
  -> BiRNA-BERT NUC full 41bp mean
  -> LoRA(Wqkv)
  -> MLP classifier
```

被调制分支：

```text
h_local = nuc_emb[:, 0:41, :].mean(dim=1)
```

这个版本不使用 CNN，不使用中心窗口。

## 主要修改点

相对 v6b：

```text
film_global_view = nuc
film_nuc_pooling = full_mean
local_window_radius = 3  # 保留配置字段，但该版本不使用中心窗口
use_lora = True
```

## 训练命令

```bash
python train.py --version v7a_nuc_global_nuc_full_mean_film_lora --dataset H_b --seed 42
```

多卡并行时可以手动指定单卡：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v7a_nuc_global_nuc_full_mean_film_lora --dataset H_b --seed 42
```

## 输出结果保存位置

```text
outputs/v7a_nuc_global_nuc_full_mean_film_lora/<dataset>/seed_<seed>/
```

默认训练结束后删除 `best_model.pt`，只保留指标、日志和预测结果。

## 与上一版本相比的变化

v6b 使用 `NUC center-window mean` 作为被调制分支；v7a 改为 `NUC full 41bp mean`。如果 v7a 明显低于 v6b，说明中心窗口仍然重要；如果 v7a 持平或更好，说明中心窗口可能不是必要结构。

## 当前版本的实验结论或备注

该版本是中心窗口消融，不验证 CNN。需要和 v6b、v7b、v7d 一起看。

