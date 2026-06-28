# v6b_nuc_global_nuc_local_film_lora

## 实验目的

作为 v6a 的关键对照，验证 FiLM/local 结构本身是否有效，以及 BPE global 是否比 NUC global 更有价值。

## 使用的数据集

默认使用 `data/m6A_41bp/<dataset>/train.csv` 训练，并使用同目录下的 `test.csv` 选择 best epoch 和最终评估。

## 使用的模型或方法

```text
BiRNA-BERT NUC global -> FiLM(gamma, beta) -> BiRNA-BERT NUC local window + LoRA(Wqkv) + MLP classifier
```

局部窗口默认使用中心 A 附近 +/-3 nt：

```text
NUC local = nuc_emb[:, 17:24, :].mean(dim=1)
```

## 主要修改点

相对 v5：

```text
use_film = True
film_global_view = nuc
local_window_radius = 3
```

## 训练命令

v6b 默认就是 test-as-val 对标协议，不需要额外 `_test_as_val` 后缀。

```bash
python train.py --version v6b_nuc_global_nuc_local_film_lora --dataset H_b --seed 42
```

## 输出结果保存位置

```text
outputs/v6b_nuc_global_nuc_local_film_lora/<dataset>/seed_<seed>/
```

## 与上一版本相比的变化

v5 只使用 NUC mean；v6b 使用 NUC global 对中心附近 NUC local 窗口进行 FiLM 调制。它不是 DFM 的完整复刻，而是判断 BPE global 是否必要的对照。

## 当前版本的实验结论或备注

如果 v6b 与 v6a 持平或更好，说明当前 41bp m6A 任务里 BPE global 的额外价值有限；如果 v6a 明显更好，说明 BPE global 有贡献。
