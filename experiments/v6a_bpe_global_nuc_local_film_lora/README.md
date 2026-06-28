# v6a_bpe_global_nuc_local_film_lora

## 实验目的

验证 BPE global 表征是否能通过 FiLM 调制 NUC local 表征，从而比简单 concat 的 BPE 双视图更有效。

## 使用的数据集

默认使用 `data/m6A_41bp/<dataset>/train.csv` 训练，并使用同目录下的 `test.csv` 选择 best epoch 和最终评估。

## 使用的模型或方法

```text
BiRNA-BERT BPE global -> FiLM(gamma, beta) -> BiRNA-BERT NUC local window + LoRA(Wqkv) + MLP classifier
```

局部窗口默认使用中心 A 附近 +/-3 nt：

```text
NUC local = nuc_emb[:, 17:24, :].mean(dim=1)
```

## 主要修改点

相对 v5：

```text
use_film = True
film_global_view = bpe
local_window_radius = 3
```

## 训练命令

v6a 默认就是 test-as-val 对标协议，不需要额外 `_test_as_val` 后缀。

```bash
python train.py --version v6a_bpe_global_nuc_local_film_lora --dataset H_b --seed 42
```

## 输出结果保存位置

```text
outputs/v6a_bpe_global_nuc_local_film_lora/<dataset>/seed_<seed>/
```

## 与上一版本相比的变化

v5 只使用 NUC mean；v6a 加入 DFM-inspired global-to-local FiLM 调制，但暂时不加入 MoE，以便先判断 FiLM 是否有效。

## 当前版本的实验结论或备注

待运行 H-b/H-k/H-l 后判断 BPE global 是否提供稳定增益。
