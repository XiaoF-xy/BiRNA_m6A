# v11e_v9a_lora_r16_a64

基于 v9a，只调整 LoRA 参数：`r=16`，`alpha=64`，`dropout=0.05`，`lr=1e-4`，`weight_decay=0.01`。目的：同时提高 rank 和 alpha。

完整 v11 矩阵见 `../v11_lora_sweep_README.md`。

```bash
python train.py --version v11e_v9a_lora_r16_a64 --dataset H_b --seed 42
```

输出：`outputs/v11e_v9a_lora_r16_a64/<dataset>/seed_<seed>/`
