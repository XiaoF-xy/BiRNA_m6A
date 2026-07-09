# v11b_v9a_lora_r8_a16

基于 v9a，只调整 LoRA 参数：`r=8`，`alpha=16`，`dropout=0.05`，`lr=1e-4`，`weight_decay=0.01`。目的：保持当前 rank，降低 alpha。

完整 v11 矩阵见 `../v11_lora_sweep_README.md`。

```bash
python train.py --version v11b_v9a_lora_r8_a16 --dataset H_b --seed 42
```

输出：`outputs/v11b_v9a_lora_r8_a16/<dataset>/seed_<seed>/`
