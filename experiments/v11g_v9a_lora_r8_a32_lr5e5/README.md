# v11g_v9a_lora_r8_a32_lr5e5

基于 v9a，只调整优化学习率：`r=8`，`alpha=32`，`dropout=0.05`，`lr=5e-5`，`weight_decay=0.01`。目的：当前 LoRA 容量，只降低 lr。

完整 v11 矩阵见 `../v11_lora_sweep_README.md`。

```bash
python train.py --version v11g_v9a_lora_r8_a32_lr5e5 --dataset H_b --seed 42
```

输出：`outputs/v11g_v9a_lora_r8_a32_lr5e5/<dataset>/seed_<seed>/`
