# v11h_v9a_lora_r16_a32_lr5e5

基于 v9a，调整 LoRA 容量和学习率：`r=16`，`alpha=32`，`dropout=0.05`，`lr=5e-5`，`weight_decay=0.01`。目的：中等 rank + 低 lr。

完整 v11 矩阵见 `../v11_lora_sweep_README.md`。

```bash
python train.py --version v11h_v9a_lora_r16_a32_lr5e5 --dataset H_b --seed 42
```

输出：`outputs/v11h_v9a_lora_r16_a32_lr5e5/<dataset>/seed_<seed>/`
