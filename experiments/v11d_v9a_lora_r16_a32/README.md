# v11d_v9a_lora_r16_a32

基于 v9a，只调整 LoRA 参数：`r=16`，`alpha=32`，`dropout=0.05`，`lr=1e-4`，`weight_decay=0.01`。目的：提高 rank，alpha 保守。

完整 v11 矩阵见 `../v11_lora_sweep_README.md`。

```bash
python train.py --version v11d_v9a_lora_r16_a32 --dataset H_b --seed 42
```

输出：`outputs/v11d_v9a_lora_r16_a32/<dataset>/seed_<seed>/`
