# v11l_v9a_lora_r16_a64_lr5e5_wd003

基于 v9a，调整 LoRA 容量、学习率和正则：`r=16`，`alpha=64`，`dropout=0.05`，`lr=5e-5`，`weight_decay=0.03`。目的：中高容量 + 更强正则。

完整 v11 矩阵见 `../v11_lora_sweep_README.md`。

```bash
python train.py --version v11l_v9a_lora_r16_a64_lr5e5_wd003 --dataset H_b --seed 42
```

输出：`outputs/v11l_v9a_lora_r16_a64_lr5e5_wd003/<dataset>/seed_<seed>/`
