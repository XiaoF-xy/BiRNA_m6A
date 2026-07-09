# v11a_v9a_lora_r4_a16

基于 v9a，只调整 LoRA 参数：`r=4`，`alpha=16`，`dropout=0.05`，`lr=1e-4`，`weight_decay=0.01`。目的：测试更小 LoRA 是否比默认 `r=8/alpha=32` 更稳。

完整 v11 矩阵见 `../v11_lora_sweep_README.md`。

```bash
python train.py --version v11a_v9a_lora_r4_a16 --dataset H_b --seed 42
```

输出：`outputs/v11a_v9a_lora_r4_a16/<dataset>/seed_<seed>/`
