# BiRNA_m6A Cloud Environment

推荐云服务器环境：

```text
OS: Linux
Python: 3.10
GPU: NVIDIA CUDA 12.1 兼容驱动
PyTorch: 2.3.1 + cu121
Transformers: 4.41.2
```

## Conda 方式，推荐

```bash
cd BiRNA_m6A
bash env/create_conda_env_cuda121.sh
conda activate birna_m6a
```

脚本会自动执行：

```bash
python env/check_runtime.py
```

看到最后一行：

```text
runtime_check: OK
```

再开始正式训练。

## venv 方式

服务器已有 `python3.10` 时可以用：

```bash
cd BiRNA_m6A
bash env/create_venv_cuda121.sh
source .venv-birna/bin/activate
```

## CPU 调试方式

CPU 只适合检查代码链路，不建议跑完整训练：

```bash
cd BiRNA_m6A
bash env/create_venv_cpu.sh
source .venv-birna-cpu/bin/activate
```

## 正式训练

```bash
cd BiRNA_m6A

python src/train_birna_nuc_baseline.py \
  --model_dir ./pretrained/birna-bert-model \
  --tokenizer_dir ./pretrained/birna-bert-model \
  --data_dir ./data/m6A_41bp \
  --output_dir ./outputs/birna_nuc_baseline \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 42 \
  --freeze_backbone
```

如果服务器 CUDA 不是 12.1，不要硬用 `cu121` 脚本。请按服务器驱动支持的 CUDA wheel 修改脚本里的 PyTorch index URL，例如 `cu118`、`cu124` 或 CPU。
