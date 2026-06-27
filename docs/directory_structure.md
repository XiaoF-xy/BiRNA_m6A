# Recommended Directory Structure

```text
BiRNA_m6A/
├── configs/                 # Global Python dataclass configs
├── data/                    # Raw, processed, and split datasets
├── docs/                    # Project notes, templates, comparison records
├── env/                     # Environment setup scripts
├── experiments/             # Versioned experiment configs and README files
├── pretrained/              # Local pretrained models and tokenizers
├── scripts/                 # Runnable command entrypoints
├── src/                     # Core implementation code
├── outputs/                 # Versioned run outputs
└── train.py                 # Unified experiment launcher
```

## Key Principles

- Keep common parameters in `configs/configarg.py`.
- Keep only version-specific differences in `experiments/v*/config_v*.py`.
- Keep each version self-documented with `experiments/v*/README.md`.
- Keep old stable scripts in `src/` until the new launcher is fully validated.
- Store run outputs by `version/dataset/seed` for reproducibility.

