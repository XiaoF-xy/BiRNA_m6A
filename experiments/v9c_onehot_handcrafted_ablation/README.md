# v9c_onehot_handcrafted_ablation

## Purpose

v9c keeps only ONEHOT in the handcrafted branch. It tests whether v9a gains mainly come from explicit nucleotide identity rather than physicochemical attributes.

## Method

```text
BiRNA branch: v7b NUC global -> FiLM -> NUC center-window multi-scale CNN + LoRA
Handcrafted branch: ONEHOT -> multi-scale CNN -> mean pooling
Fusion: concat(BiRNA feature, handcrafted feature) -> MLP classifier
```

## Feature Set

| Feature | Channels |
|---|---:|
| ONEHOT | 4 |
| NCP | 0 |
| EIIP | 0 |
| ENAC | 0 |
| Total | 4 |

## Run

```bash
python train.py --version v9c_onehot_handcrafted_ablation --dataset H_b --seed 42
python train.py --version v9c_onehot_handcrafted_ablation --dataset H_k --seed 42
python train.py --version v9c_onehot_handcrafted_ablation --dataset H_l --seed 42
```

## Output

```text
outputs/v9c_onehot_handcrafted_ablation/<dataset>/seed_<seed>/
```

## Interpretation

If v9c is close to v9a, the handcrafted branch is mostly adding base identity information. If v9c is much lower than v9a, NCP/EIIP/ENAC provide extra value.

