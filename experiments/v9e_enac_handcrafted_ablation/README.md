# v9e_enac_handcrafted_ablation

## Purpose

v9e keeps only ENAC in the handcrafted branch. It tests whether local nucleotide composition alone explains the v9a gains.

## Method

```text
BiRNA branch: v7b NUC global -> FiLM -> NUC center-window multi-scale CNN + LoRA
Handcrafted branch: ENAC(window=5) -> multi-scale CNN -> mean pooling
Fusion: concat(BiRNA feature, handcrafted feature) -> MLP classifier
```

## Feature Set

| Feature | Channels |
|---|---:|
| ONEHOT | 0 |
| NCP | 0 |
| EIIP | 0 |
| ENAC | 4 |
| Total | 4 |

## Run

```bash
python train.py --version v9e_enac_handcrafted_ablation --dataset H_b --seed 42
python train.py --version v9e_enac_handcrafted_ablation --dataset H_k --seed 42
python train.py --version v9e_enac_handcrafted_ablation --dataset H_l --seed 42
```

## Output

```text
outputs/v9e_enac_handcrafted_ablation/<dataset>/seed_<seed>/
```

## Interpretation

If v9e is close to v9a, local composition is the main contributor. If v9e is weak but v9b is strong, ENAC is not essential.

