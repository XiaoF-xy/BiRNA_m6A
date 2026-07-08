# v9b_no_enac_handcrafted_ablation

## Purpose

v9b removes ENAC from v9a to test whether the sliding-window local composition feature is responsible for the v9a gains.

## Method

```text
BiRNA branch: v7b NUC global -> FiLM -> NUC center-window multi-scale CNN + LoRA
Handcrafted branch: ONEHOT + NCP + EIIP -> multi-scale CNN -> mean pooling
Fusion: concat(BiRNA feature, handcrafted feature) -> MLP classifier
```

## Feature Set

| Feature | Channels |
|---|---:|
| ONEHOT | 4 |
| NCP | 3 |
| EIIP | 1 |
| ENAC | 0 |
| Total | 8 |

## Run

```bash
python train.py --version v9b_no_enac_handcrafted_ablation --dataset H_b --seed 42
python train.py --version v9b_no_enac_handcrafted_ablation --dataset H_k --seed 42
python train.py --version v9b_no_enac_handcrafted_ablation --dataset H_l --seed 42
```

## Output

```text
outputs/v9b_no_enac_handcrafted_ablation/<dataset>/seed_<seed>/
```

## Interpretation

Compare directly with v9a. If v9b is close to v9a, ENAC is not necessary. If v9b drops clearly, ENAC contributes useful local composition information.

