# v9d_ncp_eiip_handcrafted_ablation

## Purpose

v9d keeps only NCP and EIIP in the handcrafted branch. It tests whether the physicochemical encodings themselves add signal beyond BiRNA-BERT NUC features.

## Method

```text
BiRNA branch: v7b NUC global -> FiLM -> NUC center-window multi-scale CNN + LoRA
Handcrafted branch: NCP + EIIP -> multi-scale CNN -> mean pooling
Fusion: concat(BiRNA feature, handcrafted feature) -> MLP classifier
```

## Feature Set

| Feature | Channels |
|---|---:|
| ONEHOT | 0 |
| NCP | 3 |
| EIIP | 1 |
| ENAC | 0 |
| Total | 4 |

## Run

```bash
python train.py --version v9d_ncp_eiip_handcrafted_ablation --dataset H_b --seed 42
python train.py --version v9d_ncp_eiip_handcrafted_ablation --dataset H_k --seed 42
python train.py --version v9d_ncp_eiip_handcrafted_ablation --dataset H_l --seed 42
```

## Output

```text
outputs/v9d_ncp_eiip_handcrafted_ablation/<dataset>/seed_<seed>/
```

## Interpretation

If v9d is competitive with v9c, physicochemical attributes are useful. If v9d is weak while v9c is strong, the handcrafted branch is mainly learning nucleotide identity.

