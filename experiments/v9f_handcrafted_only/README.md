# v9f_handcrafted_only

## Purpose

v9f removes the BiRNA-BERT branch entirely and trains only the handcrafted multi-scale CNN. It estimates how much signal the explicit handcrafted features can provide without pretrained sequence embeddings.

## Method

```text
Input sequence
-> ONEHOT + NCP + EIIP + ENAC [12, 41]
-> multi-scale CNN kernel 3/5/7
-> mean pooling
-> MLP classifier
```

No BiRNA-BERT backbone, no FiLM, no LoRA.

## Feature Set

| Feature | Channels |
|---|---:|
| ONEHOT | 4 |
| NCP | 3 |
| EIIP | 1 |
| ENAC | 4 |
| Total | 12 |

## Run

```bash
python train.py --version v9f_handcrafted_only --dataset H_b --seed 42
python train.py --version v9f_handcrafted_only --dataset H_k --seed 42
python train.py --version v9f_handcrafted_only --dataset H_l --seed 42
```

## Output

```text
outputs/v9f_handcrafted_only/<dataset>/seed_<seed>/
```

## Interpretation

If v9f is close to v9a, handcrafted features alone are strong and BiRNA-BERT contributes less than expected. If v9f is much weaker than v9a, the full model gain comes from combining pretrained BiRNA features with handcrafted attributes.

