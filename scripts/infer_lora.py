from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Reserved LoRA inference entrypoint.")
    parser.add_argument("--version", type=str, default="v3_birna_bert_lora")
    parser.add_argument("--dataset", type=str, default="H_b")
    parser.add_argument("--seed", type=int, default=42)
    parser.parse_args()
    raise NotImplementedError(
        "LoRA inference is reserved for the next stage. Current experiments evaluate through "
        "src/train_birna_nuc_cv.py and save fold-level test_predictions.csv files."
    )


if __name__ == "__main__":
    main()

