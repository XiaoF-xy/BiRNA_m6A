from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.configarg import load_experiment_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Print the cv_summary.csv for a versioned experiment run.")
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="H_b")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_experiment_config(args.version, args.dataset, args.seed)
    summary_path = Path(config.training.output_dir) / "cv_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"CV summary not found: {summary_path}")
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            print(",".join(row))


if __name__ == "__main__":
    main()

