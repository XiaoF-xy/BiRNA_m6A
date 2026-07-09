from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configs.configarg import canonical_dataset_name, load_experiment_config  # noqa: E402


METRIC_KEYS = ["ACC", "MCC", "AUC", "AUPRC", "F1", "Precision", "Recall"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average prediction probabilities from existing test_predictions.csv files. "
            "This is an analysis-only script: it does not load models or run training."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--versions",
        nargs="+",
        help=(
            "Experiment versions to ensemble. Prediction paths are resolved as "
            "outputs/<version>/<dataset>/seed_<seed>/fold_<fold>/test_predictions.csv."
        ),
    )
    input_group.add_argument(
        "--prediction_paths",
        nargs="+",
        type=Path,
        help="Explicit prediction CSV paths. Useful for temporary or external comparisons.",
    )
    parser.add_argument("--dataset", type=str, default="H_b", help="Dataset alias or canonical dataset name.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Ensemble run name. Defaults to a name derived from dataset and input versions.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to outputs/ensembles/<name>/<dataset>/seed_<seed>/.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._-")
    return value or "unnamed"


def short_source_name(source: str) -> str:
    source_path = Path(source)
    if source_path.name == "test_predictions.csv":
        parts = source_path.parts
        if len(parts) >= 4:
            return sanitize_name("_".join(parts[-4:-1]))
    return sanitize_name(source_path.stem or source)


def resolve_prediction_paths(args: argparse.Namespace) -> tuple[list[str], list[Path]]:
    if args.versions:
        names = args.versions
        paths = []
        for version in args.versions:
            config = load_experiment_config(version_name=version, dataset_name=args.dataset, seed=args.seed)
            prediction_path = Path(config.training.output_dir) / f"fold_{args.fold}" / "test_predictions.csv"
            paths.append(prediction_path)
        return names, paths

    paths = [path if path.is_absolute() else (Path.cwd() / path).resolve() for path in args.prediction_paths]
    names = [short_source_name(str(path)) for path in paths]
    return names, paths


def read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sequence", "label", "prob"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Prediction file {path} is missing required columns: {sorted(missing)}")
        for row_idx, row in enumerate(reader, start=2):
            try:
                sequence = str(row["sequence"])
                label = int(row["label"])
                prob = float(row["prob"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row in {path} at CSV line {row_idx}: {row}") from exc
            if not math.isfinite(prob):
                raise ValueError(f"Non-finite probability in {path} at CSV line {row_idx}: {prob}")
            rows.append({"sequence": sequence, "label": label, "prob": prob})

    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    return rows


def rows_match_by_position(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> bool:
    if len(reference) != len(candidate):
        return False
    return all(
        ref["sequence"] == cand["sequence"] and ref["label"] == cand["label"]
        for ref, cand in zip(reference, candidate)
    )


def build_unique_index(rows: list[dict[str, Any]], source_name: str) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates: set[tuple[str, int]] = set()
    for row in rows:
        key = (row["sequence"], row["label"])
        if key in index:
            duplicates.add(key)
        index[key] = row
    if duplicates:
        examples = ", ".join(f"{seq}/{label}" for seq, label in list(duplicates)[:3])
        raise ValueError(
            f"Cannot align {source_name} by sequence/label because duplicate keys exist. "
            f"Examples: {examples}. Ensure all prediction files use the same row order."
        )
    return index


def align_predictions(
    source_names: list[str],
    prediction_rows: list[list[dict[str, Any]]],
) -> tuple[list[str], list[int], list[list[float]]]:
    reference = prediction_rows[0]
    if all(rows_match_by_position(reference, rows) for rows in prediction_rows[1:]):
        sequences = [row["sequence"] for row in reference]
        labels = [row["label"] for row in reference]
        probs_by_source = [[row["prob"] for row in rows] for rows in prediction_rows]
        return sequences, labels, probs_by_source

    reference_index = build_unique_index(reference, source_names[0])
    reference_keys = [(row["sequence"], row["label"]) for row in reference]
    aligned_rows = [reference]
    for source_name, rows in zip(source_names[1:], prediction_rows[1:]):
        index = build_unique_index(rows, source_name)
        missing = [key for key in reference_keys if key not in index]
        extra = [key for key in index if key not in reference_index]
        if missing or extra:
            raise ValueError(
                f"Prediction files do not contain the same sequence/label keys for {source_name}. "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        aligned_rows.append([index[key] for key in reference_keys])

    sequences = [row["sequence"] for row in reference]
    labels = [row["label"] for row in reference]
    probs_by_source = [[row["prob"] for row in rows] for rows in aligned_rows]
    return sequences, labels, probs_by_source


def average_probabilities(probs_by_source: list[list[float]]) -> list[float]:
    if not probs_by_source:
        raise ValueError("At least one prediction source is required.")
    length = len(probs_by_source[0])
    if any(len(probs) != length for probs in probs_by_source):
        raise ValueError("Prediction sources have different lengths after alignment.")
    return [
        float(sum(source_probs[idx] for source_probs in probs_by_source) / len(probs_by_source))
        for idx in range(length)
    ]


def write_predictions(
    path: Path,
    sequences: list[str],
    labels: list[int],
    ensemble_probs: list[float],
    probs_by_source: list[list[float]],
    source_names: list[str],
    threshold: float,
) -> None:
    source_columns = [f"prob_{sanitize_name(name)}" for name in source_names]
    fieldnames = ["sequence", "label", "prob", "pred"] + source_columns
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (sequence, label, prob) in enumerate(zip(sequences, labels, ensemble_probs)):
            row = {
                "sequence": sequence,
                "label": label,
                "prob": prob,
                "pred": int(prob >= threshold),
            }
            for column, source_probs in zip(source_columns, probs_by_source):
                row[column] = source_probs[idx]
            writer.writerow(row)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["source", "type"] + METRIC_KEYS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_metric_helpers():
    from metrics_utils import compute_binary_metrics, json_safe_metrics

    return compute_binary_metrics, json_safe_metrics


def default_ensemble_name(args: argparse.Namespace, source_names: list[str]) -> str:
    dataset = canonical_dataset_name(args.dataset)
    if args.versions:
        compact_versions = "_plus_".join(source_names)
        return sanitize_name(f"{dataset}_{compact_versions}")
    return sanitize_name(f"{dataset}_{'_plus_'.join(source_names)}")


def resolve_output_dir(args: argparse.Namespace, ensemble_name: str) -> Path:
    if args.output_dir:
        return args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir).resolve()
    dataset = canonical_dataset_name(args.dataset)
    return PROJECT_ROOT / "outputs" / "ensembles" / ensemble_name / dataset / f"seed_{args.seed}"


def main() -> None:
    args = parse_args()
    if args.fold <= 0:
        raise ValueError("--fold must be a positive integer.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1].")

    source_names, prediction_paths = resolve_prediction_paths(args)
    if len(source_names) < 2:
        raise ValueError("At least two prediction sources are required for an ensemble.")

    compute_binary_metrics, json_safe_metrics = load_metric_helpers()
    prediction_rows = [read_predictions(path) for path in prediction_paths]
    sequences, labels, probs_by_source = align_predictions(source_names, prediction_rows)
    ensemble_probs = average_probabilities(probs_by_source)
    ensemble_metrics = compute_binary_metrics(labels, ensemble_probs, threshold=args.threshold)
    member_metrics = [
        {
            "source": source_name,
            "type": "member",
            **compute_binary_metrics(labels, probs, threshold=args.threshold),
        }
        for source_name, probs in zip(source_names, probs_by_source)
    ]
    summary_rows = member_metrics + [{"source": args.name or "ensemble", "type": "ensemble", **ensemble_metrics}]

    ensemble_name = sanitize_name(args.name) if args.name else default_ensemble_name(args, source_names)
    output_dir = resolve_output_dir(args, ensemble_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "ensemble_predictions.csv"
    metrics_path = output_dir / "ensemble_metrics.json"
    summary_path = output_dir / "ensemble_summary.csv"

    write_predictions(
        predictions_path,
        sequences,
        labels,
        ensemble_probs,
        probs_by_source,
        source_names,
        threshold=args.threshold,
    )
    payload = {
        "name": ensemble_name,
        "dataset": canonical_dataset_name(args.dataset),
        "seed": args.seed,
        "fold": args.fold,
        "threshold": args.threshold,
        "sources": [
            {"name": source_name, "path": str(path)}
            for source_name, path in zip(source_names, prediction_paths)
        ],
        "num_samples": len(labels),
        "metrics": ensemble_metrics,
        "member_metrics": member_metrics,
        "outputs": {
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "summary": str(summary_path),
        },
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe_metrics(payload), handle, indent=2, ensure_ascii=False)
    write_summary_csv(summary_path, summary_rows)

    print(f"ensemble_name: {ensemble_name}")
    print(f"output_dir: {output_dir}")
    print(f"num_samples: {len(labels)}")
    print("sources:")
    for source_name, path in zip(source_names, prediction_paths):
        print(f"  - {source_name}: {path}")
    print("ensemble metrics:")
    for key in METRIC_KEYS:
        print(f"  {key}: {ensemble_metrics[key]:.6f}")


if __name__ == "__main__":
    main()
