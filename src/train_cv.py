from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from dataset_utils import SequenceSample, read_samples_from_file
from metrics_utils import format_metrics, json_safe_metrics
from model_birna_dual_view import BiRNADualViewClassifier
from model_birna_nuc import BiRNANucClassifier, load_birna_tokenizer
from training_utils import (
    DualViewDataCollator,
    RNANucDataset,
    NucDataCollator,
    append_train_log,
    evaluate,
    metric_score,
    remove_best_model_if_requested,
    resolve_path,
    save_predictions,
    select_device,
    set_seed,
    train_one_epoch,
    write_train_log_header,
)


METRIC_KEYS = ["ACC", "MCC", "AUC", "AUPRC", "F1", "Precision", "Recall"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run stratified k-fold cross validation on one 41bp m6A dataset with BiRNA-BERT."
    )
    parser.add_argument("--model_dir", type=Path, default=Path("./pretrained/birna-bert-model"))
    parser.add_argument("--tokenizer_dir", type=Path, default=Path("./pretrained/birna-bert-model"))
    parser.add_argument("--data_dir", type=Path, default=Path("./data/m6A_41bp/Human_Brain"))
    parser.add_argument("--output_dir", type=Path, default=Path("./outputs/human_brain_5fold_birna_nuc"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--use_bpe_view", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="Wqkv")
    parser.add_argument(
        "--keep_best_model",
        action="store_true",
        help="Keep each fold's best_model.pt after evaluation. By default checkpoints are deleted to save disk space.",
    )
    return parser.parse_args()


def load_single_dataset_train_test(data_dir: Path) -> tuple[list[SequenceSample], list[SequenceSample], dict]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Dataset train.csv not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Dataset test.csv not found: {test_path}")

    train_samples, train_stats = read_samples_from_file(train_path, expected_length=41)
    test_samples, test_stats = read_samples_from_file(test_path, expected_length=41)
    if not train_samples:
        raise ValueError(f"No valid training samples loaded from: {train_path}")
    if not test_samples:
        raise ValueError(f"No valid independent test samples loaded from: {test_path}")

    stats = {
        "data_dir": str(data_dir),
        "cv_source": str(train_path),
        "independent_test_source": str(test_path),
        "train_raw_records": train_stats["raw_records"],
        "test_raw_records": test_stats["raw_records"],
        "train_skipped": train_stats["skipped"],
        "test_skipped": test_stats["skipped"],
        "train_size": len(train_samples),
        "test_size": len(test_samples),
        "train_label_counts": dict(Counter(sample.label for sample in train_samples)),
        "test_label_counts": dict(Counter(sample.label for sample in test_samples)),
    }
    return train_samples, test_samples, stats


def make_loader(
    samples: list[SequenceSample],
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    use_bpe_view: bool,
):
    collator_cls = DualViewDataCollator if use_bpe_view else NucDataCollator
    return DataLoader(
        RNANucDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator_cls(tokenizer=tokenizer, max_length=max_length),
        num_workers=0,
    )


def train_one_fold(
    fold_idx: int,
    train_samples: list[SequenceSample],
    val_samples: list[SequenceSample],
    independent_test_samples: list[SequenceSample],
    tokenizer,
    args,
    device: torch.device,
) -> dict:
    fold_dir = args.output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = fold_dir / "train_log.csv"
    best_model_path = fold_dir / "best_model.pt"
    write_train_log_header(train_log_path)

    set_seed(args.seed + fold_idx - 1)
    lora_target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    model_cls = BiRNADualViewClassifier if args.use_bpe_view else BiRNANucClassifier
    model = model_cls(
        model_dir=args.model_dir,
        freeze_backbone=args.freeze_backbone,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
    )
    model.to(device)
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Fold {fold_idx} trainable_params: {trainable_params:,} / total_params: {total_params:,}")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(
        train_samples,
        tokenizer,
        args.max_length,
        args.batch_size,
        shuffle=True,
        use_bpe_view=args.use_bpe_view,
    )
    val_loader = make_loader(
        val_samples,
        tokenizer,
        args.max_length,
        args.batch_size,
        shuffle=False,
        use_bpe_view=args.use_bpe_view,
    )
    test_loader = make_loader(
        independent_test_samples,
        tokenizer,
        args.max_length,
        args.batch_size,
        shuffle=False,
        use_bpe_view=args.use_bpe_view,
    )

    best_score = -math.inf
    best_epoch = None
    best_val_metrics = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            freeze_backbone=args.freeze_backbone,
        )
        val_loss, val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            desc=f"Fold {fold_idx} epoch {epoch} val",
        )
        score = metric_score(val_metrics)
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save(
                {
                    "fold": fold_idx,
                    "epoch": epoch,
                    "best_score": best_score,
                    "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "fold_sizes": {
                        "train": len(train_samples),
                        "val": len(val_samples),
                        "independent_test": len(independent_test_samples),
                    },
                },
                best_model_path,
            )

        print(
            f"Fold {fold_idx} epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"{format_metrics(val_metrics)} best_score={best_score:.4f}"
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_score": best_score,
            "is_best": int(is_best),
        }
        row.update(val_metrics)
        append_train_log(train_log_path, row)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics, test_predictions = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        desc=f"Fold {fold_idx} independent test",
        return_predictions=True,
    )
    save_predictions(fold_dir / "test_predictions.csv", test_predictions)

    fold_payload = {
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_model_path": str(best_model_path),
        "best_model_deleted": False,
        "val_metrics_at_best": best_val_metrics,
        "independent_test_loss": test_loss,
        "independent_test_metrics": test_metrics,
        "fold_sizes": {
            "train": len(train_samples),
            "val": len(val_samples),
            "independent_test": len(independent_test_samples),
        },
        "label_counts": {
            "train": dict(Counter(sample.label for sample in train_samples)),
            "val": dict(Counter(sample.label for sample in val_samples)),
            "independent_test": dict(Counter(sample.label for sample in independent_test_samples)),
        },
    }
    metrics_path = fold_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe_metrics(fold_payload), handle, indent=2, ensure_ascii=False)

    best_model_deleted = remove_best_model_if_requested(best_model_path, args.keep_best_model)
    fold_payload["best_model_deleted"] = best_model_deleted
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe_metrics(fold_payload), handle, indent=2, ensure_ascii=False)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Fold {fold_idx} test_loss={test_loss:.4f} {format_metrics(test_metrics)}")
    if best_model_deleted:
        print(f"Fold {fold_idx} deleted best model checkpoint after evaluation: {best_model_path}")
    return fold_payload


def summarize_folds(fold_results: list[dict]) -> dict:
    summary = {"folds": fold_results, "independent_test_mean": {}, "independent_test_std": {}}
    for key in METRIC_KEYS:
        values = [
            result["independent_test_metrics"].get(key)
            for result in fold_results
            if result["independent_test_metrics"].get(key) is not None
        ]
        numeric_values = [float(value) for value in values if not math.isnan(float(value))]
        if numeric_values:
            summary["independent_test_mean"][key] = float(np.mean(numeric_values))
            summary["independent_test_std"][key] = float(np.std(numeric_values, ddof=1)) if len(numeric_values) > 1 else 0.0
        else:
            summary["independent_test_mean"][key] = math.nan
            summary["independent_test_std"][key] = math.nan
    return summary


def save_cv_summary_csv(path: Path, fold_results: list[dict], summary: dict):
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["fold", "best_epoch", "best_score", "independent_test_loss"] + METRIC_KEYS
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in fold_results:
            row = {
                "fold": result["fold"],
                "best_epoch": result["best_epoch"],
                "best_score": result["best_score"],
                "independent_test_loss": result["independent_test_loss"],
            }
            row.update(result["independent_test_metrics"])
            writer.writerow(row)
        mean_row = {"fold": "mean", "best_epoch": "", "best_score": "", "independent_test_loss": ""}
        mean_row.update(summary["independent_test_mean"])
        writer.writerow(mean_row)
        std_row = {"fold": "std", "best_epoch": "", "best_score": "", "independent_test_loss": ""}
        std_row.update(summary["independent_test_std"])
        writer.writerow(std_row)


def main():
    args = parse_args()
    args.model_dir = resolve_path(args.model_dir)
    args.tokenizer_dir = resolve_path(args.tokenizer_dir)
    args.data_dir = resolve_path(args.data_dir)
    args.output_dir = resolve_path(args.output_dir)

    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer.")
    if args.max_length < 43:
        raise ValueError("--max_length must be at least 43 for 41 NUC tokens plus CLS/SEP.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    print(f"model_dir: {args.model_dir}")
    print(f"tokenizer_dir: {args.tokenizer_dir}")
    print(f"data_dir: {args.data_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"folds: {args.folds}")
    print(f"freeze_backbone: {args.freeze_backbone}")
    print(f"use_bpe_view: {args.use_bpe_view}")
    print(f"use_lora: {args.use_lora}")
    print(f"keep_best_model: {args.keep_best_model}")
    if args.use_lora:
        print(
            "lora_config: "
            f"r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, "
            f"target_modules={args.lora_target_modules}"
        )

    cv_samples, independent_test_samples, data_stats = load_single_dataset_train_test(args.data_dir)
    labels = np.asarray([sample.label for sample in cv_samples])
    min_class_count = min(Counter(labels).values())
    if args.folds > min_class_count:
        raise ValueError(f"--folds={args.folds} is larger than the smallest class count: {min_class_count}")

    tokenizer = load_birna_tokenizer(args.tokenizer_dir, max_length=args.max_length)
    device = select_device()
    print(f"device: {device}")
    print("Data stats:")
    print(json.dumps(json_safe_metrics(data_stats), indent=2, ensure_ascii=False))

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []
    indices = np.arange(len(cv_samples))
    for fold_idx, (train_indices, val_indices) in enumerate(splitter.split(indices, labels), start=1):
        fold_train = [cv_samples[int(index)] for index in train_indices]
        fold_val = [cv_samples[int(index)] for index in val_indices]
        print(
            f"Fold {fold_idx}/{args.folds}: "
            f"train={len(fold_train)} val={len(fold_val)} independent_test={len(independent_test_samples)}"
        )
        fold_results.append(
            train_one_fold(
                fold_idx=fold_idx,
                train_samples=fold_train,
                val_samples=fold_val,
                independent_test_samples=independent_test_samples,
                tokenizer=tokenizer,
                args=args,
                device=device,
            )
        )

    summary = summarize_folds(fold_results)
    summary["data_stats"] = data_stats
    summary["args"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    with (args.output_dir / "cv_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe_metrics(summary), handle, indent=2, ensure_ascii=False)
    save_cv_summary_csv(args.output_dir / "cv_summary.csv", fold_results, summary)

    print("5-fold independent test mean:")
    print(format_metrics(summary["independent_test_mean"]))
    print("5-fold independent test std:")
    print(format_metrics(summary["independent_test_std"]))
    print(f"Saved CV outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
