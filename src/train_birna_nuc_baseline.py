from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset_utils import SequenceSample, load_dataset_splits, sequence_to_nuc_text
from metrics_utils import compute_binary_metrics, format_metrics, json_safe_metrics
from model_birna_nuc import BiRNANucClassifier, load_birna_tokenizer


class RNANucDataset(Dataset):
    def __init__(self, samples: list[SequenceSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "sequence": sample.sequence,
            "label": sample.label,
        }


class NucDataCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        sequences = [item["sequence"] for item in batch]
        texts = [sequence_to_nuc_text(sequence) for sequence in sequences]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        encoded["sequences"] = sequences
        return encoded


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the first BiRNA-BERT NUC baseline for 41bp RNA m6A binary classification."
    )
    parser.add_argument("--model_dir", type=Path, default=Path("./pretrained/birna-bert-model"))
    parser.add_argument("--tokenizer_dir", type=Path, default=Path("./pretrained/birna-bert-model"))
    parser.add_argument("--data_dir", type=Path, default=Path("./data/m6A_41bp"))
    parser.add_argument("--output_dir", type=Path, default=Path("./outputs/birna_nuc_baseline"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="Wqkv")
    parser.add_argument(
        "--keep_best_model",
        action="store_true",
        help="Keep best_model.pt after final evaluation. By default the checkpoint is deleted to save disk space.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch: dict, device: torch.device) -> tuple[dict, torch.Tensor, list[str]]:
    labels = batch["labels"].to(device)
    sequences = batch["sequences"]
    model_inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }
    return model_inputs, labels, sequences


def train_one_epoch(model, loader, optimizer, criterion, device, epoch: int, freeze_backbone: bool) -> float:
    model.train()
    if freeze_backbone and not getattr(model, "use_lora", False):
        model.birna_model.eval()

    total_loss = 0.0
    total_samples = 0
    progress = tqdm(loader, desc=f"Epoch {epoch} train", ncols=100)
    for batch in progress:
        model_inputs, labels, _ = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(**model_inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc: str, return_predictions: bool = False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_probs = []
    predictions = []

    for batch in tqdm(loader, desc=desc, ncols=100):
        model_inputs, labels, sequences = move_batch_to_device(batch, device)
        logits = model(**model_inputs)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= 0.5).long()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_labels.extend(labels.detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())

        if return_predictions:
            for sequence, label, prob, pred in zip(
                sequences,
                labels.detach().cpu().tolist(),
                probs.detach().cpu().tolist(),
                preds.detach().cpu().tolist(),
            ):
                predictions.append({
                    "sequence": sequence,
                    "label": int(label),
                    "prob": float(prob),
                    "pred": int(pred),
                })

    avg_loss = total_loss / max(total_samples, 1)
    metrics = compute_binary_metrics(all_labels, all_probs)
    if return_predictions:
        return avg_loss, metrics, predictions
    return avg_loss, metrics


def metric_score(metrics: dict[str, float]) -> float:
    mcc = metrics.get("MCC", math.nan)
    if not math.isnan(mcc):
        return mcc
    auc = metrics.get("AUC", math.nan)
    if not math.isnan(auc):
        return auc
    return -math.inf


def write_train_log_header(path: Path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "ACC",
            "MCC",
            "AUC",
            "AUPRC",
            "F1",
            "Precision",
            "Recall",
            "best_score",
            "is_best",
        ])


def append_train_log(path: Path, row: dict):
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "epoch",
            "train_loss",
            "val_loss",
            "ACC",
            "MCC",
            "AUC",
            "AUPRC",
            "F1",
            "Precision",
            "Recall",
            "best_score",
            "is_best",
        ])
        writer.writerow(row)


def save_predictions(path: Path, predictions: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "label", "prob", "pred"])
        writer.writeheader()
        writer.writerows(predictions)


def remove_best_model_if_requested(best_model_path: Path, keep_best_model: bool) -> bool:
    if keep_best_model:
        return False
    if best_model_path.exists():
        best_model_path.unlink()
        return True
    return False


def main():
    args = parse_args()
    args.model_dir = resolve_path(args.model_dir)
    args.tokenizer_dir = resolve_path(args.tokenizer_dir)
    args.data_dir = resolve_path(args.data_dir)
    args.output_dir = resolve_path(args.output_dir)

    if args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer.")
    if args.max_length < 43:
        raise ValueError("--max_length must be at least 43 for 41 NUC tokens plus CLS/SEP.")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"model_dir: {args.model_dir}")
    print(f"tokenizer_dir: {args.tokenizer_dir}")
    print(f"data_dir: {args.data_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"freeze_backbone: {args.freeze_backbone}")
    print(f"use_lora: {args.use_lora}")
    print(f"keep_best_model: {args.keep_best_model}")
    if args.use_lora:
        print(
            "lora_config: "
            f"r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, "
            f"target_modules={args.lora_target_modules}"
        )

    splits = load_dataset_splits(args.data_dir, seed=args.seed, expected_length=41)
    print("Data stats:")
    print(json.dumps(json_safe_metrics(splits.stats), indent=2, ensure_ascii=False))

    tokenizer = load_birna_tokenizer(args.tokenizer_dir, max_length=args.max_length)
    train_dataset = RNANucDataset(splits.train)
    val_dataset = RNANucDataset(splits.val)
    test_dataset = RNANucDataset(splits.test)
    collator = NucDataCollator(tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    device = select_device()
    print(f"device: {device}")
    lora_target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    model = BiRNANucClassifier(
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
    print(f"trainable_params: {trainable_params:,} / total_params: {total_params:,}")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()
    best_score = -math.inf
    best_epoch = None
    best_model_path = args.output_dir / "best_model.pt"
    train_log_path = args.output_dir / "train_log.csv"
    write_train_log_header(train_log_path)

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
            desc=f"Epoch {epoch} val",
        )
        score = metric_score(val_metrics)
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "best_score": best_score,
                    "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                },
                best_model_path,
            )

        print(
            f"Epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"{format_metrics(val_metrics)} best_score={best_score:.4f}"
        )
        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_score": best_score,
            "is_best": int(is_best),
        }
        log_row.update(val_metrics)
        append_train_log(train_log_path, log_row)

    if not best_model_path.exists():
        raise RuntimeError(f"Best model was not saved: {best_model_path}")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics, test_predictions = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        desc="Test",
        return_predictions=True,
    )
    save_predictions(args.output_dir / "test_predictions.csv", test_predictions)

    metrics_payload = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_metric_priority": "MCC first, AUC fallback if MCC is nan",
        "best_model_path": str(best_model_path),
        "best_model_deleted": False,
        "val_metrics_at_best": checkpoint.get("val_metrics", {}),
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "data_stats": splits.stats,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe_metrics(metrics_payload), handle, indent=2, ensure_ascii=False)

    best_model_deleted = remove_best_model_if_requested(best_model_path, args.keep_best_model)
    metrics_payload["best_model_deleted"] = best_model_deleted
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe_metrics(metrics_payload), handle, indent=2, ensure_ascii=False)

    print(f"Best epoch: {best_epoch}, best_score={best_score:.4f}")
    print(f"Test loss={test_loss:.4f} {format_metrics(test_metrics)}")
    if best_model_deleted:
        print(f"Deleted best model checkpoint after evaluation: {best_model_path}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
