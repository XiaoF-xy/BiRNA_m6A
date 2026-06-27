from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset_utils import SequenceSample, sequence_to_nuc_text
from metrics_utils import compute_binary_metrics


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
