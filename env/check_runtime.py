from __future__ import annotations

import sys
from pathlib import Path

import torch
import peft


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset_utils import load_dataset_splits, sequence_to_nuc_text  # noqa: E402
from model_birna_nuc import BiRNANucClassifier, load_birna_tokenizer  # noqa: E402


def main():
    model_dir = PROJECT_DIR / "pretrained" / "birna-bert-model"
    tokenizer_dir = PROJECT_DIR / "pretrained" / "birna-bert-model"
    data_dir = PROJECT_DIR / "data" / "m6A_41bp"

    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"peft: {peft.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")

    splits = load_dataset_splits(data_dir, seed=42, expected_length=41)
    print(f"split_sizes: {splits.stats['split_sizes']}")
    print(f"skipped: {splits.stats['skipped']}")

    tokenizer = load_birna_tokenizer(tokenizer_dir, max_length=64)
    sample_sequence = splits.train[0].sequence if splits.train else "A" * 41
    encoded = tokenizer(
        [sequence_to_nuc_text(sample_sequence)],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    print(f"tokenized_input_shape: {tuple(encoded['input_ids'].shape)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiRNANucClassifier(model_dir=model_dir, freeze_backbone=True)
    model.to(device)
    model.eval()
    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }
    with torch.no_grad():
        logits = model(**model_inputs)
        prob = torch.softmax(logits, dim=1)[:, 1]

    print(f"classifier_logits_shape: {tuple(logits.shape)}")
    print(f"sample_positive_prob: {prob.detach().cpu().tolist()[0]:.6f}")
    print("runtime_check: OK")


if __name__ == "__main__":
    main()
