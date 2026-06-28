from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model_birna_dual_view import mask_aware_mean_pool
from model_birna_nuc import apply_lora_to_birna, load_birna_backbone


class FiLM(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim * 2),
        )

    def forward(self, global_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.mlp(global_feat)
        return torch.chunk(params, 2, dim=-1)


class BiRNAFiLMLocalClassifier(nn.Module):
    def __init__(
        self,
        model_dir: Path,
        freeze_backbone: bool = True,
        dropout: float = 0.2,
        center_index: int = 20,
        local_window_radius: int = 3,
        film_global_view: str = "bpe",
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
    ):
        super().__init__()
        if film_global_view not in {"bpe", "nuc"}:
            raise ValueError(f"film_global_view must be 'bpe' or 'nuc', got: {film_global_view}")
        if local_window_radius < 0:
            raise ValueError(f"local_window_radius must be non-negative, got: {local_window_radius}")

        self.birna_model = load_birna_backbone(model_dir)
        self.use_lora = use_lora
        self.center_index = center_index
        self.local_window_radius = local_window_radius
        self.film_global_view = film_global_view
        hidden_size = int(getattr(self.birna_model.config, "hidden_size", 768))

        if use_lora:
            self.birna_model = apply_lora_to_birna(
                birna_model=self.birna_model,
                target_modules=lora_target_modules or ["Wqkv"],
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        self.film = FiLM(input_dim=hidden_size, output_dim=hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

        if freeze_backbone and not use_lora:
            for parameter in self.birna_model.parameters():
                parameter.requires_grad = False

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.birna_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        emb = outputs.logits
        if emb.ndim != 3:
            raise ValueError(f"Expected BiRNA-BERT token embeddings with shape [B, L, H], got: {tuple(emb.shape)}")
        return emb

    def _nuc_content_embeddings(self, nuc_emb: torch.Tensor, nuc_content_mask: torch.Tensor) -> torch.Tensor:
        token_counts = nuc_content_mask.sum(dim=1)
        expected_tokens = int(token_counts.min().item())
        local_start = self.center_index - self.local_window_radius
        local_end = self.center_index + self.local_window_radius + 1
        if local_start < 0 or expected_tokens < local_end:
            raise ValueError(
                "BiRNA-BERT NUC output is too short for local FiLM pooling: "
                f"min_content_token_count={expected_tokens}, local_window=[{local_start}, {local_end}). "
                "Check NUC tokenization, sequence length, and max_length."
            )
        if int(token_counts.max().item()) != expected_tokens:
            raise ValueError(
                "FiLM local pooling expects fixed-length NUC content tokens within a batch. "
                f"Observed min={expected_tokens}, max={int(token_counts.max().item())}."
            )
        return nuc_emb[:, 1 : 1 + expected_tokens, :]

    def forward(
        self,
        nuc_input_ids=None,
        nuc_attention_mask=None,
        nuc_token_type_ids=None,
        nuc_content_mask=None,
        bpe_input_ids=None,
        bpe_attention_mask=None,
        bpe_token_type_ids=None,
        bpe_content_mask=None,
    ):
        if nuc_input_ids is None or nuc_content_mask is None:
            raise ValueError("FiLM classifier requires DualViewDataCollator outputs with nuc_input_ids and nuc_content_mask.")

        nuc_emb = self._encode(
            input_ids=nuc_input_ids,
            attention_mask=nuc_attention_mask,
            token_type_ids=nuc_token_type_ids,
        )
        nuc_token_emb = self._nuc_content_embeddings(nuc_emb, nuc_content_mask)
        local_start = self.center_index - self.local_window_radius
        local_end = self.center_index + self.local_window_radius + 1
        h_local = nuc_token_emb[:, local_start:local_end, :].mean(dim=1)

        if self.film_global_view == "bpe":
            if bpe_input_ids is None or bpe_content_mask is None:
                raise ValueError("BPE-global FiLM requires bpe_input_ids and bpe_content_mask.")
            bpe_emb = self._encode(
                input_ids=bpe_input_ids,
                attention_mask=bpe_attention_mask,
                token_type_ids=bpe_token_type_ids,
            )
            if int(bpe_content_mask.sum(dim=1).min().item()) == 0:
                raise ValueError("BiRNA-BERT BPE output has no non-special tokens. Check BPE tokenization.")
            h_global = mask_aware_mean_pool(bpe_emb, bpe_content_mask)
        else:
            h_global = mask_aware_mean_pool(nuc_emb, nuc_content_mask)

        gamma, beta = self.film(h_global)
        h_mod = gamma * h_local + beta
        feat = torch.cat([h_global, h_mod], dim=1)
        return self.classifier(feat)
