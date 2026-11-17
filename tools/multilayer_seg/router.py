"""Leakage-free feature construction and lightweight dynamic SEG router."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def trajectory_features(states: torch.Tensor, sketch_dim: int = 64) -> torch.Tensor:
    """Compress [28,D] SEG states using only inference-visible quantities."""
    if states.ndim != 2 or states.shape[0] != 28:
        raise ValueError(f"states must have shape [28,D], got {tuple(states.shape)}")
    if states.shape[1] < sketch_dim or sketch_dim < 1:
        raise ValueError("sketch_dim must be in [1,D]")
    values = states.detach().float()
    anchor = values[-1]
    anchor_unit = F.normalize(anchor, dim=0, eps=1e-8)
    width = values.shape[1]
    padded_width = ((width + sketch_dim - 1) // sketch_dim) * sketch_dim
    semantic = F.pad(anchor_unit, (0, padded_width - width)).reshape(sketch_dim, -1).mean(1)

    residuals = values[:-1] - anchor.unsqueeze(0)
    anchor_norm = anchor.norm().clamp_min(1e-8)
    relative_norm = residuals.norm(dim=1) / anchor_norm
    anchor_cosine = F.cosine_similarity(values[:-1], anchor.unsqueeze(0), dim=1, eps=1e-8)
    adjacent_cosine = F.cosine_similarity(values[1:], values[:-1], dim=1, eps=1e-8)
    residual_anchor_cosine = F.cosine_similarity(
        residuals, anchor.unsqueeze(0), dim=1, eps=1e-8
    )
    layer_mean = values.mean(dim=1)
    layer_std = values.std(dim=1, unbiased=False)
    features = torch.cat((
        semantic,
        relative_norm,
        anchor_cosine,
        adjacent_cosine,
        residual_anchor_cosine,
        layer_mean,
        layer_std,
    ))
    if not torch.isfinite(features).all():
        raise RuntimeError("trajectory features contain NaN or Inf")
    return features.cpu()


class DynamicSegRouter(nn.Module):
    """Predict K, residual layers and residual budget from trajectory features."""

    def __init__(self, input_dim: int, hidden_dim: int = 48, dropout: float = 0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.k_head = nn.Linear(hidden_dim, 4)
        self.layer_head = nn.Linear(hidden_dim, 27)
        self.budget_head = nn.Linear(hidden_dim, 4)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.trunk(features)
        return self.k_head(hidden), self.layer_head(hidden), self.budget_head(hidden)


def route_from_logits(
    k_logits: torch.Tensor, layer_logits: torch.Tensor, budget_logits: torch.Tensor
) -> tuple[torch.Tensor, list[tuple[int, ...]], torch.Tensor]:
    """Decode batches into K, one-based residual layers, and budget class."""
    if k_logits.ndim != 2 or k_logits.shape[1] != 4:
        raise ValueError("k_logits must have shape [B,4]")
    if layer_logits.shape != (k_logits.shape[0], 27):
        raise ValueError("layer_logits must have shape [B,27]")
    if budget_logits.shape != (k_logits.shape[0], 4):
        raise ValueError("budget_logits must have shape [B,4]")
    k = k_logits.argmax(1) + 1
    budget = budget_logits.argmax(1)
    layers = []
    for row, count in zip(layer_logits, k - 1):
        chosen = row.topk(int(count), largest=True).indices.add(1).sort().values.tolist()
        layers.append(tuple(int(value) for value in chosen))
    return k, layers, budget
