"""Small, baseline-preserving fusion modules for multi-layer [SEG] states."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class FinalAnchoredResidualFusion(nn.Module):
    """Fuse selected layers while preserving the final layer at initialization.

    Input shape is ``[..., K, D]``. The last selected layer is the anchor. For
    each earlier layer ``i`` the module learns a bounded residual coefficient::

        fused = h_final + sum(tanh(raw_i) * (h_i - h_final))

    All raw coefficients start at zero, so the initial output is bit-for-bit
    the anchor tensor. Only ``K - 1`` scalar parameters are introduced.
    """

    def __init__(
        self, layer_indices: Sequence[int], weight_scale: float = 1.0
    ) -> None:
        super().__init__()
        layers = tuple(int(layer) for layer in layer_indices)
        if len(layers) < 2:
            raise ValueError("fusion requires at least two selected layers")
        if len(set(layers)) != len(layers):
            raise ValueError("layer_indices contains duplicates")
        if tuple(sorted(layers)) != layers:
            raise ValueError("layer_indices must be strictly increasing")
        if not 0.0 < float(weight_scale) <= 1.0:
            raise ValueError("weight_scale must be in (0, 1]")
        self.layer_indices = layers
        self.weight_scale = float(weight_scale)
        self.raw_residual_weights = nn.Parameter(torch.zeros(len(layers) - 1))

    @property
    def effective_residual_weights(self) -> Tensor:
        return self.weight_scale * self.raw_residual_weights.tanh()

    def forward(self, states: Tensor) -> Tensor:
        if states.ndim < 2:
            raise ValueError("states must have shape [..., K, D]")
        if states.shape[-2] != len(self.layer_indices):
            raise ValueError(
                f"expected {len(self.layer_indices)} layers, got {states.shape[-2]}"
            )
        anchor = states[..., -1, :]
        residuals = states[..., :-1, :] - anchor.unsqueeze(-2)
        weights = self.effective_residual_weights.to(
            device=states.device, dtype=states.dtype
        )
        return anchor + (residuals * weights.unsqueeze(-1)).sum(dim=-2)

    def diagnostics(self) -> dict[str, object]:
        return {
            "layer_indices": list(self.layer_indices),
            "anchor_layer": self.layer_indices[-1],
            "weight_scale": self.weight_scale,
            "raw_residual_weights": self.raw_residual_weights.detach().float().cpu().tolist(),
            "effective_residual_weights": self.effective_residual_weights.detach().float().cpu().tolist(),
        }
