"""Memory refinement class extracted verbatim from the archived joint implementation."""
from __future__ import annotations
import torch
from torch import nn
PREFIX = 5

class JointMemoryFusion(nn.Module):
    """Exp053 residual with a live memory input and a detached RMS scale."""

    def __init__(self, channels: int = 64, hidden: int = 16, bound: float = 0.1):
        super().__init__()
        self.channels = channels
        self.bound = bound
        self.net = nn.Sequential(
            nn.Conv2d(4 * channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, memories: torch.Tensor) -> dict[str, torch.Tensor]:
        if memories.ndim != 5 or memories.shape[:3] != (4, PREFIX, self.channels):
            raise ValueError(f"expected [4,{PREFIX},{self.channels},H,W], got {tuple(memories.shape)}")
        if memories.is_inference() or not bool(torch.isfinite(memories).all()):
            raise ValueError("memory inputs must be ordinary finite tensors")
        device = next(self.parameters()).device
        with torch.autocast(device_type=device.type, enabled=False):
            values = memories.to(device).float()
            anchor = values[-1]
            x = torch.cat([anchor, *(values[i] - anchor for i in range(3))], dim=1)
            raw = self.net(x)
            anchor_power = anchor.square().mean((1, 2, 3), keepdim=True).detach()
            scale = anchor_power.sqrt().clamp_min(1e-6)
            correction = self.bound * scale * torch.tanh(raw)
            fused = (anchor + correction).to(memories.dtype)
            penalty = (correction.square().mean((1, 2, 3)) /
                       (anchor_power.flatten() + 1e-6)).mean()
        return {"fused": fused, "anchor": anchor, "correction": correction,
                "anchor_penalty": penalty}
