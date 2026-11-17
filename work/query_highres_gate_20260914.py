"""h28 query-conditioned residual adapter for SAM2 high-resolution skips."""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


class _LevelGate(nn.Module):
    def __init__(self, channels: int, query_dim: int = 256, hidden: int = 16, key_dim: int = 16):
        super().__init__()
        self.feature_key = nn.Conv2d(channels, key_dim, 1, bias=False)
        self.query_key = nn.Linear(query_dim, key_dim, bias=False)
        self.in_proj = nn.Conv2d(channels, hidden, 1)
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.out_proj = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, feature: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # SAM2's cached features are frozen.  Work in FP32 and return its native
        # dtype so zero-output residuals preserve the exact decoder interface.
        native_dtype = feature.dtype
        value = feature.detach().clone().float()
        q = query.detach().float().mean(dim=1)
        key = F.normalize(self.feature_key(value), dim=1, eps=1e-6)
        q_key = F.normalize(self.query_key(q), dim=-1, eps=1e-6)[:, :, None, None]
        gate = torch.sigmoid((key * q_key).sum(dim=1, keepdim=True))
        residual = self.out_proj(F.gelu(self.local(F.gelu(self.in_proj(value)))))
        correction = gate * residual
        return (value + correction).to(native_dtype), gate, correction


class QueryHighResGate(nn.Module):
    """Spatially selects two native SAM2 skip maps using the h28 SEG query."""
    def __init__(self, channels: tuple[int, int] = (32, 64), query_dim: int = 256, hidden: int = 16, key_dim: int = 16):
        super().__init__()
        self.channels = tuple(channels)
        self.query_dim = int(query_dim)
        self.levels = nn.ModuleList([_LevelGate(c, query_dim, hidden, key_dim) for c in channels])
        self.last_stats: list[dict[str, float]] = []

    def forward(self, features: list[torch.Tensor], query: torch.Tensor) -> list[torch.Tensor]:
        if len(features) != len(self.levels) or tuple(x.shape[1] for x in features) != self.channels:
            raise ValueError(f"unexpected SAM2 high-res feature contract: {[tuple(x.shape) for x in features]}")
        if query.ndim != 3 or query.shape[1:] != (1, self.query_dim):
            raise ValueError(f"unexpected h28 query contract: {tuple(query.shape)}")
        outputs, stats = [], []
        for level, feature in zip(self.levels, features):
            output, gate, correction = level(feature, query)
            outputs.append(output)
            stats.append({"gate_mean": float(gate.detach().mean()), "gate_std": float(gate.detach().std()),
                          "correction_l1": float(correction.detach().abs().mean())})
        self.last_stats = stats
        return outputs


class DecoderHighResPatch:
    """Temporarily injects a query-conditioned residual before SAM2 upsampling."""
    def __init__(self, decoder: nn.Module, gate: QueryHighResGate):
        self.decoder, self.gate = decoder, gate
        self.query: torch.Tensor | None = None
        self.original = decoder.forward

        def patched(_decoder, *args, **kwargs):
            high_res = kwargs.get("high_res_features")
            if self.query is not None:
                if high_res is None:
                    raise RuntimeError("query gate requested but SAM2 high-res features are disabled")
                kwargs = dict(kwargs)
                kwargs["high_res_features"] = self.gate(high_res, self.query)
            return self.original(*args, **kwargs)

        decoder.forward = MethodType(patched, decoder)

    @contextmanager
    def active(self, query: torch.Tensor):
        old = self.query
        self.query = query
        try:
            yield
        finally:
            self.query = old

    def close(self) -> None:
        self.decoder.forward = self.original
