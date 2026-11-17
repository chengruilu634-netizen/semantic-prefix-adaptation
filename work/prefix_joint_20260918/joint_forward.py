"""Prefix Q/M joint path. Keeps all changes outside the frozen Sa2VA sources."""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from work.prefix5_memoryspace import (
    LAYERS,
    PREFIX,
    MemoryRollout,
    autocast_for,
    check_backend,
    materialize,
    register_object,
)
from work.query_highres_gate_20260914 import DecoderHighResPatch, QueryHighResGate


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


class JointPrefix:
    """One frozen backend with independent Q and M switches.

    ``decode`` keeps the Q-derived memory graph live. ``rollout`` commits only
    the h28 prefix states, replacing their memory features for M/J.
    """

    def __init__(self, backend: nn.Module, gate: QueryHighResGate | None = None,
                 fusion: JointMemoryFusion | None = None):
        check_backend(backend)
        self.backend = backend
        self.gate = gate
        self.fusion = fusion
        self.patch = DecoderHighResPatch(backend.sam_mask_decoder, gate) if gate is not None else None
        self.roll = MemoryRollout(backend)

    def close(self) -> None:
        if self.patch is not None:
            self.patch.close()

    def decode(self, images: torch.Tensor, embeddings: torch.Tensor, arm: str) -> dict:
        if arm not in ("P", "Q", "M", "J"):
            raise ValueError(f"unknown arm: {arm}")
        use_q, use_m = arm in ("Q", "J"), arm in ("M", "J")
        if use_q and self.patch is None or use_m and self.fusion is None:
            raise ValueError("requested adapter is unavailable")
        if len(images) < PREFIX or embeddings.ndim != 2 or embeddings.shape != (4, 256):
            raise ValueError("expected >=5 images and four 256D SEG embeddings")
        state = materialize(self.backend.init_state(images[:PREFIX]))
        obj = register_object(state)
        query = embeddings[-1].reshape(1, 1, -1)
        native: list[dict] = []
        layers = [[] for _ in LAYERS]
        # Only the h28 branch is needed without M. The same h28 query is used
        # by every auxiliary branch when Q and M are jointly active.
        selected = range(4) if use_m else (3,)
        grad = nullcontext() if use_q else torch.no_grad()
        with grad, autocast_for(state):
            for frame in range(PREFIX):
                for index in selected:
                    context = self.patch.active(query) if use_q else nullcontext()
                    with context:
                        current, _ = self.backend._run_single_frame_inference(
                            inference_state=state,
                            output_dict=state["output_dict_per_obj"][obj],
                            frame_idx=frame, batch_size=1, is_init_cond_frame=True,
                            point_inputs=None, mask_inputs=None, reverse=False,
                            run_mem_encoder=False, prev_sam_mask_logits=None,
                            language_embd=embeddings[index].reshape(1, 1, -1),
                        )
                    logits = current["pred_masks"].float()
                    high = F.interpolate(logits, (self.backend.image_size, self.backend.image_size),
                                         mode="bilinear", align_corners=False)
                    if self.backend.non_overlap_masks_for_mem_enc:
                        high = self.backend._apply_non_overlapping_constraints(high)
                    memory, positions = self.backend._run_memory_encoder(
                        inference_state=state, frame_idx=frame, batch_size=1,
                        high_res_masks=high, is_mask_from_pts=True,
                    )
                    layers[index].append(memory[0])
                    if index == 3:
                        native.append({
                            "pred_masks": current["pred_masks"].detach(),
                            "obj_ptr": current["obj_ptr"].detach(),
                            "maskmem_features": memory.detach(),
                            "maskmem_pos_enc": [p.detach() for p in positions],
                        })
        if len(native) != PREFIX:
            raise RuntimeError("h28 prefix incomplete")
        anchor = torch.stack(layers[3])
        if use_m:
            memories = torch.stack([torch.stack(row) for row in layers])
            plan = self.fusion(memories)
            if plan["fused"].dtype != anchor.dtype:
                raise RuntimeError("memory dtype changed")
        else:
            zero = anchor.float().sum() * 0.0
            plan = {"fused": anchor, "anchor": anchor.float(),
                    "correction": torch.zeros_like(anchor.float()), "anchor_penalty": zero}
        return {"native": native, "plan": plan, "arm": arm,
                "prefix_decodes": PREFIX * len(selected),
                "prefix_memory_encodes": PREFIX * len(selected)}

    def propagate(self, images: torch.Tensor, decoded: dict, *, total_video_frames: int | None = None,
                  backprop_start: int = 12, native_loop: bool = False, cpu_output: bool = False) -> dict:
        if decoded["arm"] not in ("P", "Q", "M", "J"):
            raise ValueError("decode result has unknown arm")
        return self.roll(images, decoded["native"], decoded["plan"]["fused"],
                         total_video_frames=total_video_frames,
                         backprop_start=backprop_start, native_loop=native_loop,
                         cpu_output=cpu_output)
