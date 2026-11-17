"""Prefix-5 Hierarchical Memory-Space Fusion. One state, native suffix.

Uses the actual exp050/052 consolidation contract and exp054's locally unwrapped
native propagation bodies. No SAM2 source changes, prompts, or pointer fusion.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F

from work.prefix5_propagation_training import (
    GradPropagationProxy, materialize, materialize_backend_caches, register_object,
)
from work.train_spatial_trajectory_admission import frame_losses

REVISION = "exp053_prefix5_hierarchical_memoryspace_v1"
LAYERS, PREFIX = (21, 24, 25, 28), 5


class MemoryFusion(nn.Module):
    def __init__(self, channels=64, hidden=16, bound=0.1):
        super().__init__()
        if channels <= 0 or hidden <= 0 or not math.isfinite(bound) or bound <= 0:
            raise ValueError("invalid memory fusion configuration")
        self.channels, self.hidden, self.bound = channels, hidden, float(bound)
        self.net = nn.Sequential(nn.Conv2d(4 * channels, hidden, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(hidden, channels, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, memories):
        # [layer, conditioning frame, channel, height, width]; frames are a batch.
        if memories.ndim != 5 or memories.shape[:3] != (4, 5, self.channels):
            raise ValueError("exactly four layers and five conditioning memories required")
        if memories.requires_grad or memories.is_inference() or not torch.isfinite(memories).all():
            raise ValueError("ordinary finite frozen memories required")
        device = next(self.parameters()).device
        with torch.autocast(device_type=device.type, enabled=False):
            values = memories.to(device).float()
            anchor = values[-1]
            x = torch.cat([anchor, *(values[i] - anchor for i in range(3))], dim=1)
            raw = self.net(x)
            # Fixed per-frame RMS bound, not a learned acceptance gate.
            scale = anchor.square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
            correction = self.bound * scale * torch.tanh(raw)
            fused_float = anchor + correction
            # Native state memory is bf16. Preserve that contract in BOTH modes.
            fused = fused_float.to(memories.dtype)
            regularizer = (correction.square().mean((1, 2, 3)) /
                           (anchor.square().mean((1, 2, 3)) + 1e-6)).mean()
        return {"fused": fused, "anchor": anchor, "correction": correction,
                "anchor_penalty": regularizer}


def check_backend(backend):
    if any(m.training for m in backend.modules()) or any(p.requires_grad for p in backend.parameters()):
        raise RuntimeError("SAM2 must remain frozen and eval")
    if getattr(backend, "binarize_mask_from_pts_for_mem_enc", False) or getattr(backend, "fill_hole_area", 0):
        raise RuntimeError("unaudited hard-mask/connected-component path")


def autocast_for(state):
    return torch.autocast("cuda", dtype=torch.bfloat16) if torch.device(state["device"]).type == "cuda" else nullcontext()


@torch.no_grad()
def decode_prefix_memories(backend, images, embeddings):
    """20 independent decodes/encodes, no auxiliary suffix propagation at all."""
    check_backend(backend)
    if len(images) < PREFIX or embeddings.ndim != 2 or len(embeddings) != 4:
        raise ValueError("five images and four projected SEG vectors required")
    state = backend.init_state(images[:PREFIX])
    obj = register_object(state)
    device = state["device"]
    memories, lows, highs, native = [[] for _ in LAYERS], [[] for _ in LAYERS], [[] for _ in LAYERS], []
    raw_positions = []
    # Inspect the REAL encoder output before predictor's shared-position cache.
    def capture_positions(module, args, output):
        raw_positions.append([x.detach().clone() for x in output["vision_pos_enc"]])
    handle = backend.memory_encoder.register_forward_hook(capture_positions)
    try:
        with autocast_for(state):
            for frame in range(PREFIX):
                current_frame = []
                for layer_index in range(4):
                    current, _ = backend._run_single_frame_inference(
                        inference_state=state, output_dict=state["output_dict_per_obj"][obj],
                        frame_idx=frame, batch_size=1, is_init_cond_frame=True,
                        point_inputs=None, mask_inputs=None, reverse=False, run_mem_encoder=False,
                        prev_sam_mask_logits=None, language_embd=embeddings[layer_index].reshape(1, 1, -1))
                    logits = current["pred_masks"].to(device, dtype=torch.float32)
                    high = F.interpolate(logits, (backend.image_size, backend.image_size), mode="bilinear", align_corners=False)
                    if backend.non_overlap_masks_for_mem_enc:
                        high = backend._apply_non_overlapping_constraints(high)
                    memory, positions = backend._run_memory_encoder(
                        inference_state=state, frame_idx=frame, batch_size=1,
                        high_res_masks=high, is_mask_from_pts=True)
                    memories[layer_index].append(memory[0].cpu())
                    lows[layer_index].append(logits[0, 0].cpu())
                    highs[layer_index].append(high[0, 0].half().cpu())
                    current_frame.append({"pred_masks": logits.cpu(), "obj_ptr": current["obj_ptr"].float().cpu(),
                        "maskmem_features": memory.cpu(), "maskmem_pos_enc": [p.cpu() for p in positions]})
                native.append(current_frame[-1])
    finally:
        handle.remove()
    if len(raw_positions) != 20:
        raise RuntimeError("expected exactly 20 real Memory Encoder calls")
    reference = raw_positions[-1]
    if any(len(row) != len(reference) or any(not torch.equal(a, b) for a, b in zip(row, reference))
           for row in raw_positions):
        raise RuntimeError("layer-dependent raw position encodings: STOP before fusion")
    return materialize({"memories": torch.stack([torch.stack(x) for x in memories]),
        "prefix_low": torch.stack([torch.stack(x) for x in lows]),
        "prefix_high": torch.stack([torch.stack(x) for x in highs]), "native": native,
        "raw_positions_all_equal": True, "prefix_decodes": 20, "prefix_memory_encodes": 20,
        "auxiliary_suffix_decodes": 0})


class MemoryRollout:
    def __init__(self, backend):
        self.backend = backend
        backend.eval()
        for parameter in backend.parameters():
            parameter.requires_grad_(False)
        self.proxy = GradPropagationProxy(backend)
        check_backend(backend)

    def __call__(self, images, native, fused, *, total_video_frames=None, native_loop=False,
                 backprop_start=5, cpu_output=False):
        check_backend(self.backend)
        if len(native) != 5 or len(fused) != 5 or len(images) <= 5:
            raise ValueError("five conditioning memories plus suffix required")
        if images.requires_grad or images.is_inference():
            raise ValueError("materialized frozen images required")
        if not native_loop and torch.is_inference_mode_enabled():
            raise ValueError("training cannot run in inference mode")
        if not 5 <= backprop_start < len(images):
            raise ValueError("invalid truncation boundary")
        backend = self.backend
        if not native_loop:
            materialize_backend_caches(backend)
        state = backend.init_state(images)
        if total_video_frames is not None:
            if total_video_frames < len(images):
                raise ValueError("invalid original video length")
            state["num_frames"] = int(total_video_frames)
        obj = register_object(state)
        for i, old in enumerate(native):
            if fused[i].shape != old["maskmem_features"].shape[1:] or fused.dtype != old["maskmem_features"].dtype:
                raise ValueError("memory shape/dtype differs from native state")
            committed = {"pred_masks": old["pred_masks"].to(state["storage_device"]),
                         "obj_ptr": old["obj_ptr"].to(state["device"]),
                         "maskmem_pos_enc": [p.to(state["storage_device"]) for p in old["maskmem_pos_enc"]],
                         "maskmem_features": fused[i:i+1].to(state["storage_device"])}
            state["output_dict"]["cond_frame_outputs"][i] = committed
            backend._add_output_per_object(state, i, committed, "cond_frame_outputs")
            state["consolidated_frame_inds"]["cond_frame_outputs"].add(i)
        loop = backend.propagate_in_video if native_loop else self.proxy.propagate_in_video
        generator = loop(state, start_frame_idx=0, max_frame_num_to_track=len(images) - 1)
        outputs, indices = [], []
        with autocast_for(state):
            for expected in range(len(images)):
                # Burn-in forward values are native, but recurrent history before
                # backprop_start is detached. The five prefix memories stay LIVE.
                context = torch.no_grad() if expected < backprop_start else nullcontext()
                with context:
                    index, _, high = next(generator)
                    output = high[0, 0].half().float()
                if cpu_output:
                    output = output.detach().cpu()
                outputs.append(output)
                indices.append(int(index))
        generator.close()
        if indices != list(range(len(images))):
            raise RuntimeError("incomplete native propagation")
        return {"logits": torch.stack(outputs), "state": state, "indices": indices}


def objective(logits, truth, plan, *, start=5, anchor_lambda=0.01):
    if not 5 <= start < len(logits) or len(logits) != len(truth) or not ((truth == 0) | (truth == 1)).all():
        raise ValueError("invalid future-only supervision")
    if not math.isfinite(anchor_lambda) or anchor_lambda < 0:
        raise ValueError("anchor must be a nonnegative penalty")
    resized = F.interpolate(logits[start:, None].float(), truth.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    loss = frame_losses(resized, truth[start:].to(resized.device)).mean()
    return {"future": loss, "anchor": plan["anchor_penalty"],
            "total": loss + anchor_lambda * plan["anchor_penalty"]}
