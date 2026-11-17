"""Prefix-five fusion with gradients through a frozen, eval-mode SAM2 rollout.

No monkeypatching of SAM2 or exp052. Only two inference-mode decorators are
bypassed by an instance-local proxy; their original propagation bodies are reused.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import inspect
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from work.train_two_stage_candidate_then_gate import CandidateFusion
from work.train_spatial_trajectory_admission import RESIDUAL_CLIP, frame_losses

PREFIX = 5
LAYERS = (21, 24, 25, 28)
REVISION = "prefix5_propagation_training_v1"


def materialize(value: Any) -> Any:
    """Turn frozen inference-cache tensors into ordinary, detached tensors."""
    if torch.is_inference_mode_enabled():
        raise RuntimeError("materialize must run outside inference_mode")
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: materialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [materialize(v) for v in value]
    if isinstance(value, tuple):
        return tuple(materialize(v) for v in value)
    return value


class PrefixFiveFusion(nn.Module):
    """Same residual architecture, but the planner cannot accept suffix frames."""

    def __init__(self, query_dim: int, *, plan_size: int = 128, state_size: int = 256):
        super().__init__()
        self.candidate = CandidateFusion(query_dim)
        self.query_dim = query_dim
        self.plan_size, self.state_size = plan_size, state_size

    def forward(self, query: torch.Tensor, high: torch.Tensor, low: torch.Tensor):
        if high.ndim != 4 or low.ndim != 4 or high.shape[:2] != (4, PREFIX) or low.shape[:2] != (4, PREFIX):
            raise ValueError("exactly four layers and five frames are required; suffix input forbidden")
        if low.shape[-2:] != (self.state_size, self.state_size):
            raise ValueError("incorrect SAM2 state resolution")
        if query.shape != (self.query_dim,):
            raise ValueError("incorrect query dimension")
        if any(t.requires_grad for t in (query, high, low)):
            raise ValueError("query and auxiliary trajectories must remain frozen")
        if any(not torch.isfinite(t).all() for t in (query, high, low)):
            raise ValueError("non-finite frozen input")
        device = next(self.parameters()).device
        if torch.is_grad_enabled() and any(t.is_inference() for t in (query, high, low)):
            raise ValueError("materialize inference-cache tensors before training")
        # Match exp052's float16 resize-cache roundtrip and float32 planner.
        with torch.autocast(device_type=device.type, enabled=False):
            resized = F.interpolate(high.float().to(device).flatten(0, 1)[:, None],
                size=(self.plan_size, self.plan_size), mode="bilinear", align_corners=False)
            resized = resized[:, 0].reshape(4, PREFIX, self.plan_size, self.plan_size).half().float()
            _, plan = self.candidate(query.float().to(device), resized)
            gates = F.interpolate(plan["spatial_gates"].flatten(0, 1)[:, None].float(),
                size=(self.state_size, self.state_size), mode="bilinear", align_corners=False)
            gates = gates[:, 0].reshape(3, PREFIX, self.state_size, self.state_size)
            values = low.float().to(device)
            residuals = (values[:-1] - values[-1]).clamp(-RESIDUAL_CLIP, RESIDUAL_CLIP)
            correction = (plan["layer_weights"][:, None, None, None] * gates * residuals).sum(0)
            fused = values[-1] + correction
        return {"fused_low": fused, "native_low": values[-1], "correction": correction,
                "spatial_gates": plan["spatial_gates"], "layer_weights": plan["layer_weights"]}


class GradPropagationProxy:
    """Reuse the exact native loop without changing the backend's methods/mode."""

    def __init__(self, backend):
        self.backend = backend
        for name in ("propagate_in_video", "propagate_in_video_preflight"):
            function = getattr(type(backend), name)
            if not hasattr(function, "__wrapped__"):
                raise RuntimeError(f"unexpected unwrapped native method: {name}")

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def propagate_in_video_preflight(self, state):
        function = inspect.unwrap(type(self.backend).propagate_in_video_preflight)
        return function(self, state)

    def propagate_in_video(self, state, **kwargs):
        function = inspect.unwrap(type(self.backend).propagate_in_video)
        return function(self, state, **kwargs)


def materialize_backend_caches(backend):
    """Keep cache values/device/dtype, replace only inference-tensor storage.

    SAM2's RoPE freqs_cis and positional-encoding dictionaries may have been
    populated by an earlier inference-mode candidate pass. They are constants,
    but autograd must be allowed to save them for input-gradient computation.
    Never replace parameters, modules, or already-normal live graph tensors.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("cache materialization requires ordinary tensor mode")
    count = 0

    def convert(value):
        nonlocal count
        if isinstance(value, nn.Parameter):
            return value
        if isinstance(value, torch.Tensor):
            if value.is_inference():
                count += 1
                return value.detach().clone()
            return value
        if isinstance(value, dict):
            new = {k: convert(v) for k, v in value.items()}
            return new if any(new[k] is not v for k, v in value.items()) else value
        if isinstance(value, (list, tuple)):
            new = [convert(v) for v in value]
            return type(value)(new) if any(a is not b for a, b in zip(new, value)) else value
        return value

    with torch.no_grad():
        for module in backend.modules():
            for name, value in list(vars(module).items()):
                if name not in ("_parameters", "_modules"):
                    replacement = convert(value)
                    if replacement is not value:
                        setattr(module, name, replacement)
    return count


def register_object(state, object_id=100):
    if state["tracking_has_started"] or state["obj_ids"]:
        raise ValueError("rollout must start from a fresh single-object state")
    obj = 0
    state["obj_id_to_idx"][object_id] = obj
    state["obj_idx_to_id"][obj] = object_id
    state["obj_ids"] = [object_id]
    state["point_inputs_per_obj"][obj] = {}
    state["mask_inputs_per_obj"][obj] = {}
    for name in ("output_dict_per_obj", "temp_output_dict_per_obj"):
        state[name][obj] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    return obj


class FrozenSAM2Rollout:
    def __init__(self, backend, *, state_size=256, autocast_dtype=torch.bfloat16):
        self.backend = backend
        self.state_size, self.autocast_dtype = state_size, autocast_dtype
        backend.eval()
        for parameter in backend.parameters():
            parameter.requires_grad_(False)
        self.proxy = GradPropagationProxy(backend)
        self.check_backend()

    def check_backend(self):
        if any(module.training for module in self.backend.modules()):
            raise RuntimeError("SAM2 must stay in eval mode to preserve inference behavior")
        if any(p.requires_grad for p in self.backend.parameters()):
            raise RuntimeError("SAM2 parameters must remain frozen")
        # Refuse a hard threshold rather than silently changing it to a sigmoid.
        if getattr(self.backend, "binarize_mask_from_pts_for_mem_enc", False):
            raise RuntimeError("hard-binary memory input is incompatible with this exact-forward gradient path")
        if getattr(self.backend, "fill_hole_area", 0) != 0:
            raise RuntimeError("connected-component postprocessing has not been gradient-audited")

    def __call__(self, images, embedding, fused_low=None, *, native_loop=False, total_video_frames=None):
        self.check_backend()
        if torch.is_inference_mode_enabled() and not native_loop:
            raise RuntimeError("gradient rollout cannot run inside inference_mode")
        if len(images) <= PREFIX:
            raise ValueError("propagation supervision requires at least one suffix frame")
        if total_video_frames is not None and total_video_frames < len(images):
            raise ValueError("full video cannot be shorter than its training window")
        if images.requires_grad or embedding.requires_grad:
            raise ValueError("image and language backbone inputs must be frozen")
        if images.is_inference() or embedding.is_inference():
            raise ValueError("materialize image/embedding caches before rollout")
        if fused_low is not None and fused_low.shape != (PREFIX, self.state_size, self.state_size):
            raise ValueError("only five fused conditioning masks may be supplied")
        predictor = self.backend
        if not native_loop:
            materialize_backend_caches(predictor)
        state = predictor.init_state(images)
        # Preserve original video-length-dependent memory/pointer behavior while
        # limiting unrolling to the supplied window. Future images are never read.
        if total_video_frames is not None:
            state["num_frames"] = int(total_video_frames)
        obj = register_object(state)
        device = torch.device(state["device"])
        autocast = torch.autocast("cuda", dtype=self.autocast_dtype) if device.type == "cuda" else nullcontext()
        native_pointers, memories = [], []
        with autocast:
            for i in range(PREFIX):
                # Native sparse prompts/pointers and backbone features are constants.
                # no_grad creates ordinary tensors, unlike inference_mode.
                with torch.no_grad():
                    current, _ = predictor._run_single_frame_inference(
                        inference_state=state, output_dict=state["output_dict_per_obj"][obj],
                        frame_idx=i, batch_size=1, is_init_cond_frame=True,
                        point_inputs=None, mask_inputs=None, reverse=False, run_mem_encoder=False,
                        prev_sam_mask_logits=None, language_embd=embedding.reshape(1, 1, -1))
                chosen = current["pred_masks"] if fused_low is None else fused_low[i][None, None]
                logits = chosen.to(state["storage_device"], dtype=torch.float32)
                high = F.interpolate(logits.to(device), size=(predictor.image_size, predictor.image_size),
                                     mode="bilinear", align_corners=False)
                if predictor.non_overlap_masks_for_mem_enc:
                    high = predictor._apply_non_overlapping_constraints(high)
                memory, positions = predictor._run_memory_encoder(
                    inference_state=state, frame_idx=i, batch_size=1,
                    high_res_masks=high, is_mask_from_pts=True)
                pointer = current["obj_ptr"].to(device, dtype=torch.float32)
                committed = {"pred_masks": logits, "obj_ptr": pointer,
                             "maskmem_features": memory, "maskmem_pos_enc": positions}
                state["output_dict"]["cond_frame_outputs"][i] = committed
                predictor._add_output_per_object(state, i, committed, "cond_frame_outputs")
                state["consolidated_frame_inds"]["cond_frame_outputs"].add(i)
                native_pointers.append(pointer)
                memories.append(memory)
            loop = predictor.propagate_in_video if native_loop else self.proxy.propagate_in_video
            indices, outputs = [], []
            for i, _, high in loop(state, start_frame_idx=0, max_frame_num_to_track=len(images) - 1):
                indices.append(int(i))
                # Match exp052's output precision WITHOUT detaching the graph.
                outputs.append(high[0, 0].half().float())
        if indices != list(range(len(images))):
            raise RuntimeError("incomplete or reordered native rollout")
        return {"logits": torch.stack(outputs), "conditioning_memories": memories,
                "native_pointers": native_pointers, "state": state}


@dataclass(frozen=True)
class LossWeights:
    prefix: float = 0.25
    suffix: float = 0.75
    residual: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.prefix, self.suffix, self.residual)) or self.prefix < 0 or self.suffix <= 0 or self.residual < 0:
            raise ValueError("nonnegative weights and strictly positive suffix supervision required")


def propagation_objective(logits, truth, correction, weights=LossWeights()):
    """GT is used only here, after forward; no teacher-forced GT memory writes."""
    if logits.ndim != 3 or truth.ndim != 3 or len(logits) != len(truth) or len(logits) <= PREFIX:
        raise ValueError("matching prefix+suffix predictions/targets required")
    if not torch.isfinite(logits).all() or not torch.isfinite(truth).all():
        raise ValueError("non-finite prediction or target")
    if not ((truth == 0) | (truth == 1)).all():
        raise ValueError("binary segmentation targets required")
    # Loss resolution may be smaller, but masks enter SAM2 at its native resolution.
    resized = F.interpolate(logits[:, None].float(), size=truth.shape[-2:],
                            mode="bilinear", align_corners=False)[:, 0]
    per_frame = frame_losses(resized, truth.to(resized.device))
    prefix, suffix = per_frame[:PREFIX].mean(), per_frame[PREFIX:].mean()
    residual = correction.float().square().mean()
    total = weights.prefix * prefix + weights.suffix * suffix + weights.residual * residual
    return {"total": total, "prefix": prefix, "suffix": suffix, "residual": residual}


def train_step(fusion, rollout, optimizer, clip, *, weights=LossWeights(), max_grad_norm=2.0):
    expected = {id(p) for p in fusion.parameters() if p.requires_grad}
    supplied = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if not expected or supplied != expected:
        raise ValueError("optimizer must contain exactly the trainable fusion parameters")
    fusion.train()
    optimizer.zero_grad(set_to_none=True)
    plan = fusion(clip["query"], clip["prefix_high"], clip["prefix_low"])
    result = rollout(clip["images"], clip["embedding"], plan["fused_low"],
                     total_video_frames=clip.get("total_video_frames"))
    losses = propagation_objective(result["logits"], clip["truth"], plan["correction"], weights)
    losses["total"].backward()
    parameters = [p for p in fusion.parameters() if p.grad is not None]
    if not parameters or any(not torch.isfinite(p.grad).all() for p in parameters):
        raise RuntimeError("missing or non-finite fusion gradient")
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    if float(grad_norm) == 0:
        raise RuntimeError("all fusion gradients are zero; refusing silent no-op training")
    optimizer.step()
    rollout.check_backend()
    return {**{k: float(v.detach()) for k, v in losses.items()}, "grad_norm": float(grad_norm),
            "frames": len(clip["images"])}


def checkpoint_payload(fusion, *, epoch, protocol_sha256, development_loss):
    return {"revision": REVISION, "epoch": epoch,
            "state_dict": {k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()},
            "query_dim": fusion.query_dim, "plan_size": fusion.plan_size, "state_size": fusion.state_size,
            "layers": list(LAYERS), "prefix_frames": PREFIX,
            "protocol_sha256": protocol_sha256, "development_loss": development_loss,
            "final_validation_performed": False}


def load_prefix5_checkpoint(path, *, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("revision") != REVISION or payload.get("layers") != list(LAYERS) or payload.get("prefix_frames") != PREFIX:
        raise ValueError("not a matching prefix-five propagation-trained checkpoint")
    fusion = PrefixFiveFusion(payload["query_dim"], plan_size=payload["plan_size"], state_size=payload["state_size"])
    fusion.load_state_dict(payload["state_dict"], strict=True)
    return fusion.to(device).eval(), payload


@torch.no_grad()
def predict_clip(fusion, rollout, query, prefix_high, prefix_low, images, embedding, *, total_video_frames=None):
    """Inference uses the SAME fusion forward and the original decorated SAM2 loop."""
    fusion.eval()
    plan = fusion(query, prefix_high, prefix_low)
    return rollout(images, embedding, plan["fused_low"], native_loop=True,
                   total_video_frames=total_video_frames)["logits"]
