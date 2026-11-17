#!/usr/bin/env python3
"""Exp_050 Stage 1: one frozen fused-mask write into SAM2 memory."""

from __future__ import annotations

import argparse
import faulthandler
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from tools.multilayer_seg.evaluate_llm_spatial_mask_prompt import register_object
from tools.multilayer_seg.evaluate_spatial_map_affine_gate import build_event
from tools.multilayer_seg.extract_spatial_adapter_features import extract_states
from tools.multilayer_seg.train_eval_fusion_standard import preprocess_video
from work.evaluate_layer_policy_oracle_feasibility import (
    atomic_json,
    atomic_torch,
    pack_binary_mask,
    paired_summary,
    unpack_binary_mask,
)
from work.evaluate_recurrent_direct_memory_identity import sequence_metrics
from work.evaluate_tgif_disjoint_confirmation import (
    score_action,
    sha256_json,
)
from work.evaluate_tgif_query_causality_ablation import preprocessor_from_checkpoint
from work.train_evaluate_tgif_seg_router import (
    ALL_LAYERS,
    TgifSegRouter,
    decode_frame_mask,
    error_rates,
    query_embedding,
    segment_metrics,
    transform_query,
)
from work.train_spatial_trajectory_admission import AUX_LAYERS, LAYERS, RESIDUAL_CLIP
from work.train_two_stage_candidate_then_gate import CandidateFusion


EXPERIMENT_ID = "exp_050_recurrent_direct_memory_mask_fusion"
STAGE = "stage1_one_shot_fused_memory"
CONDITIONING_FRAMES = 5
INTERVENTION_FRAME = 2
SAMPLES = 3
EXPECTED_CANDIDATE_SHA256 = "48cfb9a1915ecb57bd6987c2573bf20d2e836e03974fbbb2b0078ee982f46eaf"
PIPELINE_REVISION = "stage1_v2_explicit_freeze_score_boundary"
ACTIONS = ("manual_h28_direct_memory", "exp049_output_only", "one_shot_fused_memory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", nargs="?")
    parser.add_argument("--data-root")
    parser.add_argument("--parent-dir")
    parser.add_argument("--candidate-checkpoint")
    parser.add_argument("--stage0-selection")
    parser.add_argument("--stage0-summary")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--candidate-only", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_hash(
    selection: dict[str, Any],
    parent_sha: str,
    candidate_sha: str,
    seed: int = 20260902,
) -> str:
    return sha256_json(
        {
            "experiment_id": EXPERIMENT_ID,
            "stage": STAGE,
            "pipeline_revision": PIPELINE_REVISION,
            "seed": seed,
            "row_rng": "seed_plus_sha256_key_prefix_mod_2pow32",
            "selection_sha256": selection["selected_sha256"],
            "layers": list(LAYERS),
            "layer_preprocessing": "per_sample_rms_align_to_h28",
            "candidate": "frozen_exp049_epoch12_candidate",
            "candidate_checkpoint_sha256": candidate_sha,
            "query_preprocessor_checkpoint_sha256": parent_sha,
            "candidate_planning_resolution": 128,
            "candidate_state_resolution": 256,
            "query_precision": "exp049_float16_cache_roundtrip",
            "fused_logit_precision": "float32_no_post_fusion_quantization",
            "intervention_frame": INTERVENTION_FRAME,
            "intervention_count": 1,
            "other_conditioning_frames": "manual_native_h28_direct_memory",
            "pointer": "native_h28_on_all_conditioning_frames",
            "memory": "direct__run_memory_encoder_is_mask_from_pts_true",
            "controls": ["native_h28", *ACTIONS[:-1]],
            "training": "none",
            "candidate_phase": "all_binary_trajectories_frozen_before_train_gt",
            "valid_access": False,
            "sealed_test_access": False,
        }
    )


def state_contract_diagnostics(
    state: dict[str, Any], object_index: int, yielded_frames: list[int]
) -> dict[str, Any]:
    frame_count = int(state["num_frames"])
    cond = list(range(min(CONDITIONING_FRAMES, frame_count)))
    suffix = list(range(len(cond), frame_count))
    top = state["output_dict"]
    per_object = state["output_dict_per_obj"][object_index]
    temp = state["temp_output_dict_per_obj"][object_index]
    finite = True
    memory_present = True
    for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
        for output in top[key].values():
            values = [output["pred_masks"], output["obj_ptr"]]
            memory = output["maskmem_features"]
            positions = output["maskmem_pos_enc"]
            memory_present = memory_present and memory is not None and positions is not None
            if memory is not None:
                values.append(memory)
            if positions is not None:
                values.extend(positions)
            finite = finite and all(bool(torch.isfinite(value.float()).all()) for value in values)
    details = {
        "top_cond_frames": sorted(int(x) for x in top["cond_frame_outputs"]),
        "top_non_cond_frames": sorted(int(x) for x in top["non_cond_frame_outputs"]),
        "object_cond_frames": sorted(int(x) for x in per_object["cond_frame_outputs"]),
        "object_non_cond_frames": sorted(int(x) for x in per_object["non_cond_frame_outputs"]),
        "consolidated_cond_frames": sorted(
            int(x) for x in state["consolidated_frame_inds"]["cond_frame_outputs"]
        ),
        "consolidated_non_cond_frames": sorted(
            int(x) for x in state["consolidated_frame_inds"]["non_cond_frame_outputs"]
        ),
        "temp_cond_frames": sorted(int(x) for x in temp["cond_frame_outputs"]),
        "temp_non_cond_frames": sorted(int(x) for x in temp["non_cond_frame_outputs"]),
        "tracked_frames": sorted(int(x) for x in state["frames_already_tracked"]),
        "yielded_frames": yielded_frames,
    }
    contract_ok = bool(
        details["top_cond_frames"] == cond
        and details["object_cond_frames"] == cond
        and details["top_non_cond_frames"] == suffix
        and details["object_non_cond_frames"] == suffix
        and details["consolidated_cond_frames"] == cond
        and details["consolidated_non_cond_frames"] == []
        and details["temp_cond_frames"] == []
        and details["temp_non_cond_frames"] == []
        and details["tracked_frames"] == list(range(frame_count))
        and yielded_frames == list(range(frame_count))
    )
    return {
        "frame_count": frame_count,
        "state_contract_ok": contract_ok,
        "all_tensors_finite": finite,
        "all_memory_present": memory_present,
        "details": details,
    }

@torch.inference_mode()
def infer_independent_trajectory(
    grounding: Any,
    images: torch.Tensor,
    embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predictor = grounding.sam2_model
    state = predictor.init_state(images)
    language = [[embedding] for _ in range(min(CONDITIONING_FRAMES, len(images)))]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        high_logits = grounding.language_embd_inference(state, language)
    low_logits = []
    for frame_index in range(len(images)):
        key = (
            "cond_frame_outputs"
            if frame_index in state["output_dict"]["cond_frame_outputs"]
            else "non_cond_frame_outputs"
        )
        low_logits.append(state["output_dict"][key][frame_index]["pred_masks"][0, 0].float().cpu())
    high = high_logits[:, 0].to(torch.float16).cpu()
    low = torch.stack(low_logits).to(torch.float16)
    conditioning_pointers = torch.stack([
        state["output_dict"]["cond_frame_outputs"][index]["obj_ptr"].float().cpu()
        for index in range(min(CONDITIONING_FRAMES, len(images)))
    ])
    del state, high_logits
    gc.collect()
    torch.cuda.empty_cache()
    return high, low, conditioning_pointers


@torch.inference_mode()
def resize_layer_logits(
    raw: torch.Tensor, size: tuple[int, int], device: torch.device, chunk: int = 8
) -> torch.Tensor:
    layers, frames, _, _ = raw.shape
    result = []
    for start in range(0, frames, chunk):
        stop = min(start + chunk, frames)
        value = raw[:, start:stop].float().to(device)
        resized = F.interpolate(
            value.flatten(0, 1)[:, None],
            size=size,
            mode="bilinear",
            align_corners=False,
        )[:, 0].reshape(layers, stop - start, *size)
        result.append(resized.to(torch.float16).cpu())
    return torch.cat(result, dim=1)


@torch.inference_mode()
def apply_candidate_at_resolution(
    raw: torch.Tensor,
    gates_128: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    chunk: int = 8,
) -> torch.Tensor:
    _, frames, height, width = raw.shape
    outputs = []
    for start in range(0, frames, chunk):
        stop = min(start + chunk, frames)
        values = raw[:, start:stop].float().to(device)
        gates = F.interpolate(
            gates_128[:, start:stop].flatten(0, 1)[:, None].float().to(device),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[:, 0].reshape(len(AUX_LAYERS), stop - start, height, width)
        residuals = (values[:-1] - values[-1]).clamp(-RESIDUAL_CLIP, RESIDUAL_CLIP)
        correction = (weights[:, None, None, None].to(device) * gates * residuals).sum(0)
        outputs.append((values[-1] + correction).float().cpu())
    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def candidate_trajectories(
    candidate: CandidateFusion,
    query: torch.Tensor,
    layer_high: torch.Tensor,
    layer_low: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    train_logits = resize_layer_logits(layer_high, (128, 128), device)
    _, plan = candidate(query.to(device), train_logits.float().to(device))
    gates = plan["spatial_gates"].float().cpu()
    weights = plan["layer_weights"].float().cpu()
    output_high = apply_candidate_at_resolution(layer_high, gates, weights, device)
    state_low = apply_candidate_at_resolution(layer_low, gates, weights, device)
    diagnostics = {
        "layer_weights": {
            str(layer): float(weights[index]) for index, layer in enumerate(AUX_LAYERS)
        },
        "spatial_gate_mean": float(gates.mean()),
        "spatial_gate_mean_abs": float(gates.abs().mean()),
        "spatial_gate_std": float(gates.std(unbiased=False)),
        "state_intervention_vs_h28": sequence_metrics(
            [layer_low[-1, INTERVENTION_FRAME]],
            [state_low[INTERVENTION_FRAME]],
        ),
    }
    del train_logits, plan, gates, weights
    torch.cuda.empty_cache()
    return output_high, state_low, diagnostics


@torch.inference_mode()
def infer_direct_memory_branch(
    predictor: Any,
    images: torch.Tensor,
    h28_embedding: torch.Tensor,
    intervention_low: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    state = predictor.init_state(images)
    object_index = register_object(state, 100)
    object_output = state["output_dict_per_obj"][object_index]
    language = h28_embedding.reshape(1, 1, -1)
    intervention: dict[str, torch.Tensor] = {}

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for frame_index in range(min(CONDITIONING_FRAMES, len(images))):
            current, _ = predictor._run_single_frame_inference(
                inference_state=state,
                output_dict=object_output,
                frame_idx=frame_index,
                batch_size=1,
                is_init_cond_frame=True,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=False,
                prev_sam_mask_logits=None,
                language_embd=language,
            )
            selected = current["pred_masks"]
            if intervention_low is not None and frame_index == INTERVENTION_FRAME:
                selected = intervention_low.reshape(1, 1, *intervention_low.shape).to(
                    state["device"], dtype=torch.float32
                )
            consolidated_logits = selected.to(
                state["storage_device"], dtype=torch.float32, non_blocking=True
            )
            consolidated_pointer = current["obj_ptr"].to(
                state["device"], dtype=torch.float32, non_blocking=True
            )
            high_res_masks = F.interpolate(
                consolidated_logits.to(state["device"], non_blocking=True),
                size=(predictor.image_size, predictor.image_size),
                mode="bilinear",
                align_corners=False,
            )
            if predictor.non_overlap_masks_for_mem_enc:
                high_res_masks = predictor._apply_non_overlapping_constraints(high_res_masks)
            memory, positions = predictor._run_memory_encoder(
                inference_state=state,
                frame_idx=frame_index,
                batch_size=1,
                high_res_masks=high_res_masks,
                is_mask_from_pts=True,
            )
            committed = {
                "maskmem_features": memory,
                "maskmem_pos_enc": positions,
                "pred_masks": consolidated_logits,
                "obj_ptr": consolidated_pointer,
            }
            state["output_dict"]["cond_frame_outputs"][frame_index] = committed
            predictor._add_output_per_object(state, frame_index, committed, "cond_frame_outputs")
            state["consolidated_frame_inds"]["cond_frame_outputs"].add(frame_index)
            if frame_index == INTERVENTION_FRAME:
                intervention = {
                    "decoded_h28_logits": current["pred_masks"].float().cpu(),
                    "committed_logits": consolidated_logits.float().cpu(),
                    "decoded_h28_pointer": current["obj_ptr"].float().cpu(),
                    "committed_pointer": consolidated_pointer.float().cpu(),
                    "memory": memory.float().cpu(),
                }

        outputs = []
        yielded_frames = []
        for frame_index, _, logits in predictor.propagate_in_video(state):
            yielded_frames.append(int(frame_index))
            outputs.append(logits[0, 0].to(torch.float16).cpu())
    diagnostics = state_contract_diagnostics(state, object_index, yielded_frames)
    trajectory = torch.stack(outputs)
    del state, outputs
    gc.collect()
    torch.cuda.empty_cache()
    return trajectory, diagnostics, intervention


@torch.inference_mode()
def binary_at_output(
    logits: torch.Tensor, output_size: tuple[int, int], device: torch.device, chunk: int = 8
) -> np.ndarray:
    width, height = output_size
    masks = []
    for start in range(0, len(logits), chunk):
        value = logits[start : start + chunk].float().to(device)
        resized = F.interpolate(
            value[:, None], size=(height, width), mode="bilinear", align_corners=False
        )[:, 0]
        masks.append((resized > 0).cpu().numpy())
    return np.concatenate(masks, axis=0)


def invariant_audit(
    native_mask: np.ndarray,
    manual_mask: np.ndarray,
    candidate_diag: dict[str, Any],
    manual_diag: dict[str, Any],
    one_shot_diag: dict[str, Any],
    manual_intervention: dict[str, torch.Tensor],
    one_shot_intervention: dict[str, torch.Tensor],
) -> dict[str, Any]:
    pointer = sequence_metrics(
        [manual_intervention["committed_pointer"]],
        [one_shot_intervention["committed_pointer"]],
    )
    memory = sequence_metrics(
        [manual_intervention["memory"]], [one_shot_intervention["memory"]]
    )
    fused = candidate_diag["state_intervention_vs_h28"]
    gates = {
        "native_manual_binary_exact": bool(np.array_equal(native_mask, manual_mask)),
        "manual_state_contract": bool(
            manual_diag["state_contract_ok"]
            and manual_diag["all_tensors_finite"]
            and manual_diag["all_memory_present"]
        ),
        "one_shot_state_contract": bool(
            one_shot_diag["state_contract_ok"]
            and one_shot_diag["all_tensors_finite"]
            and one_shot_diag["all_memory_present"]
        ),
        "fused_mask_finite": bool(
            np.isfinite(fused["max_abs_error"])
            and np.isfinite(fused["mean_abs_error"])
        ),
        "h28_pointer_preserved_exactly": pointer["exact_equal"],
        "memory_finite": bool(
            np.isfinite(memory["max_abs_error"])
            and np.isfinite(memory["mean_abs_error"])
        ),
    }
    return {
        "pointer_identity": pointer,
        "intervention_memory_change": memory,
        "fused_mask_nontrivial": bool(
            fused["max_abs_error"] > 0.0 and fused["mean_abs_error"] > 1e-5
        ),
        "intervention_memory_nontrivial": bool(
            memory["max_abs_error"] > 0.0 and memory["mean_abs_error"] > 0.0
        ),
        "gates": gates,
        "all_invariants_pass": all(gates.values()),
    }


def packed_masks_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare actual pixels without allocating two full-resolution videos.

    np.packbits uses big-endian bits; unused low bits in the final byte are
    ignored, exactly as unpack_binary_mask(..., count=total) ignores them.
    Payload dimensions and lengths are validated separately by the caller.
    """
    if left["shape"] != right["shape"]:
        return False
    total = int(np.prod(left["shape"]))
    full_bytes, remaining_bits = divmod(total, 8)
    first = left["packed"].cpu().numpy()
    second = right["packed"].cpu().numpy()
    if not np.array_equal(first[:full_bytes], second[:full_bytes]):
        return False
    if remaining_bits:
        pixel_bits = (0xFF << (8 - remaining_bits)) & 0xFF
        return (int(first[full_bytes]) & pixel_bits) == (int(second[full_bytes]) & pixel_bits)
    return True


def validate_frozen_row(
    row: dict[str, Any],
    item: dict[str, str],
    selection: dict[str, Any],
    protocol: str,
    candidate_sha: str,
    parent_sha: str,
) -> None:
    """Fail closed on incomplete, swapped, stale, or mislabelled no-GT rows."""
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "stage": STAGE,
        "key": item["key"],
        "video_id": item["video_id"],
        "exp_id": item["exp_id"],
        "selection_sha256": selection["selected_sha256"],
        "protocol_hash": protocol,
        "candidate_checkpoint_sha256": candidate_sha,
        "query_preprocessor_checkpoint_sha256": parent_sha,
        "intervention_frame": INTERVENTION_FRAME,
        "fusion_writes": 1,
        "pointer_source": "native_h28",
    }
    if any(row.get(name) != value for name, value in expected.items()):
        raise RuntimeError(f"frozen metadata contract mismatch: {item['key']}")
    for name in ("gt_access", "valid_access", "sealed_test_access", "training_performed", "dense_mask_redecode"):
        if row.get(name) is not False:
            raise RuntimeError(f"invalid {name} flag: {item['key']}")
    audit = row.get("invariant_audit", {})
    gates = audit.get("gates", {})
    required_gates = {
        "native_manual_binary_exact", "manual_state_contract", "one_shot_state_contract",
        "fused_mask_finite", "h28_pointer_preserved_exactly", "memory_finite",
    }
    if (
        row.get("all_invariants_pass") is not True
        or audit.get("all_invariants_pass") is not True
        or set(gates) != required_gates
        or any(value is not True for value in gates.values())
    ):
        raise RuntimeError(f"frozen invariant contract mismatch: {item['key']}")
    frame_count = int(row.get("frame_count", 0))
    size = row.get("output_size_wh", [])
    if frame_count <= CONDITIONING_FRAMES or len(size) != 2 or min(size) <= 0:
        raise RuntimeError(f"invalid full-video dimensions: {item['key']}")
    if not row.get("annotation_ids"):
        raise RuntimeError(f"missing annotation IDs: {item['key']}")
    expected_shape = [frame_count, int(size[1]), int(size[0])]
    masks = row.get("masks", {})
    if set(masks) != {"native_h28", *ACTIONS}:
        raise RuntimeError(f"missing/mislabelled frozen actions: {item['key']}")
    for payload in masks.values():
        packed = payload.get("packed")
        if (
            payload.get("shape") != expected_shape
            or not isinstance(packed, torch.Tensor)
            or packed.dtype != torch.uint8
            or packed.ndim != 1
            or packed.numel() != (int(np.prod(expected_shape)) + 7) // 8
        ):
            raise RuntimeError(f"invalid packed mask: {item['key']}")
    if not packed_masks_equal(masks["native_h28"], masks["manual_h28_direct_memory"]):
        raise RuntimeError(f"stored masks violate native/manual identity: {item['key']}")


def score_frozen_rows(
    frozen_rows: list[dict[str, Any]],
    split_root: Path,
    output: Path,
    selection: dict[str, Any],
    protocol: str,
    candidate_sha: str,
    parent_sha: str,
) -> dict[str, Any]:
    # Validate again at the GT boundary, including calls made outside run().
    if len(frozen_rows) != SAMPLES or len({row["key"] for row in frozen_rows}) != SAMPLES:
        raise RuntimeError("all distinct registered rows are required before GT access")
    selected = {item["key"]: item for item in selection["selected"]}
    if set(selected) != {row["key"] for row in frozen_rows}:
        raise RuntimeError("score cohort differs from locked selection")
    for frozen in frozen_rows:
        validate_frozen_row(frozen, selected[frozen["key"]], selection, protocol, candidate_sha, parent_sha)
    if not any(row["invariant_audit"]["intervention_memory_nontrivial"] for row in frozen_rows):
        raise RuntimeError("no intervention changed memory; GT remains closed")
    if not any(row["invariant_audit"]["fused_mask_nontrivial"] for row in frozen_rows):
        raise RuntimeError("no non-trivial fused mask; GT remains closed")
    print("EXP050_STAGE1_GT_READ selected_rows=3", flush=True)
    masks = json.loads((split_root / "mask_dict.json").read_text())
    rows = []
    rows_dir = output / "scored_rows"
    rows_dir.mkdir(exist_ok=True)
    for frozen in frozen_rows:
        print(f"EXP050_STAGE1_SCORE {frozen['key']} baseline", flush=True)
        width, height = frozen["output_size_wh"]
        truth = np.zeros((frozen["frame_count"], height, width), dtype=np.uint8)
        for frame_index in range(frozen["frame_count"]):
            truth[frame_index] = decode_frame_mask(
                masks, frozen["annotation_ids"], frame_index, (height, width)
            )
        truth = truth.astype(bool)
        native = unpack_binary_mask(frozen["masks"]["native_h28"])
        native_metrics = segment_metrics(truth, native)
        native_conditioning = segment_metrics(
            truth[:CONDITIONING_FRAMES], native[:CONDITIONING_FRAMES]
        )
        native_suffix = segment_metrics(
            truth[CONDITIONING_FRAMES:], native[CONDITIONING_FRAMES:]
        )
        native_errors = (
            *error_rates(truth[:CONDITIONING_FRAMES], native[:CONDITIONING_FRAMES]),
            *error_rates(truth[CONDITIONING_FRAMES:], native[CONDITIONING_FRAMES:]),
        )
        native_frame_jf = [
            float(segment_metrics(truth[i : i + 1], native[i : i + 1])["J&F"])
            for i in range(len(truth))
        ]
        native_areas = native.mean(axis=(1, 2))
        actions = {}
        for action in ACTIONS:
            print(f"EXP050_STAGE1_SCORE {frozen['key']} action={action}", flush=True)
            candidate = unpack_binary_mask(frozen["masks"][action])
            scored = score_action(
                truth,
                native,
                candidate,
                native_metrics,
                native_conditioning,
                native_suffix,
                native_errors,
            )
            divergence = np.flatnonzero(np.any(candidate != native, axis=(1, 2)))
            scored["first_binary_divergence_frame"] = (
                None if not len(divergence) else int(divergence[0])
            )
            suffix_divergence = divergence[divergence >= CONDITIONING_FRAMES]
            scored["first_suffix_binary_divergence_frame"] = (
                None if not len(suffix_divergence) else int(suffix_divergence[0])
            )
            frame_deltas = [
                100.0 * (
                    float(segment_metrics(truth[i : i + 1], candidate[i : i + 1])["J&F"])
                    - native_frame_jf[i]
                )
                for i in range(len(truth))
            ]
            scored["per_frame_delta_jf_points"] = frame_deltas
            scored["per_frame_area_delta_image_points"] = (
                100.0 * (candidate.mean(axis=(1, 2)) - native_areas)
            ).tolist()
            scored["intervention_frame_delta_jf_points"] = frame_deltas[INTERVENTION_FRAME]
            scored["next_frame_delta_jf_points"] = frame_deltas[INTERVENTION_FRAME + 1]
            scored["first_suffix_frame_delta_jf_points"] = frame_deltas[CONDITIONING_FRAMES]
            actions[action] = scored
        row = {
            "key": frozen["key"],
            "video_id": frozen["video_id"],
            "exp_id": frozen["exp_id"],
            "baseline_h28": native_metrics,
            "baseline_conditioning": native_conditioning,
            "baseline_suffix": native_suffix,
            "actions": actions,
            "candidate_diagnostics": frozen["candidate_diagnostics"],
            "invariant_audit": frozen["invariant_audit"],
            "selection_sha256": selection["selected_sha256"],
            "protocol_hash": protocol,
            "gt_access_after_all_candidate_freeze": True,
            "valid_access": False,
            "sealed_test_access": False,
        }
        atomic_json(rows_dir / f"{frozen['key']}.json", row)
        rows.append(row)

    action_summary = {}
    for action in ACTIONS:
        deltas = [float(row["actions"][action]["delta_J&F_points"]) for row in rows]
        suffix = [
            float(row["actions"][action]["suffix_delta_J&F_points"]) for row in rows
        ]
        action_summary[action] = {
            "mean_metrics": {
                metric: float(np.mean([row["actions"][action]["metrics"][metric] for row in rows]))
                for metric in ("J", "F", "J&F")
            },
            "delta_jf_points": paired_summary(deltas),
            "conditioning_delta_jf_points": paired_summary(
                [float(row["actions"][action]["conditioning_delta_J&F_points"]) for row in rows]
            ),
            "suffix_delta_jf_points": paired_summary(suffix),
            "suffix_nonnegative_count": sum(value >= 0.0 for value in suffix),
            "declines_at_or_below_minus_5": sum(value <= -5.0 for value in deltas),
            "leave_best_out_mean_delta_jf_points": (sum(deltas) - max(deltas)) / (len(deltas) - 1),
        }
    one = action_summary["one_shot_fused_memory"]
    stage2_gates = {
        "mean_full_delta_nonnegative": one["delta_jf_points"]["mean"] >= 0.0,
        "mean_suffix_delta_nonnegative": one["suffix_delta_jf_points"]["mean"] >= 0.0,
        "suffix_nonnegative_count_ge_2": one["suffix_nonnegative_count"] >= 2,
        "no_full_decline_at_or_below_minus_5": one["declines_at_or_below_minus_5"] == 0,
        "all_no_gt_invariants_pass": all(row["invariant_audit"]["all_invariants_pass"] for row in rows),
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "stage": STAGE,
        "completed": len(rows),
        "native_mean_metrics": {
            metric: float(np.mean([row["baseline_h28"][metric] for row in rows]))
            for metric in ("J", "F", "J&F")
        },
        "actions": action_summary,
        "stage2_gates": stage2_gates,
        "stage2_authorized": all(stage2_gates.values()),
        "selection_sha256": selection["selected_sha256"],
        "protocol_hash": protocol,
        "training_performed": False,
        "gt_access_after_all_candidate_freeze": True,
        "valid_access": False,
        "sealed_test_access": False,
    }


def self_test() -> None:
    candidate = CandidateFusion(16)
    query = torch.randn(16)
    logits = torch.randn(4, 3, 16, 16)
    fused, plan = candidate(query, logits)
    assert torch.equal(fused, logits[-1].float())
    assert torch.equal(plan["spatial_gates"], torch.zeros_like(plan["spatial_gates"]))
    metrics = sequence_metrics([logits[-1, 0]], [logits[-1, 0].clone()])
    assert metrics["exact_equal"] and metrics["sign_xor"] == 0.0
    print(json.dumps({"state": "self_test_passed", "intervention_frame": INTERVENTION_FRAME}, indent=2))


def run(args: argparse.Namespace) -> None:
    required = (
        args.model_path,
        args.data_root,
        args.parent_dir,
        args.candidate_checkpoint,
        args.stage0_selection,
        args.stage0_summary,
        args.output,
    )
    if not all(required):
        raise ValueError("all model/data/checkpoint/selection/output paths are required")
    if args.candidate_only == args.score_only:
        raise ValueError("choose exactly one of --candidate-only or --score-only")
    if args.limit is not None and not 1 <= args.limit <= SAMPLES:
        raise ValueError("limit must be between one and three")
    if args.score_only and args.limit is not None:
        raise ValueError("GT scoring cannot be combined with a partial limit")

    output = Path(args.output)
    frozen_dir = output / "frozen_trajectories"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(args.stage0_selection).read_text())
    stage0 = json.loads(Path(args.stage0_summary).read_text())
    if (
        len(selection["selected"]) != SAMPLES
        or sha256_json(selection["selected"]) != selection["selected_sha256"]
        or len({item["video_id"] for item in selection["selected"]}) != SAMPLES
        or stage0.get("all_identity_gates_pass") is not True
        or stage0.get("completed") != SAMPLES
        or stage0.get("gt_access") is not False
        or stage0.get("selected_sha256") != selection["selected_sha256"]
    ):
        raise RuntimeError("Stage 0 authorization contract is not satisfied")
    parent_path = Path(args.parent_dir) / "checkpoint_best.pt"
    candidate_path = Path(args.candidate_checkpoint)
    parent_sha = file_sha256(parent_path)
    candidate_sha = file_sha256(candidate_path)
    if candidate_sha != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError("frozen exp049 checkpoint hash differs from pre-registration")
    protocol = protocol_hash(selection, parent_sha, candidate_sha, args.seed)
    split_root = Path(args.data_root) / "video_datas" / "mevis" / "train"
    metadata = json.loads((split_root / "meta_expressions.json").read_text())["videos"]
    targets = selection["selected"][: args.limit] if args.limit else selection["selected"]

    frozen_paths = []
    pending = []
    for number, item in enumerate(targets, 1):
        path = frozen_dir / f"{item['key']}.pt"
        if path.exists():
            row = torch.load(path, map_location="cpu", weights_only=False)
            validate_frozen_row(row, item, selection, protocol, candidate_sha, parent_sha)
            frozen_paths.append(path)
            print(f"EXP050_STAGE1_RESUME [{number}/{len(targets)}] {item['key']}", flush=True)
        else:
            pending.append((number, item, path))

    if pending:
        if args.score_only:
            raise RuntimeError("score-only cannot generate candidates; run stage1_freeze first")
        if (output / "gt_access_receipt.json").exists():
            raise RuntimeError("candidate generation is forbidden after this cohort's GT boundary")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for missing candidates")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device("cuda:0")
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        if tuple(parent["layer_indices"]) != ALL_LAYERS or int(parent["best_epoch"]) != 3:
            raise RuntimeError("exp037 query preprocessor checkpoint contract mismatch")
        preprocessor = preprocessor_from_checkpoint(parent)
        checkpoint = torch.load(candidate_path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("best_epoch", -1)) != 12:
            raise RuntimeError("exp049 diagnostic candidate must be selected epoch 12")
        candidate = CandidateFusion(int(checkpoint["query_dim"])).to(device)
        candidate.load_state_dict(checkpoint["state_dict"])
        candidate.eval()
        for parameter in candidate.parameters():
            parameter.requires_grad_(False)

        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model.preparing_for_generation(tokenizer=tokenizer)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        grounding = model.grounding_encoder
        predictor = grounding.sam2_model
        projection = model.text_hidden_fcs
        projection_dtype = next(projection.parameters()).dtype

        for number, item, path in pending:
            print(f"EXP050_STAGE1_BEGIN [{number}/{len(targets)}] {item['key']}", flush=True)
            # Resume must not change the generation RNG for the remaining rows.
            row_seed = (args.seed + int(hashlib.sha256(item["key"].encode()).hexdigest()[:8], 16)) % (2**32)
            torch.manual_seed(row_seed)
            np.random.seed(row_seed)
            event = build_event(Path(args.data_root), item, metadata)
            states, _, _, _ = extract_states(
                model, tokenizer, event, CONDITIONING_FRAMES, ALL_LAYERS
            )
            aligned = TgifSegRouter.rms_align(states.float().to(device))
            embeddings = projection(aligned.to(dtype=projection_dtype)).reshape(len(ALL_LAYERS), -1)
            query = query_embedding(model, tokenizer, event["prompt"])
            reduced_query = transform_query(query[None], preprocessor)[0].to(torch.float16).float()
            images, output_size = preprocess_video(
                grounding, model.extra_image_processor, model.torch_dtype, event["frame_paths"]
            )
            images = images.cpu()

            high_by_layer = []
            low_by_layer = []
            pointers_by_layer = []
            for layer in LAYERS:
                high, low, pointers = infer_independent_trajectory(
                    grounding, images, embeddings[layer - 1]
                )
                high_by_layer.append(high)
                low_by_layer.append(low)
                pointers_by_layer.append(pointers)
                print(f"EXP050_STAGE1_EXPERT {item['key']} layer={layer} frames={len(images)}", flush=True)
            layer_high = torch.stack(high_by_layer)
            layer_low = torch.stack(low_by_layer)
            del high_by_layer, low_by_layer
            if (
                layer_high.shape[:2] != (len(LAYERS), len(images))
                or layer_low.shape != (len(LAYERS), len(images), 256, 256)
                or not torch.isfinite(layer_high).all()
                or not torch.isfinite(layer_low).all()
            ):
                raise RuntimeError("independent expert logits must be finite and frame-complete")
            output_candidate, state_candidate, candidate_diag = candidate_trajectories(
                candidate, reduced_query, layer_high, layer_low, device
            )
            candidate_diag["auxiliary_vs_h28_conditioning_pointers"] = {
                str(layer): sequence_metrics(
                    list(pointers_by_layer[-1]), list(pointers_by_layer[index])
                )
                for index, layer in enumerate(AUX_LAYERS)
            }
            del pointers_by_layer
            print(f"EXP050_STAGE1_CANDIDATE {item['key']}", flush=True)

            manual_high, manual_diag, manual_intervention = infer_direct_memory_branch(
                predictor, images, embeddings[-1], None
            )
            print(f"EXP050_STAGE1_MANUAL_H28 {item['key']}", flush=True)
            one_shot_high, one_shot_diag, one_shot_intervention = infer_direct_memory_branch(
                predictor,
                images,
                embeddings[-1],
                state_candidate[INTERVENTION_FRAME],
            )
            print(f"EXP050_STAGE1_ONE_SHOT {item['key']}", flush=True)
            native_mask = binary_at_output(layer_high[-1], output_size, device)
            manual_mask = binary_at_output(manual_high, output_size, device)
            output_mask = binary_at_output(output_candidate, output_size, device)
            one_shot_mask = binary_at_output(one_shot_high, output_size, device)
            invariants = invariant_audit(
                native_mask,
                manual_mask,
                candidate_diag,
                manual_diag,
                one_shot_diag,
                manual_intervention,
                one_shot_intervention,
            )
            if not invariants["all_invariants_pass"]:
                raise RuntimeError(f"Stage 1 no-GT invariants failed for {event['key']}: {invariants['gates']}")
            frozen = {
                "experiment_id": EXPERIMENT_ID,
                "stage": STAGE,
                "key": event["key"],
                "video_id": event["video_id"],
                "exp_id": event["exp_id"],
                "annotation_ids": event["annotation_ids"],
                "frame_count": len(images),
                "row_seed": row_seed,
                "output_size_wh": list(output_size),
                "selection_sha256": selection["selected_sha256"],
                "protocol_hash": protocol,
                "candidate_checkpoint_sha256": candidate_sha,
                "query_preprocessor_checkpoint_sha256": parent_sha,
                "candidate_diagnostics": candidate_diag,
                "manual_state_diagnostics": manual_diag,
                "one_shot_state_diagnostics": one_shot_diag,
                "invariant_audit": invariants,
                "all_invariants_pass": True,
                "masks": {
                    "native_h28": pack_binary_mask(native_mask),
                    "manual_h28_direct_memory": pack_binary_mask(manual_mask),
                    "exp049_output_only": pack_binary_mask(output_mask),
                    "one_shot_fused_memory": pack_binary_mask(one_shot_mask),
                },
                "intervention_frame": INTERVENTION_FRAME,
                "fusion_writes": 1,
                "dense_mask_redecode": False,
                "pointer_source": "native_h28",
                "training_performed": False,
                "gt_access": False,
                "valid_access": False,
                "sealed_test_access": False,
            }
            validate_frozen_row(frozen, item, selection, protocol, candidate_sha, parent_sha)
            atomic_torch(path, frozen)
            frozen_paths.append(path)
            print(
                f"EXP050_STAGE1_FREEZE [{number}/{len(targets)}] {event['key']} "
                f"mask_xor={candidate_diag['state_intervention_vs_h28']['sign_xor']:.8f} "
                f"memory_maxerr={invariants['intervention_memory_change']['max_abs_error']:.8f}",
                flush=True,
            )
            del event, states, aligned, embeddings, query, reduced_query, images
            del layer_high, layer_low
            del output_candidate, state_candidate, candidate_diag
            del manual_high, manual_diag, manual_intervention
            del one_shot_high, one_shot_diag, one_shot_intervention
            del native_mask, manual_mask, output_mask, one_shot_mask, invariants, frozen
            gc.collect()
            torch.cuda.empty_cache()

        del candidate, model, tokenizer, grounding, predictor, projection
        gc.collect()
        torch.cuda.empty_cache()

    frozen_paths = sorted(set(frozen_paths))
    all_registered = sorted(frozen_dir / f"{item['key']}.pt" for item in selection["selected"])
    all_frozen = all(path.exists() for path in all_registered)
    frozen_audits = []
    verified_rows = []
    if all_frozen:
        selected_by_key = {item["key"]: item for item in selection["selected"]}
        for path in all_registered:
            row = torch.load(path, map_location="cpu", weights_only=False)
            validate_frozen_row(row, selected_by_key[path.stem], selection, protocol, candidate_sha, parent_sha)
            print(f"EXP050_STAGE1_VERIFIED {path.stem}", flush=True)
            frozen_audits.append(row["invariant_audit"])
            verified_rows.append(row)
    cohort_invariants_pass = bool(
        all_frozen
        and all(item["all_invariants_pass"] for item in frozen_audits)
        and any(item["fused_mask_nontrivial"] for item in frozen_audits)
        and any(item["intervention_memory_nontrivial"] for item in frozen_audits)
    )
    atomic_json(
        output / "status_candidates.json",
        {
            "state": "complete",
            "phase": "all_stage1_candidates_frozen_no_gt" if all_frozen else "partial_stage1_candidates_frozen_no_gt",
            "completed": sum(path.exists() for path in all_registered),
            "registered": SAMPLES,
            "selection_sha256": selection["selected_sha256"],
            "protocol_hash": protocol,
            "all_invariants_pass": cohort_invariants_pass,
            "nontrivial_fused_mask_rows": sum(
                int(item["fused_mask_nontrivial"]) for item in frozen_audits
            ),
            "nontrivial_intervention_memory_rows": sum(
                int(item["intervention_memory_nontrivial"]) for item in frozen_audits
            ),
            "training_performed": False,
            "gt_access": False,
            "valid_access": False,
            "sealed_test_access": False,
        },
    )
    freeze_manifest = {
        "selection_sha256": selection["selected_sha256"],
        "protocol_hash": protocol,
        "candidate_checkpoint_sha256": candidate_sha,
        "files": [
            {"key": path.stem, "filename": path.name, "sha256": file_sha256(path)}
            for path in all_registered if path.exists()
        ],
        "all_invariants_pass": cohort_invariants_pass,
        "gt_access": False,
    }
    manifest_path = output / "freeze_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != freeze_manifest:
        raise RuntimeError("frozen candidate file hashes changed")
    if args.candidate_only:
        if cohort_invariants_pass and not manifest_path.exists():
            atomic_json(manifest_path, freeze_manifest)
        return
    if not cohort_invariants_pass:
        raise RuntimeError(
            "GT scoring is blocked until all rows are frozen and at least one "
            "fused mask changes direct memory"
        )
    if not manifest_path.exists():
        raise RuntimeError("score-only requires the completed no-GT freeze manifest")
    summary_path = output / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous.get("protocol_hash") != protocol:
            raise RuntimeError("existing score belongs to another protocol")
        print("EXP050_STAGE1_SCORE_ALREADY_COMPLETE", flush=True)
        return
    atomic_json(
        output / "gt_access_receipt.json",
        {
            "phase": "entering_selected_train_gt_scoring",
            "selection_sha256": selection["selected_sha256"],
            "protocol_hash": protocol,
            "freeze_manifest_sha256": file_sha256(manifest_path),
            "selected_keys": [row["key"] for row in verified_rows],
            "valid_access": False,
            "sealed_test_access": False,
        },
    )
    summary = score_frozen_rows(
        verified_rows, split_root, output, selection, protocol, candidate_sha, parent_sha
    )
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "status.json",
        {
            "state": "complete",
            "phase": "stage1_scored_stage2_authorized" if summary["stage2_authorized"] else "stage1_scored_stage2_blocked",
            "completed": SAMPLES,
            "stage2_authorized": summary["stage2_authorized"],
            "selection_sha256": selection["selected_sha256"],
            "protocol_hash": protocol,
            "training_performed": False,
            "gt_access_after_all_candidate_freeze": True,
            "valid_access": False,
            "sealed_test_access": False,
        },
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    trace_seconds = int(os.environ.get("EXP050_TRACEBACK_SECONDS", "0"))
    if trace_seconds > 0:
        faulthandler.dump_traceback_later(trace_seconds, repeat=True)
    try:
        run(args)
    finally:
        if trace_seconds > 0:
            faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
