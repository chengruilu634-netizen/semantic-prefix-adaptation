#!/usr/bin/env python3
"""Audit native h28 against a manual direct-memory SAM2 state commit.

This is exp_050 Stage 0.  It never reads segmentation ground truth and it does
not construct a multi-layer candidate.  The only tested intervention is the
state-transfer implementation needed by later recurrent mask-fusion stages.
"""

from __future__ import annotations

import argparse
import gc
import json
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
from work.evaluate_layer_policy_oracle_feasibility import atomic_json
from work.evaluate_tgif_disjoint_confirmation import (
    select_items,
    sha256_json,
    video_id_from_key,
)


EXPERIMENT_ID = "exp_050_recurrent_direct_memory_mask_fusion"
STAGE = "stage0_h28_manual_direct_memory_identity"
CONDITIONING_FRAMES = 5
IDENTITY_SAMPLES = 3

GATE_LIMITS = {
    "stored_logit_sign_xor_max": 0.001,
    "stored_logit_cosine_min": 0.99999,
    "stored_logit_max_abs_error_max": 0.05,
    "pointer_cosine_min": 0.999999,
    "pointer_max_abs_error_max": 0.001,
    "memory_cosine_min": 0.999999,
    "memory_max_abs_error_max": 0.001,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", nargs="?")
    parser.add_argument("--data-root")
    parser.add_argument("--router-split")
    parser.add_argument("--calibration-manifest")
    parser.add_argument("--exp039-selection")
    parser.add_argument("--exp043-selection")
    parser.add_argument("--exp044-selection")
    parser.add_argument("--exp049-prior-exclusion")
    parser.add_argument("--exp049-split")
    parser.add_argument("--output")
    parser.add_argument("--samples", type=int, default=IDENTITY_SAMPLES)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--selection-salt",
        default="exp050_recurrent_direct_memory_identity_v1",
    )
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(item["video_id"]) for item in rows}


def build_selection_manifest(
    metadata: dict[str, Any],
    router_split: dict[str, Any],
    calibration: dict[str, Any],
    exp039: dict[str, Any],
    exp043: dict[str, Any],
    exp044: dict[str, Any],
    exp049_prior: dict[str, Any],
    exp049_split: dict[str, Any],
    samples: int,
    salt: str,
) -> dict[str, Any]:
    historical = {
        "exp037_train": {
            video_id_from_key(str(key)) for key in router_split["train_keys"]
        },
        "exp037_validation": {
            video_id_from_key(str(key)) for key in router_split["validation_keys"]
        },
        "historical_calibration": _ids(calibration["calibration"]),
        "exp039": _ids(exp039["selected"]),
        "exp043": _ids(exp043["selected"]),
        "exp044": _ids(exp044["selected"]),
        "exp049_prior": _ids(exp049_prior["selected"]),
    }
    pre_exp049 = set().union(*historical.values())
    registered = exp049_split["excluded"]
    if len(pre_exp049) != int(registered["total_unique_count"]):
        raise RuntimeError("reconstructed exp049 exclusion count mismatch")
    if sha256_json(sorted(pre_exp049)) != registered["video_ids_sha256"]:
        raise RuntimeError("reconstructed exp049 exclusion hash mismatch")

    exp049_groups = {
        name: _ids(exp049_split[name])
        for name in ("train", "validation", "prospective")
    }
    if any(
        exp049_groups[left] & exp049_groups[right]
        for index, left in enumerate(exp049_groups)
        for right in list(exp049_groups)[index + 1 :]
    ):
        raise RuntimeError("exp049 split groups overlap")
    if pre_exp049 & set().union(*exp049_groups.values()):
        raise RuntimeError("exp049 split overlaps its registered exclusions")

    excluded = pre_exp049 | set().union(*exp049_groups.values())
    selected = select_items(metadata, excluded, samples, salt)
    return {
        "state": "locked_before_model_load_no_gt",
        "selection_salt": salt,
        "selection_rule": {
            "video": "ascending sha256(<salt>:<video_id>)",
            "expression": (
                "ascending sha256(<salt>:<video_id>:<expression_id>) "
                "among annotated expressions"
            ),
            "one_expression_per_video": True,
        },
        "selected": selected,
        "selected_sha256": sha256_json(selected),
        "excluded": {
            **{f"{name}_count": len(values) for name, values in historical.items()},
            **{f"exp049_{name}_count": len(values) for name, values in exp049_groups.items()},
            "pre_exp049_unique_count": len(pre_exp049),
            "pre_exp049_video_ids_sha256": sha256_json(sorted(pre_exp049)),
            "total_unique_count": len(excluded),
            "video_ids_sha256": sha256_json(sorted(excluded)),
        },
        "metadata_video_count": len(metadata),
        "gt_access": False,
        "valid_access": False,
        "sealed_test_access": False,
    }


def protocol_hash(selection: dict[str, Any]) -> str:
    return sha256_json(
        {
            "experiment_id": EXPERIMENT_ID,
            "stage": STAGE,
            "selection_sha256": selection["selected_sha256"],
            "model_prompt": "native_projected_h28",
            "conditioning_frames": CONDITIONING_FRAMES,
            "native_reference": "grounding.language_embd_inference",
            "manual_decode": "_run_single_frame_inference_run_mem_encoder_false",
            "manual_memory": (
                "low_res_h28_logits_bilinear_to_image_size_then_"
                "_run_memory_encoder_is_mask_from_pts_true"
            ),
            "manual_pointer": "unchanged_h28_decoder_obj_ptr",
            "manual_storage": "direct_cond_frame_output_and_per_object_slice",
            "propagation": "native_propagate_in_video_after_five_cond_states",
            "training": "none",
            "gt_access": False,
            "gate_limits": GATE_LIMITS,
        }
    )


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to("cpu", non_blocking=False).contiguous()


def snapshot_state(
    state: dict[str, Any], object_index: int, conditioning_frames: int
) -> dict[str, Any]:
    frame_count = int(state["num_frames"])
    top = state["output_dict"]
    per_object = state["output_dict_per_obj"][object_index]
    tensors: dict[str, list[torch.Tensor]] = {
        "logits": [],
        "pointers": [],
        "memories": [],
        "memory_positional_encodings": [],
    }
    assignments: list[str] = []
    all_finite = True
    all_memory_present = True
    for frame_index in range(frame_count):
        hits = [
            key
            for key in ("cond_frame_outputs", "non_cond_frame_outputs")
            if frame_index in top[key]
        ]
        if len(hits) != 1:
            raise RuntimeError(
                f"frame {frame_index} has {len(hits)} top-level state assignments"
            )
        storage_key = hits[0]
        assignments.append(storage_key)
        output = top[storage_key][frame_index]
        object_output = per_object[storage_key].get(frame_index)
        if object_output is None:
            raise RuntimeError(f"frame {frame_index} missing per-object state")
        logits = _cpu_tensor(output["pred_masks"])
        pointer = _cpu_tensor(output["obj_ptr"])
        memory = output["maskmem_features"]
        positions = output["maskmem_pos_enc"]
        if memory is None or positions is None:
            all_memory_present = False
            raise RuntimeError(f"frame {frame_index} has no encoded memory")
        memory_cpu = _cpu_tensor(memory)
        position_cpu = torch.cat(
            [_cpu_tensor(value).reshape(-1) for value in positions]
        )
        tensors["logits"].append(logits)
        tensors["pointers"].append(pointer)
        tensors["memories"].append(memory_cpu)
        tensors["memory_positional_encodings"].append(position_cpu)
        all_finite = all_finite and all(
            bool(torch.isfinite(value.float()).all())
            for value in (logits, pointer, memory_cpu, position_cpu)
        )

    expected_cond = list(range(min(conditioning_frames, frame_count)))
    expected_non_cond = list(range(len(expected_cond), frame_count))
    temp = state["temp_output_dict_per_obj"][object_index]
    state_contract = {
        "top_cond_frames": sorted(int(x) for x in top["cond_frame_outputs"]),
        "top_non_cond_frames": sorted(int(x) for x in top["non_cond_frame_outputs"]),
        "object_cond_frames": sorted(
            int(x) for x in per_object["cond_frame_outputs"]
        ),
        "object_non_cond_frames": sorted(
            int(x) for x in per_object["non_cond_frame_outputs"]
        ),
        "consolidated_cond_frames": sorted(
            int(x) for x in state["consolidated_frame_inds"]["cond_frame_outputs"]
        ),
        "consolidated_non_cond_frames": sorted(
            int(x)
            for x in state["consolidated_frame_inds"]["non_cond_frame_outputs"]
        ),
        "temp_cond_frames": sorted(int(x) for x in temp["cond_frame_outputs"]),
        "temp_non_cond_frames": sorted(
            int(x) for x in temp["non_cond_frame_outputs"]
        ),
        "tracked_frames": sorted(int(x) for x in state["frames_already_tracked"]),
        "assignments": assignments,
    }
    state_contract_ok = bool(
        state_contract["top_cond_frames"] == expected_cond
        and state_contract["object_cond_frames"] == expected_cond
        and state_contract["top_non_cond_frames"] == expected_non_cond
        and state_contract["object_non_cond_frames"] == expected_non_cond
        and state_contract["consolidated_cond_frames"] == expected_cond
        and state_contract["consolidated_non_cond_frames"] == []
        and state_contract["temp_cond_frames"] == []
        and state_contract["temp_non_cond_frames"] == []
        and state_contract["tracked_frames"] == list(range(frame_count))
    )
    return {
        "tensors": tensors,
        "diagnostics": {
            "frame_count": frame_count,
            "conditioning_frames": len(expected_cond),
            "all_tensors_finite": all_finite,
            "all_memory_present": all_memory_present,
            "state_contract_ok": state_contract_ok,
            "state_contract": state_contract,
        },
    }


def sequence_metrics(
    reference: list[torch.Tensor], candidate: list[torch.Tensor]
) -> dict[str, Any]:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("tensor sequences must be non-empty and have equal length")
    dot = ref_sq = cand_sq = abs_sum = diff_sq = 0.0
    max_abs = 0.0
    sign_different = 0
    element_count = 0
    exact = True
    frame_cosines: list[float] = []
    frame_sign_xor: list[float] = []
    for left, right in zip(reference, candidate):
        if left.shape != right.shape:
            raise ValueError(f"tensor shape mismatch: {left.shape} versus {right.shape}")
        exact = exact and torch.equal(left, right)
        left64 = left.double().reshape(-1)
        right64 = right.double().reshape(-1)
        diff = right64 - left64
        dot_i = float(torch.dot(left64, right64))
        ref_sq_i = float(torch.dot(left64, left64))
        cand_sq_i = float(torch.dot(right64, right64))
        denom_i = max((ref_sq_i * cand_sq_i) ** 0.5, 1e-30)
        frame_cosines.append(dot_i / denom_i)
        xor_i = int(((left64 > 0) != (right64 > 0)).sum())
        frame_sign_xor.append(xor_i / max(left64.numel(), 1))
        dot += dot_i
        ref_sq += ref_sq_i
        cand_sq += cand_sq_i
        abs_sum += float(diff.abs().sum())
        diff_sq += float(torch.dot(diff, diff))
        max_abs = max(max_abs, float(diff.abs().max()))
        sign_different += xor_i
        element_count += left64.numel()
    denom = max((ref_sq * cand_sq) ** 0.5, 1e-30)
    return {
        "frames": len(reference),
        "elements": element_count,
        "exact_equal": exact,
        "cosine": dot / denom,
        "min_frame_cosine": min(frame_cosines),
        "mean_abs_error": abs_sum / element_count,
        "rmse": (diff_sq / element_count) ** 0.5,
        "max_abs_error": max_abs,
        "sign_xor": sign_different / element_count,
        "max_frame_sign_xor": max(frame_sign_xor),
    }


@torch.inference_mode()
def infer_native_reference(
    grounding: Any,
    predictor: Any,
    images: torch.Tensor,
    backbone_cache: dict[int, Any],
    embedding: torch.Tensor,
) -> dict[str, Any]:
    state = predictor.init_state(images)
    state["cached_features"] = backbone_cache.copy()
    language = [[embedding] for _ in range(min(CONDITIONING_FRAMES, len(images)))]
    logits = grounding.language_embd_inference(state, language)
    wrapper_diagnostics = {
        "frame_count": int(logits.shape[0]),
        "all_logits_finite": bool(torch.isfinite(logits.float()).all()),
    }
    snapshot = snapshot_state(state, 0, CONDITIONING_FRAMES)
    snapshot["diagnostics"]["wrapper"] = wrapper_diagnostics
    del state, logits
    gc.collect()
    torch.cuda.empty_cache()
    return snapshot


@torch.inference_mode()
def infer_manual_direct_memory(
    predictor: Any,
    images: torch.Tensor,
    backbone_cache: dict[int, Any],
    embedding: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, list[torch.Tensor]]]:
    state = predictor.init_state(images)
    state["cached_features"] = backbone_cache.copy()
    object_index = register_object(state, 100)
    object_output = state["output_dict_per_obj"][object_index]
    language = embedding.reshape(1, 1, -1)
    decode_tensors: dict[str, list[torch.Tensor]] = {"logits": [], "pointers": []}
    frame_count = min(CONDITIONING_FRAMES, len(images))

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for frame_index in range(frame_count):
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
            if current["maskmem_features"] is not None:
                raise RuntimeError("manual candidate decode unexpectedly wrote memory")
            decode_tensors["logits"].append(_cpu_tensor(current["pred_masks"]))
            decode_tensors["pointers"].append(_cpu_tensor(current["obj_ptr"]))

            # Native consolidation first copies each per-object bfloat16 mask and
            # pointer into float32 batch tensors.  Preserve that dtype/storage
            # contract before invoking the direct memory interface.
            consolidated_logits = current["pred_masks"].to(
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
                high_res_masks = predictor._apply_non_overlapping_constraints(
                    high_res_masks
                )
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
            predictor._add_output_per_object(
                state, frame_index, committed, "cond_frame_outputs"
            )
            state["consolidated_frame_inds"]["cond_frame_outputs"].add(frame_index)

        yielded_frames: list[int] = []
        yielded_finite = True
        for frame_index, _, logits in predictor.propagate_in_video(state):
            yielded_frames.append(int(frame_index))
            yielded_finite = yielded_finite and bool(torch.isfinite(logits.float()).all())

    snapshot = snapshot_state(state, object_index, CONDITIONING_FRAMES)
    snapshot["diagnostics"]["manual_yield"] = {
        "frames": yielded_frames,
        "complete_in_order": yielded_frames == list(range(len(images))),
        "all_logits_finite": yielded_finite,
    }
    del state
    gc.collect()
    torch.cuda.empty_cache()
    return snapshot, decode_tensors


def audit_identity(
    native: dict[str, Any],
    manual: dict[str, Any],
    manual_decode: dict[str, list[torch.Tensor]],
) -> dict[str, Any]:
    comparisons = {
        name: sequence_metrics(native["tensors"][name], manual["tensors"][name])
        for name in (
            "logits",
            "pointers",
            "memories",
            "memory_positional_encodings",
        )
    }
    count = min(CONDITIONING_FRAMES, len(manual["tensors"]["logits"]))
    direct_commit = {
        "logits": sequence_metrics(
            manual_decode["logits"], manual["tensors"]["logits"][:count]
        ),
        "pointers": sequence_metrics(
            manual_decode["pointers"], manual["tensors"]["pointers"][:count]
        ),
    }
    gates = {
        "native_state_contract": bool(
            native["diagnostics"]["state_contract_ok"]
            and native["diagnostics"]["all_tensors_finite"]
            and native["diagnostics"]["all_memory_present"]
            and native["diagnostics"]["wrapper"]["all_logits_finite"]
        ),
        "manual_state_contract": bool(
            manual["diagnostics"]["state_contract_ok"]
            and manual["diagnostics"]["all_tensors_finite"]
            and manual["diagnostics"]["all_memory_present"]
            and manual["diagnostics"]["manual_yield"]["complete_in_order"]
            and manual["diagnostics"]["manual_yield"]["all_logits_finite"]
        ),
        "direct_commit_preserves_h28_logits": direct_commit["logits"]["exact_equal"],
        "direct_commit_preserves_h28_pointer": direct_commit["pointers"]["exact_equal"],
        "stored_logit_sign": (
            comparisons["logits"]["sign_xor"]
            <= GATE_LIMITS["stored_logit_sign_xor_max"]
        ),
        "stored_logit_values": (
            comparisons["logits"]["cosine"]
            >= GATE_LIMITS["stored_logit_cosine_min"]
            and comparisons["logits"]["max_abs_error"]
            <= GATE_LIMITS["stored_logit_max_abs_error_max"]
        ),
        "object_pointer_identity": (
            comparisons["pointers"]["cosine"]
            >= GATE_LIMITS["pointer_cosine_min"]
            and comparisons["pointers"]["max_abs_error"]
            <= GATE_LIMITS["pointer_max_abs_error_max"]
        ),
        "memory_feature_identity": (
            comparisons["memories"]["cosine"]
            >= GATE_LIMITS["memory_cosine_min"]
            and comparisons["memories"]["max_abs_error"]
            <= GATE_LIMITS["memory_max_abs_error_max"]
        ),
        "memory_position_identity": comparisons[
            "memory_positional_encodings"
        ]["exact_equal"],
    }
    return {
        "comparisons": comparisons,
        "manual_decode_to_direct_commit": direct_commit,
        "gates": gates,
        "identity_gate_pass": all(gates.values()),
    }


def summarize(rows: list[dict[str, Any]], selection: dict[str, Any], protocol: str) -> dict[str, Any]:
    metric_names = ("logits", "pointers", "memories", "memory_positional_encodings")
    aggregate = {}
    for name in metric_names:
        values = [row["audit"]["comparisons"][name] for row in rows]
        aggregate[name] = {
            "all_exact": all(item["exact_equal"] for item in values),
            "max_sign_xor": max(item["sign_xor"] for item in values),
            "min_cosine": min(item["cosine"] for item in values),
            "max_abs_error": max(item["max_abs_error"] for item in values),
        }
    return {
        "experiment_id": EXPERIMENT_ID,
        "stage": STAGE,
        "status": "identity_gate_passed" if all(row["audit"]["identity_gate_pass"] for row in rows) else "identity_gate_failed",
        "completed": len(rows),
        "selected_sha256": selection["selected_sha256"],
        "protocol_hash": protocol,
        "gate_limits": GATE_LIMITS,
        "aggregate": aggregate,
        "rows_passed": sum(int(row["audit"]["identity_gate_pass"]) for row in rows),
        "all_identity_gates_pass": all(row["audit"]["identity_gate_pass"] for row in rows),
        "fusion_implemented": False,
        "training_performed": False,
        "gt_access": False,
        "valid_access": False,
        "sealed_test_access": False,
    }


def self_test() -> None:
    names = [f"v{index:02d}" for index in range(14)]
    metadata = {
        name: {"expressions": {"0": {"anno_id": [f"a{index}"]}}}
        for index, name in enumerate(names)
    }
    router = {"train_keys": ["v00_0"], "validation_keys": ["v01_0"]}
    calibration = {"calibration": [{"video_id": "v02"}]}
    exp039 = {"selected": [{"video_id": "v03"}]}
    exp043 = {"selected": [{"video_id": "v04"}]}
    exp044 = {"selected": [{"video_id": "v05"}]}
    prior = {"selected": [{"video_id": "v06"}]}
    pre = {f"v{index:02d}" for index in range(7)}
    exp049 = {
        "train": [{"video_id": "v07"}],
        "validation": [{"video_id": "v08"}],
        "prospective": [{"video_id": "v09"}],
        "excluded": {
            "total_unique_count": len(pre),
            "video_ids_sha256": sha256_json(sorted(pre)),
        },
    }
    manifest = build_selection_manifest(
        metadata,
        router,
        calibration,
        exp039,
        exp043,
        exp044,
        prior,
        exp049,
        3,
        "test",
    )
    assert len(manifest["selected"]) == 3
    assert manifest["excluded"]["total_unique_count"] == 10
    left = [torch.tensor([[-1.0, 2.0]]), torch.tensor([[3.0, -4.0]])]
    same = sequence_metrics(left, [value.clone() for value in left])
    assert same["exact_equal"] and same["sign_xor"] == 0.0
    assert abs(same["cosine"] - 1.0) < 1e-12
    changed = sequence_metrics(left, [left[0], torch.tensor([[3.0, 4.0]])])
    assert changed["sign_xor"] == 0.25 and not changed["exact_equal"]
    print(
        json.dumps(
            {
                "state": "self_test_passed",
                "selected": manifest["selected"],
                "gate_limits": GATE_LIMITS,
            },
            indent=2,
        )
    )


def run(args: argparse.Namespace) -> None:
    required = (
        args.model_path,
        args.data_root,
        args.router_split,
        args.calibration_manifest,
        args.exp039_selection,
        args.exp043_selection,
        args.exp044_selection,
        args.exp049_prior_exclusion,
        args.exp049_split,
        args.output,
    )
    if not all(required):
        raise ValueError("all model/data/manifest/output paths are required")
    if args.samples != IDENTITY_SAMPLES:
        raise ValueError(f"the registered identity cohort has {IDENTITY_SAMPLES} videos")
    if args.limit is not None and not 1 <= args.limit <= args.samples:
        raise ValueError("limit must be between one and samples")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output = Path(args.output)
    rows_dir = output / "rows"
    output.mkdir(parents=True, exist_ok=True)
    rows_dir.mkdir(exist_ok=True)
    data_root = Path(args.data_root)
    split_root = data_root / "video_datas" / "mevis" / "train"
    metadata = json.loads((split_root / "meta_expressions.json").read_text())["videos"]
    selection = build_selection_manifest(
        metadata,
        json.loads(Path(args.router_split).read_text()),
        json.loads(Path(args.calibration_manifest).read_text()),
        json.loads(Path(args.exp039_selection).read_text()),
        json.loads(Path(args.exp043_selection).read_text()),
        json.loads(Path(args.exp044_selection).read_text()),
        json.loads(Path(args.exp049_prior_exclusion).read_text()),
        json.loads(Path(args.exp049_split).read_text()),
        args.samples,
        args.selection_salt,
    )
    selection_path = output / "selection_manifest.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise RuntimeError("existing selection manifest differs from locked protocol")
    atomic_json(selection_path, selection)
    protocol = protocol_hash(selection)
    atomic_json(
        output / "status.json",
        {
            "state": "running",
            "phase": "selection_locked_before_model_load_no_gt",
            "selected_sha256": selection["selected_sha256"],
            "protocol_hash": protocol,
            "gt_access": False,
        },
    )

    targets = selection["selected"][: args.limit] if args.limit else selection["selected"]
    completed_rows: list[dict[str, Any]] = []
    pending: list[tuple[int, dict[str, str], Path]] = []
    for number, item in enumerate(targets, 1):
        row_path = rows_dir / f"{item['key']}.json"
        if row_path.exists():
            row = json.loads(row_path.read_text())
            if (
                row.get("selected_sha256") != selection["selected_sha256"]
                or row.get("protocol_hash") != protocol
                or row.get("gt_access") is not False
            ):
                raise RuntimeError(f"invalid resume row: {row_path}")
            completed_rows.append(row)
            print(f"EXP050_RESUME [{number}/{len(targets)}] {item['key']}", flush=True)
        else:
            pending.append((number, item, row_path))

    if pending:
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval().to("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model.preparing_for_generation(tokenizer=tokenizer)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        projection = model.text_hidden_fcs
        projection_dtype = next(projection.parameters()).dtype
        grounding = model.grounding_encoder
        predictor = grounding.sam2_model

        for number, item, row_path in pending:
            event = build_event(data_root, item, metadata)
            states, _, _, _ = extract_states(
                model, tokenizer, event, CONDITIONING_FRAMES, (28,)
            )
            native_embedding = projection(
                states.float().to("cuda:0", dtype=projection_dtype)
            ).reshape(-1)
            images, output_size = preprocess_video(
                grounding,
                model.extra_image_processor,
                model.torch_dtype,
                event["frame_paths"],
            )
            images = images.cpu()
            backbone_cache: dict[int, Any] = {}
            torch.cuda.empty_cache()

            native = infer_native_reference(
                grounding, predictor, images, backbone_cache, native_embedding
            )
            manual, manual_decode = infer_manual_direct_memory(
                predictor, images, backbone_cache, native_embedding
            )
            audit = audit_identity(native, manual, manual_decode)
            row = {
                "experiment_id": EXPERIMENT_ID,
                "stage": STAGE,
                "key": event["key"],
                "video_id": event["video_id"],
                "exp_id": event["exp_id"],
                "frame_count": len(images),
                "output_size_wh": list(output_size),
                "selected_sha256": selection["selected_sha256"],
                "protocol_hash": protocol,
                "native_diagnostics": native["diagnostics"],
                "manual_diagnostics": manual["diagnostics"],
                "audit": audit,
                "fusion_implemented": False,
                "training_performed": False,
                "gt_access": False,
                "valid_access": False,
                "sealed_test_access": False,
            }
            atomic_json(row_path, row)
            completed_rows.append(row)
            print(
                f"EXP050_IDENTITY [{number}/{len(targets)}] {event['key']} "
                f"pass={audit['identity_gate_pass']} "
                f"xor={audit['comparisons']['logits']['sign_xor']:.8f} "
                f"cos={audit['comparisons']['logits']['cosine']:.9f} "
                f"maxerr={audit['comparisons']['logits']['max_abs_error']:.8f}",
                flush=True,
            )
            del event, states, native_embedding, images, backbone_cache
            del native, manual, manual_decode, audit, row
            gc.collect()
            torch.cuda.empty_cache()

        del model, tokenizer, projection, grounding, predictor
        gc.collect()
        torch.cuda.empty_cache()

    completed_rows.sort(key=lambda row: row["key"])
    if args.limit is not None and args.limit < args.samples:
        atomic_json(
            output / "status.json",
            {
                "state": "complete",
                "phase": "stage0_smoke_complete_no_gt",
                "completed": len(completed_rows),
                "registered_samples": args.samples,
                "selected_sha256": selection["selected_sha256"],
                "protocol_hash": protocol,
                "all_completed_rows_pass": all(
                    row["audit"]["identity_gate_pass"] for row in completed_rows
                ),
                "fusion_implemented": False,
                "training_performed": False,
                "gt_access": False,
                "valid_access": False,
                "sealed_test_access": False,
            },
        )
        return

    if len(completed_rows) != args.samples:
        raise RuntimeError("not every registered identity row completed")
    summary = summarize(completed_rows, selection, protocol)
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "status.json",
        {
            "state": "complete",
            "phase": summary["status"],
            "completed": len(completed_rows),
            "selected_sha256": selection["selected_sha256"],
            "protocol_hash": protocol,
            "all_identity_gates_pass": summary["all_identity_gates_pass"],
            "stage1_authorized": summary["all_identity_gates_pass"],
            "fusion_implemented": False,
            "training_performed": False,
            "gt_access": False,
            "valid_access": False,
            "sealed_test_access": False,
        },
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
