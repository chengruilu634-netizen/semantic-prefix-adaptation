#!/usr/bin/env python3
"""Frozen prefix-only fusion, five direct SAM2 memory writes, native suffix."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from work import evaluate_recurrent_one_shot_fused_memory as base
from work.evaluate_recurrent_direct_memory_identity import build_selection_manifest as old_selection
from work.evaluate_tgif_disjoint_confirmation import select_items, video_id_from_key

EXPERIMENT = "exp_052_prefix5_direct_memory_fusion"
SAMPLES = 8
PREFIX = 5
SALT = "exp052_prefix5_direct_memory_fusion_v1"
SEED = 20260902
ACTIONS = ("manual_h28_direct_memory", "prefix_output_only", "prefix5_fused_memory")
SOURCE_PATHS = {
    "router": "exp_037_tgif_seg_layer_router/pilot8/router_split.json",
    "calibration": "multilayer_seg/spatial_map_affine_safety_v1/manifest.json",
    "exp039": "exp_039_tgif_disjoint_confirmation/selection_manifest.json",
    "exp043": "exp_043_anchored_lowrank_disjoint_trajectory/selection_manifest.json",
    "exp044": "exp_044_mask_logit_state_rebuild/pilot8/selection_manifest.json",
    "exp049_prior": "exp_049_two_stage_candidate_then_admission/prior_exclusion_manifest.json",
    "exp049_split": "exp_049_two_stage_candidate_then_admission/pilot_cache/split_manifest.json",
    "exp050": "exp_050_recurrent_direct_memory_mask_fusion/stage0_identity3/selection_manifest.json",
}
GATE_NAMES = {
    "prefix_h28_high_exact", "prefix_h28_low_exact", "prefix_h28_pointer_exact",
    "native_manual_high_exact", "native_manual_low_exact", "native_manual_memory_exact",
    "native_manual_pointer_exact", "fused_pointer_preserved", "all_five_logits_committed",
    "all_states_valid", "prefix_output_suffix_exact", "all_fused_logits_finite",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def immutable_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"locked artifact changed: {path}")
    else:
        base.atomic_json(path, value)


def selection_manifest(root, metadata):
    derived = root / "work/derived"
    source = {name: read_json(derived / path) for name, path in SOURCE_PATHS.items()}
    old = old_selection(metadata, source["router"], source["calibration"], source["exp039"],
                        source["exp043"], source["exp044"], source["exp049_prior"],
                        source["exp049_split"], 3, "exp050_recurrent_direct_memory_identity_v1")
    if old["selected_sha256"] != source["exp050"]["selected_sha256"]:
        raise RuntimeError("exp050 selection reconstruction failed")
    excluded = {video_id_from_key(str(k)) for group in ("train_keys", "validation_keys")
                for k in source["router"][group]}
    excluded.update(str(x["video_id"]) for x in source["calibration"]["calibration"])
    for name in ("exp039", "exp043", "exp044", "exp049_prior", "exp050"):
        excluded.update(str(x["video_id"]) for x in source[name]["selected"])
    for group in ("train", "validation", "prospective"):
        excluded.update(str(x["video_id"]) for x in source["exp049_split"][group])
    if len(excluded) != 946 or old["excluded"]["total_unique_count"] != 943:
        raise RuntimeError("expected exact 943 + 3 historical video exclusions")
    eligible = {k: v for k, v in metadata.items() if len(v["frames"]) > PREFIX}
    selected = select_items(eligible, excluded, SAMPLES, SALT)
    return {
        "experiment_id": EXPERIMENT, "selection_salt": SALT,
        "selection_rule": "ascending sha256 salt:video then salt:video:expression; one annotated expression/video; T>5",
        "selected": selected, "selected_sha256": base.sha256_json(selected),
        "excluded_video_ids": sorted(excluded), "excluded_count": len(excluded),
        "excluded_sha256": base.sha256_json(sorted(excluded)),
        "source_sha256": {name: base.file_sha256(derived / path) for name, path in SOURCE_PATHS.items()},
        "gt_access": False, "valid_access": False, "sealed_test_access": False,
    }


def protocol_manifest(selection, parent_path, candidate_path):
    candidate_sha = base.file_sha256(candidate_path)
    if candidate_sha != base.EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError("frozen exp049 checkpoint does not match")
    return {
        "experiment_id": EXPERIMENT, "revision": "v1", "seed": SEED, "samples": SAMPLES,
        "selection_sha256": selection["selected_sha256"], "layers": list(base.LAYERS),
        "candidate_checkpoint_sha256": candidate_sha,
        "query_preprocessor_checkpoint_sha256": base.file_sha256(parent_path),
        "conditioning_frames": PREFIX, "auxiliary_trajectory_frames": PREFIX,
        "candidate_temporal_pooling_frames": PREFIX, "fusion_writes": list(range(PREFIX)),
        "planner": "frozen exp049 epoch12; first-five 128x128 logits only; no suffix evidence",
        "state": "float32 fused 256x256 logits; direct memory encoder is_mask_from_pts=True",
        "pointer_source": "native h28 on each of five conditioning frames",
        "suffix": "unaltered native propagate_in_video; no language fusion after frame index 4",
        "precision": "existing fp16 candidate cache and query roundtrip; fused logits remain float32",
        "actions": ["native_h28", *ACTIONS], "training_performed": False,
        "safety_gate": {"mean_full_min": 0, "mean_suffix_min": 0,
                        "suffix_nonnegative_count_min": 6, "full_decline_le_minus5_max": 0},
        "utility_gate": {"mean_full_min": 0.5, "full_improved_count_min": 5},
        "interpretation": "diagnostic pilot only; checkpoint was trained on full trajectories, not prefix-only",
        "gt_boundary": "all eight candidates and invariants frozen before selected train GT",
        "valid_access": False, "sealed_test_access": False,
    }


def check_prefix(high, low):
    if high.ndim != 4 or low.ndim != 4 or high.shape[:2] != (4, PREFIX) or low.shape[:2] != (4, PREFIX):
        raise RuntimeError("planner requires exactly four layers and five frames; suffix is forbidden")
    if not torch.isfinite(high).all() or not torch.isfinite(low).all():
        raise RuntimeError("nonfinite prefix candidate")


@torch.inference_mode()
def plan_prefix(candidate, query, high, low, device):
    check_prefix(high, low)
    resized = base.resize_layer_logits(high, (128, 128), device)
    _, plan = candidate(query.to(device), resized.float().to(device))
    gates = plan["spatial_gates"].float().cpu()
    weights = plan["layer_weights"].float().cpu()
    fused_high = base.apply_candidate_at_resolution(high, gates, weights, device)
    fused_low = base.apply_candidate_at_resolution(low, gates, weights, device)
    return fused_high, fused_low, {
        "candidate_frames": PREFIX, "temporal_pooling_frames": PREFIX,
        "layer_weights": weights.tolist(), "gate_mean_abs": float(gates.abs().mean()),
        "per_frame_state_change": [base.sequence_metrics([low[-1, i]], [fused_low[i]]) for i in range(PREFIX)],
    }


def snapshot_conditioning(state):
    outputs = state["output_dict"]["cond_frame_outputs"]
    return {name: [outputs[i][name].float().cpu() for i in range(PREFIX)]
            for name in ("pred_masks", "obj_ptr", "maskmem_features")}


@torch.inference_mode()
def native_trajectory(grounding, images, embedding):
    predictor = grounding.sam2_model
    state = predictor.init_state(images)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        high = grounding.language_embd_inference(state, [[embedding] for _ in range(PREFIX)])
    high = high[:, 0].half().cpu()
    cond = snapshot_conditioning(state)
    low = torch.stack([state["output_dict"]["cond_frame_outputs" if i < PREFIX else "non_cond_frame_outputs"][i]["pred_masks"][0, 0].float().cpu()
                       for i in range(len(images))]).half()
    del state
    gc.collect()
    torch.cuda.empty_cache()
    return high, low, cond


@torch.inference_mode()
def direct_trajectory(predictor, images, embedding, fused_low=None):
    if fused_low is not None and fused_low.shape != (PREFIX, 256, 256):
        raise RuntimeError("five 256x256 fused states are required")
    state = predictor.init_state(images)
    obj = base.register_object(state, 100)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(PREFIX):
            current, _ = predictor._run_single_frame_inference(
                inference_state=state, output_dict=state["output_dict_per_obj"][obj],
                frame_idx=i, batch_size=1, is_init_cond_frame=True,
                point_inputs=None, mask_inputs=None, reverse=False, run_mem_encoder=False,
                prev_sam_mask_logits=None, language_embd=embedding.reshape(1, 1, -1))
            selected = current["pred_masks"] if fused_low is None else fused_low[i][None, None]
            logits = selected.to(state["storage_device"], dtype=torch.float32)
            high = F.interpolate(logits.to(state["device"]), size=(predictor.image_size, predictor.image_size),
                                 mode="bilinear", align_corners=False)
            if predictor.non_overlap_masks_for_mem_enc:
                high = predictor._apply_non_overlapping_constraints(high)
            memory, positions = predictor._run_memory_encoder(
                inference_state=state, frame_idx=i, batch_size=1, high_res_masks=high, is_mask_from_pts=True)
            committed = {"pred_masks": logits, "obj_ptr": current["obj_ptr"].to(state["device"], dtype=torch.float32),
                         "maskmem_features": memory, "maskmem_pos_enc": positions}
            state["output_dict"]["cond_frame_outputs"][i] = committed
            predictor._add_output_per_object(state, i, committed, "cond_frame_outputs")
            state["consolidated_frame_inds"]["cond_frame_outputs"].add(i)
        yielded, outputs = [], []
        for i, _, high in predictor.propagate_in_video(state):
            yielded.append(int(i))
            outputs.append(high[0, 0].half().cpu())
    diag = base.state_contract_diagnostics(state, obj, yielded)
    cond = snapshot_conditioning(state)
    low = torch.stack([state["output_dict"]["cond_frame_outputs" if i < PREFIX else "non_cond_frame_outputs"][i]["pred_masks"][0, 0].float().cpu()
                       for i in range(len(images))]).half()
    del state
    gc.collect()
    torch.cuda.empty_cache()
    return torch.stack(outputs), low, cond, diag


def validate_row(row, item, selection, protocol):
    expected = {"experiment_id": EXPERIMENT, **item, "selection_sha256": selection["selected_sha256"],
                "protocol_hash": base.sha256_json(protocol), "fusion_writes": list(range(PREFIX)),
                "candidate_frames": PREFIX, "pointer_source": "native_h28"}
    if any(row.get(k) != v for k, v in expected.items()):
        raise RuntimeError("frozen row metadata mismatch")
    for flag in ("gt_access", "valid_access", "sealed_test_access", "training_performed", "dense_mask_redecode"):
        if row.get(flag) is not False:
            raise RuntimeError(f"invalid flag {flag}")
    gates = row.get("invariants", {})
    if set(gates) != GATE_NAMES or any(v is not True for v in gates.values()):
        raise RuntimeError(f"invariant failure: {gates}")
    if row["frame_count"] <= PREFIX or not row.get("annotation_ids"):
        raise RuntimeError("invalid video or annotation metadata")
    width, height = row["output_size_wh"]
    shape = [row["frame_count"], height, width]
    if min(shape) <= 0 or set(row["masks"]) != {"native_h28", *ACTIONS}:
        raise RuntimeError("invalid masks/dimensions")
    for payload in row["masks"].values():
        value = payload["packed"]
        if payload["shape"] != shape or value.dtype != torch.uint8 or value.ndim != 1 or value.numel() != (int(np.prod(shape)) + 7) // 8:
            raise RuntimeError("invalid packed mask")
    if not base.packed_masks_equal(row["masks"]["native_h28"], row["masks"][ACTIONS[0]]):
        raise RuntimeError("native/manual stored binary masks differ")


@torch.inference_mode()
def generate_row(item, model, tokenizer, candidate, preprocessor, root, metadata, selection, protocol):
    device = torch.device("cuda:0")
    row_seed = (SEED + int(hashlib.sha256(item["key"].encode()).hexdigest()[:8], 16)) % (2**32)
    torch.manual_seed(row_seed)
    np.random.seed(row_seed)
    event = base.build_event(root / "data", item, metadata)
    states, _, _, _ = base.extract_states(model, tokenizer, event, PREFIX, base.ALL_LAYERS)
    aligned = base.TgifSegRouter.rms_align(states.float().to(device))
    projection = model.text_hidden_fcs
    embeddings = projection(aligned.to(next(projection.parameters()).dtype)).reshape(len(base.ALL_LAYERS), -1)
    query = base.transform_query(base.query_embedding(model, tokenizer, event["prompt"])[None], preprocessor)[0].half().float()
    grounding = model.grounding_encoder
    images, size = base.preprocess_video(grounding, model.extra_image_processor, model.torch_dtype, event["frame_paths"])
    images = images.cpu()
    highs, lows, prefix_h28 = [], [], None
    for layer in base.LAYERS:
        # The native wrapper receives ONLY five images. Auxiliary suffixes never run.
        high, low, cond = native_trajectory(grounding, images[:PREFIX], embeddings[layer - 1])
        highs.append(high)
        lows.append(low)
        if layer == 28:
            prefix_h28 = cond
        print(f"EXP052_PREFIX {item['key']} layer={layer} frames={len(high)}", flush=True)
    layer_high, layer_low = torch.stack(highs), torch.stack(lows)
    fused_high, fused_low, candidate_diag = plan_prefix(candidate, query, layer_high, layer_low, device)
    native_high, native_low, native_cond = native_trajectory(grounding, images, embeddings[-1])
    print(f"EXP052_NATIVE {item['key']} frames={len(images)}", flush=True)
    manual_high, manual_low, manual_cond, manual_diag = direct_trajectory(grounding.sam2_model, images, embeddings[-1])
    fused_trajectory, _, fused_cond, fused_diag = direct_trajectory(grounding.sam2_model, images, embeddings[-1], fused_low)
    eq = lambda x, y: base.sequence_metrics(x, y)["exact_equal"]
    output_only = native_high.float().clone()
    output_only[:PREFIX] = fused_high
    memory_changes = [base.sequence_metrics([manual_cond["maskmem_features"][i]], [fused_cond["maskmem_features"][i]]) for i in range(PREFIX)]
    gates = {
        "prefix_h28_high_exact": torch.equal(layer_high[-1], native_high[:PREFIX]),
        "prefix_h28_low_exact": torch.equal(layer_low[-1], native_low[:PREFIX]),
        "prefix_h28_pointer_exact": eq(prefix_h28["obj_ptr"], native_cond["obj_ptr"]),
        "native_manual_high_exact": torch.equal(native_high, manual_high),
        "native_manual_low_exact": torch.equal(native_low, manual_low),
        "native_manual_memory_exact": eq(native_cond["maskmem_features"], manual_cond["maskmem_features"]),
        "native_manual_pointer_exact": eq(native_cond["obj_ptr"], manual_cond["obj_ptr"]),
        "fused_pointer_preserved": eq(native_cond["obj_ptr"], fused_cond["obj_ptr"]),
        "all_five_logits_committed": eq([fused_low[i][None, None] for i in range(PREFIX)], fused_cond["pred_masks"]),
        "all_states_valid": all(d[k] for d in (manual_diag, fused_diag) for k in ("state_contract_ok", "all_tensors_finite", "all_memory_present")),
        "prefix_output_suffix_exact": torch.equal(output_only[PREFIX:], native_high[PREFIX:].float()),
        "all_fused_logits_finite": bool(torch.isfinite(fused_low).all() and torch.isfinite(fused_high).all()),
    }
    if not all(gates.values()):
        raise RuntimeError(f"no-GT contract failed: {item['key']} {gates}")
    packed = {}
    for name, logits in zip(("native_h28", *ACTIONS), (native_high, manual_high, output_only, fused_trajectory)):
        packed[name] = base.pack_binary_mask(base.binary_at_output(logits, size, device))
    row = {
        "experiment_id": EXPERIMENT, **item, "row_seed": row_seed,
        "selection_sha256": selection["selected_sha256"], "protocol_hash": base.sha256_json(protocol),
        "annotation_ids": event["annotation_ids"], "frame_count": len(images), "output_size_wh": list(size),
        "candidate_frames": PREFIX, "fusion_writes": list(range(PREFIX)), "pointer_source": "native_h28",
        "candidate_diagnostics": candidate_diag, "memory_changes": memory_changes,
        "state_diagnostics": {"manual": manual_diag, "fused": fused_diag}, "invariants": gates,
        "masks": packed, "gt_access": False, "valid_access": False, "sealed_test_access": False,
        "training_performed": False, "dense_mask_redecode": False,
    }
    validate_row(row, item, selection, protocol)
    return row


def validate_cohort(rows, selection, protocol):
    selected = {x["key"]: x for x in selection["selected"]}
    if len(rows) != SAMPLES or len(selected) != SAMPLES or {r["key"] for r in rows} != set(selected):
        raise RuntimeError("all eight locked videos must be frozen before GT")
    for row in rows:
        validate_row(row, selected[row["key"]], selection, protocol)
    if not any(m["max_abs_error"] > 0 for row in rows for m in row["memory_changes"]):
        raise RuntimeError("all memory interventions were trivial; GT stays closed")


def score(rows, root, output, selection, protocol):
    validate_cohort(rows, selection, protocol)
    # This is the only segmentation GT read in this program.
    masks = read_json(root / "data/video_datas/mevis/train/mask_dict.json")
    scored_rows = []
    for row in rows:
        print(f"EXP052_SCORE {row['key']}", flush=True)
        width, height = row["output_size_wh"]
        truth = np.stack([base.decode_frame_mask(masks, row["annotation_ids"], i, (height, width)) for i in range(row["frame_count"])]).astype(bool)
        native = base.unpack_binary_mask(row["masks"]["native_h28"])
        metrics = base.segment_metrics(truth, native)
        cond = base.segment_metrics(truth[:PREFIX], native[:PREFIX])
        suffix = base.segment_metrics(truth[PREFIX:], native[PREFIX:])
        errors = (*base.error_rates(truth[:PREFIX], native[:PREFIX]), *base.error_rates(truth[PREFIX:], native[PREFIX:]))
        frame_native = [base.segment_metrics(truth[i:i+1], native[i:i+1])["J&F"] for i in range(len(truth))]
        actions = {}
        for action in ACTIONS:
            value = base.unpack_binary_mask(row["masks"][action])
            result = base.score_action(truth, native, value, metrics, cond, suffix, errors)
            result["per_frame_delta_jf_points"] = [100 * (base.segment_metrics(truth[i:i+1], value[i:i+1])["J&F"] - frame_native[i]) for i in range(len(truth))]
            divergence = np.flatnonzero(np.any(value[PREFIX:] != native[PREFIX:], axis=(1, 2)))
            result["first_suffix_divergence_index"] = int(divergence[0]) + PREFIX if len(divergence) else None
            actions[action] = result
        scored = {"key": row["key"], "baseline_h28": metrics, "actions": actions,
                  "protocol_hash": base.sha256_json(protocol), "gt_access_after_freeze": True}
        base.atomic_json(output / "scored_rows" / f"{row['key']}.json", scored)
        scored_rows.append(scored)
    summaries = {}
    for action in ACTIONS:
        full = [r["actions"][action]["delta_J&F_points"] for r in scored_rows]
        suffix = [r["actions"][action]["suffix_delta_J&F_points"] for r in scored_rows]
        summaries[action] = {
            "delta_jf_points": base.paired_summary(full), "suffix_delta_jf_points": base.paired_summary(suffix),
            "conditioning_delta_jf_points": base.paired_summary([r["actions"][action]["conditioning_delta_J&F_points"] for r in scored_rows]),
            "mean_jf": float(np.mean([r["actions"][action]["metrics"]["J&F"] for r in scored_rows])),
            "full_improved_count": sum(x > 1e-9 for x in full), "suffix_nonnegative_count": sum(x >= 0 for x in suffix),
            "decline_le_minus5_count": sum(x <= -5 for x in full), "leave_best_out_mean": (sum(full) - max(full)) / (SAMPLES - 1),
        }
    primary = summaries[ACTIONS[-1]]
    safety = {"mean_full_nonnegative": primary["delta_jf_points"]["mean"] >= 0,
              "mean_suffix_nonnegative": primary["suffix_delta_jf_points"]["mean"] >= 0,
              "six_suffix_nonnegative": primary["suffix_nonnegative_count"] >= 6,
              "no_full_decline_le_minus5": primary["decline_le_minus5_count"] == 0}
    utility = {"mean_full_ge_half_point": primary["delta_jf_points"]["mean"] >= .5,
               "five_full_improved": primary["full_improved_count"] >= 5}
    return {"experiment_id": EXPERIMENT, "completed": len(rows), "protocol_hash": base.sha256_json(protocol),
            "selection_sha256": selection["selected_sha256"], "native_mean_jf": float(np.mean([r["baseline_h28"]["J&F"] for r in scored_rows])),
            "actions": summaries, "safety_gates": safety, "utility_gates": utility,
            "confirmation_supported": all(safety.values()) and all(utility.values()),
            "training_performed": False, "gt_access_after_all_candidate_freeze": True,
            "valid_access": False, "sealed_test_access": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--model", default="pretrained/Sa2VA-InternVL3-2B")
    parser.add_argument("--mode", choices=("lock", "freeze", "score"), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and (args.mode != "freeze" or not 1 <= args.limit <= SAMPLES):
        raise ValueError("limit is only allowed for candidate freeze, within [1,8]")
    root = args.root
    output = root / "work/derived" / EXPERIMENT / "pilot8"
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen_trajectories").mkdir(exist_ok=True)
    (output / "scored_rows").mkdir(exist_ok=True)
    metadata = read_json(root / "data/video_datas/mevis/train/meta_expressions.json")["videos"]
    selection = selection_manifest(root, metadata)
    parent_path = root / "work/derived/exp_037_tgif_seg_layer_router/pilot8/checkpoint_best.pt"
    candidate_path = root / "work/derived/exp_049_two_stage_candidate_then_admission/two_stage/candidate_best.pt"
    protocol = protocol_manifest(selection, parent_path, candidate_path)
    immutable_json(output / "selection_manifest.json", selection)
    immutable_json(output / "protocol.json", protocol)
    if args.mode == "lock":
        print(json.dumps({"selected": selection["selected"], "excluded": selection["excluded_count"], "protocol_hash": base.sha256_json(protocol)}), flush=True)
        return
    targets = selection["selected"][:args.limit] if args.limit else selection["selected"]
    pending = []
    for item in targets:
        path = output / "frozen_trajectories" / f"{item['key']}.pt"
        if path.exists():
            validate_row(torch.load(path, map_location="cpu", weights_only=False), item, selection, protocol)
        else:
            pending.append((item, path))
    if pending:
        if args.mode != "freeze" or (output / "gt_access_receipt.json").exists():
            raise RuntimeError("missing candidates; score mode/GT receipt prohibits candidate generation")
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        if tuple(parent["layer_indices"]) != base.ALL_LAYERS or parent["best_epoch"] != 3:
            raise RuntimeError("query preprocessor contract mismatch")
        preprocessor = base.preprocessor_from_checkpoint(parent)
        checkpoint = torch.load(candidate_path, map_location="cpu", weights_only=False)
        if checkpoint["best_epoch"] != 12:
            raise RuntimeError("candidate epoch mismatch")
        torch.manual_seed(SEED)
        candidate = base.CandidateFusion(checkpoint["query_dim"]).cuda().eval()
        candidate.load_state_dict(checkpoint["state_dict"])
        model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                         use_flash_attn=True, trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model.preparing_for_generation(tokenizer=tokenizer)
        for module in (model, candidate):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        for item, path in pending:
            print(f"EXP052_BEGIN {item['key']}", flush=True)
            row = generate_row(item, model, tokenizer, candidate, preprocessor, root, metadata, selection, protocol)
            base.atomic_torch(path, row)
            print(f"EXP052_FREEZE {item['key']} frames={row['frame_count']} all_invariants_pass=True", flush=True)
            del row
            gc.collect()
            torch.cuda.empty_cache()
        del candidate, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    paths = [output / "frozen_trajectories" / f"{item['key']}.pt" for item in selection["selected"]]
    completed = sum(p.exists() for p in paths)
    base.atomic_json(output / "status_candidates.json", {"completed": completed, "registered": SAMPLES, "gt_access": False,
                                                       "protocol_hash": base.sha256_json(protocol)})
    if completed != SAMPLES:
        return
    rows = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    validate_cohort(rows, selection, protocol)
    freeze = {"protocol_hash": base.sha256_json(protocol), "files": [{"key": p.stem, "sha256": base.file_sha256(p)} for p in paths],
              "all_invariants_pass": True, "gt_access": False}
    freeze_path = output / "freeze_manifest.json"
    if args.mode == "freeze":
        immutable_json(freeze_path, freeze)
        return
    if not freeze_path.exists() or read_json(freeze_path) != freeze:
        raise RuntimeError("missing or changed no-GT freeze manifest")
    if (output / "summary.json").exists():
        if read_json(output / "summary.json")["protocol_hash"] != base.sha256_json(protocol):
            raise RuntimeError("summary protocol mismatch")
        print("EXP052_ALREADY_SCORED", flush=True)
        return
    immutable_json(output / "gt_access_receipt.json", {"protocol_hash": base.sha256_json(protocol),
                   "freeze_manifest_sha256": base.file_sha256(freeze_path), "selected_keys": [r["key"] for r in rows],
                   "valid_access": False, "sealed_test_access": False})
    base.atomic_json(output / "summary.json", score(rows, root, output, selection, protocol))
    print("EXP052_SCORED", flush=True)


if __name__ == "__main__":
    main()
