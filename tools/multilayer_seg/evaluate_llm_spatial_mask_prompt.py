#!/usr/bin/env python3
"""Evaluate the real LLM spatial-map -> SAM2 dense-mask-prompt path.

The official final-layer ``[SEG]`` embedding remains SAM2's sparse language
prompt.  Each selected LLM layer is tested only as an additional spatial cue:

1. the configured anchor layer supplies the projected sparse language prompt;
2. the selected layer's cosine similarity to its image tokens becomes a soft
   dense mask prompt.

The prompts are injected on the first five frames and SAM2 propagates over the
complete video.  Crucially, this bypasses ``add_new_mask`` and locally disables
``use_mask_input_as_output_without_sam`` so the soft map actually passes through
SAM2's prompt encoder and mask decoder together with the language embedding.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from tools.multilayer_seg.run_dynamic_k_oracle import exact_metrics
from tools.multilayer_seg.train_eval_fusion_standard import (
    cache_video_backbones,
    ground_truth_video,
    infer_video_embedding,
    preprocess_video,
)
from tools.multilayer_seg.train_fusion_official_video_spsa import collect_video_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--map-cache", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--layers", type=int, nargs="+", default=(28, 4, 8, 12, 16, 20, 24))
    parser.add_argument("--language-anchor-layer", type=int, default=28)
    parser.add_argument("--quantiles", type=float, nargs="+", default=(0.75,))
    parser.add_argument("--logit-scale", type=float, default=2.0)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


def register_object(state: dict[str, Any], obj_id: int) -> int:
    """Mirror SAM2's private module-level _obj_id_to_idx helper."""
    existing = state["obj_id_to_idx"].get(obj_id)
    if existing is not None:
        return existing
    if state["tracking_has_started"]:
        raise RuntimeError("cannot register an object after propagation starts")
    obj_idx = len(state["obj_id_to_idx"])
    state["obj_id_to_idx"][obj_id] = obj_idx
    state["obj_idx_to_id"][obj_idx] = obj_id
    state["obj_ids"] = list(state["obj_id_to_idx"])
    state["point_inputs_per_obj"][obj_idx] = {}
    state["mask_inputs_per_obj"][obj_idx] = {}
    state["output_dict_per_obj"][obj_idx] = {
        "cond_frame_outputs": {}, "non_cond_frame_outputs": {},
    }
    state["temp_output_dict_per_obj"][obj_idx] = {
        "cond_frame_outputs": {}, "non_cond_frame_outputs": {},
    }
    return obj_idx


def calibrated_mask_logits(spatial_map: torch.Tensor, quantile: float, scale: float) -> torch.Tensor:
    """Turn an uncalibrated cosine map into a soft SAM mask-logit prompt."""
    flat = spatial_map.float().flatten()
    mean = flat.mean()
    std = flat.std(unbiased=False).clamp_min(1e-5)
    standardized = (spatial_map.float() - mean) / std
    foreground_boundary = torch.quantile(standardized.flatten(), quantile)
    return (standardized - foreground_boundary) * scale


@torch.inference_mode()
def add_joint_prompt(
    predictor: Any,
    state: dict[str, Any],
    frame_idx: int,
    obj_id: int,
    language_embedding: torch.Tensor,
    mask_logits: torch.Tensor,
) -> None:
    """Run SAM2 decoder with language and dense mask prompts simultaneously."""
    obj_idx = register_object(state, obj_id)
    obj_output = state["output_dict_per_obj"][obj_idx]
    obj_temp = state["temp_output_dict_per_obj"][obj_idx]
    is_init = frame_idx not in state["frames_already_tracked"]
    storage_key = "cond_frame_outputs" if (
        is_init or predictor.add_all_frames_to_correct_as_cond
    ) else "non_cond_frame_outputs"
    language = language_embedding.reshape(1, 1, -1)
    dense_mask = mask_logits.reshape(1, 1, *mask_logits.shape[-2:]).to(
        device=language.device, dtype=torch.float32
    )
    current_out, _ = predictor._run_single_frame_inference(
        inference_state=state,
        output_dict=obj_output,
        frame_idx=frame_idx,
        batch_size=1,
        is_init_cond_frame=is_init,
        point_inputs=None,
        mask_inputs=dense_mask,
        reverse=False,
        run_mem_encoder=False,
        prev_sam_mask_logits=None,
        language_embd=language,
    )
    obj_temp[storage_key][frame_idx] = current_out
    # Record the frame as a true mask-input frame for SAM2 state invariants/debugging.
    state["mask_inputs_per_obj"][obj_idx][frame_idx] = dense_mask.cpu()


@torch.inference_mode()
def infer_joint_prompt(
    grounding: Any,
    images: torch.Tensor,
    backbone_cache: dict[int, Any],
    embedding: torch.Tensor,
    maps: torch.Tensor,
    output_size: tuple[int, int],
    quantile: float,
    scale: float,
) -> np.ndarray:
    predictor = grounding.sam2_model
    state = predictor.init_state(images)
    state["cached_features"] = backbone_cache.copy()
    conditioning_frames = min(5, len(images), maps.shape[0])
    original_bypass = predictor.use_mask_input_as_output_without_sam
    predictor.use_mask_input_as_output_without_sam = False
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for frame_idx in range(conditioning_frames):
                prompt = calibrated_mask_logits(maps[frame_idx], quantile, scale)
                add_joint_prompt(predictor, state, frame_idx, 100, embedding, prompt)
            by_frame: dict[int, torch.Tensor] = {}
            for frame_idx, _, logits in predictor.propagate_in_video(state):
                by_frame[int(frame_idx)] = logits
            logits = torch.cat([by_frame[index] for index in range(len(images))], dim=0)
    finally:
        predictor.use_mask_input_as_output_without_sam = original_bypass
    width, height = output_size
    logits = F.interpolate(logits.float(), size=(height, width), mode="bilinear", align_corners=False)
    return (logits[:, 0].sigmoid() > 0.5).cpu().numpy()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 1 <= args.samples <= 32:
        raise ValueError("samples must be in [1,32]")
    if any(not 0.0 < value < 1.0 for value in args.quantiles):
        raise ValueError("quantiles must be in (0,1)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    atomic_json(status_path, {"state": "running", "phase": "load_caches"})

    maps_payload = torch.load(args.map_cache, map_location="cpu", weights_only=False)
    feature_payload = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    map_by_key = {
        key: maps_payload["compact_alignment_maps"][index].float()
        for index, key in enumerate(maps_payload["keys"])
    }
    projected_by_key = {
        key: feature_payload["projected_seg"][index].float()
        for index, key in enumerate(feature_payload["keys"])
    }
    projected_layer_count = int(feature_payload["projected_seg"].shape[1])
    if not 1 <= args.language_anchor_layer <= projected_layer_count:
        raise RuntimeError(
            f"language anchor layer {args.language_anchor_layer} is outside the "
            f"projected SEG cache range [1,{projected_layer_count}]"
        )
    map_layer_to_index = {
        int(layer): index for index, layer in enumerate(maps_payload["layers"])
    }
    missing = [layer for layer in args.layers if layer not in map_layer_to_index]
    if missing:
        raise RuntimeError(f"requested layers absent from map cache: {missing}")

    events = collect_video_events(Path(args.data_root), args.samples, 1, args.seed)
    for event in events:
        if event["key"] not in map_by_key or event["key"] not in projected_by_key:
            raise RuntimeError(f"cache does not contain {event['key']}")
    mask_dict = json.loads(
        (Path(args.data_root) / "video_datas" / "mevis" / "train" / "mask_dict.json").read_text()
    )

    atomic_json(status_path, {"state": "running", "phase": "model_load"})
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True,
    ).eval().to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model.preparing_for_generation(tokenizer=tokenizer)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    grounding = model.grounding_encoder
    rows: list[dict[str, Any]] = []

    for number, event in enumerate(events, 1):
        atomic_json(status_path, {
            "state": "running", "phase": "joint_prompt_evaluation",
            "completed": number - 1, "total": len(events), "key": event["key"],
        })
        images, output_size = preprocess_video(
            grounding, model.extra_image_processor, model.torch_dtype, event["frame_paths"]
        )
        width, height = output_size
        gt = ground_truth_video(
            mask_dict, event["annotation_ids"], len(event["frame_paths"]), (height, width)
        )
        backbone_cache = cache_video_backbones(grounding, images)
        projected = projected_by_key[event["key"]].to(device="cuda:0", dtype=model.torch_dtype)
        baseline_prediction = infer_video_embedding(
            grounding, images, backbone_cache, projected[27], output_size
        )
        baseline = exact_metrics(gt, baseline_prediction)
        candidates = []
        for layer in args.layers:
            layer_map = map_by_key[event["key"]][map_layer_to_index[layer]]
            for quantile in args.quantiles:
                prediction = infer_joint_prompt(
                    grounding, images, backbone_cache,
                    projected[args.language_anchor_layer - 1], layer_map,
                    output_size, quantile, args.logit_scale,
                )
                metrics = exact_metrics(gt, prediction)
                candidates.append({
                    "candidate_id": (
                        f"map_l{layer:02d}_lang_l{args.language_anchor_layer:02d}_"
                        f"q{quantile:.2f}"
                    ),
                    "map_layer": layer,
                    "language_anchor_layer": args.language_anchor_layer,
                    "quantile": quantile,
                    "logit_scale": args.logit_scale,
                    "metrics": metrics,
                    "delta_J&F": metrics["J&F"] - baseline["J&F"],
                })
                del prediction
        row = {"key": event["key"], "baseline_h28": baseline, "candidates": candidates}
        rows.append(row)
        atomic_json(output / "rows.json", {"rows": rows})
        print(
            f"JOINT_PROMPT [{number}/{len(events)}] {event['key']} "
            f"baseline={baseline['J&F']:.4f} best_delta="
            f"{max(item['delta_J&F'] for item in candidates):+.4f}", flush=True,
        )
        del images, backbone_cache, projected, baseline_prediction
        gc.collect()
        torch.cuda.empty_cache()

    deltas: dict[str, list[float]] = {}
    scores: dict[str, list[float]] = {}
    for row in rows:
        for item in row["candidates"]:
            deltas.setdefault(item["candidate_id"], []).append(item["delta_J&F"])
            scores.setdefault(item["candidate_id"], []).append(item["metrics"]["J&F"])
    summary = {
        "samples": len(rows),
        "baseline_mean_J&F": float(np.mean([row["baseline_h28"]["J&F"] for row in rows])),
        "candidates": {
            key: {
                "mean_J&F": float(np.mean(scores[key])),
                "mean_delta_J&F": float(np.mean(values)),
                "positive_rate": float(np.mean(np.asarray(values) > 0.0)),
            }
            for key, values in deltas.items()
        },
        "contract": {
            "conditioning": (
                f"fixed layer-{args.language_anchor_layer} projected SEG language prompt + "
                "selected-layer LLM image-token spatial map"
            ),
            "injection": "first 5 frames",
            "propagation": "complete video",
            "sam2_path": "prompt_encoder + mask_decoder; binary mask bypass disabled",
        },
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(status_path, {"state": "complete", "phase": "done", "samples": len(rows)})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
