#!/usr/bin/env python3
"""Evaluate affine-calibrated l28 adapter maps with exact frozen SAM2 propagation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from tools.multilayer_seg.evaluate_llm_spatial_mask_prompt import add_joint_prompt
from tools.multilayer_seg.extract_spatial_adapter_features import extract_states
from tools.multilayer_seg.run_dynamic_k_oracle import exact_metrics
from tools.multilayer_seg.train_eval_fusion_standard import (
    cache_video_backbones,
    ground_truth_video,
    preprocess_video,
    prompt_for,
)
from tools.multilayer_seg.train_spatial_map_adapters import SpatialMapAdapters


DEFAULT_CONFIGS = [
    {"layer": 28, "alpha": alpha, "beta": beta}
    for alpha in (0.125, 0.25, 0.5)
    for beta in (-1.0, 0.0, 1.0)
] + [{"layer": 28, "alpha": 1.0, "beta": 0.0}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=("calibration", "test"), required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260913)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


def config_id(config: dict[str, Any]) -> str:
    return f"l{int(config['layer']):02d}_a{float(config['alpha']):g}_b{float(config['beta']):+g}"


def protocol_hash(manifest: dict[str, Any], split: str, configs: list[dict[str, Any]]) -> str:
    payload = {
        "split_hash": manifest[f"{split}_sha256"],
        "configs": configs,
        "language_layer": 28,
        "conditioning_frames": 5,
        "propagation": "complete_video",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_event(data_root: Path, item: dict[str, str], metadata: dict[str, Any]) -> dict[str, Any]:
    video = metadata[item["video_id"]]
    expression = video["expressions"][item["exp_id"]]
    image_root = data_root / "video_datas" / "mevis" / "train" / "JPEGImages"
    return {
        **item,
        "prompt": prompt_for(expression["exp"]),
        "annotation_ids": expression.get("anno_id") or [],
        "frame_paths": [str(image_root / item["video_id"] / f"{name}.jpg") for name in video["frames"]],
    }


@torch.inference_mode()
def infer_baseline_logits(
    grounding: Any,
    images: torch.Tensor,
    backbone_cache: dict[int, Any],
    embedding: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    state = grounding.sam2_model.init_state(images)
    state["cached_features"] = backbone_cache.copy()
    language = [[embedding] for _ in range(min(5, len(images)))]
    logits = grounding.language_embd_inference(state, language)
    width, height = output_size
    return F.interpolate(logits.float(), size=(height, width), mode="bilinear", align_corners=False)[:, 0].cpu()


@torch.inference_mode()
def infer_map_logits(
    grounding: Any,
    images: torch.Tensor,
    backbone_cache: dict[int, Any],
    embedding: torch.Tensor,
    map_logits: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    predictor = grounding.sam2_model
    state = predictor.init_state(images)
    state["cached_features"] = backbone_cache.copy()
    original_bypass = predictor.use_mask_input_as_output_without_sam
    predictor.use_mask_input_as_output_without_sam = False
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for frame_index in range(min(5, len(images), len(map_logits))):
                add_joint_prompt(
                    predictor, state, frame_index, 100, embedding, map_logits[frame_index]
                )
            by_frame: dict[int, torch.Tensor] = {}
            for frame_index, _, logits in predictor.propagate_in_video(state):
                by_frame[int(frame_index)] = logits
            logits = torch.cat([by_frame[index] for index in range(len(images))], dim=0)
    finally:
        predictor.use_mask_input_as_output_without_sam = original_bypass
    width, height = output_size
    return F.interpolate(logits.float(), size=(height, width), mode="bilinear", align_corners=False)[:, 0].cpu()


def temporal_iou(mask: np.ndarray) -> float:
    if len(mask) < 2:
        return 1.0
    values = []
    for left, right in zip(mask[:-1], mask[1:]):
        union = np.logical_or(left, right).sum()
        values.append(1.0 if union == 0 else float(np.logical_and(left, right).sum() / union))
    return float(np.mean(values))


def deployable_features(
    baseline_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    raw_map: torch.Tensor,
) -> dict[str, float]:
    base = baseline_logits.numpy()
    cand = candidate_logits.numpy()
    base_mask = base > 0.0
    cand_mask = cand > 0.0
    frame_intersection = np.logical_and(base_mask, cand_mask).sum(axis=(1, 2))
    frame_union = np.logical_or(base_mask, cand_mask).sum(axis=(1, 2))
    agreement = np.where(frame_union == 0, 1.0, frame_intersection / np.maximum(frame_union, 1))
    xor = np.logical_xor(base_mask, cand_mask).mean(axis=(1, 2))
    base_area = base_mask.mean(axis=(1, 2))
    cand_area = cand_mask.mean(axis=(1, 2))
    log_area_ratio = np.abs(np.log((cand_area + 1e-6) / (base_area + 1e-6)))
    raw = raw_map.float().numpy().reshape(-1)
    return {
        "agreement_iou_mean": float(agreement.mean()),
        "agreement_iou_min": float(agreement.min()),
        "xor_fraction_mean": float(xor.mean()),
        "xor_fraction_max": float(xor.max()),
        "area_delta_abs_mean": float(np.abs(cand_area - base_area).mean()),
        "area_delta_abs_max": float(np.abs(cand_area - base_area).max()),
        "log_area_ratio_mean": float(log_area_ratio.mean()),
        "candidate_margin_minus_baseline": float(np.abs(cand).mean() - np.abs(base).mean()),
        "candidate_near_boundary_minus_baseline": float((np.abs(cand) < 1.0).mean() - (np.abs(base) < 1.0).mean()),
        "candidate_temporal_iou_minus_baseline": float(temporal_iou(cand_mask) - temporal_iou(base_mask)),
        "raw_map_mean": float(raw.mean()),
        "raw_map_std": float(raw.std()),
        "raw_map_positive_fraction": float((raw > 0.0).mean()),
    }


def load_configs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if args.split == "calibration":
        if args.policy:
            raise ValueError("calibration must not receive a policy")
        return DEFAULT_CONFIGS, None
    if not args.policy:
        raise ValueError("test evaluation requires a locked policy")
    policy = json.loads(Path(args.policy).read_text())
    if policy.get("status") != "GO":
        raise RuntimeError("policy is not GO")
    return [policy["candidate"]], policy


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index/count")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    row_dir = output / "rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    manifest = json.loads(Path(args.manifest).read_text())
    configs, policy = load_configs(args)
    events_spec = manifest[args.split]
    selected = [event for index, event in enumerate(events_spec) if index % args.num_shards == args.shard_index]
    current_hash = protocol_hash(manifest, args.split, configs)
    if policy is not None and policy.get("test_protocol_hash") != current_hash:
        raise RuntimeError("locked policy/test protocol hash mismatch")
    atomic_json(status_path, {
        "state": "running", "phase": "model_load", "split": args.split,
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "samples": len(selected), "protocol_hash": current_hash,
    })

    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint["best_epoch"]) != 10:
        raise RuntimeError("adapter checkpoint is not epoch 10")
    layers = [int(value) for value in checkpoint["layers"]]
    if 28 not in layers:
        raise RuntimeError("adapter checkpoint has no layer 28")
    layer_position = layers.index(28)
    adapter = SpatialMapAdapters(
        len(layers), int(checkpoint["input_dim"]), int(checkpoint["projection_dim"])
    ).eval()
    adapter.load_state_dict(checkpoint["state_dict"], strict=True)

    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True,
    ).eval().to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model.preparing_for_generation(tokenizer=tokenizer)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    grounding = model.grounding_encoder
    projection = model.text_hidden_fcs
    projection_dtype = next(projection.parameters()).dtype
    data_root = Path(args.data_root)
    split_root = data_root / "video_datas" / "mevis" / "train"
    metadata = json.loads((split_root / "meta_expressions.json").read_text())["videos"]
    mask_dict = json.loads((split_root / "mask_dict.json").read_text())

    completed = 0
    for number, item in enumerate(selected, 1):
        row_path = row_dir / f"{item['key']}.json"
        if row_path.exists():
            existing = json.loads(row_path.read_text())
            if existing.get("protocol_hash") != current_hash:
                raise RuntimeError(f"resume protocol mismatch for {item['key']}")
            completed += 1
            continue
        event = build_event(data_root, item, metadata)
        atomic_json(status_path, {
            "state": "running", "phase": "evaluation", "completed": completed,
            "total": len(selected), "key": event["key"], "protocol_hash": current_hash,
        })
        seg, image_states, _, side = extract_states(model, tokenizer, event, 5, tuple(layers))
        with torch.inference_mode():
            raw_maps = adapter(seg[None], image_states[None])[0]
            h28_embedding = projection(seg[layer_position].to(device="cuda:0", dtype=projection_dtype)).reshape(-1)
        if h28_embedding.numel() != 256:
            raise RuntimeError("projected h28 embedding is not 256-dimensional")
        images, output_size = preprocess_video(
            grounding, model.extra_image_processor, model.torch_dtype, event["frame_paths"]
        )
        width, height = output_size
        truth = ground_truth_video(mask_dict, event["annotation_ids"], len(images), (height, width))
        backbone_cache = cache_video_backbones(grounding, images)
        baseline_logits = infer_baseline_logits(
            grounding, images, backbone_cache, h28_embedding, output_size
        )
        baseline_prediction = baseline_logits.numpy() > 0.0
        baseline_metrics = exact_metrics(truth, baseline_prediction)
        candidates = []
        raw_l28 = raw_maps[layer_position].reshape(raw_maps.shape[1], side, side)
        for config in configs:
            calibrated = float(config["alpha"]) * raw_l28 + float(config["beta"])
            candidate_logits = infer_map_logits(
                grounding, images, backbone_cache, h28_embedding, calibrated, output_size
            )
            metrics = exact_metrics(truth, candidate_logits.numpy() > 0.0)
            candidates.append({
                "candidate_id": config_id(config),
                "config": config,
                "metrics": metrics,
                "delta_J_points": (metrics["J"] - baseline_metrics["J"]) * 100.0,
                "delta_F_points": (metrics["F"] - baseline_metrics["F"]) * 100.0,
                "delta_J&F_points": (metrics["J&F"] - baseline_metrics["J&F"]) * 100.0,
                "features": deployable_features(baseline_logits, candidate_logits, raw_l28),
            })
            del candidate_logits
        row = {
            "key": event["key"], "video_id": event["video_id"], "exp_id": event["exp_id"],
            "split": args.split, "protocol_hash": current_hash,
            "baseline_h28_no_map": baseline_metrics, "candidates": candidates,
        }
        atomic_json(row_path, row)
        completed += 1
        print(
            f"AFFINE_GATE [{number}/{len(selected)}] {event['key']} baseline={baseline_metrics['J&F']:.4f} "
            + " ".join(f"{c['candidate_id']}={c['delta_J&F_points']:+.3f}pt" for c in candidates),
            flush=True,
        )
        del seg, image_states, raw_maps, raw_l28, h28_embedding, images, backbone_cache
        del baseline_logits, baseline_prediction, truth
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(status_path, {
        "state": "complete", "phase": "done", "completed": completed,
        "total": len(selected), "protocol_hash": current_hash,
    })


if __name__ == "__main__":
    main()
