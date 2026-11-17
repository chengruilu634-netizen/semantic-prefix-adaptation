#!/usr/bin/env python3
"""Larger four-layer fusion training and full MeViS valid_u evaluation."""

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
from PIL import Image
from pycocotools import mask as cocomask
from transformers import AutoModel, AutoTokenizer

from third_parts.revos.utils.metircs import db_eval_boundary, db_eval_iou
from tools.multilayer_seg import FinalAnchoredResidualFusion
from tools.multilayer_seg.train_eval_fusion_pilot_v3 import (
    LAYERS,
    SEG_PROMPT,
    SEG_PROMPT_QUESTION,
    atomic_json,
    atomic_torch,
    build_event,
    extract_event_states,
    make_sam_cache,
    prompt_logits,
    public_event,
    segmentation_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-events-per-video", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--residual-layers", type=int, nargs="+", default=(14, 18, 21, 24),
        choices=tuple(range(1, 28)),
    )
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--coefficient-penalty", type=float, default=0.01)
    parser.add_argument("--validation-video-limit", type=int)
    parser.add_argument("--validation-expression-limit", type=int)
    return parser.parse_args()


def collect_large_train_events(
    data_root: Path,
    train_samples: int,
    max_events_per_video: int,
    seed: int,
) -> list[dict[str, Any]]:
    split_root = data_root / "video_datas" / "mevis" / "train"
    metadata = json.loads((split_root / "meta_expressions.json").read_text())["videos"]
    mask_dict = json.loads((split_root / "mask_dict.json").read_text())
    image_root = split_root / "JPEGImages"
    video_ids = sorted(metadata)
    random.Random(seed).shuffle(video_ids)
    events: list[dict[str, Any]] = []
    for video_id in video_ids:
        video = metadata[video_id]
        from_video = 0
        for expression_id in sorted(video["expressions"], key=lambda value: int(value)):
            event = build_event(
                image_root,
                mask_dict,
                video_id,
                video,
                expression_id,
                video["expressions"][expression_id],
                2,
            )
            if event is None:
                continue
            event["embedding_image_paths"] = [event["image_path"]]
            events.append(event)
            from_video += 1
            if len(events) == train_samples:
                return events
            if from_video >= max_events_per_video:
                break
    raise RuntimeError(f"only collected {len(events)}/{train_samples} train events")


def validation_videos(data_root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    split_root = data_root / "video_datas" / "mevis" / "valid_u"
    metadata = json.loads((split_root / "meta_expressions.json").read_text())["videos"]
    mask_dict = json.loads((split_root / "mask_dict.json").read_text())
    return split_root / "JPEGImages", metadata, mask_dict


def prompt_for(text: str) -> str:
    return SEG_PROMPT.format(text) if "?" not in text else SEG_PROMPT_QUESTION.format(text)


@torch.no_grad()
def extract_states_without_sam(
    model: Any,
    tokenizer: Any,
    image_paths: list[str],
    prompt: str,
    layer_indices: tuple[int, ...],
) -> torch.Tensor:
    event = {
        "embedding_image_paths": image_paths,
        "prompt": prompt,
    }
    grounding = model.grounding_encoder
    original = grounding.language_embd_inference

    def dummy_language_inference(state: Any, language: Any, return_obj_ptrs: bool = False):
        frame_count = len(language)
        device = next(grounding.parameters()).device
        masks = torch.zeros(frame_count, 1, 8, 8, device=device, dtype=torch.bfloat16)
        if not return_obj_ptrs:
            return masks
        pointers = torch.zeros(frame_count, 1, 256, device=device, dtype=torch.bfloat16)
        return masks, pointers

    grounding.language_embd_inference = dummy_language_inference
    try:
        states, _ = extract_event_states(model, tokenizer, event, layer_indices)
    finally:
        grounding.language_embd_inference = original
    return states


@torch.no_grad()
def preprocess_video(
    grounding: Any,
    processor: Any,
    dtype: torch.dtype,
    paths: list[str],
) -> tuple[torch.Tensor, tuple[int, int]]:
    tensors = []
    size = None
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if size is None:
                size = rgb.size
            raw = np.asarray(rgb)
        resized = processor.apply_image(raw)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous()
        tensors.append(grounding.preprocess_image(tensor))
    if size is None:
        raise RuntimeError("video contains no frames")
    return torch.stack(tensors).to(device="cuda", dtype=dtype), size


@torch.no_grad()
def cache_video_backbones(grounding: Any, images: torch.Tensor) -> dict[int, Any]:
    cache = {}
    for frame_index in range(len(images)):
        image = images[frame_index].unsqueeze(0)
        cache[frame_index] = (image, grounding.sam2_model.forward_image(image))
    return cache


@torch.no_grad()
def infer_video_embedding(
    grounding: Any,
    images: torch.Tensor,
    backbone_cache: dict[int, Any],
    embedding: torch.Tensor,
    output_size: tuple[int, int],
) -> np.ndarray:
    state = grounding.sam2_model.init_state(images)
    state["cached_features"] = backbone_cache.copy()
    # Match Sa2VA.predict_forward exactly: the MLLM observes at most the first
    # five video frames, so language prompts are injected only on those frames.
    # SAM2 then propagates the prompted objects through the complete video state.
    conditioning_frame_count = min(5, len(images))
    language = [[embedding] for _ in range(conditioning_frame_count)]
    logits = grounding.language_embd_inference(state, language)
    width, height = output_size
    logits = F.interpolate(logits.float(), size=(height, width), mode="bilinear", align_corners=False)
    return (logits[:, 0].sigmoid() > 0.5).cpu().numpy()


def ground_truth_video(
    mask_dict: dict[str, Any],
    annotation_ids: list[Any],
    frame_count: int,
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    result = np.zeros((frame_count, height, width), dtype=np.uint8)
    for frame_index in range(frame_count):
        for annotation_id in annotation_ids:
            track = mask_dict.get(str(annotation_id))
            if track is None or frame_index >= len(track) or not track[frame_index]:
                continue
            result[frame_index] |= cocomask.decode(track[frame_index]).astype(np.uint8)
    return result


def encode_masks(masks: np.ndarray) -> list[dict[str, Any]]:
    encoded = []
    for mask in masks:
        rle = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        encoded.append(rle)
    return encoded


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 1 <= args.train_samples <= 1024:
        raise ValueError("train-samples must be in [1, 1024]")
    if not 1 <= args.epochs <= 5:
        raise ValueError("epochs must be in [1, 5]")
    residual_layers = tuple(args.residual_layers)
    if len(set(residual_layers)) != len(residual_layers):
        raise ValueError("residual layers contain duplicates")
    if tuple(sorted(residual_layers)) != residual_layers:
        raise ValueError("residual layers must be strictly increasing")
    if not 0.0 < args.weight_scale <= 1.0:
        raise ValueError("weight-scale must be in (0, 1]")
    if args.coefficient_penalty < 0.0:
        raise ValueError("coefficient-penalty must be non-negative")
    layer_indices = (*residual_layers, 28)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    atomic_json(status_path, {"state": "running", "phase": "data_index"})

    train_events = collect_large_train_events(
        Path(args.data_root), args.train_samples, args.max_events_per_video, args.seed
    )
    val_image_root, val_metadata, val_mask_dict = validation_videos(Path(args.data_root))
    atomic_json(
        output / "train_events.json",
        {"events": [public_event(event) for event in train_events]},
    )

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    atomic_json(status_path, {"state": "running", "phase": "train_embedding_cache"})
    state_cache = {}
    for number, event in enumerate(train_events, 1):
        state_cache[event["key"]] = extract_states_without_sam(
            model, tokenizer, event["embedding_image_paths"], event["prompt"],
            layer_indices,
        )
        print(f"TRAIN_EMBED [{number}/{len(train_events)}] {event['key']}", flush=True)
        if number % 8 == 0:
            atomic_json(status_path, {"state": "running", "phase": "train_embedding_cache", "completed": number, "total": len(train_events)})
            gc.collect()
            torch.cuda.empty_cache()

    fusion = FinalAnchoredResidualFusion(
        layer_indices, weight_scale=args.weight_scale
    ).to("cuda:0")
    initial_error = max(float((fusion(value.cuda()) - value[-1].cuda()).abs().max()) for value in state_cache.values())
    if initial_error != 0.0:
        raise RuntimeError(f"initial fusion differs from baseline: {initial_error}")
    atomic_torch(output / "train_embedding_cache.pt", {"layer_indices": layer_indices, "states": state_cache})

    grounding = model.grounding_encoder
    processor = model.extra_image_processor
    text_hidden_fcs = model.text_hidden_fcs
    dtype = model.torch_dtype
    projection_dtype = next(text_hidden_fcs.parameters()).dtype
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=args.learning_rate, weight_decay=0.0)
    history = []
    best_loss = float("inf")
    best_state = None
    first_gradient_norm = None
    atomic_json(status_path, {"state": "running", "phase": "fusion_training"})
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_events)))
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        for position, index in enumerate(order, 1):
            event = train_events[index]
            states = state_cache[event["key"]].cuda()
            fused = fusion(states)
            embedding = text_hidden_fcs(fused.to(dtype=projection_dtype))
            cached = make_sam_cache(grounding, processor, dtype, event["image_path"])
            logits = prompt_logits(grounding, cached, embedding)
            gt = torch.from_numpy(event["gt"].astype(np.float32)).cuda()
            mask_loss, _, _ = segmentation_loss(logits, gt)
            preserve = F.smooth_l1_loss(fused.float(), states[-1].float(), beta=0.02)
            penalty = fusion.effective_residual_weights.square().mean()
            total = (
                mask_loss + 0.02 * preserve
                + args.coefficient_penalty * penalty
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if first_gradient_norm is None:
                first_gradient_norm = float(fusion.raw_residual_weights.grad.float().norm())
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 2.0)
            optimizer.step()
            losses.append(float(total.detach()))
            print(f"TRAIN epoch={epoch} [{position}/{len(order)}] loss={losses[-1]:.6f}", flush=True)
            del cached, logits, gt, embedding, fused, states, total
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "fusion": fusion.diagnostics()}
        history.append(row)
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in fusion.state_dict().items()}
        atomic_torch(output / "checkpoint_last.pt", {"state_dict": fusion.state_dict(), "history": history, "layer_indices": layer_indices, "weight_scale": args.weight_scale})
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    fusion.load_state_dict(best_state)
    atomic_torch(output / "checkpoint_best.pt", {"state_dict": best_state, "history": history, "layer_indices": layer_indices, "weight_scale": args.weight_scale})

    atomic_json(status_path, {"state": "running", "phase": "standard_valid_u"})
    rows = []
    baseline_results: dict[str, Any] = {}
    fused_results: dict[str, Any] = {}
    video_ids = sorted(val_metadata)
    if args.validation_video_limit is not None:
        video_ids = video_ids[: args.validation_video_limit]
    for video_number, video_id in enumerate(video_ids, 1):
        video = val_metadata[video_id]
        frame_paths = [str(val_image_root / video_id / f"{name}.jpg") for name in video["frames"]]
        llm_paths = frame_paths[:5]
        expression_items = sorted(video["expressions"].items(), key=lambda item: int(item[0]))
        if args.validation_expression_limit is not None:
            expression_items = expression_items[: args.validation_expression_limit]
        hidden_by_expression = {}
        for expression_id, expression in expression_items:
            hidden_by_expression[expression_id] = extract_states_without_sam(
                model, tokenizer, llm_paths, prompt_for(expression["exp"]),
                layer_indices,
            )
        images, output_size = preprocess_video(grounding, processor, dtype, frame_paths)
        backbone_cache = cache_video_backbones(grounding, images)
        baseline_results[video_id] = {}
        fused_results[video_id] = {}
        for expression_number, (expression_id, expression) in enumerate(expression_items, 1):
            states = hidden_by_expression[expression_id].cuda()
            baseline_embedding = text_hidden_fcs(states[-1].to(dtype=projection_dtype))
            fused_embedding = text_hidden_fcs(fusion(states).to(dtype=projection_dtype))
            baseline_masks = infer_video_embedding(grounding, images, backbone_cache, baseline_embedding, output_size)
            fused_masks = infer_video_embedding(grounding, images, backbone_cache, fused_embedding, output_size)
            width, height = output_size
            gt = ground_truth_video(val_mask_dict, expression["anno_id"], len(frame_paths), (height, width))
            baseline_j = float(db_eval_iou(gt, baseline_masks).mean())
            baseline_f = float(db_eval_boundary(gt, baseline_masks).mean())
            fused_j = float(db_eval_iou(gt, fused_masks).mean())
            fused_f = float(db_eval_boundary(gt, fused_masks).mean())
            row = {
                "video_id": video_id,
                "exp_id": expression_id,
                "baseline": {"J": baseline_j, "F": baseline_f, "J&F": (baseline_j + baseline_f) / 2},
                "fused": {"J": fused_j, "F": fused_f, "J&F": (fused_j + fused_f) / 2},
                "delta_J&F": (fused_j + fused_f - baseline_j - baseline_f) / 2,
            }
            rows.append(row)
            baseline_results[video_id][expression_id] = {"prediction_masks": encode_masks(baseline_masks)}
            fused_results[video_id][expression_id] = {"prediction_masks": encode_masks(fused_masks)}
            print(f"VALID video=[{video_number}/{len(video_ids)}] expression=[{expression_number}/{len(expression_items)}] baseline={row['baseline']['J&F']:.6f} fused={row['fused']['J&F']:.6f}", flush=True)
            del states, baseline_embedding, fused_embedding, baseline_masks, fused_masks, gt
        atomic_json(output / "heldout_rows.json", {"rows": rows})
        atomic_json(output / "baseline_results.json", baseline_results)
        atomic_json(output / "results.json", fused_results)
        del images, backbone_cache, hidden_by_expression
        gc.collect()
        torch.cuda.empty_cache()

    def mean_metrics(name: str) -> dict[str, float]:
        return {metric: float(np.mean([row[name][metric] for row in rows])) for metric in ("J", "F", "J&F")}

    baseline_mean = mean_metrics("baseline")
    fused_mean = mean_metrics("fused")
    deltas = np.asarray([row["delta_J&F"] for row in rows])
    by_video: dict[str, list[float]] = {}
    for row in rows:
        by_video.setdefault(row["video_id"], []).append(row["delta_J&F"])
    video_means = np.asarray([np.mean(values) for values in by_video.values()])
    rng = np.random.default_rng(args.seed + 5000)
    boot = np.asarray([rng.choice(video_means, len(video_means), replace=True).mean() for _ in range(10000)])
    signs = np.asarray([(video_means * rng.choice((-1.0, 1.0), len(video_means))).mean() for _ in range(20000)])
    observed = float(video_means.mean())
    p_value = float((1 + np.count_nonzero(np.abs(signs) >= abs(observed))) / (len(signs) + 1))
    metrics = {
        "status": "complete",
        "scope": "MeViS train larger-scale training; full annotated valid_u standard all-frame evaluation",
        "train_samples": len(train_events),
        "train_videos": len({event["video_id"] for event in train_events}),
        "validation_videos": len(video_ids),
        "validation_expressions": len(rows),
        "validation_frames": sum(len(val_metadata[video_id]["frames"]) for video_id in video_ids),
        "epochs": args.epochs,
        "layer_indices": list(layer_indices),
        "trainable_parameters": len(residual_layers),
        "pretrained_trainable_parameters": 0,
        "weight_scale": args.weight_scale,
        "coefficient_penalty": args.coefficient_penalty,
        "initial_baseline_max_abs_error": initial_error,
        "first_gradient_norm": first_gradient_norm,
        "training_history": history,
        "fusion": fusion.diagnostics(),
        "valid_u": {
            "baseline": baseline_mean,
            "fused": fused_mean,
            "paired_delta_J&F": float(deltas.mean()),
            "paired_median_delta_J&F": float(np.median(deltas)),
            "expressions_improved": int((deltas > 1e-6).sum()),
            "expressions_declined": int((deltas < -1e-6).sum()),
            "expressions_equal": int((np.abs(deltas) <= 1e-6).sum()),
            "video_mean_delta_J&F": observed,
            "videos_improved": int((video_means > 1e-6).sum()),
            "videos_declined": int((video_means < -1e-6).sum()),
            "cluster_bootstrap_95pct_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "sign_flip_randomization_p_two_sided": p_value,
        },
    }
    atomic_json(output / "metrics.json", metrics)
    atomic_json(status_path, {"state": "complete", "phase": "complete"})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    try:
        run(args)
    except Exception as error:
        atomic_json(output / "status.json", {"state": "failed", "phase": "failed", "error_type": type(error).__name__, "error": str(error)})
        atomic_json(output / "metrics.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
