#!/usr/bin/env python3
"""Train and evaluate a 200-expression multi-video fusion pilot.

The pretrained Sa2VA model, text projection, and SAM2 are frozen. Generation
hidden states are cached once. Only final-layer-anchored residual coefficients
are optimized with BCE + Dice supervision on MeViS-train frames. Evaluation is
video-disjoint, caps expressions per video, and reports paired baseline/fused
J, F, and J&F over multiple visible frames per expression.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as cocomask
from transformers import AutoModel, AutoTokenizer

from third_parts.revos.utils.metircs import db_eval_boundary, db_eval_iou
from tools.multilayer_seg import (
    FinalAnchoredResidualFusion,
    extract_generation_seg_states,
)


LAYERS = (14, 18, 21, 24, 28)
EXPECTED_TRANSFORMER_LAYERS = 28
SEG_PROMPT = "<image>\nPlease segment {}."
SEG_PROMPT_QUESTION = "<image>\n {} Please respond with a segmentation mask."


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".tmp_", suffix=".json", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".tmp_", suffix=".pt", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-samples", type=int, default=48)
    parser.add_argument("--eval-samples", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-events-per-video", type=int, default=2)
    parser.add_argument("--eval-frames-per-event", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--residual-layer", type=int, choices=tuple(range(1, 28)))
    parser.add_argument(
        "--residual-layers", type=int, nargs="+", choices=tuple(range(1, 28))
    )
    parser.add_argument("--embedding-cache")
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--coefficient-penalty", type=float, default=0.01)
    return parser.parse_args()


def decode_expression_mask(
    mask_dict: dict[str, Any], annotation_ids: list[Any], frame_index: int
) -> np.ndarray | None:
    masks = []
    for annotation_id in annotation_ids:
        track = mask_dict.get(str(annotation_id))
        if track is None or frame_index >= len(track):
            continue
        rle = track[frame_index]
        if not rle:
            continue
        masks.append(cocomask.decode(rle).astype(bool))
    if not masks:
        return None
    merged = masks[0]
    for mask in masks[1:]:
        merged = np.logical_or(merged, mask)
    return merged


def build_event(
    image_root: Path,
    mask_dict: dict[str, Any],
    video_id: str,
    video: dict[str, Any],
    expression_id: str,
    expression: dict[str, Any],
    eval_frames_per_event: int,
) -> dict[str, Any] | None:
    annotation_ids = expression.get("anno_id") or []
    if not annotation_ids:
        return None
    frames = video["frames"]
    visible = []
    for frame_index in range(len(frames)):
        has_visible_object = False
        for annotation_id in annotation_ids:
            track = mask_dict.get(str(annotation_id))
            if track is None or frame_index >= len(track) or not track[frame_index]:
                continue
            if float(np.asarray(cocomask.area(track[frame_index])).sum()) > 0.0:
                has_visible_object = True
                break
        if has_visible_object:
            visible.append(frame_index)
    if not visible:
        return None
    frame_index = visible[len(visible) // 2]
    gt = decode_expression_mask(mask_dict, annotation_ids, frame_index)
    if gt is None or not bool(gt.any()):
        return None
    text = expression["exp"]
    prompt = (
        SEG_PROMPT.format(text)
        if "?" not in text
        else SEG_PROMPT_QUESTION.format(text)
    )
    frame_name = frames[frame_index]
    eval_positions = np.linspace(
        0, len(visible) - 1, num=min(eval_frames_per_event, len(visible)), dtype=int
    )
    eval_frame_indices = [visible[int(position)] for position in eval_positions]
    eval_frames = []
    for eval_frame_index in eval_frame_indices:
        eval_gt = decode_expression_mask(mask_dict, annotation_ids, eval_frame_index)
        if eval_gt is None or not bool(eval_gt.any()):
            continue
        eval_frame_name = frames[eval_frame_index]
        eval_frames.append({
            "frame_index": eval_frame_index,
            "frame_name": eval_frame_name,
            "image_path": str(image_root / video_id / f"{eval_frame_name}.jpg"),
            "gt": eval_gt,
        })
    if not eval_frames:
        return None
    return {
        "key": f"{video_id}_{expression_id}_{frame_index:05d}",
        "video_id": video_id,
        "exp_id": str(expression_id),
        "frame_index": frame_index,
        "frame_name": frame_name,
        "image_path": str(image_root / video_id / f"{frame_name}.jpg"),
        "prompt": prompt,
        "gt": gt,
        "eval_frames": eval_frames,
    }


def collect_video_disjoint_events(
    data_root: Path,
    train_samples: int,
    eval_samples: int,
    seed: int,
    max_events_per_video: int,
    eval_frames_per_event: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_root = data_root / "video_datas" / "mevis" / "train"
    metadata = json.loads((split_root / "meta_expressions.json").read_text())[
        "videos"
    ]
    mask_dict = json.loads((split_root / "mask_dict.json").read_text())
    image_root = split_root / "JPEGImages"
    video_ids = sorted(metadata)
    random.Random(seed).shuffle(video_ids)
    split_at = max(1, int(round(0.8 * len(video_ids))))
    train_video_ids = video_ids[:split_at]
    eval_video_ids = video_ids[split_at:]

    def collect(video_subset: list[str], limit: int) -> list[dict[str, Any]]:
        events = []
        for video_id in video_subset:
            video = metadata[video_id]
            events_from_video = 0
            for expression_id in sorted(video["expressions"], key=lambda value: int(value)):
                event = build_event(
                    image_root,
                    mask_dict,
                    video_id,
                    video,
                    expression_id,
                    video["expressions"][expression_id],
                    eval_frames_per_event,
                )
                if event is not None:
                    events.append(event)
                    events_from_video += 1
                    if len(events) >= limit:
                        return events
                    if events_from_video >= max_events_per_video:
                        break
        return events

    train_events = collect(train_video_ids, train_samples)
    eval_events = collect(eval_video_ids, eval_samples)
    if len(train_events) != train_samples or len(eval_events) != eval_samples:
        raise RuntimeError(
            "not enough annotated events: "
            f"train {len(train_events)}/{train_samples}, "
            f"eval {len(eval_events)}/{eval_samples}"
        )
    overlap = {event["video_id"] for event in train_events} & {
        event["video_id"] for event in eval_events
    }
    if overlap:
        raise RuntimeError(f"train/eval video overlap: {sorted(overlap)}")
    for event in train_events:
        event["embedding_image_paths"] = [event["image_path"]]
    for event in eval_events:
        event["embedding_image_paths"] = [
            frame["image_path"] for frame in event["eval_frames"]
        ]
    return train_events, eval_events


@torch.no_grad()
def extract_event_states(
    model: Any,
    tokenizer: Any,
    event: dict[str, Any],
    layer_indices: tuple[int, ...],
) -> tuple[torch.Tensor, bool]:
    images = [
        Image.open(path).convert("RGB") for path in event["embedding_image_paths"]
    ]
    capture: dict[str, Any] = {}
    original_generate = model.generate

    def capture_generate(*args: Any, **kwargs: Any):
        result = original_generate(*args, **kwargs)
        capture["result"] = result
        return result

    model.generate = capture_generate
    try:
        prediction = model.predict_forward(
            video=images, text=event["prompt"], tokenizer=tokenizer
        )
    finally:
        model.generate = original_generate
        for image in images:
            image.close()
    if "result" not in capture:
        raise RuntimeError("predict_forward did not call generate")
    output = capture["result"]
    extracted = extract_generation_seg_states(
        output.hidden_states,
        output.sequences,
        int(model.seg_token_idx),
        layer_indices,
        expected_transformer_layers=EXPECTED_TRANSFORMER_LAYERS,
    )
    if extracted.states.shape[:2] != (1, len(layer_indices)):
        raise RuntimeError(
            f"expected one SEG with {len(layer_indices)} layers, "
            f"got {tuple(extracted.states.shape)}"
        )
    return extracted.states[0].detach().float().cpu(), bool(
        prediction["prediction_masks"]
    )


@torch.no_grad()
def make_sam_cache(
    grounding: Any, processor: Any, dtype: torch.dtype, image_path: str
) -> dict[str, Any]:
    with Image.open(image_path) as image:
        raw = np.asarray(image.convert("RGB"))
    resized = processor.apply_image(raw)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous()
    device = next(grounding.parameters()).device
    tensor = grounding.preprocess_image(tensor).to(
        device=device, dtype=dtype
    ).unsqueeze(0)
    backbone = grounding.sam2_model.forward_image(tensor)
    return {"image": tensor, "backbone": backbone}


def fresh_state(grounding: Any, cached: dict[str, Any]) -> dict[str, Any]:
    state = grounding.sam2_model.init_state(cached["image"])
    state["cached_features"] = {0: (cached["image"], cached["backbone"])}
    return state


def prompt_logits(
    grounding: Any, cached: dict[str, Any], embedding: torch.Tensor
) -> torch.Tensor:
    state = fresh_state(grounding, cached)
    language = embedding.to(dtype=cached["image"].dtype).reshape(1, 1, -1)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, _, logits = grounding.sam2_model.add_language_embd(
            state,
            frame_idx=0,
            obj_id=100,
            language_embd=language,
            inference=False,
        )
    return logits


def segmentation_loss(
    logits: torch.Tensor, gt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = F.interpolate(
        gt[None, None].float(), size=logits.shape[-2:], mode="nearest"
    )
    positive = target.sum()
    negative = target.numel() - positive
    positive_weight = torch.clamp(negative / positive.clamp_min(1.0), 1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(
        logits.float(), target, pos_weight=positive_weight
    )
    probability = logits.float().sigmoid()
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice, bce, dice


@torch.no_grad()
def evaluate_embedding(
    grounding: Any,
    cached: dict[str, Any],
    embedding: torch.Tensor,
    gt: np.ndarray,
) -> dict[str, float | int]:
    logits = prompt_logits(grounding, cached, embedding)
    resized = F.interpolate(
        logits.float(), size=gt.shape, mode="bilinear", align_corners=False
    )
    prediction = (resized[0, 0].sigmoid() > 0.5).cpu().numpy()
    j = float(db_eval_iou(gt[None], prediction[None]).mean())
    f = float(db_eval_boundary(gt[None], prediction[None]).mean())
    return {
        "J": j,
        "F": f,
        "J&F": (j + f) / 2.0,
        "pred_area": int(prediction.sum()),
        "gt_area": int(gt.sum()),
    }


def public_event(event: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in event.items()
        if key not in {"gt", "eval_frames"}
    }
    result["eval_frames"] = [
        {key: value for key, value in frame.items() if key != "gt"}
        for frame in event["eval_frames"]
    ]
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fusion training pilot")
    if not 1 <= args.train_samples <= 192:
        raise ValueError("train-samples must be in [1, 192]")
    if not 1 <= args.eval_samples <= 96:
        raise ValueError("eval-samples must be in [1, 96]")
    if not 1 <= args.epochs <= 5:
        raise ValueError("epochs must be in [1, 5]")
    if not 1 <= args.max_events_per_video <= 4:
        raise ValueError("max-events-per-video must be in [1, 4]")
    if not 2 <= args.eval_frames_per_event <= 5:
        raise ValueError("eval-frames-per-event must be in [2, 5]")
    if not 0.0 < args.weight_scale <= 1.0:
        raise ValueError("weight-scale must be in (0, 1]")
    if args.coefficient_penalty < 0.0:
        raise ValueError("coefficient-penalty must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    metrics_path = output / "metrics.json"
    device = torch.device("cuda:0")
    if args.residual_layer is not None and args.residual_layers is not None:
        raise ValueError("use only one of --residual-layer and --residual-layers")
    selected_residual_layers = (
        tuple(args.residual_layers)
        if args.residual_layers is not None
        else ((args.residual_layer,) if args.residual_layer is not None else LAYERS[:-1])
    )
    if len(set(selected_residual_layers)) != len(selected_residual_layers):
        raise ValueError("residual layers contain duplicates")
    if tuple(sorted(selected_residual_layers)) != selected_residual_layers:
        raise ValueError("residual layers must be strictly increasing")
    layer_indices = (*selected_residual_layers, LAYERS[-1])

    train_events, eval_events = collect_video_disjoint_events(
        Path(args.data_root),
        args.train_samples,
        args.eval_samples,
        args.seed,
        args.max_events_per_video,
        args.eval_frames_per_event,
    )
    atomic_json(
        output / "event_split.json",
        {
            "train": [public_event(event) for event in train_events],
            "heldout": [public_event(event) for event in eval_events],
        },
    )
    atomic_json(
        status_path,
        {"state": "running", "phase": "model_load_and_embedding_cache"},
    )

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if args.embedding_cache:
        # predict_forward normally performs this lazy initialization.  Cached
        # hidden states bypass predict_forward, but SAM preprocessing still
        # needs the same model-owned processor and generation configuration.
        model.preparing_for_generation(tokenizer)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    all_events = train_events + eval_events
    state_cache: dict[str, torch.Tensor] = {}
    prediction_mask_count = 0
    if args.embedding_cache:
        payload = torch.load(args.embedding_cache, map_location="cpu", weights_only=False)
        cached_layers = tuple(int(value) for value in payload["layer_indices"])
        positions = [cached_layers.index(layer) for layer in layer_indices]
        missing = [event["key"] for event in all_events if event["key"] not in payload["states"]]
        if missing:
            raise RuntimeError(f"embedding cache missing {len(missing)} event keys")
        for event in all_events:
            state_cache[event["key"]] = payload["states"][event["key"]][positions].float()
        prediction_mask_count = len(all_events)
        print(
            f"EMBED_CACHE loaded={len(state_cache)} layers={layer_indices} "
            f"source={args.embedding_cache}",
            flush=True,
        )
    else:
        for number, event in enumerate(all_events, 1):
            states, has_masks = extract_event_states(
                model, tokenizer, event, layer_indices
            )
            state_cache[event["key"]] = states
            prediction_mask_count += int(has_masks)
            print(
                f"EMBED [{number}/{len(all_events)}] {event['key']} "
                f"shape={tuple(states.shape)} masks={has_masks}",
                flush=True,
            )
            atomic_json(
                status_path,
                {
                    "state": "running",
                    "phase": "model_load_and_embedding_cache",
                    "completed": number,
                    "total": len(all_events),
                },
            )
            gc.collect()
            torch.cuda.empty_cache()

    fusion = FinalAnchoredResidualFusion(
        layer_indices, weight_scale=args.weight_scale
    ).to(device)
    initial_error = max(
        float(
            (fusion(states.to(device)) - states[-1].to(device))
            .abs()
            .max()
            .item()
        )
        for states in state_cache.values()
    )
    if initial_error != 0.0:
        raise RuntimeError(f"initial fusion is not baseline-equivalent: {initial_error}")
    atomic_torch(
        output / "embedding_cache.pt",
        {
            "layer_indices": layer_indices,
            "states": state_cache,
            "events": [public_event(event) for event in all_events],
        },
    )

    grounding = model.grounding_encoder
    processor = model.extra_image_processor
    text_hidden_fcs = model.text_hidden_fcs
    dtype = model.torch_dtype
    grounding.eval()
    text_hidden_fcs.eval()
    projection_dtype = next(text_hidden_fcs.parameters()).dtype
    model.grounding_encoder = None
    model.text_hidden_fcs = None
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    atomic_json(status_path, {"state": "running", "phase": "fusion_training"})
    optimizer = torch.optim.AdamW(
        fusion.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    history = []
    first_gradient_norm = None
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_events)))
        random.Random(args.seed + epoch).shuffle(order)
        epoch_loss, epoch_bce, epoch_dice = [], [], []
        for position, event_index in enumerate(order, 1):
            event = train_events[event_index]
            states = state_cache[event["key"]].to(device)
            fused = fusion(states)
            embedding = text_hidden_fcs(fused.to(dtype=projection_dtype))
            cached = make_sam_cache(
                grounding, processor, dtype, event["image_path"]
            )
            logits = prompt_logits(grounding, cached, embedding)
            gt = torch.from_numpy(event["gt"].astype(np.float32)).to(device)
            mask_loss, bce, dice = segmentation_loss(logits, gt)
            preserve = F.smooth_l1_loss(fused.float(), states[-1].float(), beta=0.02)
            coefficient_penalty = fusion.effective_residual_weights.square().mean()
            total = (
                mask_loss
                + 0.02 * preserve
                + args.coefficient_penalty * coefficient_penalty
            )

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if first_gradient_norm is None:
                first_gradient_norm = float(
                    fusion.raw_residual_weights.grad.float().norm().item()
                )
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 2.0)
            optimizer.step()
            epoch_loss.append(float(total.detach()))
            epoch_bce.append(float(bce.detach()))
            epoch_dice.append(float(dice.detach()))
            print(
                f"TRAIN epoch={epoch} [{position}/{len(order)}] "
                f"loss={epoch_loss[-1]:.6f} key={event['key']}",
                flush=True,
            )
            del cached, logits, embedding, fused, states, gt, total
            if position % 4 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        row = {
            "epoch": epoch,
            "loss": float(np.mean(epoch_loss)),
            "bce": float(np.mean(epoch_bce)),
            "dice": float(np.mean(epoch_dice)),
            "fusion": fusion.diagnostics(),
        }
        history.append(row)
        print("EPOCH " + json.dumps(row), flush=True)
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in fusion.state_dict().items()
            }
        atomic_torch(
            output / "checkpoint_last.pt",
            {
                "state_dict": fusion.state_dict(),
                "layer_indices": layer_indices,
                "history": history,
            },
        )
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    fusion.load_state_dict(best_state)
    atomic_torch(
        output / "checkpoint_best.pt",
        {
            "state_dict": best_state,
            "layer_indices": layer_indices,
            "history": history,
        },
    )

    atomic_json(status_path, {"state": "running", "phase": "heldout_evaluation"})
    rows = []
    for number, event in enumerate(eval_events, 1):
        states = state_cache[event["key"]].to(device)
        with torch.no_grad():
            baseline_embedding = text_hidden_fcs(
                states[-1].to(dtype=projection_dtype)
            )
            fused_embedding = text_hidden_fcs(
                fusion(states).to(dtype=projection_dtype)
            )
        for frame_number, frame in enumerate(event["eval_frames"], 1):
            cached = make_sam_cache(
                grounding, processor, dtype, frame["image_path"]
            )
            baseline = evaluate_embedding(
                grounding, cached, baseline_embedding, frame["gt"]
            )
            fused_metrics = evaluate_embedding(
                grounding, cached, fused_embedding, frame["gt"]
            )
            row = {
                "key": event["key"],
                "video_id": event["video_id"],
                "exp_id": event["exp_id"],
                "frame_index": frame["frame_index"],
                "frame_name": frame["frame_name"],
                "baseline": baseline,
                "fused": fused_metrics,
                "delta_J&F": fused_metrics["J&F"] - baseline["J&F"],
            }
            rows.append(row)
            atomic_json(output / "heldout_rows.json", {"rows": rows})
            print(
                f"EVAL expression=[{number}/{len(eval_events)}] "
                f"frame=[{frame_number}/{len(event['eval_frames'])}] {event['key']} "
                f"baseline={baseline['J&F']:.6f} "
                f"fused={fused_metrics['J&F']:.6f}",
                flush=True,
            )
            del cached
        del states, baseline_embedding, fused_embedding
        gc.collect()
        torch.cuda.empty_cache()

    def mean_metrics(name: str) -> dict[str, float]:
        return {
            metric: float(np.mean([row[name][metric] for row in rows]))
            for metric in ("J", "F", "J&F")
        }

    baseline_mean = mean_metrics("baseline")
    fused_mean = mean_metrics("fused")
    deltas = np.asarray([row["delta_J&F"] for row in rows])
    expression_deltas: dict[str, list[float]] = {}
    video_deltas: dict[str, list[float]] = {}
    for row in rows:
        expression_deltas.setdefault(row["key"], []).append(row["delta_J&F"])
        video_deltas.setdefault(row["video_id"], []).append(row["delta_J&F"])
    expression_means = np.asarray([
        np.mean(values) for values in expression_deltas.values()
    ])
    video_means = np.asarray([np.mean(values) for values in video_deltas.values()])

    # Cluster-level uncertainty: resample held-out videos, keeping all their
    # frames together.  A sign-flip randomization test complements the CI.
    rng = np.random.default_rng(args.seed + 3000)
    bootstrap_means = np.asarray([
        rng.choice(video_means, size=len(video_means), replace=True).mean()
        for _ in range(10000)
    ])
    observed_video_mean = float(video_means.mean())
    sign_flip_means = np.asarray([
        (video_means * rng.choice((-1.0, 1.0), size=len(video_means))).mean()
        for _ in range(20000)
    ])
    randomization_p = float(
        (1 + np.count_nonzero(np.abs(sign_flip_means) >= abs(observed_video_mean)))
        / (len(sign_flip_means) + 1)
    )

    metrics = {
        "status": "complete",
        "experiment": "multilayer_seg_fusion_pilot200_v3",
        "scope": "MeViS-train 200-expression video-disjoint multi-frame pilot",
        "train_samples": len(train_events),
        "eval_samples": len(eval_events),
        "eval_frames": len(rows),
        "train_videos": len({event["video_id"] for event in train_events}),
        "eval_videos": len({event["video_id"] for event in eval_events}),
        "max_events_per_video": args.max_events_per_video,
        "eval_frames_per_event": args.eval_frames_per_event,
        "epochs": args.epochs,
        "layer_indices": list(layer_indices),
        "trainable_parameters": sum(
            parameter.numel() for parameter in fusion.parameters()
            if parameter.requires_grad
        ),
        "pretrained_trainable_parameters": 0,
        "weight_scale": args.weight_scale,
        "coefficient_penalty": args.coefficient_penalty,
        "initial_baseline_max_abs_error": initial_error,
        "generation_expressions_with_masks": prediction_mask_count,
        "first_gradient_norm": first_gradient_norm,
        "training_history": history,
        "fusion": fusion.diagnostics(),
        "heldout": {
            "baseline": baseline_mean,
            "fused": fused_mean,
            "paired_delta_J&F": float(deltas.mean()),
            "paired_median_delta_J&F": float(np.median(deltas)),
            "improved": int((deltas > 1e-6).sum()),
            "declined": int((deltas < -1e-6).sum()),
            "equal": int((np.abs(deltas) <= 1e-6).sum()),
            "expression_level": {
                "mean_delta_J&F": float(expression_means.mean()),
                "median_delta_J&F": float(np.median(expression_means)),
                "improved": int((expression_means > 1e-6).sum()),
                "declined": int((expression_means < -1e-6).sum()),
                "equal": int((np.abs(expression_means) <= 1e-6).sum()),
            },
            "video_level": {
                "mean_delta_J&F": observed_video_mean,
                "median_delta_J&F": float(np.median(video_means)),
                "improved": int((video_means > 1e-6).sum()),
                "declined": int((video_means < -1e-6).sum()),
                "equal": int((np.abs(video_means) <= 1e-6).sum()),
                "cluster_bootstrap_95pct_ci": [
                    float(np.quantile(bootstrap_means, 0.025)),
                    float(np.quantile(bootstrap_means, 0.975)),
                ],
                "sign_flip_randomization_p_two_sided": randomization_p,
                "bootstrap_replicates": 10000,
                "randomization_replicates": 20000,
            },
        },
    }
    atomic_json(metrics_path, metrics)
    atomic_json(status_path, {"state": "complete", "phase": "complete"})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    try:
        run(args)
    except Exception as error:
        atomic_json(
            output / "status.json",
            {
                "state": "failed",
                "phase": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        atomic_json(
            output / "metrics.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
