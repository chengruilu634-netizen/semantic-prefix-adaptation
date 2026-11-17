#!/usr/bin/env python3
"""Train final-anchored SEG fusion against exact full-video Sa2VA inference."""

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

from tools.multilayer_seg import FinalAnchoredResidualFusion
from tools.multilayer_seg.train_eval_fusion_pilot_v3 import atomic_json, atomic_torch
from tools.multilayer_seg.train_eval_fusion_standard import (
    extract_states_without_sam,
    ground_truth_video,
    preprocess_video,
    prompt_for,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-events-per-video", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--residual-layers",
        type=int,
        nargs="+",
        default=(14, 18, 21, 24),
        choices=tuple(range(1, 28)),
    )
    parser.add_argument("--weight-scale", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--perturbation", type=float, default=0.08)
    parser.add_argument("--coefficient-penalty", type=float, default=0.02)
    parser.add_argument("--max-raw-weight", type=float, default=2.0)
    return parser.parse_args()


def collect_video_events(
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
    events = []
    for video_id in video_ids:
        video = metadata[video_id]
        expression_ids = sorted(video["expressions"], key=lambda value: int(value))
        random.Random(f"{seed}:{video_id}").shuffle(expression_ids)
        selected = 0
        for expression_id in expression_ids:
            expression = video["expressions"][expression_id]
            annotation_ids = expression.get("anno_id") or []
            if not annotation_ids:
                continue
            frame_paths = [
                str(image_root / video_id / f"{frame_name}.jpg")
                for frame_name in video["frames"]
            ]
            events.append({
                "key": f"{video_id}_{expression_id}",
                "video_id": video_id,
                "exp_id": str(expression_id),
                "prompt": prompt_for(expression["exp"]),
                "frame_paths": frame_paths,
                "annotation_ids": annotation_ids,
            })
            selected += 1
            if len(events) == train_samples:
                return events
            if selected >= max_events_per_video:
                break
    raise RuntimeError(f"only collected {len(events)}/{train_samples} events")


@torch.inference_mode()
def exact_video_logits(
    grounding: Any,
    images: torch.Tensor,
    embedding: torch.Tensor,
) -> torch.Tensor:
    state = grounding.get_sam2_embeddings(images)
    language = [[embedding] for _ in range(min(5, len(images)))]
    return grounding.language_embd_inference(state, language).float()


@torch.inference_mode()
def video_objective(
    grounding: Any,
    text_hidden_fcs: Any,
    states: torch.Tensor,
    theta: torch.Tensor,
    weight_scale: float,
    images: torch.Tensor,
    target: torch.Tensor,
    coefficient_penalty: float,
) -> float:
    anchor = states[-1]
    residuals = states[:-1] - anchor.unsqueeze(0)
    weights = weight_scale * theta.tanh()
    fused = anchor + (residuals * weights[:, None]).sum(dim=0)
    projection_dtype = next(text_hidden_fcs.parameters()).dtype
    embedding = text_hidden_fcs(fused.to(dtype=projection_dtype))
    logits = exact_video_logits(grounding, images, embedding)
    resized_target = F.interpolate(
        target[:, None].float(), size=logits.shape[-2:], mode="nearest"
    )
    positive = resized_target.sum()
    negative = resized_target.numel() - positive
    positive_weight = torch.clamp(
        negative / positive.clamp_min(1.0), 1.0, 20.0
    )
    bce = F.binary_cross_entropy_with_logits(
        logits, resized_target, pos_weight=positive_weight
    )
    probability = logits.sigmoid()
    intersection = (probability * resized_target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + resized_target.sum(dim=(-2, -1))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    penalty = coefficient_penalty * weights.square().mean()
    return float((bce + dice + penalty).cpu())


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    residual_layers = tuple(args.residual_layers)
    if tuple(sorted(set(residual_layers))) != residual_layers:
        raise ValueError("residual layers must be unique and strictly increasing")
    if not 0.0 < args.weight_scale <= 1.0:
        raise ValueError("weight-scale must be in (0, 1]")
    layer_indices = (*residual_layers, 28)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    atomic_json(status_path, {"state": "running", "phase": "data_index"})

    events = collect_video_events(
        Path(args.data_root),
        args.train_samples,
        args.max_events_per_video,
        args.seed,
    )
    train_mask_dict = json.loads(
        (Path(args.data_root) / "video_datas" / "mevis" / "train"
         / "mask_dict.json").read_text()
    )
    atomic_json(output / "train_events.json", {
        "events": [{
            "key": event["key"],
            "video_id": event["video_id"],
            "exp_id": event["exp_id"],
            "frame_count": len(event["frame_paths"]),
        } for event in events]
    })

    atomic_json(status_path, {"state": "running", "phase": "model_load"})
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

    atomic_json(status_path, {"state": "running", "phase": "embedding_cache"})
    states_by_key = {}
    for index, event in enumerate(events, 1):
        states_by_key[event["key"]] = extract_states_without_sam(
            model,
            tokenizer,
            event["frame_paths"][:5],
            event["prompt"],
            layer_indices,
        ).float().cpu()
        print(f"EMBED [{index}/{len(events)}] {event['key']}", flush=True)
        atomic_json(status_path, {
            "state": "running",
            "phase": "embedding_cache",
            "completed": index,
            "total": len(events),
        })
    atomic_torch(output / "train_embedding_cache.pt", {
        "layer_indices": layer_indices,
        "states": states_by_key,
    })

    grounding = model.grounding_encoder
    processor = model.extra_image_processor
    dtype = model.torch_dtype
    theta = torch.zeros(len(residual_layers), device="cuda", dtype=torch.float32)
    first_moment = torch.zeros_like(theta)
    second_moment = torch.zeros_like(theta)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + 1000)
    history = []
    step = 0

    atomic_json(status_path, {"state": "running", "phase": "official_video_training"})
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(events)))
        random.Random(args.seed + epoch).shuffle(order)
        epoch_losses = []
        for position, event_index in enumerate(order, 1):
            event = events[event_index]
            states = states_by_key[event["key"]].cuda()
            images, output_size = preprocess_video(
                grounding, processor, dtype, event["frame_paths"]
            )
            width, height = output_size
            gt_numpy = ground_truth_video(
                train_mask_dict,
                event["annotation_ids"],
                len(event["frame_paths"]),
                (height, width),
            )
            target = torch.from_numpy(gt_numpy.astype(np.float32)).cuda()
            step += 1
            perturbation_scale = args.perturbation / (step ** 0.101)
            learning_rate = args.learning_rate / (step ** 0.602)
            delta = torch.randint(
                0, 2, theta.shape, generator=generator, device="cuda"
            ).float().mul_(2.0).sub_(1.0)
            loss_plus = video_objective(
                grounding,
                model.text_hidden_fcs,
                states,
                theta + perturbation_scale * delta,
                args.weight_scale,
                images,
                target,
                args.coefficient_penalty,
            )
            loss_minus = video_objective(
                grounding,
                model.text_hidden_fcs,
                states,
                theta - perturbation_scale * delta,
                args.weight_scale,
                images,
                target,
                args.coefficient_penalty,
            )
            gradient = (
                (loss_plus - loss_minus) / (2.0 * perturbation_scale)
            ) * delta
            first_moment.mul_(0.9).add_(gradient, alpha=0.1)
            second_moment.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
            corrected_first = first_moment / (1.0 - 0.9 ** step)
            corrected_second = second_moment / (1.0 - 0.999 ** step)
            theta.addcdiv_(
                corrected_first,
                corrected_second.sqrt().add_(1e-8),
                value=-learning_rate,
            )
            theta.clamp_(-args.max_raw_weight, args.max_raw_weight)
            mean_loss = (loss_plus + loss_minus) / 2.0
            epoch_losses.append(mean_loss)
            effective = (args.weight_scale * theta.tanh()).detach().cpu().tolist()
            print(
                f"TRAIN epoch={epoch} [{position}/{len(order)}] "
                f"loss={mean_loss:.6f} plus={loss_plus:.6f} minus={loss_minus:.6f} "
                f"weights={effective}",
                flush=True,
            )
            checkpoint = {
                "state_dict": {
                    "raw_residual_weights": theta.detach().cpu().clone()
                },
                "layer_indices": layer_indices,
                "weight_scale": args.weight_scale,
                "optimizer": {
                    "step": step,
                    "first_moment": first_moment.detach().cpu(),
                    "second_moment": second_moment.detach().cpu(),
                },
                "history": history,
            }
            atomic_torch(output / "checkpoint_last.pt", checkpoint)
            atomic_json(status_path, {
                "state": "running",
                "phase": "official_video_training",
                "epoch": epoch,
                "epochs": args.epochs,
                "completed": position,
                "total": len(order),
                "step": step,
                "last_loss": mean_loss,
                "effective_residual_weights": effective,
            })
            del states, images, target, gt_numpy
            gc.collect()
            torch.cuda.empty_cache()

        history.append({
            "epoch": epoch,
            "mean_loss": float(np.mean(epoch_losses)),
            "raw_residual_weights": theta.detach().cpu().tolist(),
            "effective_residual_weights": (
                args.weight_scale * theta.tanh()
            ).detach().cpu().tolist(),
        })

    fusion = FinalAnchoredResidualFusion(
        layer_indices, weight_scale=args.weight_scale
    )
    fusion.raw_residual_weights.data.copy_(theta.detach().cpu())
    final_checkpoint = {
        "state_dict": fusion.state_dict(),
        "layer_indices": layer_indices,
        "weight_scale": args.weight_scale,
        "history": history,
        "training_protocol": "SPSA over exact official full-video inference",
    }
    atomic_torch(output / "checkpoint_best.pt", final_checkpoint)
    metrics = {
        "status": "complete",
        "protocol": "first-five-frame conditioning and exact full-video SAM2 propagation",
        "optimizer": "SPSA-Adam",
        "train_samples": len(events),
        "epochs": args.epochs,
        "layer_indices": list(layer_indices),
        "weight_scale": args.weight_scale,
        "training_history": history,
        "fusion": fusion.diagnostics(),
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
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "status.json", {
            "state": "failed",
            "phase": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        })
        raise


if __name__ == "__main__":
    main()
