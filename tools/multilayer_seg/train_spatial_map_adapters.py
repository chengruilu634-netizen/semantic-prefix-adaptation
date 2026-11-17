#!/usr/bin/env python3
"""Train layer-specific SEG/image-token alignment and mask-logit calibration heads."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache")
    parser.add_argument("--output")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


class SpatialMapAdapters(nn.Module):
    def __init__(self, layers: int, input_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.q_projections = nn.ModuleList(
            [nn.Linear(input_dim, projection_dim, bias=False) for _ in range(layers)]
        )
        self.k_projections = nn.ModuleList(
            [nn.Linear(input_dim, projection_dim, bias=False) for _ in range(layers)]
        )
        for q_projection, k_projection in zip(self.q_projections, self.k_projections):
            nn.init.normal_(q_projection.weight, std=1.0 / math.sqrt(input_dim))
            k_projection.weight.data.copy_(q_projection.weight.data)
        self.log_temperature = nn.Parameter(torch.full((layers,), math.log(5.0)))
        self.bias = nn.Parameter(torch.full((layers,), -1.5))

    def forward(self, seg: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        outputs = []
        for layer, (q_projection, k_projection) in enumerate(
            zip(self.q_projections, self.k_projections)
        ):
            q_input = F.layer_norm(seg[:, layer].float(), (seg.shape[-1],))
            k_input = F.layer_norm(image[:, layer].float(), (image.shape[-1],))
            q = F.normalize(q_projection(q_input), dim=-1, eps=1e-6)
            k = F.normalize(k_projection(k_input), dim=-1, eps=1e-6)
            similarity = torch.einsum("bd,bfpd->bfp", q, k)
            logits = similarity * self.log_temperature[layer].exp().clamp(0.25, 40.0)
            logits = logits + self.bias[layer]
            outputs.append(logits)
        return torch.stack(outputs, dim=1)

    def alignment_regularizer(self) -> torch.Tensor:
        values = [
            (q.weight - k.weight).square().mean()
            for q, k in zip(self.q_projections, self.k_projections)
        ]
        return torch.stack(values).mean()


def map_loss(
    logits: torch.Tensor, targets: torch.Tensor, valid_frames: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    target = targets.float().flatten(2)[:, None].expand_as(logits)
    valid = valid_frames[:, None, :, None].expand_as(logits)
    positive_fraction = target.mean(dim=-1, keepdim=True)
    positive_weight = ((1.0 - positive_fraction) / positive_fraction.clamp_min(1e-3)).clamp(1.0, 20.0)
    bce = -(
        positive_weight * target * F.logsigmoid(logits)
        + (1.0 - target) * F.logsigmoid(-logits)
    )
    bce = (bce * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
    probabilities = logits.sigmoid()
    intersection = (probabilities * target * valid).sum(dim=-1)
    denominator = ((probabilities + target) * valid).sum(dim=-1)
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    per_layer = (bce + dice).mean(dim=(0, 2))
    return per_layer.mean(), per_layer


def raw_logits(seg: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    seg = F.normalize(seg.float(), dim=-1, eps=1e-6)
    image = F.normalize(image.float(), dim=-1, eps=1e-6)
    similarity = torch.einsum("bld,blfpd->blfp", seg, image)
    mean = similarity.mean(dim=-1, keepdim=True)
    std = similarity.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-5)
    standardized = (similarity - mean) / std
    boundary = torch.quantile(standardized, 0.75, dim=-1, keepdim=True)
    return (standardized - boundary) * 2.0


@torch.inference_mode()
def evaluate(
    model: SpatialMapAdapters | None,
    seg: torch.Tensor,
    image: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    all_logits = []
    for start in range(0, len(indices), batch_size):
        index = indices[start : start + batch_size]
        batch_seg = seg[index].to(device, non_blocking=True)
        batch_image = image[index].to(device, non_blocking=True)
        logits = raw_logits(batch_seg, batch_image) if model is None else model(batch_seg, batch_image)
        all_logits.append(logits.cpu())
    logits = torch.cat(all_logits)
    selected_targets = targets[indices].float()
    selected_valid = valid[indices]
    total_loss, per_layer_loss = map_loss(logits, selected_targets, selected_valid)
    expanded_target = selected_targets.flatten(2)[:, None].expand_as(logits)
    expanded_valid = selected_valid[:, None, :, None].expand_as(logits)
    probabilities = logits.sigmoid()
    intersection = (probabilities * expanded_target * expanded_valid).sum(dim=(0, 2, 3))
    denominator = ((probabilities + expanded_target) * expanded_valid).sum(dim=(0, 2, 3))
    soft_dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    prediction = probabilities >= 0.5
    foreground = expanded_target >= 0.5
    hard_intersection = (prediction & foreground & expanded_valid).sum(dim=(0, 2, 3)).float()
    hard_union = ((prediction | foreground) & expanded_valid).sum(dim=(0, 2, 3)).float()
    hard_iou = hard_intersection / hard_union.clamp_min(1.0)
    return {
        "loss": float(total_loss),
        "per_layer_loss": per_layer_loss.tolist(),
        "soft_dice": soft_dice.tolist(),
        "hard_iou": hard_iou.tolist(),
        "logits": logits.half(),
    }


def synthetic_self_test() -> None:
    torch.manual_seed(7)
    model = SpatialMapAdapters(4, 32, 8)
    seg = torch.randn(3, 4, 32)
    image = torch.randn(3, 4, 5, 16, 32)
    targets = (torch.rand(3, 5, 16) > 0.8).float()
    valid = torch.ones(3, 5, dtype=torch.bool)
    logits = model(seg, image)
    assert logits.shape == (3, 4, 5, 16)
    loss, per_layer = map_loss(logits, targets, valid)
    assert per_layer.shape == (4,) and torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    print("SELF_TEST_OK", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        synthetic_self_test()
        return
    if not args.feature_cache or not args.output:
        raise ValueError("--feature-cache and --output are required unless --self-test is used")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status = output / "status.json"
    atomic_json(status, {"state": "running", "phase": "load_cache"})
    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    seg = cache["seg_states"]
    image = cache["image_states"]
    targets = cache["targets"]
    valid = cache["valid_frames"]
    layers = [int(layer) for layer in cache["layers"]]
    train_index = torch.tensor([i for i, split in enumerate(cache["splits"]) if split == "train"])
    validation_index = torch.tensor(
        [i for i, split in enumerate(cache["splits"]) if split == "validation"]
    )
    if not len(train_index) or not len(validation_index):
        raise RuntimeError("cache must contain non-empty train and validation splits")
    train_videos = {cache["video_ids"][i] for i in train_index.tolist()}
    validation_videos = {cache["video_ids"][i] for i in validation_index.tolist()}
    if train_videos & validation_videos:
        raise RuntimeError("train/validation video leakage detected")

    model = SpatialMapAdapters(len(layers), seg.shape[-1], args.projection_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    raw_validation = evaluate(
        None, seg, image, targets, valid, validation_index, args.batch_size, device
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    generator = torch.Generator().manual_seed(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_index[torch.randperm(len(train_index), generator=generator)]
        epoch_losses = []
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            batch_seg = seg[index].to(device, non_blocking=True)
            batch_image = image[index].to(device, non_blocking=True)
            batch_target = targets[index].to(device, non_blocking=True)
            batch_valid = valid[index].to(device, non_blocking=True)
            logits = model(batch_seg, batch_image)
            loss, _ = map_loss(logits, batch_target, batch_valid)
            loss = loss + 1e-3 * model.alignment_regularizer()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))

        model.eval()
        validation = evaluate(
            model, seg, image, targets, valid, validation_index, args.batch_size, device
        )
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_loss": validation["loss"],
        })
        improved = validation["loss"] < best_loss - 1e-5
        if improved:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        atomic_json(status, {
            "state": "running", "phase": "adapter_training", "epoch": epoch,
            "best_epoch": best_epoch, "best_validation_loss": best_loss,
        })
        print(
            f"ADAPTER_TRAIN epoch={epoch:03d} train={np.mean(epoch_losses):.5f} "
            f"val={validation['loss']:.5f} best={best_loss:.5f}@{best_epoch}",
            flush=True,
        )
        if epoch - best_epoch >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    trained_train = evaluate(model, seg, image, targets, valid, train_index, args.batch_size, device)
    trained_validation = evaluate(
        model, seg, image, targets, valid, validation_index, args.batch_size, device
    )
    checkpoint = {
        "layers": layers,
        "input_dim": int(seg.shape[-1]),
        "projection_dim": args.projection_dim,
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "best_epoch": best_epoch,
        "feature_contract": cache.get("contract"),
    }
    torch.save(checkpoint, output / "spatial_map_adapters.pt")
    torch.save(
        {
            "keys": [cache["keys"][i] for i in validation_index.tolist()],
            "layers": layers,
            "logits": trained_validation.pop("logits"),
            "targets": targets[validation_index],
            "valid_frames": valid[validation_index],
        },
        output / "validation_maps.pt",
    )
    raw_validation.pop("logits")
    trained_train.pop("logits")
    summary = {
        "train_samples": len(train_index),
        "validation_samples": len(validation_index),
        "video_disjoint": True,
        "layers": layers,
        "best_epoch": best_epoch,
        "raw_validation": raw_validation,
        "trained_validation": trained_validation,
        "trained_train": trained_train,
        "per_layer": {
            str(layer): {
                "raw_loss": raw_validation["per_layer_loss"][i],
                "trained_loss": trained_validation["per_layer_loss"][i],
                "raw_soft_dice": raw_validation["soft_dice"][i],
                "trained_soft_dice": trained_validation["soft_dice"][i],
                "raw_hard_iou": raw_validation["hard_iou"][i],
                "trained_hard_iou": trained_validation["hard_iou"][i],
            }
            for i, layer in enumerate(layers)
        },
        "contract": {
            "frozen": "Sa2VA LLM, official h28 projector, and SAM2",
            "trained": "per-layer Wq/Wk, temperature, and bias",
            "supervision": "first-five-frame Gaussian-softened GT masks; BCE + Dice",
            "next_gate": "run frozen-SAM2 full-video evaluation only if held-out map metrics improve",
        },
        "history": history,
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(status, {
        "state": "complete", "phase": "done", "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
    })
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
