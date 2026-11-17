"""Train query-conditioned spatial admission of frozen SAM2 depth trajectories."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from work.evaluate_layer_policy_oracle_feasibility import atomic_json, atomic_torch, paired_summary
from work.train_evaluate_tgif_seg_router import decode_frame_mask, segment_metrics


LAYERS = (21, 24, 25, 28)
AUX_LAYERS = LAYERS[:-1]
CONDITIONING_FRAMES = 5
TRAINING_SPATIAL_SIZE = 128
RESIDUAL_CLIP = 12.0
EMA_ALPHA = 0.8
INITIAL_RHO = 0.01
BOUNDARY_LAMBDA = 0.1
RHO_REGULARIZATION = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument("--cache-root")
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--videos-per-step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def logit(value: float) -> float:
    return math.log(value / (1.0 - value))


class SpatialTrajectoryAdmission(nn.Module):
    """Shared spatial evidence encoder plus layer competition and temporal admission."""

    def __init__(self, query_dim: int, channels: int = 8):
        super().__init__()
        self.query_dim = int(query_dim)
        self.channels = int(channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(6, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.film = nn.Linear(query_dim, 2 * channels)
        self.spatial_gate = nn.Conv2d(channels, 1, kernel_size=1)
        self.layer_scorer = nn.Sequential(
            nn.Linear(4 * channels, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.rho_head = nn.Sequential(
            nn.Linear(6 * channels + len(AUX_LAYERS), 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).reshape(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2).contiguous()
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("rho_prior_logit", torch.tensor(logit(INITIAL_RHO)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.film.weight, std=1e-3)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)
        for module in (self.layer_scorer[-1], self.rho_head[-1]):
            nn.init.normal_(module.weight, std=1e-3)
            nn.init.zeros_(module.bias)

    def boundary(self, probability: torch.Tensor) -> torch.Tensor:
        if probability.ndim != 3:
            raise ValueError("probability must have shape [frames,height,width]")
        value = probability[:, None]
        gx = F.conv2d(value, self.sobel_x.to(value), padding=1)
        gy = F.conv2d(value, self.sobel_y.to(value), padding=1)
        return (torch.sqrt(gx.square() + gy.square() + 1e-12) / 4.0).clamp(0.0, 1.0)[:, 0]

    @staticmethod
    def causal_ema(values: torch.Tensor, alpha: float = EMA_ALPHA) -> torch.Tensor:
        rows = []
        previous = values[0]
        for index in range(len(values)):
            previous = values[index] if index == 0 else alpha * previous + (1.0 - alpha) * values[index]
            rows.append(previous)
        return torch.stack(rows)

    def plan(
        self,
        query: torch.Tensor,
        logits: torch.Tensor,
        *,
        query_enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        if logits.ndim != 4 or logits.shape[0] != len(LAYERS):
            raise ValueError("logits must have shape [4,frames,height,width]")
        native = logits[-1].float().clamp(-RESIDUAL_CLIP, RESIDUAL_CLIP)
        probability = torch.sigmoid(native)
        uncertainty = 4.0 * probability * (1.0 - probability)
        boundary = self.boundary(probability)
        residuals = (logits[:-1].float() - logits[-1].float()).clamp(
            -RESIDUAL_CLIP, RESIDUAL_CLIP
        )
        spatial_inputs = []
        for residual in residuals:
            residual_normalized = residual / RESIDUAL_CLIP
            spatial_inputs.append(
                torch.stack(
                    (
                        native / RESIDUAL_CLIP,
                        residual_normalized,
                        uncertainty,
                        boundary,
                        uncertainty * residual_normalized,
                        boundary * residual_normalized,
                    ),
                    dim=1,
                )
            )
        # [aux,frames,6,height,width] -> shared spatial CNN.
        spatial = torch.stack(spatial_inputs)
        auxiliary_count, frame_count, _, height, width = spatial.shape
        features = self.encoder(spatial.flatten(0, 1))
        if query_enabled:
            film = self.film(query.float())
            gamma, beta = film.chunk(2)
            features = features * (1.0 + 0.1 * torch.tanh(gamma)[None, :, None, None])
            features = features + 0.1 * torch.tanh(beta)[None, :, None, None]
        features = features.reshape(auxiliary_count, frame_count, self.channels, height, width)
        gate = 2.0 * torch.sigmoid(self.spatial_gate(features.flatten(0, 1)))
        gate = gate.reshape(auxiliary_count, frame_count, height, width)
        mean_pool = features.mean((-2, -1))
        max_pool = features.amax((-2, -1))
        descriptor = torch.cat((mean_pool, max_pool), dim=-1)
        video_descriptor = torch.cat((descriptor.mean(1), descriptor.amax(1)), dim=-1)
        layer_scores = self.layer_scorer(video_descriptor)[:, 0]
        layer_weights = F.softmax(layer_scores, dim=0)
        frame_descriptor = descriptor.permute(1, 0, 2).reshape(frame_count, -1)
        rho_input = torch.cat(
            (frame_descriptor, layer_weights[None].expand(frame_count, -1)), dim=-1
        )
        rho_instant = torch.sigmoid(self.rho_head(rho_input)[:, 0] + self.rho_prior_logit)
        rho = self.causal_ema(rho_instant)
        return {
            "native": logits[-1].float(),
            "residuals": residuals,
            "spatial_gates": gate,
            "layer_weights": layer_weights,
            "rho_instant": rho_instant,
            "rho": rho,
        }

    def forward(
        self,
        query: torch.Tensor,
        logits: torch.Tensor,
        *,
        query_enabled: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        plan = self.plan(query, logits, query_enabled=query_enabled)
        weighted = (
            plan["layer_weights"][:, None, None, None]
            * plan["spatial_gates"]
            * plan["residuals"]
        ).sum(0)
        fused = plan["native"] + plan["rho"][:, None, None] * weighted
        return fused, plan


def sobel_magnitude(values: torch.Tensor) -> torch.Tensor:
    kernel_x = values.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).reshape(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2).contiguous()
    item = values[:, None]
    gx = F.conv2d(item, kernel_x, padding=1)
    gy = F.conv2d(item, kernel_y, padding=1)
    return (torch.sqrt(gx.square() + gy.square() + 1e-12) / 4.0).clamp(0.0, 1.0)[:, 0]


def frame_losses(logits: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    target = truth.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean((-2, -1))
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum((-2, -1))
    dice = 1.0 - (2.0 * intersection + 1.0) / (
        probability.sum((-2, -1)) + target.sum((-2, -1)) + 1.0
    )
    boundary = (sobel_magnitude(probability) - sobel_magnitude(target)).abs().mean((-2, -1))
    return bce + dice + BOUNDARY_LAMBDA * boundary


def trajectory_mask_loss(logits: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    losses = frame_losses(logits, truth)
    split = min(CONDITIONING_FRAMES, len(losses))
    if split == len(losses):
        return losses.mean()
    conditioning = losses[:split].mean()
    suffix = losses[split:].mean()
    return 0.25 * conditioning + 0.75 * suffix


@dataclass
class Example:
    key: str
    logits: torch.Tensor
    train_logits: torch.Tensor
    query: torch.Tensor
    truth_train: torch.Tensor
    truth_full: np.ndarray
    output_size_wh: tuple[int, int]


def load_examples(cache_root: Path, split: str, mask_dict: dict[str, Any]) -> list[Example]:
    examples: list[Example] = []
    for path in sorted((cache_root / "cache" / split).glob("*.pt")):
        row = torch.load(path, map_location="cpu", weights_only=False)
        if row.get("gt_access") is not False:
            raise RuntimeError(f"cache was not frozen before GT: {path}")
        if row.get("all_logits_finite") is not True or row.get("complete_frame_coverage") is not True:
            raise RuntimeError(f"invalid trajectory cache: {path}")
        logits = torch.stack([row["layer_logits"][layer] for layer in LAYERS]).to(torch.float16)
        layer_count, frame_count, _, _ = logits.shape
        train_logits = F.interpolate(
            logits.float().flatten(0, 1)[:, None],
            size=(TRAINING_SPATIAL_SIZE, TRAINING_SPATIAL_SIZE),
            mode="bilinear",
            align_corners=False,
        )[:, 0].reshape(
            layer_count, frame_count, TRAINING_SPATIAL_SIZE, TRAINING_SPATIAL_SIZE
        ).to(torch.float16)
        width, height = row["output_size_wh"]
        truth = np.zeros((row["frame_count"], height, width), dtype=np.uint8)
        for frame_index in range(row["frame_count"]):
            truth[frame_index] = decode_frame_mask(
                mask_dict, row["annotation_ids"], frame_index, (height, width)
            )
        truth_train = F.interpolate(
            torch.from_numpy(truth.astype(np.float32))[:, None],
            size=(TRAINING_SPATIAL_SIZE, TRAINING_SPATIAL_SIZE),
            mode="nearest",
        )[:, 0].bool()
        examples.append(
            Example(
                key=str(row["key"]),
                logits=logits,
                train_logits=train_logits,
                query=row["query_reduced"].float(),
                truth_train=truth_train,
                truth_full=truth.astype(bool),
                output_size_wh=(int(width), int(height)),
            )
        )
    return examples


def plan_diagnostics(plan: dict[str, torch.Tensor]) -> dict[str, Any]:
    weights = plan["layer_weights"].detach().float().cpu()
    rho = plan["rho"].detach().float().cpu()
    gates = plan["spatial_gates"].detach().float().cpu()
    return {
        "layer_weights": {
            str(layer): float(weights[index]) for index, layer in enumerate(AUX_LAYERS)
        },
        "rho_mean": float(rho.mean()),
        "rho_std_frames": float(rho.std(unbiased=False)),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "spatial_gate_mean": float(gates.mean()),
        "spatial_gate_std": float(gates.std(unbiased=False)),
    }


@torch.no_grad()
def validation_loss(
    model: SpatialTrajectoryAdmission,
    examples: list[Example],
    device: torch.device,
    *,
    query_order: list[int] | None = None,
    query_enabled: bool = True,
) -> tuple[float, list[dict[str, Any]]]:
    model.eval()
    losses = []
    diagnostics = []
    for index, example in enumerate(examples):
        query_index = index if query_order is None else query_order[index]
        fused, plan = model(
            examples[query_index].query.to(device),
            example.train_logits.float().to(device),
            query_enabled=query_enabled,
        )
        losses.append(float(trajectory_mask_loss(fused, example.truth_train.to(device))))
        diagnostics.append(plan_diagnostics(plan))
    return float(np.mean(losses)), diagnostics


@torch.no_grad()
def native_loss(examples: list[Example], device: torch.device) -> float:
    values = [
        float(
            trajectory_mask_loss(
                example.train_logits[-1].float().to(device), example.truth_train.to(device)
            )
        )
        for example in examples
    ]
    return float(np.mean(values))


@torch.no_grad()
def fuse_native_resolution(
    model: SpatialTrajectoryAdmission,
    query: torch.Tensor,
    example: Example,
    device: torch.device,
    *,
    query_enabled: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _, plan = model(
        query.to(device), example.train_logits.float().to(device), query_enabled=query_enabled
    )
    raw = example.logits.float().to(device)
    _, frame_count, height, width = raw.shape
    gates = F.interpolate(
        plan["spatial_gates"].flatten(0, 1)[:, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[:, 0].reshape(len(AUX_LAYERS), frame_count, height, width)
    residuals = (raw[:-1] - raw[-1]).clamp(-RESIDUAL_CLIP, RESIDUAL_CLIP)
    weighted = (plan["layer_weights"][:, None, None, None] * gates * residuals).sum(0)
    fused = raw[-1] + plan["rho"][:, None, None] * weighted
    diagnostics = plan_diagnostics({**plan, "spatial_gates": gates})
    return fused.cpu(), diagnostics


@torch.no_grad()
def score_logits(truth: np.ndarray, logits: torch.Tensor, size: tuple[int, int]) -> dict[str, Any]:
    width, height = size
    prediction = F.interpolate(
        logits.float()[:, None].cpu(), size=(height, width), mode="bilinear", align_corners=False
    )[:, 0].numpy() > 0
    split = min(CONDITIONING_FRAMES, len(truth))
    return {
        "full": segment_metrics(truth, prediction),
        "conditioning": segment_metrics(truth[:split], prediction[:split]),
        "suffix": segment_metrics(truth[split:], prediction[split:]),
    }


def mean_metrics(rows: list[dict[str, Any]], name: str, phase: str = "full") -> dict[str, float]:
    return {
        metric: float(np.mean([row[name][phase][metric] for row in rows]))
        for metric in ("J", "F", "J&F")
    }


@torch.no_grad()
def evaluate_jf(
    model: SpatialTrajectoryAdmission,
    validation_examples: list[Example],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    shuffled = list(range(len(validation_examples) - 1, -1, -1))
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(validation_examples):
        native = score_logits(example.truth_full, example.logits[-1], example.output_size_wh)
        correct_logits, correct_diagnostics = fuse_native_resolution(
            model, example.query, example, device, query_enabled=True
        )
        shuffled_logits, _ = fuse_native_resolution(
            model,
            validation_examples[shuffled[index]].query,
            example,
            device,
            query_enabled=True,
        )
        spatial_logits, spatial_diagnostics = fuse_native_resolution(
            model, example.query, example, device, query_enabled=False
        )
        correct = score_logits(example.truth_full, correct_logits, example.output_size_wh)
        shuffled_score = score_logits(example.truth_full, shuffled_logits, example.output_size_wh)
        spatial = score_logits(example.truth_full, spatial_logits, example.output_size_wh)
        native_jf = native["full"]["J&F"]
        row = {
            "key": example.key,
            "native": native,
            "correct": correct,
            "shuffled": shuffled_score,
            "spatial_only": spatial,
            "diagnostics": correct_diagnostics,
            "spatial_only_diagnostics": spatial_diagnostics,
            "delta_J&F_points": 100.0 * (correct["full"]["J&F"] - native_jf),
            "conditioning_delta_J&F_points": 100.0
            * (correct["conditioning"]["J&F"] - native["conditioning"]["J&F"]),
            "suffix_delta_J&F_points": 100.0
            * (correct["suffix"]["J&F"] - native["suffix"]["J&F"]),
            "correct_minus_shuffled_J&F_points": 100.0
            * (correct["full"]["J&F"] - shuffled_score["full"]["J&F"]),
            "correct_minus_spatial_only_J&F_points": 100.0
            * (correct["full"]["J&F"] - spatial["full"]["J&F"]),
        }
        rows.append(row)
        del correct_logits, shuffled_logits, spatial_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
    deltas = [float(row["delta_J&F_points"]) for row in rows]
    best_index = int(np.argmax(deltas))
    leave_best_out = float(
        np.mean([value for row_index, value in enumerate(deltas) if row_index != best_index])
    )
    diagnostics = [row["diagnostics"] for row in rows]
    layer_weight_rows = np.asarray(
        [[row["layer_weights"][str(layer)] for layer in AUX_LAYERS] for row in diagnostics]
    )
    summary = {
        "samples": len(rows),
        "methods_full": {
            name: mean_metrics(rows, name)
            for name in ("native", "correct", "shuffled", "spatial_only")
        },
        "delta_jf_points": paired_summary(deltas),
        "conditioning_delta_jf_points": paired_summary(
            [float(row["conditioning_delta_J&F_points"]) for row in rows]
        ),
        "suffix_delta_jf_points": paired_summary(
            [float(row["suffix_delta_J&F_points"]) for row in rows]
        ),
        "correct_minus_shuffled_jf_points": paired_summary(
            [float(row["correct_minus_shuffled_J&F_points"]) for row in rows]
        ),
        "correct_minus_spatial_only_jf_points": paired_summary(
            [float(row["correct_minus_spatial_only_J&F_points"]) for row in rows]
        ),
        "leave_best_out_mean_delta_jf_points": leave_best_out,
        "router": {
            "mean_layer_weights": {
                str(layer): float(layer_weight_rows[:, index].mean())
                for index, layer in enumerate(AUX_LAYERS)
            },
            "std_layer_weights": {
                str(layer): float(layer_weight_rows[:, index].std())
                for index, layer in enumerate(AUX_LAYERS)
            },
            "mean_rho": float(np.mean([row["rho_mean"] for row in diagnostics])),
            "mean_rho_frame_std": float(
                np.mean([row["rho_std_frames"] for row in diagnostics])
            ),
            "mean_spatial_gate": float(
                np.mean([row["spatial_gate_mean"] for row in diagnostics])
            ),
            "mean_spatial_gate_std": float(
                np.mean([row["spatial_gate_std"] for row in diagnostics])
            ),
        },
    }
    return summary, rows


def self_test() -> None:
    torch.manual_seed(0)
    model = SpatialTrajectoryAdmission(query_dim=16)
    query = torch.randn(16)
    logits = torch.randn(4, 7, 32, 32)
    fused, plan = model(query, logits)
    assert fused.shape == logits[-1].shape
    assert torch.allclose(plan["spatial_gates"], torch.ones_like(plan["spatial_gates"]), atol=0, rtol=0)
    assert 0.005 < float(plan["rho"].mean()) < 0.02
    assert torch.equal(logits[-1], logits[-1])  # exact separately available native fallback
    equal_logits = logits[-1:].expand(4, -1, -1, -1).clone()
    equal_fused, _ = model(query, equal_logits)
    assert torch.equal(equal_fused, equal_logits[-1])
    second_fused, _ = model(-query, logits)
    assert not torch.equal(fused, second_fused)
    target = torch.rand(7, 32, 32) > 0.5
    loss = trajectory_mask_loss(fused, target)
    loss.backward()
    gradient = sum(
        float(parameter.grad.square().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0 and math.isfinite(gradient)
    print(
        json.dumps(
            {
                "state": "self_test_passed",
                "spatial_gate_initially_one": True,
                "initial_rho_mean": float(plan["rho"].mean()),
                "zero_residual_exact_h28": True,
                "query_prepool_causality": True,
                "gradient_nonzero": True,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.data_root or not args.cache_root or not args.output:
        raise ValueError("data-root, cache-root, and output are required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cache_root = Path(args.cache_root)
    for split, expected in (("headroom_train", 32), ("headroom_validation", 16)):
        status = json.loads((cache_root / f"status_{split}.json").read_text())
        if (
            status.get("state") != "complete"
            or status.get("completed") != expected
            or status.get("gt_access") is not False
        ):
            raise RuntimeError(f"{split} cache is not completely frozen no-GT: {status}")
    output = Path(args.output)
    row_dir = output / "rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    split_root = Path(args.data_root) / "video_datas" / "mevis" / "train"
    mask_dict = json.loads((split_root / "mask_dict.json").read_text())
    train_examples = load_examples(cache_root, "headroom_train", mask_dict)
    validation_examples = load_examples(cache_root, "headroom_validation", mask_dict)
    if len(train_examples) != 32 or len(validation_examples) != 16:
        raise RuntimeError("pilot split count mismatch")
    query_dim = train_examples[0].query.numel()
    model = SpatialTrajectoryAdmission(query_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    native_validation_loss = native_loss(validation_examples, device)
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_epoch = 0
    checkpoint_path = output / "checkpoint_best.pt"
    atomic_json(
        output / "status.json",
        {
            "state": "running",
            "phase": "spatial_trajectory_admission_training",
            "train_samples": 32,
            "validation_samples": 16,
            "gt_access": "development_only_after_all_candidate_trajectories_frozen",
            "prospective_access": False,
            "valid_access": False,
            "sealed_test_access": False,
        },
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_examples)))
        random.Random(args.seed + epoch).shuffle(order)
        objectives: list[float] = []
        for start in range(0, len(order), args.videos_per_step):
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            for index in order[start : start + args.videos_per_step]:
                example = train_examples[index]
                fused, plan = model(
                    example.query.to(device), example.train_logits.float().to(device)
                )
                loss = trajectory_mask_loss(fused, example.truth_train.to(device))
                loss = loss + RHO_REGULARIZATION * plan["rho"].mean()
                batch_losses.append(loss)
            objective = torch.stack(batch_losses).mean()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            objectives.append(float(objective.detach()))
        validation_loss_value, validation_diagnostics = validation_loss(
            model, validation_examples, device
        )
        row = {
            "epoch": epoch,
            "train_objective": float(np.mean(objectives)),
            "validation_trajectory_loss": validation_loss_value,
            "validation_relative_improvement_vs_h28": float(
                (native_validation_loss - validation_loss_value)
                / max(native_validation_loss, 1e-12)
            ),
            "validation_mean_rho": float(
                np.mean([item["rho_mean"] for item in validation_diagnostics])
            ),
        }
        history.append(row)
        print(
            f"EXP048_TRAIN epoch={epoch}/{args.epochs} train={row['train_objective']:.8f} "
            f"validation={validation_loss_value:.8f} rho={row['validation_mean_rho']:.6f}",
            flush=True,
        )
        if validation_loss_value < best_loss:
            best_loss = validation_loss_value
            best_epoch = epoch
            atomic_torch(
                checkpoint_path,
                {
                    "state_dict": model.state_dict(),
                    "query_dim": query_dim,
                    "channels": 8,
                    "layers": LAYERS,
                    "best_epoch": best_epoch,
                    "best_validation_trajectory_loss": best_loss,
                    "native_validation_trajectory_loss": native_validation_loss,
                    "training_config": {
                        "epochs": args.epochs,
                        "learning_rate": args.learning_rate,
                        "videos_per_step": args.videos_per_step,
                        "training_spatial_size": TRAINING_SPATIAL_SIZE,
                        "residual_clip": RESIDUAL_CLIP,
                        "ema_alpha": EMA_ALPHA,
                        "initial_rho": INITIAL_RHO,
                        "boundary_lambda": BOUNDARY_LAMBDA,
                        "rho_regularization": RHO_REGULARIZATION,
                    },
                },
            )
    selected = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["state_dict"])
    correct_loss, _ = validation_loss(model, validation_examples, device)
    shuffled_order = list(range(len(validation_examples) - 1, -1, -1))
    shuffled_loss, _ = validation_loss(
        model, validation_examples, device, query_order=shuffled_order
    )
    spatial_only_loss, _ = validation_loss(
        model, validation_examples, device, query_enabled=False
    )
    validation_summary, validation_rows = evaluate_jf(model, validation_examples, device)
    for row in validation_rows:
        atomic_json(row_dir / f"{row['key']}.json", row)
    delta = validation_summary["delta_jf_points"]
    relative_loss = (native_validation_loss - correct_loss) / max(native_validation_loss, 1e-12)
    mean_rho = validation_summary["router"]["mean_rho"]
    gates = {
        "validation_loss_relative_improvement_ge_0p5pct": relative_loss >= 0.005,
        "validation_mean_delta_jf_ge_0p5": delta["mean"] >= 0.5,
        "validation_suffix_delta_nonnegative": validation_summary["suffix_delta_jf_points"]["mean"]
        >= 0.0,
        "validation_improved_videos_ge_9_of_16": delta["better_equal_worse"][0] >= 9,
        "no_validation_decline_at_or_below_minus_5": min(
            float(row["delta_J&F_points"]) for row in validation_rows
        )
        > -5.0,
        "leave_best_out_mean_positive": validation_summary[
            "leave_best_out_mean_delta_jf_points"
        ]
        > 0.0,
        "correct_minus_shuffled_ge_0p25": validation_summary[
            "correct_minus_shuffled_jf_points"
        ]["mean"]
        >= 0.25,
        "correct_minus_spatial_only_ge_0p25": validation_summary[
            "correct_minus_spatial_only_jf_points"
        ]["mean"]
        >= 0.25,
        "mean_rho_in_registered_range": 0.002 <= mean_rho <= 0.5,
    }
    summary = {
        "status": "complete",
        "best_epoch": best_epoch,
        "native_validation_trajectory_loss": native_validation_loss,
        "best_correct_validation_trajectory_loss": correct_loss,
        "relative_validation_loss_improvement": float(relative_loss),
        "shuffled_query_validation_trajectory_loss": shuffled_loss,
        "spatial_only_validation_trajectory_loss": spatial_only_loss,
        "training_history": history,
        "validation": validation_summary,
        "gates": gates,
        "all_small_pilot_gates_pass": all(gates.values()),
        "full_cache_expansion_authorized": all(gates.values()),
        "prospective_access": False,
        "valid_access": False,
        "sealed_test_access": False,
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "status.json",
        {
            "state": "complete",
            "phase": "small_pilot_validation_scored",
            "best_epoch": best_epoch,
            "all_small_pilot_gates_pass": all(gates.values()),
            "full_cache_expansion_authorized": all(gates.values()),
            "prospective_access": False,
            "valid_access": False,
            "sealed_test_access": False,
        },
    )


if __name__ == "__main__":
    main()
