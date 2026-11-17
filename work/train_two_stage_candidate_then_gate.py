"""Train full-strength candidate fusion first, then freeze it and train admission."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from work.evaluate_layer_policy_oracle_feasibility import atomic_json, atomic_torch, paired_summary
from work.train_spatial_trajectory_admission import (
    AUX_LAYERS,
    CONDITIONING_FRAMES,
    EMA_ALPHA,
    INITIAL_RHO,
    LAYERS,
    RESIDUAL_CLIP,
    Example,
    load_examples,
    mean_metrics,
    score_logits,
    trajectory_mask_loss,
)


CANDIDATE_EPOCHS = 12
GATE_EPOCHS = 8
GATE_MAX = 0.25
INTERNAL_TRAIN_COUNT = 24
INTERNAL_DEVELOPMENT_COUNT = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument("--cache-root")
    parser.add_argument("--output")
    parser.add_argument("--candidate-epochs", type=int, default=CANDIDATE_EPOCHS)
    parser.add_argument("--gate-epochs", type=int, default=GATE_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--videos-per-step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def inverse_sigmoid(value: float) -> float:
    return math.log(value / (1.0 - value))


class CandidateFusion(nn.Module):
    """Query-modulated bounded spatial residual fusion with no admission rho."""

    def __init__(self, query_dim: int, channels: int = 8):
        super().__init__()
        self.query_dim = int(query_dim)
        self.channels = int(channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(6, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )
        self.film = nn.Linear(query_dim, 2 * channels)
        self.spatial_gate = nn.Conv2d(channels, 1, 1)
        self.layer_scorer = nn.Sequential(
            nn.Linear(4 * channels, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).reshape(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_x.transpose(-1, -2).contiguous())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.film.weight, std=1e-3)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)
        nn.init.normal_(self.layer_scorer[-1].weight, std=1e-3)
        nn.init.zeros_(self.layer_scorer[-1].bias)

    def boundary(self, probability: torch.Tensor) -> torch.Tensor:
        value = probability[:, None]
        gx = F.conv2d(value, self.sobel_x.to(value), padding=1)
        gy = F.conv2d(value, self.sobel_y.to(value), padding=1)
        return (torch.sqrt(gx.square() + gy.square() + 1e-12) / 4.0).clamp(0, 1)[:, 0]

    def plan(
        self, query: torch.Tensor, logits: torch.Tensor, *, query_enabled: bool = True
    ) -> dict[str, torch.Tensor]:
        if logits.ndim != 4 or logits.shape[0] != len(LAYERS):
            raise ValueError("logits must be [4,frames,height,width]")
        native_clipped = logits[-1].float().clamp(-RESIDUAL_CLIP, RESIDUAL_CLIP)
        probability = torch.sigmoid(native_clipped)
        uncertainty = 4.0 * probability * (1.0 - probability)
        boundary = self.boundary(probability)
        residuals = (logits[:-1].float() - logits[-1].float()).clamp(
            -RESIDUAL_CLIP, RESIDUAL_CLIP
        )
        inputs = []
        for residual in residuals:
            normalized = residual / RESIDUAL_CLIP
            inputs.append(
                torch.stack(
                    (
                        native_clipped / RESIDUAL_CLIP,
                        normalized,
                        uncertainty,
                        boundary,
                        uncertainty * normalized,
                        boundary * normalized,
                    ),
                    dim=1,
                )
            )
        spatial = torch.stack(inputs)
        auxiliary_count, frame_count, _, height, width = spatial.shape
        features = self.encoder(spatial.flatten(0, 1))
        if query_enabled:
            gamma, beta = self.film(query.float()).chunk(2)
            features = features * (1.0 + 0.1 * torch.tanh(gamma)[None, :, None, None])
            features = features + 0.1 * torch.tanh(beta)[None, :, None, None]
        features = features.reshape(auxiliary_count, frame_count, self.channels, height, width)
        gates = GATE_MAX * torch.tanh(self.spatial_gate(features.flatten(0, 1)))
        gates = gates.reshape(auxiliary_count, frame_count, height, width)
        descriptor = torch.cat(
            (features.mean((-2, -1)), features.amax((-2, -1))), dim=-1
        )
        video_descriptor = torch.cat((descriptor.mean(1), descriptor.amax(1)), dim=-1)
        layer_weights = F.softmax(self.layer_scorer(video_descriptor)[:, 0], dim=0)
        return {
            "native": logits[-1].float(),
            "residuals": residuals,
            "spatial_gates": gates,
            "descriptor": descriptor,
            "layer_weights": layer_weights,
        }

    def forward(
        self, query: torch.Tensor, logits: torch.Tensor, *, query_enabled: bool = True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        plan = self.plan(query, logits, query_enabled=query_enabled)
        residual = (
            plan["layer_weights"][:, None, None, None]
            * plan["spatial_gates"]
            * plan["residuals"]
        ).sum(0)
        return plan["native"] + residual, plan


class FrozenCandidateAdmission(nn.Module):
    """Causal frame-level gate trained only after candidate fusion is frozen."""

    def __init__(self, channels: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6 * channels + len(AUX_LAYERS), 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("prior_logit", torch.tensor(inverse_sigmoid(INITIAL_RHO)))

    @staticmethod
    def causal_ema(values: torch.Tensor) -> torch.Tensor:
        rows = []
        previous = values[0]
        for index in range(len(values)):
            previous = values[index] if index == 0 else EMA_ALPHA * previous + (1 - EMA_ALPHA) * values[index]
            rows.append(previous)
        return torch.stack(rows)

    def forward(self, plan: dict[str, torch.Tensor]) -> torch.Tensor:
        descriptor = plan["descriptor"].permute(1, 0, 2).reshape(
            plan["descriptor"].shape[1], -1
        )
        weights = plan["layer_weights"][None].expand(len(descriptor), -1)
        instant = torch.sigmoid(self.mlp(torch.cat((descriptor, weights), dim=-1))[:, 0] + self.prior_logit)
        return self.causal_ema(instant)


def native_loss(examples: list[Example], device: torch.device) -> float:
    with torch.no_grad():
        return float(
            np.mean(
                [
                    float(
                        trajectory_mask_loss(
                            item.train_logits[-1].float().to(device), item.truth_train.to(device)
                        )
                    )
                    for item in examples
                ]
            )
        )


def candidate_diagnostics(plan: dict[str, torch.Tensor]) -> dict[str, Any]:
    gates = plan["spatial_gates"].detach().float().cpu()
    weights = plan["layer_weights"].detach().float().cpu()
    return {
        "mean_abs_spatial_gate": float(gates.abs().mean()),
        "spatial_gate_std": float(gates.std(unbiased=False)),
        "spatial_gate_min": float(gates.min()),
        "spatial_gate_max": float(gates.max()),
        "layer_weights": {
            str(layer): float(weights[index]) for index, layer in enumerate(AUX_LAYERS)
        },
    }


@torch.no_grad()
def candidate_loss(
    model: CandidateFusion,
    examples: list[Example],
    device: torch.device,
    *,
    query_order: list[int] | None = None,
    query_enabled: bool = True,
) -> tuple[float, list[dict[str, Any]]]:
    model.eval()
    losses, diagnostics = [], []
    for index, item in enumerate(examples):
        query_index = index if query_order is None else query_order[index]
        fused, plan = model(
            examples[query_index].query.to(device),
            item.train_logits.float().to(device),
            query_enabled=query_enabled,
        )
        losses.append(float(trajectory_mask_loss(fused, item.truth_train.to(device))))
        diagnostics.append(candidate_diagnostics(plan))
    return float(np.mean(losses)), diagnostics


@torch.no_grad()
def gated_loss(
    candidate_model: CandidateFusion,
    gate: FrozenCandidateAdmission,
    examples: list[Example],
    device: torch.device,
) -> tuple[float, list[float]]:
    candidate_model.eval()
    gate.eval()
    losses, rho_means = [], []
    for item in examples:
        candidate, plan = candidate_model(item.query.to(device), item.train_logits.float().to(device))
        rho = gate(plan)
        fused = plan["native"] + rho[:, None, None] * (candidate - plan["native"])
        losses.append(float(trajectory_mask_loss(fused, item.truth_train.to(device))))
        rho_means.append(float(rho.mean()))
    return float(np.mean(losses)), rho_means


@torch.no_grad()
def native_resolution_outputs(
    candidate_model: CandidateFusion,
    gate: FrozenCandidateAdmission,
    query: torch.Tensor,
    item: Example,
    device: torch.device,
    *,
    query_enabled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    _, plan = candidate_model(
        query.to(device), item.train_logits.float().to(device), query_enabled=query_enabled
    )
    raw = item.logits.float().to(device)
    _, frames, height, width = raw.shape
    spatial_gates = F.interpolate(
        plan["spatial_gates"].flatten(0, 1)[:, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[:, 0].reshape(len(AUX_LAYERS), frames, height, width)
    residuals = (raw[:-1] - raw[-1]).clamp(-RESIDUAL_CLIP, RESIDUAL_CLIP)
    correction = (
        plan["layer_weights"][:, None, None, None] * spatial_gates * residuals
    ).sum(0)
    candidate = raw[-1] + correction
    rho = gate(plan)
    final = raw[-1] + rho[:, None, None] * correction
    diagnostics = candidate_diagnostics({**plan, "spatial_gates": spatial_gates})
    diagnostics.update(
        {
            "rho_mean": float(rho.mean()),
            "rho_std_frames": float(rho.std(unbiased=False)),
            "rho_min": float(rho.min()),
            "rho_max": float(rho.max()),
        }
    )
    return candidate.cpu(), final.cpu(), diagnostics


@torch.no_grad()
def evaluate_final(
    candidate_model: CandidateFusion,
    gate: FrozenCandidateAdmission,
    examples: list[Example],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    shuffled = list(range(len(examples) - 1, -1, -1))
    rows = []
    for index, item in enumerate(examples):
        candidate_logits, correct_logits, diagnostics = native_resolution_outputs(
            candidate_model, gate, item.query, item, device
        )
        _, shuffled_logits, _ = native_resolution_outputs(
            candidate_model, gate, examples[shuffled[index]].query, item, device
        )
        _, disabled_logits, _ = native_resolution_outputs(
            candidate_model, gate, item.query, item, device, query_enabled=False
        )
        native = score_logits(item.truth_full, item.logits[-1], item.output_size_wh)
        candidate = score_logits(item.truth_full, candidate_logits, item.output_size_wh)
        correct = score_logits(item.truth_full, correct_logits, item.output_size_wh)
        shuffled_score = score_logits(item.truth_full, shuffled_logits, item.output_size_wh)
        disabled = score_logits(item.truth_full, disabled_logits, item.output_size_wh)
        row = {
            "key": item.key,
            "native": native,
            "candidate": candidate,
            "correct": correct,
            "shuffled": shuffled_score,
            "query_disabled": disabled,
            "diagnostics": diagnostics,
            "delta_J&F_points": 100 * (correct["full"]["J&F"] - native["full"]["J&F"]),
            "candidate_delta_J&F_points": 100
            * (candidate["full"]["J&F"] - native["full"]["J&F"]),
            "conditioning_delta_J&F_points": 100
            * (correct["conditioning"]["J&F"] - native["conditioning"]["J&F"]),
            "suffix_delta_J&F_points": 100
            * (correct["suffix"]["J&F"] - native["suffix"]["J&F"]),
            "correct_minus_shuffled_J&F_points": 100
            * (correct["full"]["J&F"] - shuffled_score["full"]["J&F"]),
            "correct_minus_query_disabled_J&F_points": 100
            * (correct["full"]["J&F"] - disabled["full"]["J&F"]),
        }
        rows.append(row)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    deltas = [float(row["delta_J&F_points"]) for row in rows]
    best_index = int(np.argmax(deltas))
    leave_best_out = float(np.mean([x for i, x in enumerate(deltas) if i != best_index]))
    summary = {
        "samples": len(rows),
        "methods_full": {
            name: mean_metrics(rows, name)
            for name in ("native", "candidate", "correct", "shuffled", "query_disabled")
        },
        "delta_jf_points": paired_summary(deltas),
        "candidate_delta_jf_points": paired_summary(
            [float(row["candidate_delta_J&F_points"]) for row in rows]
        ),
        "conditioning_delta_jf_points": paired_summary(
            [float(row["conditioning_delta_J&F_points"]) for row in rows]
        ),
        "suffix_delta_jf_points": paired_summary(
            [float(row["suffix_delta_J&F_points"]) for row in rows]
        ),
        "correct_minus_shuffled_jf_points": paired_summary(
            [float(row["correct_minus_shuffled_J&F_points"]) for row in rows]
        ),
        "correct_minus_query_disabled_jf_points": paired_summary(
            [float(row["correct_minus_query_disabled_J&F_points"]) for row in rows]
        ),
        "leave_best_out_mean_delta_jf_points": leave_best_out,
        "mean_rho": float(np.mean([row["diagnostics"]["rho_mean"] for row in rows])),
        "mean_abs_spatial_gate": float(
            np.mean([row["diagnostics"]["mean_abs_spatial_gate"] for row in rows])
        ),
    }
    return summary, rows


def self_test() -> None:
    torch.manual_seed(0)
    candidate_model = CandidateFusion(16)
    gate = FrozenCandidateAdmission()
    query = torch.randn(16)
    logits = torch.randn(4, 7, 32, 32)
    candidate, plan = candidate_model(query, logits)
    assert torch.equal(candidate, logits[-1].float())
    assert torch.equal(plan["spatial_gates"], torch.zeros_like(plan["spatial_gates"]))
    target = torch.rand(7, 32, 32) > 0.5
    loss = trajectory_mask_loss(candidate, target)
    loss.backward()
    spatial_gradient = float(candidate_model.spatial_gate.weight.grad.square().sum())
    assert spatial_gradient > 0 and math.isfinite(spatial_gradient)
    candidate_model.zero_grad(set_to_none=True)
    with torch.no_grad():
        candidate_model.spatial_gate.bias.fill_(0.1)
    for parameter in candidate_model.parameters():
        parameter.requires_grad_(False)
    candidate, plan = candidate_model(query, logits)
    rho = gate(plan)
    assert 0.005 < float(rho.mean()) < 0.02
    final = plan["native"] + rho[:, None, None] * (candidate - plan["native"])
    gated = trajectory_mask_loss(final, target)
    gated.backward()
    gate_gradient = sum(
        float(parameter.grad.square().sum())
        for parameter in gate.parameters()
        if parameter.grad is not None
    )
    assert gate_gradient > 0 and math.isfinite(gate_gradient)
    print(
        json.dumps(
            {
                "state": "self_test_passed",
                "candidate_initial_exact_h28": True,
                "candidate_spatial_gradient_nonzero_without_rho": True,
                "initial_gate_rho": float(rho.mean()),
                "gate_gradient_nonzero_with_frozen_candidate": True,
                "internal_split": [24, 8],
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
        raise ValueError("data-root, cache-root, output required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cache_root = Path(args.cache_root)
    train_status = json.loads((cache_root / "status_headroom_train.json").read_text())
    validation_status = json.loads((cache_root / "status_headroom_validation.json").read_text())
    if train_status.get("completed") != 32 or train_status.get("gt_access") is not False:
        raise RuntimeError("32-video no-GT development cache incomplete")
    if validation_status.get("completed") != 16 or validation_status.get("gt_access") is not False:
        raise RuntimeError("16-video no-GT final cache incomplete")
    output = Path(args.output)
    rows_dir = output / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    mask_dict = json.loads(
        (Path(args.data_root) / "video_datas" / "mevis" / "train" / "mask_dict.json").read_text()
    )
    development_all = load_examples(cache_root, "headroom_train", mask_dict)
    internal_train = development_all[:INTERNAL_TRAIN_COUNT]
    internal_development = development_all[INTERNAL_TRAIN_COUNT:]
    if len(internal_train) != 24 or len(internal_development) != 8:
        raise RuntimeError("internal 24/8 split mismatch")
    query_dim = internal_train[0].query.numel()
    candidate_model = CandidateFusion(query_dim).to(device)
    optimizer = torch.optim.AdamW(candidate_model.parameters(), lr=args.learning_rate, weight_decay=0)
    native_train_loss = native_loss(internal_train, device)
    native_dev_loss = native_loss(internal_development, device)
    candidate_history = []
    candidate_best = math.inf
    candidate_best_epoch = 0
    candidate_checkpoint = output / "candidate_best.pt"
    atomic_json(
        output / "status.json",
        {
            "state": "running",
            "phase": "candidate_training_without_rho",
            "final_validation_gt_access": False,
            "prospective_access": False,
            "valid_access": False,
            "sealed_test_access": False,
        },
    )
    for epoch in range(1, args.candidate_epochs + 1):
        candidate_model.train()
        order = list(range(len(internal_train)))
        random.Random(args.seed + epoch).shuffle(order)
        objectives = []
        for start in range(0, len(order), args.videos_per_step):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in order[start : start + args.videos_per_step]:
                item = internal_train[index]
                fused, _ = candidate_model(
                    item.query.to(device), item.train_logits.float().to(device)
                )
                losses.append(trajectory_mask_loss(fused, item.truth_train.to(device)))
            objective = torch.stack(losses).mean()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(candidate_model.parameters(), 2.0)
            optimizer.step()
            objectives.append(float(objective.detach()))
        dev_loss, diagnostics = candidate_loss(candidate_model, internal_development, device)
        row = {
            "epoch": epoch,
            "train_objective": float(np.mean(objectives)),
            "development_loss": dev_loss,
            "development_mean_abs_spatial_gate": float(
                np.mean([item["mean_abs_spatial_gate"] for item in diagnostics])
            ),
        }
        candidate_history.append(row)
        print(
            f"EXP049_CANDIDATE epoch={epoch}/{args.candidate_epochs} "
            f"train={row['train_objective']:.8f} dev={dev_loss:.8f} "
            f"gate={row['development_mean_abs_spatial_gate']:.6f}",
            flush=True,
        )
        if dev_loss < candidate_best:
            candidate_best = dev_loss
            candidate_best_epoch = epoch
            atomic_torch(
                candidate_checkpoint,
                {
                    "state_dict": candidate_model.state_dict(),
                    "query_dim": query_dim,
                    "best_epoch": epoch,
                    "native_train_loss": native_train_loss,
                    "native_development_loss": native_dev_loss,
                },
            )
    selected_candidate = torch.load(candidate_checkpoint, map_location="cpu", weights_only=False)
    candidate_model.load_state_dict(selected_candidate["state_dict"])
    selected_train_loss, train_diagnostics = candidate_loss(candidate_model, internal_train, device)
    selected_dev_loss, dev_diagnostics = candidate_loss(candidate_model, internal_development, device)
    reversed_order = list(range(len(internal_development) - 1, -1, -1))
    shuffled_dev_loss, _ = candidate_loss(
        candidate_model, internal_development, device, query_order=reversed_order
    )
    disabled_dev_loss, _ = candidate_loss(
        candidate_model, internal_development, device, query_enabled=False
    )
    train_improvement = (native_train_loss - selected_train_loss) / max(native_train_loss, 1e-12)
    mean_abs_gate = float(np.mean([item["mean_abs_spatial_gate"] for item in dev_diagnostics]))
    fit_gates = {
        "internal_train_loss_relative_improvement_ge_1pct": train_improvement >= 0.01,
        "mean_absolute_spatial_gate_ge_0p005": mean_abs_gate >= 0.005,
    }
    candidate_summary = {
        "best_epoch": candidate_best_epoch,
        "native_internal_train_loss": native_train_loss,
        "selected_internal_train_loss": selected_train_loss,
        "internal_train_relative_improvement": float(train_improvement),
        "native_internal_development_loss": native_dev_loss,
        "selected_internal_development_loss": selected_dev_loss,
        "shuffled_internal_development_loss": shuffled_dev_loss,
        "query_disabled_internal_development_loss": disabled_dev_loss,
        "mean_abs_spatial_gate": mean_abs_gate,
        "history": candidate_history,
        "fit_gates": fit_gates,
        "fit_gate_pass": all(fit_gates.values()),
    }
    atomic_json(output / "candidate_summary.json", candidate_summary)
    if not all(fit_gates.values()):
        summary = {
            "status": "complete",
            "decision": "stop_before_admission_training",
            "candidate": candidate_summary,
            "final_validation_gt_access": False,
            "all_final_gates_pass": False,
            "prospective_access": False,
            "valid_access": False,
            "sealed_test_access": False,
        }
        atomic_json(output / "summary.json", summary)
        atomic_json(
            output / "status.json",
            {
                "state": "complete",
                "phase": "candidate_fit_gate_failed",
                "admission_trained": False,
                "final_validation_gt_access": False,
                "prospective_access": False,
                "valid_access": False,
                "sealed_test_access": False,
            },
        )
        return
    for parameter in candidate_model.parameters():
        parameter.requires_grad_(False)
    candidate_model.eval()
    gate = FrozenCandidateAdmission().to(device)
    gate_optimizer = torch.optim.AdamW(gate.parameters(), lr=args.learning_rate, weight_decay=0)
    gate_history = []
    gate_best = math.inf
    gate_best_epoch = 0
    gate_checkpoint = output / "gate_best.pt"
    for epoch in range(1, args.gate_epochs + 1):
        gate.train()
        order = list(range(len(internal_train)))
        random.Random(args.seed + 1000 + epoch).shuffle(order)
        objectives = []
        for start in range(0, len(order), args.videos_per_step):
            gate_optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in order[start : start + args.videos_per_step]:
                item = internal_train[index]
                with torch.no_grad():
                    candidate, plan = candidate_model(
                        item.query.to(device), item.train_logits.float().to(device)
                    )
                rho = gate(plan)
                fused = plan["native"] + rho[:, None, None] * (candidate - plan["native"])
                losses.append(trajectory_mask_loss(fused, item.truth_train.to(device)))
            objective = torch.stack(losses).mean()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 2.0)
            gate_optimizer.step()
            objectives.append(float(objective.detach()))
        dev_loss, rho_means = gated_loss(candidate_model, gate, internal_development, device)
        row = {
            "epoch": epoch,
            "train_objective": float(np.mean(objectives)),
            "development_loss": dev_loss,
            "development_mean_rho": float(np.mean(rho_means)),
        }
        gate_history.append(row)
        print(
            f"EXP049_GATE epoch={epoch}/{args.gate_epochs} train={row['train_objective']:.8f} "
            f"dev={dev_loss:.8f} rho={row['development_mean_rho']:.6f}",
            flush=True,
        )
        if dev_loss < gate_best:
            gate_best = dev_loss
            gate_best_epoch = epoch
            atomic_torch(
                gate_checkpoint,
                {"state_dict": gate.state_dict(), "best_epoch": epoch, "development_loss": dev_loss},
            )
    selected_gate = torch.load(gate_checkpoint, map_location="cpu", weights_only=False)
    gate.load_state_dict(selected_gate["state_dict"])
    # The final 16-video GT is opened only after both stages and checkpoints are frozen.
    final_examples = load_examples(cache_root, "headroom_validation", mask_dict)
    final_summary, final_rows = evaluate_final(candidate_model, gate, final_examples, device)
    for row in final_rows:
        atomic_json(rows_dir / f"{row['key']}.json", row)
    delta = final_summary["delta_jf_points"]
    final_gates = {
        "validation_mean_delta_jf_ge_0p5": delta["mean"] >= 0.5,
        "validation_suffix_delta_nonnegative": final_summary["suffix_delta_jf_points"]["mean"] >= 0,
        "validation_improved_videos_ge_9_of_16": delta["better_equal_worse"][0] >= 9,
        "no_validation_decline_at_or_below_minus_5": min(
            float(row["delta_J&F_points"]) for row in final_rows
        )
        > -5,
        "leave_best_out_mean_positive": final_summary["leave_best_out_mean_delta_jf_points"] > 0,
        "correct_minus_shuffled_ge_0p25": final_summary["correct_minus_shuffled_jf_points"]["mean"] >= 0.25,
        "correct_minus_query_disabled_ge_0p25": final_summary[
            "correct_minus_query_disabled_jf_points"
        ]["mean"]
        >= 0.25,
    }
    summary = {
        "status": "complete",
        "decision": "expand" if all(final_gates.values()) else "reject_expansion",
        "candidate": candidate_summary,
        "admission": {
            "best_epoch": gate_best_epoch,
            "history": gate_history,
            "best_internal_development_loss": gate_best,
        },
        "final_validation": final_summary,
        "final_gates": final_gates,
        "all_final_gates_pass": all(final_gates.values()),
        "final_validation_gt_access": True,
        "prospective_access": False,
        "valid_access": False,
        "sealed_test_access": False,
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "status.json",
        {
            "state": "complete",
            "phase": "two_stage_final_validation_scored",
            "candidate_fit_gate_pass": True,
            "admission_trained": True,
            "all_final_gates_pass": all(final_gates.values()),
            "final_validation_gt_access": True,
            "prospective_access": False,
            "valid_access": False,
            "sealed_test_access": False,
        },
    )


if __name__ == "__main__":
    main()
