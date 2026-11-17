#!/usr/bin/env python3
"""Train and evaluate a TGIF-style query router over Sa2VA [SEG] depths."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as cocomask
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from tools.multilayer_seg.evaluate_spatial_map_affine_gate import (
    build_event,
    infer_baseline_logits,
)
from tools.multilayer_seg.extract_spatial_adapter_features import extract_states
from tools.multilayer_seg.run_dynamic_k_oracle import exact_metrics
from tools.multilayer_seg.train_eval_fusion_pilot_v3 import (
    make_sam_cache,
    prompt_logits,
    segmentation_loss,
)
from tools.multilayer_seg.train_eval_fusion_standard import (
    cache_video_backbones,
    preprocess_video,
)
from tools.multilayer_seg.train_fusion_official_video_spsa import (
    collect_video_events,
)
from work.evaluate_layer_policy_oracle_feasibility import (
    atomic_json,
    atomic_torch,
    pack_binary_mask,
    paired_summary,
    unpack_binary_mask,
)


ALL_LAYERS = tuple(range(1, 29))
ACTION = "tgif_style_query_softmax_all28_no_point"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", nargs="?")
    parser.add_argument("--data-root")
    parser.add_argument("--feature-cache")
    parser.add_argument("--manifest")
    parser.add_argument("--reference-baseline-dir")
    parser.add_argument("--output")
    parser.add_argument("--train-samples", type=int, default=96)
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--pilot-samples", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--query-dim", type=int, default=16)
    parser.add_argument("--router-hidden-dim", type=int, default=16)
    parser.add_argument("--h28-prior", type=float, default=0.98)
    parser.add_argument("--load-balance-lambda", type=float, default=1e-4)
    parser.add_argument("--preservation-lambda", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


class TgifSegRouter(nn.Module):
    """Text-conditioned softmax fusion with an h28-safe initialization."""

    def __init__(
        self,
        query_dim: int,
        hidden_dim: int,
        layer_count: int = 28,
        h28_prior: float = 0.98,
    ) -> None:
        super().__init__()
        if layer_count < 2:
            raise ValueError("layer_count must be at least two")
        if not 0.5 < h28_prior < 1.0:
            raise ValueError("h28_prior must be in (0.5, 1.0)")
        self.layer_count = layer_count
        self.h28_prior = float(h28_prior)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, layer_count),
        )
        final = self.mlp[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        remaining = (1.0 - h28_prior) / float(layer_count - 1)
        prior = torch.full((layer_count,), remaining, dtype=torch.float32)
        prior[-1] = h28_prior
        self.register_buffer("prior_logits", prior.log())

    @staticmethod
    def rms_align(states: torch.Tensor) -> torch.Tensor:
        """Match every layer's RMS to h28 while preserving its direction."""
        if states.ndim < 2:
            raise ValueError("states must have [..., layers, hidden] shape")
        rms = states.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        anchor_rms = rms[..., -1:, :]
        aligned = states.float() * (anchor_rms / rms)
        # Avoid any numerical drift for the native expert.
        aligned[..., -1, :] = states[..., -1, :].float()
        return aligned

    def layer_weights(self, query: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(query.float()) + self.prior_logits
        return F.softmax(logits, dim=-1)

    def forward(
        self, query: torch.Tensor, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.layer_weights(query)
        aligned = self.rms_align(states)
        fused = torch.einsum("...l,...lh->...h", weights, aligned)
        return fused, weights


def text_embedding_module(model: Any) -> Any:
    candidates = [
        lambda: model.get_input_embeddings(),
        lambda: model.mllm.model.language_model.get_input_embeddings(),
        lambda: model.mllm.language_model.get_input_embeddings(),
        lambda: model.model.language_model.get_input_embeddings(),
    ]
    errors = []
    for getter in candidates:
        try:
            return getter()
        except Exception as error:
            errors.append(repr(error))
    raise RuntimeError(f"cannot locate LLM input embeddings: {errors}")


@torch.inference_mode()
def query_embedding(model: Any, tokenizer: Any, prompt: str) -> torch.Tensor:
    module = text_embedding_module(model)
    device = next(module.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = encoded["input_ids"].to(device)
    mask = encoded.get("attention_mask", torch.ones_like(ids)).to(device).bool()
    values = module(ids)[0].float()
    return values[mask[0]].mean(0).cpu()


def fit_query_preprocessor(query: torch.Tensor, components: int) -> dict[str, Any]:
    values = query.float().numpy()
    if components < 1 or len(values) < 2:
        raise ValueError("query PCA requires at least two samples and one component")
    # The formal run keeps the pre-registered 16 dimensions (96 train rows).
    # Tiny smoke runs cap the rank to the available sample count.
    components = min(components, len(values) - 1, values.shape[1])
    scaler = StandardScaler().fit(values)
    scaled = scaler.transform(values)
    pca = PCA(n_components=min(components, scaled.shape[1]), random_state=0).fit(scaled)
    return {
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "pca_components": pca.components_.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
    }


def transform_query(query: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    values = query.float().numpy().astype(np.float32)
    scaled = (values - state["scaler_mean"]) / np.maximum(
        state["scaler_scale"], 1e-6
    )
    reduced = (scaled - state["pca_mean"]) @ state["pca_components"].T
    return torch.from_numpy(np.asarray(reduced, dtype=np.float32))


def decode_frame_mask(
    mask_dict: dict[str, Any], annotation_ids: list[Any], frame_index: int,
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    result = np.zeros((height, width), dtype=np.uint8)
    for annotation_id in annotation_ids:
        track = mask_dict.get(str(annotation_id))
        if track is None or frame_index >= len(track) or not track[frame_index]:
            continue
        decoded = cocomask.decode(track[frame_index]).astype(np.uint8)
        if decoded.ndim == 3:
            decoded = decoded.max(axis=2)
        if decoded.shape != result.shape:
            raise RuntimeError(
                f"decoded GT shape {decoded.shape} does not match image {result.shape}"
            )
        result |= decoded
    return result


def make_training_example(
    event: dict[str, Any], mask_dict: dict[str, Any]
) -> dict[str, Any]:
    paths = event["frame_paths"][:5]
    if not paths:
        raise RuntimeError(f"event {event['key']} has no conditioning frame")
    with Image.open(paths[0]) as image:
        width, height = image.size
    masks = [
        decode_frame_mask(mask_dict, event["annotation_ids"], index, (height, width))
        for index in range(len(paths))
    ]
    areas = np.asarray([mask.sum() for mask in masks], dtype=np.int64)
    selected = int(np.argmax(areas))
    return {
        "key": event["key"],
        "video_id": event["video_id"],
        "frame_index": selected,
        "frame_path": paths[selected],
        "gt": masks[selected],
        "all_first_five_empty": bool(areas.max() == 0),
        "selected_area": int(areas[selected]),
    }


def normalized_preservation(fused: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    scale = anchor.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    return F.smooth_l1_loss(fused.float() / scale, anchor.float() / scale, beta=0.1)


def router_diagnostics(weights: torch.Tensor) -> dict[str, Any]:
    weights = weights.float().cpu()
    mean = weights.mean(0)
    std = weights.std(0, unbiased=False)
    entropy = -(weights * weights.clamp_min(1e-9).log()).sum(-1)
    order = torch.argsort(mean, descending=True)
    return {
        "mean_by_layer": {str(i + 1): float(value) for i, value in enumerate(mean)},
        "std_by_layer": {str(i + 1): float(value) for i, value in enumerate(std)},
        "top8_mean_layers": [int(index) + 1 for index in order[:8]],
        "h28_mean_weight": float(mean[-1]),
        "mean_query_layer_weight_std": float(std.mean()),
        "mean_effective_layer_count": float(entropy.exp().mean()),
        "mean_entropy_normalized": float(entropy.mean() / math.log(weights.shape[1])),
    }


@torch.no_grad()
def evaluate_prompt_loss(
    router: TgifSegRouter,
    model: Any,
    states: torch.Tensor,
    reduced_query: torch.Tensor,
    examples: list[dict[str, Any]],
    indices: list[int],
    shuffled_query: bool = False,
) -> tuple[float, torch.Tensor]:
    grounding = model.grounding_encoder
    processor = model.extra_image_processor
    dtype = model.torch_dtype
    projection = model.text_hidden_fcs
    projection_dtype = next(projection.parameters()).dtype
    query_values = reduced_query[indices].clone()
    if shuffled_query and len(query_values) > 1:
        query_values = query_values[torch.arange(len(query_values) - 1, -1, -1)]
    losses = []
    all_weights = []
    router.eval()
    for local, index in enumerate(indices):
        state = states[index].float().to("cuda:0")
        query = query_values[local].to("cuda:0")
        fused, weights = router(query[None], state[None])
        embedding = projection(fused[0].to(dtype=projection_dtype))
        cached = make_sam_cache(
            grounding, processor, dtype, examples[index]["frame_path"]
        )
        logits = prompt_logits(grounding, cached, embedding)
        target = torch.from_numpy(examples[index]["gt"].astype(np.float32)).to("cuda:0")
        loss, _, _ = segmentation_loss(logits, target)
        losses.append(float(loss))
        all_weights.append(weights[0].cpu())
        del state, query, fused, embedding, cached, logits, target, loss
        gc.collect()
        torch.cuda.empty_cache()
    return float(np.mean(losses)), torch.stack(all_weights)


@torch.no_grad()
def evaluate_native_loss(
    model: Any,
    states: torch.Tensor,
    examples: list[dict[str, Any]],
    indices: list[int],
) -> float:
    grounding = model.grounding_encoder
    processor = model.extra_image_processor
    dtype = model.torch_dtype
    projection = model.text_hidden_fcs
    projection_dtype = next(projection.parameters()).dtype
    losses = []
    for index in indices:
        anchor = states[index, -1].float().to("cuda:0")
        embedding = projection(anchor.to(dtype=projection_dtype))
        cached = make_sam_cache(
            grounding, processor, dtype, examples[index]["frame_path"]
        )
        logits = prompt_logits(grounding, cached, embedding)
        target = torch.from_numpy(examples[index]["gt"].astype(np.float32)).to("cuda:0")
        loss, _, _ = segmentation_loss(logits, target)
        losses.append(float(loss))
        del anchor, embedding, cached, logits, target, loss
        gc.collect()
        torch.cuda.empty_cache()
    return float(np.mean(losses))


def train_router(
    args: argparse.Namespace,
    model: Any,
    states: torch.Tensor,
    query: torch.Tensor,
    examples: list[dict[str, Any]],
    output: Path,
) -> tuple[TgifSegRouter, dict[str, Any], torch.Tensor, dict[str, Any]]:
    train_count = 4 if args.smoke else args.train_samples
    validation_count = 2 if args.smoke else args.validation_samples
    epochs = 1 if args.smoke else args.epochs
    train_indices = list(range(train_count))
    validation_indices = list(range(train_count, train_count + validation_count))
    preprocessor = fit_query_preprocessor(query[train_indices], args.query_dim)
    reduced = transform_query(query, preprocessor)
    router = TgifSegRouter(
        reduced.shape[1], args.router_hidden_dim, 28, args.h28_prior
    ).to("cuda:0")
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    projection = model.text_hidden_fcs
    projection_dtype = next(projection.parameters()).dtype
    grounding = model.grounding_encoder
    processor = model.extra_image_processor
    dtype = model.torch_dtype

    baseline_validation_loss = evaluate_native_loss(
        model, states, examples, validation_indices
    )
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    first_gradient_norm = None
    step = 0
    for epoch in range(1, epochs + 1):
        order = train_indices.copy()
        random.Random(args.seed + epoch).shuffle(order)
        router.train()
        epoch_losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            batch_states = states[batch].float().to("cuda:0")
            batch_query = reduced[batch].float().to("cuda:0")
            fused, weights = router(batch_query, batch_states)
            segmentation_terms = []
            for local, index in enumerate(batch):
                embedding = projection(fused[local].to(dtype=projection_dtype))
                cached = make_sam_cache(
                    grounding, processor, dtype, examples[index]["frame_path"]
                )
                logits = prompt_logits(grounding, cached, embedding)
                target = torch.from_numpy(
                    examples[index]["gt"].astype(np.float32)
                ).to("cuda:0")
                mask_loss, _, _ = segmentation_loss(logits, target)
                segmentation_terms.append(mask_loss)
            segmentation = torch.stack(segmentation_terms).mean()
            preservation = normalized_preservation(
                fused, batch_states[:, -1]
            )
            mean_weight = weights.mean(0)
            load_balance = (
                mean_weight * mean_weight.clamp_min(1e-9).log()
            ).sum()
            total = (
                segmentation
                + args.preservation_lambda * preservation
                + args.load_balance_lambda * load_balance
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if first_gradient_norm is None:
                squared = sum(
                    float(parameter.grad.float().square().sum())
                    for parameter in router.parameters()
                    if parameter.grad is not None
                )
                first_gradient_norm = math.sqrt(squared)
            torch.nn.utils.clip_grad_norm_(router.parameters(), 2.0)
            optimizer.step()
            step += 1
            epoch_losses.append(float(total.detach()))
            if step % 8 == 0 or start + len(batch) == len(order):
                print(
                    f"TGIF_TRAIN epoch={epoch} "
                    f"[{min(start + len(batch), len(order))}/{len(order)}] "
                    f"loss={epoch_losses[-1]:.6f} "
                    f"h28_weight={float(weights[:, -1].mean()):.6f}",
                    flush=True,
                )
            del batch_states, batch_query, fused, weights
            del segmentation_terms, segmentation, preservation, mean_weight
            del load_balance, total, embedding, cached, logits, target, mask_loss
            gc.collect()
            torch.cuda.empty_cache()

        validation_loss, validation_weights = evaluate_prompt_loss(
            router, model, states, reduced, examples, validation_indices
        )
        diagnostic = router_diagnostics(validation_weights)
        row = {
            "epoch": epoch,
            "mean_train_loss": float(np.mean(epoch_losses)),
            "validation_loss": validation_loss,
            "validation_relative_improvement_vs_h28": float(
                (baseline_validation_loss - validation_loss)
                / max(baseline_validation_loss, 1e-12)
            ),
            "router": diagnostic,
        }
        history.append(row)
        print(json.dumps({"TGIF_EPOCH": row}, ensure_ascii=False), flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in router.state_dict().items()
            }
        atomic_json(
            output / "status.json",
            {
                "state": "running",
                "phase": "router_training",
                "epoch": epoch,
                "epochs": epochs,
                "validation_loss": validation_loss,
            },
        )
    if best_state is None or best_epoch is None:
        raise RuntimeError("router training produced no checkpoint")
    router.load_state_dict(best_state)
    best_validation_loss, best_weights = evaluate_prompt_loss(
        router, model, states, reduced, examples, validation_indices
    )
    shuffled_loss, shuffled_weights = evaluate_prompt_loss(
        router,
        model,
        states,
        reduced,
        examples,
        validation_indices,
        shuffled_query=True,
    )
    diagnostic = router_diagnostics(best_weights)
    relative = (
        baseline_validation_loss - best_validation_loss
    ) / max(baseline_validation_loss, 1e-12)
    gates = {
        "validation_loss_relative_improvement_ge_1pct": relative >= 0.01,
        "mean_query_layer_weight_std_ge_0p002": diagnostic[
            "mean_query_layer_weight_std"
        ] >= 0.002,
        "checkpoint_finite": bool(
            np.isfinite(best_validation_loss)
            and all(torch.isfinite(value).all() for value in best_state.values())
        ),
    }
    metrics = {
        "status": "complete",
        "train_samples": train_count,
        "validation_samples": validation_count,
        "epochs": epochs,
        "trainable_parameters": sum(p.numel() for p in router.parameters()),
        "pretrained_trainable_parameters": 0,
        "first_gradient_norm": first_gradient_norm,
        "baseline_h28_validation_loss": baseline_validation_loss,
        "best_router_validation_loss": best_validation_loss,
        "relative_validation_loss_improvement": float(relative),
        "query_shuffled_validation_loss": shuffled_loss,
        "query_shuffled_minus_correct_loss": float(
            shuffled_loss - best_validation_loss
        ),
        "best_epoch": best_epoch,
        "history": history,
        "router": diagnostic,
        "shuffled_router": router_diagnostics(shuffled_weights),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "training_frame_empty_count": int(
            sum(examples[index]["all_first_five_empty"] for index in train_indices)
        ),
        "validation_frame_empty_count": int(
            sum(
                examples[index]["all_first_five_empty"]
                for index in validation_indices
            )
        ),
    }
    checkpoint = {
        "state_dict": best_state,
        "query_preprocessor": {
            key: torch.from_numpy(value)
            for key, value in preprocessor.items()
        },
        "query_dim": reduced.shape[1],
        "router_hidden_dim": args.router_hidden_dim,
        "layer_indices": ALL_LAYERS,
        "h28_prior": args.h28_prior,
        "best_epoch": best_epoch,
        "training_metrics": metrics,
    }
    atomic_torch(output / "checkpoint_best.pt", checkpoint)
    atomic_json(output / "training_metrics.json", metrics)
    return router, preprocessor, reduced, metrics


def segment_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if len(truth) == 0:
        return {"J": float("nan"), "F": float("nan"), "J&F": float("nan")}
    return {
        key: float(value) for key, value in exact_metrics(truth, prediction).items()
    }


def error_rates(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    axes = tuple(range(1, truth.ndim))
    pixels = float(np.prod(truth.shape[1:]))
    false_positive = np.logical_and(prediction, ~truth).sum(axis=axes) / pixels
    target_area = truth.sum(axis=axes)
    false_negative = np.logical_and(~prediction, truth).sum(axis=axes)
    false_negative = np.divide(
        false_negative,
        target_area,
        out=np.zeros_like(false_negative, dtype=float),
        where=target_area > 0,
    )
    return float(np.mean(false_positive)), float(np.mean(false_negative))


def protocol_hash(args: argparse.Namespace, manifest: dict[str, Any]) -> str:
    contract = {
        "split_hash": manifest["calibration_sha256"],
        "pilot_samples": args.pilot_samples,
        "layers": ALL_LAYERS,
        "fusion": "query_mlp_softmax_rms_aligned_raw_seg_then_native_projector",
        "h28_prior": args.h28_prior,
        "router_train": 4 if args.smoke else args.train_samples,
        "router_validation": 2 if args.smoke else args.validation_samples,
        "point_box_mask_map": "none",
        "gpu": 7,
    }
    raw = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def summarize_pilot(rows: list[dict[str, Any]], training: dict[str, Any]) -> dict[str, Any]:
    baseline = {
        metric: float(np.mean([row["baseline_h28"][metric] for row in rows]))
        for metric in ("J", "F", "J&F")
    }
    method = {
        metric: float(np.mean([row["candidate"]["metrics"][metric] for row in rows]))
        for metric in ("J", "F", "J&F")
    }
    deltas = [float(row["candidate"]["delta_J&F_points"]) for row in rows]
    paired = paired_summary(deltas)
    gates = {
        "mean_delta_jf_points_ge_0p2": paired["mean"] >= 0.2,
        "videos_improved_ge_5_of_8": paired["better_equal_worse"][0] >= 5,
        "no_catastrophic_decline_at_or_below_minus_10": paired[
            "catastrophic_at_or_below_minus_10"
        ] == 0,
        "all_native_baselines_pixel_exact_to_exp034": all(
            row["native_reference_xor_fraction"] == 0.0 for row in rows
        ),
    }
    return {
        "status": "complete",
        "action": ACTION,
        "samples": len(rows),
        "baseline_mean": baseline,
        "method_mean": method,
        "delta_jf_points": paired,
        "conditioning_delta_jf_points": paired_summary(
            [row["candidate"]["conditioning_delta_J&F_points"] for row in rows]
        ),
        "suffix_delta_jf_points": paired_summary(
            [row["candidate"]["suffix_delta_J&F_points"] for row in rows]
        ),
        "mean_conditioning_false_positive_delta_points": float(
            np.mean(
                [
                    row["candidate"]["conditioning_false_positive_delta_points"]
                    for row in rows
                ]
            )
        ),
        "mean_suffix_false_positive_delta_points": float(
            np.mean(
                [
                    row["candidate"]["suffix_false_positive_delta_points"]
                    for row in rows
                ]
            )
        ),
        "mean_conditioning_false_negative_delta_points": float(
            np.mean(
                [
                    row["candidate"]["conditioning_false_negative_delta_points"]
                    for row in rows
                ]
            )
        ),
        "mean_suffix_false_negative_delta_points": float(
            np.mean(
                [
                    row["candidate"]["suffix_false_negative_delta_points"]
                    for row in rows
                ]
            )
        ),
        "training": training,
        "gates": gates,
        "all_pilot_gates_pass": all(gates.values()),
        "claim_scope": "reused eight-video calibration directional pilot only",
        "sealed_test_access": False,
    }


def self_test() -> None:
    torch.manual_seed(0)
    router = TgifSegRouter(3, 4, layer_count=4, h28_prior=0.9)
    query = torch.randn(2, 3)
    states = torch.randn(2, 4, 5)
    fused, weights = router(query, states)
    assert fused.shape == (2, 5)
    assert weights.shape == (2, 4)
    assert torch.allclose(weights.sum(-1), torch.ones(2))
    expected = torch.tensor([0.1 / 3, 0.1 / 3, 0.1 / 3, 0.9])
    assert torch.allclose(weights[0], expected, atol=1e-6)
    aligned = router.rms_align(states)
    assert torch.equal(aligned[:, -1], states[:, -1].float())
    rms = aligned.square().mean(-1).sqrt()
    assert torch.allclose(rms, rms[:, -1:], atol=1e-5)
    prep_query = torch.randn(8, 9)
    prep = fit_query_preprocessor(prep_query[:6], 3)
    transformed = transform_query(prep_query, prep)
    assert transformed.shape == (8, 3)
    print(
        json.dumps(
            {
                "state": "self_test_passed",
                "action": ACTION,
                "prior_weights": weights[0].tolist(),
            },
            indent=2,
        )
    )


def run(args: argparse.Namespace) -> None:
    required = (
        args.model_path,
        args.data_root,
        args.feature_cache,
        args.manifest,
        args.reference_baseline_dir,
        args.output,
    )
    if not all(required):
        raise ValueError("all model/data/cache/manifest/reference/output paths are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.pilot_samples not in (1, 8):
        raise ValueError("pilot-samples must be 1 or 8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "status.json", {"state": "running", "phase": "cache_load"})

    payload = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    if tuple(payload["layer_indices"]) != ALL_LAYERS:
        raise RuntimeError("feature cache does not contain layers 1..28 in order")
    states = payload["states"].float()
    query = payload["query"].float()
    keys = list(payload["keys"])
    required_cache = (6 if args.smoke else args.train_samples + args.validation_samples)
    if len(keys) < required_cache:
        raise RuntimeError(f"feature cache has {len(keys)} rows, need {required_cache}")

    data_root = Path(args.data_root)
    reconstructed = collect_video_events(data_root, len(keys), 1, 20260818)
    event_by_key = {event["key"]: event for event in reconstructed}
    if set(keys) != set(event_by_key):
        missing = sorted(set(keys) - set(event_by_key))
        raise RuntimeError(f"cannot reconstruct cached events: {missing[:5]}")
    events = [event_by_key[key] for key in keys]
    if len({event["video_id"] for event in events}) != len(events):
        raise RuntimeError("router cache is not one unique video per row")
    mask_dict = json.loads(
        (data_root / "video_datas" / "mevis" / "train" / "mask_dict.json").read_text()
    )
    example_count = 6 if args.smoke else args.train_samples + args.validation_samples
    examples = [make_training_example(event, mask_dict) for event in events[:example_count]]
    atomic_json(
        output / "router_split.json",
        {
            "cache_keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
            "train_keys": keys[: 4 if args.smoke else args.train_samples],
            "validation_keys": keys[
                4 if args.smoke else args.train_samples : example_count
            ],
            "one_unique_video_per_row": True,
            "training_frame_policy": "maximum GT area among first five frames",
        },
    )

    manifest = json.loads(Path(args.manifest).read_text())
    selected_pilot = manifest["calibration"][: args.pilot_samples]
    pilot_video_ids = {str(item["video_id"]) for item in selected_pilot}
    router_video_ids = {example["video_id"] for example in examples}
    overlap = sorted(pilot_video_ids & router_video_ids)
    if overlap:
        raise RuntimeError(f"router train/validation overlaps pilot videos: {overlap}")
    protocol = protocol_hash(args, manifest)

    atomic_json(output / "status.json", {"state": "running", "phase": "model_load"})
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

    router, preprocessor, _, training_metrics = train_router(
        args, model, states, query, examples, output
    )
    allow_pilot = bool(training_metrics["all_gates_pass"] or args.smoke)
    if not allow_pilot:
        atomic_json(
            output / "summary.json",
            {
                "status": "complete",
                "phase": "training_gate_failed_no_full_video_pilot",
                "training": training_metrics,
                "sealed_test_access": False,
            },
        )
        atomic_json(
            output / "status.json",
            {
                "state": "complete",
                "phase": "training_gate_failed_no_full_video_pilot",
                "protocol_hash": protocol,
            },
        )
        return

    atomic_json(
        output / "status.json",
        {
            "state": "running",
            "phase": "full_video_candidate_freeze",
            "protocol_hash": protocol,
            "pilot_gt_access": False,
        },
    )
    row_dir = output / "rows"
    trajectory_dir = output / "frozen_trajectories"
    row_dir.mkdir(exist_ok=True)
    trajectory_dir.mkdir(exist_ok=True)
    split = data_root / "video_datas" / "mevis" / "train"
    metadata = json.loads((split / "meta_expressions.json").read_text())["videos"]
    reference_dir = Path(args.reference_baseline_dir)
    projection = model.text_hidden_fcs
    projection_dtype = next(projection.parameters()).dtype
    grounding = model.grounding_encoder
    frozen_paths = []
    router.eval()
    for number, item in enumerate(selected_pilot, 1):
        event = build_event(data_root, item, metadata)
        seg, _, _, _ = extract_states(model, tokenizer, event, 5, ALL_LAYERS)
        q = query_embedding(model, tokenizer, event["prompt"])
        q_reduced = transform_query(q[None], preprocessor).to("cuda:0")
        seg_gpu = seg.float().to("cuda:0")
        with torch.no_grad():
            fused, weights = router(q_reduced, seg_gpu[None])
            native_embedding = projection(seg_gpu[-1].to(dtype=projection_dtype)).reshape(-1)
            fused_embedding = projection(fused[0].to(dtype=projection_dtype)).reshape(-1)
            embedding_cosine = float(
                F.cosine_similarity(
                    native_embedding.float(), fused_embedding.float(), dim=0
                )
            )
            images, size = preprocess_video(
                grounding,
                model.extra_image_processor,
                model.torch_dtype,
                event["frame_paths"],
            )
            cache = cache_video_backbones(grounding, images)
            native_logits = infer_baseline_logits(
                grounding, images, cache, native_embedding, size
            )
            fused_logits = infer_baseline_logits(
                grounding, images, cache, fused_embedding, size
            )
        native_mask = native_logits.numpy() > 0
        fused_mask = fused_logits.numpy() > 0
        reference_path = reference_dir / f"{event['key']}.pt"
        if not reference_path.exists():
            raise FileNotFoundError(f"missing exp034 reference {reference_path}")
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        reference_mask = unpack_binary_mask(reference["baseline_mask"])
        reference_xor = float(np.logical_xor(native_mask, reference_mask).mean())
        if reference_xor != 0.0:
            raise RuntimeError(
                f"native h28 reproduction failed for {event['key']}: xor={reference_xor}"
            )
        trajectory = {
            "key": event["key"],
            "video_id": event["video_id"],
            "exp_id": event["exp_id"],
            "annotation_ids": event["annotation_ids"],
            "frame_count": len(images),
            "output_size_wh": list(size),
            "protocol_hash": protocol,
            "native_mask": pack_binary_mask(native_mask),
            "fused_mask": pack_binary_mask(fused_mask),
            "native_reference_xor_fraction": reference_xor,
            "layer_weights": weights[0].float().cpu(),
            "top8_layers": [
                int(index) + 1
                for index in torch.argsort(weights[0], descending=True)[:8].cpu()
            ],
            "h28_weight": float(weights[0, -1]),
            "native_fused_embedding_cosine": embedding_cosine,
            "native_embedding_norm": float(native_embedding.float().norm()),
            "fused_embedding_norm": float(fused_embedding.float().norm()),
            "contract": (
                "TGIF-style query MLP softmax over RMS-aligned raw l1-l28 SEG; "
                "native projector; no point/box/mask/map; frozen Sa2VA and SAM2"
            ),
            "pilot_gt_access": False,
            "sealed_test_access": False,
        }
        path = trajectory_dir / f"{event['key']}.pt"
        atomic_torch(path, trajectory)
        frozen_paths.append(path)
        print(
            f"TGIF_FREEZE [{number}/{len(selected_pilot)}] {event['key']} "
            f"h28_weight={float(weights[0,-1]):.6f} "
            f"embedding_cos={embedding_cosine:.6f}",
            flush=True,
        )
        del seg, q, q_reduced, seg_gpu, fused, weights
        del native_embedding, fused_embedding, images, cache
        del native_logits, fused_logits, native_mask, fused_mask
        gc.collect()
        torch.cuda.empty_cache()

    del router, model, tokenizer, grounding, projection
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(
        output / "status.json",
        {
            "state": "running",
            "phase": "all_pilot_candidates_frozen_before_gt",
            "completed": len(frozen_paths),
            "protocol_hash": protocol,
            "pilot_gt_access": False,
        },
    )

    masks = json.loads((split / "mask_dict.json").read_text())
    rows = []
    for path in frozen_paths:
        trajectory = torch.load(path, map_location="cpu", weights_only=False)
        width, height = trajectory["output_size_wh"]
        truth = np.zeros((trajectory["frame_count"], height, width), dtype=np.uint8)
        for frame_index in range(trajectory["frame_count"]):
            truth[frame_index] = decode_frame_mask(
                masks,
                trajectory["annotation_ids"],
                frame_index,
                (height, width),
            )
        truth = truth.astype(bool)
        native = unpack_binary_mask(trajectory["native_mask"])
        fused = unpack_binary_mask(trajectory["fused_mask"])
        native_metrics = segment_metrics(truth, native)
        fused_metrics = segment_metrics(truth, fused)
        native_conditioning = segment_metrics(truth[:5], native[:5])
        fused_conditioning = segment_metrics(truth[:5], fused[:5])
        native_suffix = segment_metrics(truth[5:], native[5:])
        fused_suffix = segment_metrics(truth[5:], fused[5:])
        native_cond_fp, native_cond_fn = error_rates(truth[:5], native[:5])
        fused_cond_fp, fused_cond_fn = error_rates(truth[:5], fused[:5])
        native_suffix_fp, native_suffix_fn = error_rates(truth[5:], native[5:])
        fused_suffix_fp, fused_suffix_fn = error_rates(truth[5:], fused[5:])
        candidate = {
            "candidate_id": ACTION,
            "metrics": fused_metrics,
            "delta_J&F_points": 100.0 * (
                fused_metrics["J&F"] - native_metrics["J&F"]
            ),
            "conditioning_metrics": fused_conditioning,
            "suffix_metrics": fused_suffix,
            "conditioning_delta_J&F_points": 100.0 * (
                fused_conditioning["J&F"] - native_conditioning["J&F"]
            ),
            "suffix_delta_J&F_points": 100.0 * (
                fused_suffix["J&F"] - native_suffix["J&F"]
            ),
            "conditioning_false_positive_delta_points": 100.0 * (
                fused_cond_fp - native_cond_fp
            ),
            "suffix_false_positive_delta_points": 100.0 * (
                fused_suffix_fp - native_suffix_fp
            ),
            "conditioning_false_negative_delta_points": 100.0 * (
                fused_cond_fn - native_cond_fn
            ),
            "suffix_false_negative_delta_points": 100.0 * (
                fused_suffix_fn - native_suffix_fn
            ),
        }
        row = {
            "key": trajectory["key"],
            "video_id": trajectory["video_id"],
            "exp_id": trajectory["exp_id"],
            "protocol_hash": protocol,
            "baseline_h28": native_metrics,
            "baseline_conditioning": native_conditioning,
            "baseline_suffix": native_suffix,
            "candidate": candidate,
            "native_reference_xor_fraction": trajectory[
                "native_reference_xor_fraction"
            ],
            "layer_weights": trajectory["layer_weights"].tolist(),
            "top8_layers": trajectory["top8_layers"],
            "h28_weight": trajectory["h28_weight"],
            "native_fused_embedding_cosine": trajectory[
                "native_fused_embedding_cosine"
            ],
            "native_embedding_norm": trajectory["native_embedding_norm"],
            "fused_embedding_norm": trajectory["fused_embedding_norm"],
            "pilot_gt_access_after_candidate_freeze": True,
            "sealed_test_access": False,
        }
        atomic_json(row_dir / f"{trajectory['key']}.json", row)
        rows.append(row)
        print(
            f"TGIF_SCORE {trajectory['key']} "
            f"delta={candidate['delta_J&F_points']:+.3f}pt "
            f"conditioning={candidate['conditioning_delta_J&F_points']:+.3f}pt "
            f"suffix={candidate['suffix_delta_J&F_points']:+.3f}pt",
            flush=True,
        )
    summary = summarize_pilot(rows, training_metrics)
    summary["protocol_hash"] = protocol
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "status.json",
        {
            "state": "complete",
            "phase": "done",
            "protocol_hash": protocol,
            "all_pilot_gates_pass": summary["all_pilot_gates_pass"],
            "sealed_test_access": False,
        },
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    output = Path(args.output or ".")
    try:
        run(args)
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            output / "status.json",
            {
                "state": "failed",
                "phase": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
