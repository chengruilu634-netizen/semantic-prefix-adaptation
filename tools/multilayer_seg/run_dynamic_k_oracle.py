#!/usr/bin/env python3
"""Bounded GT oracle for dynamic SEG-layer count and layer identity.

This is a feasibility upper bound, not a deployable router: ground truth is used
for the single-frame proxy and for the final per-query K/layer-set selection.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from third_parts.revos.utils.metircs import db_eval_boundary, db_eval_iou
from tools.multilayer_seg.dynamic_oracle import (
    ResidualCandidate,
    choose_dynamic_k,
    gradient_beam_candidates,
    select_budget_diverse_finalists,
)
from tools.multilayer_seg.train_eval_fusion_pilot_v3 import (
    atomic_json,
    evaluate_embedding,
    make_sam_cache,
    prompt_logits,
    segmentation_loss,
)
from tools.multilayer_seg.train_eval_fusion_standard import (
    cache_video_backbones,
    extract_states_without_sam,
    ground_truth_video,
    infer_video_embedding,
    preprocess_video,
)
from tools.multilayer_seg.train_fusion_official_video_spsa import collect_video_events


ALL_LAYERS = tuple(range(1, 29))
RESIDUAL_LAYERS = tuple(range(1, 28))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--finalists-per-budget", type=int, default=2)
    parser.add_argument("--budgets", type=float, nargs="+", default=(0.05, 0.10, 0.15))
    parser.add_argument("--complexity-penalty", type=float, default=0.0005)
    parser.add_argument(
        "--resume-rows",
        help="Optional rows.json from an identical smaller-prefix run.",
    )
    return parser.parse_args()


def candidate_state(states: torch.Tensor, candidate: ResidualCandidate) -> torch.Tensor:
    anchor = states[-1]
    fused = anchor.clone()
    for layer, weight in zip(candidate.layers, candidate.weights):
        fused = fused + float(weight) * (states[layer - 1] - anchor)
    return fused


def project(text_hidden_fcs: Any, state: torch.Tensor) -> torch.Tensor:
    dtype = next(text_hidden_fcs.parameters()).dtype
    return text_hidden_fcs(state.to(dtype=dtype))


def exact_metrics(gt: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    j = float(db_eval_iou(gt, prediction).mean())
    f = float(db_eval_boundary(gt, prediction).mean())
    return {"J": j, "F": f, "J&F": (j + f) / 2.0}


def public_candidate(candidate: ResidualCandidate) -> dict[str, Any]:
    return {
        "K": candidate.k,
        "layers": [*candidate.layers, 28],
        "residual_layers": list(candidate.layers),
        "signs": list(candidate.signs),
        "budget": candidate.budget,
        "weights": list(candidate.weights),
    }


def bootstrap_delta(values: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        rng.choice(values, len(values), replace=True).mean() for _ in range(10000)
    ])
    signs = np.asarray([
        (values * rng.choice((-1.0, 1.0), len(values))).mean() for _ in range(20000)
    ])
    observed = float(values.mean())
    p_value = float((1 + np.count_nonzero(np.abs(signs) >= abs(observed))) / (len(signs) + 1))
    return {
        "mean": observed,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "sign_flip_p": p_value,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 1 <= args.samples <= 128:
        raise ValueError("samples must be in [1, 128]")
    if args.finalists_per_budget < 1:
        raise ValueError("finalists-per-budget must be positive")
    if args.pool_size < 3:
        raise ValueError("pool-size must be at least 3")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    atomic_json(status_path, {"state": "running", "phase": "data_index"})

    data_root = Path(args.data_root)
    events = collect_video_events(data_root, args.samples, 1, args.seed)
    split_root = data_root / "video_datas" / "mevis" / "train"
    train_masks = json.loads((split_root / "mask_dict.json").read_text())
    atomic_json(output / "events.json", {"events": [
        {key: event[key] for key in ("key", "video_id", "exp_id")}
        for event in events
    ]})

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

    grounding = model.grounding_encoder
    processor = model.extra_image_processor
    text_hidden_fcs = model.text_hidden_fcs
    dtype = model.torch_dtype
    rows: list[dict[str, Any]] = []
    if args.resume_rows:
        rows = json.loads(Path(args.resume_rows).read_text())["rows"]
        row_keys = [row["key"] for row in rows]
        event_keys = [event["key"] for event in events]
        if len(set(row_keys)) != len(row_keys):
            raise RuntimeError("resume rows contain duplicate keys")
        if row_keys != event_keys[: len(row_keys)]:
            raise RuntimeError("resume rows are not an exact event-prefix match")
        atomic_json(output / "rows.json", {"rows": rows})
    completed_keys = {row["key"] for row in rows}

    for event_number, event in enumerate(events, 1):
        if event["key"] in completed_keys:
            continue
        atomic_json(status_path, {
            "state": "running", "phase": "oracle_search",
            "completed": event_number - 1, "total": len(events), "key": event["key"],
        })
        states = extract_states_without_sam(
            model, tokenizer, event["frame_paths"][:5], event["prompt"], ALL_LAYERS
        ).cuda()
        images, output_size = preprocess_video(grounding, processor, dtype, event["frame_paths"])
        width, height = output_size
        gt = ground_truth_video(
            train_masks, event["annotation_ids"], len(event["frame_paths"]), (height, width)
        )
        visible = np.flatnonzero(gt.reshape(len(gt), -1).any(axis=1))
        if len(visible) == 0:
            raise RuntimeError(f"event has no visible ground truth: {event['key']}")
        proxy_frame = int(visible[len(visible) // 2])
        proxy_cache = make_sam_cache(
            grounding, processor, dtype, event["frame_paths"][proxy_frame]
        )

        anchor = states[-1].detach()
        residuals = states[:-1] - anchor.unsqueeze(0)
        coefficients = torch.zeros(len(RESIDUAL_LAYERS), device="cuda", requires_grad=True)
        fused = anchor + (coefficients[:, None] * residuals).sum(dim=0)
        embedding = project(text_hidden_fcs, fused)
        logits = prompt_logits(grounding, proxy_cache, embedding)
        proxy_gt_tensor = torch.from_numpy(gt[proxy_frame]).to(device="cuda")
        loss, _, _ = segmentation_loss(logits, proxy_gt_tensor)
        gradients = torch.autograd.grad(loss, coefficients, only_inputs=True)[0].detach().float().cpu()

        proxy_scored: dict[int, list[tuple[ResidualCandidate, float]]] = {2: [], 3: [], 4: []}
        for k in (2, 3, 4):
            candidates = gradient_beam_candidates(
                RESIDUAL_LAYERS, gradients.tolist(), residual_count=k - 1,
                pool_size=args.pool_size, beam_size=args.beam_size, budgets=args.budgets,
            )
            for candidate in candidates:
                candidate_embedding = project(text_hidden_fcs, candidate_state(states, candidate))
                metrics = evaluate_embedding(
                    grounding, proxy_cache, candidate_embedding, gt[proxy_frame]
                )
                proxy_scored[k].append((candidate, float(metrics["J&F"])))

        finalists = {
            k: select_budget_diverse_finalists(scored, args.finalists_per_budget)
            for k, scored in proxy_scored.items()
        }
        backbone_cache = cache_video_backbones(grounding, images)
        baseline_embedding = project(text_hidden_fcs, anchor)
        baseline_prediction = infer_video_embedding(
            grounding, images, backbone_cache, baseline_embedding, output_size
        )
        baseline = exact_metrics(gt, baseline_prediction)
        exact_by_k: dict[int, list[dict[str, Any]]] = {1: [{
            "candidate": {"K": 1, "layers": [28], "residual_layers": [], "signs": [], "budget": 0.0, "weights": []},
            "metrics": baseline,
        }]}
        for k in (2, 3, 4):
            exact_by_k[k] = []
            for candidate in finalists[k]:
                candidate_embedding = project(text_hidden_fcs, candidate_state(states, candidate))
                prediction = infer_video_embedding(
                    grounding, images, backbone_cache, candidate_embedding, output_size
                )
                exact_by_k[k].append({
                    "candidate": public_candidate(candidate),
                    "metrics": exact_metrics(gt, prediction),
                })
        best_by_k = {
            k: max(values, key=lambda value: value["metrics"]["J&F"])
            for k, values in exact_by_k.items()
        }
        scores_by_k = {k: value["metrics"]["J&F"] for k, value in best_by_k.items()}
        raw_k = min(scores_by_k, key=lambda k: (-scores_by_k[k], k))
        selected_k = choose_dynamic_k(scores_by_k, args.complexity_penalty)
        row = {
            "key": event["key"], "video_id": event["video_id"], "exp_id": event["exp_id"],
            "proxy_frame": proxy_frame,
            "gradient_top8": [
                {"layer": RESIDUAL_LAYERS[index], "gradient": float(gradients[index])}
                for index in sorted(range(len(gradients)), key=lambda i: -abs(float(gradients[i])))[:8]
            ],
            "baseline": baseline,
            "exact_by_k": exact_by_k,
            "best_by_k": best_by_k,
            "raw_oracle_k": int(raw_k),
            "raw_oracle": best_by_k[raw_k],
            "selected_k": int(selected_k),
            "selected": best_by_k[selected_k],
            "raw_delta_J&F": scores_by_k[raw_k] - baseline["J&F"],
            "selected_delta_J&F": scores_by_k[selected_k] - baseline["J&F"],
        }
        rows.append(row)
        atomic_json(output / "rows.json", {"rows": rows})
        print(
            f"ORACLE [{event_number}/{len(events)}] {event['key']} "
            f"baseline={baseline['J&F']:.6f} rawK={raw_k} "
            f"rawDelta={row['raw_delta_J&F']:+.6f} selectedK={selected_k} "
            f"selectedDelta={row['selected_delta_J&F']:+.6f}", flush=True,
        )
        del states, images, gt, proxy_cache, coefficients, fused, embedding, logits
        del proxy_gt_tensor, gradients, backbone_cache, baseline_embedding, baseline_prediction
        gc.collect()
        torch.cuda.empty_cache()

    if len(rows) != len(events):
        raise RuntimeError(f"incomplete oracle rows: {len(rows)}/{len(events)}")
    raw_deltas = np.asarray([row["raw_delta_J&F"] for row in rows])
    selected_deltas = np.asarray([row["selected_delta_J&F"] for row in rows])
    baseline_mean = float(np.mean([row["baseline"]["J&F"] for row in rows]))
    raw_mean = float(np.mean([row["raw_oracle"]["metrics"]["J&F"] for row in rows]))
    selected_mean = float(np.mean([row["selected"]["metrics"]["J&F"] for row in rows]))
    selected_counts = Counter(str(row["selected_k"]) for row in rows)
    fraction_multilayer = float(np.mean([row["selected_k"] > 1 for row in rows]))
    metrics = {
        "status": "complete",
        "protocol": "train-split unique-video bounded GT oracle; first-5-frame SEG extraction; full-video SAM2 propagation",
        "warning": "Ground truth selects candidates and K; this is an upper bound, not a deployable router.",
        "samples": len(rows), "seed": args.seed,
        "candidate_search": {
            "layers": list(ALL_LAYERS), "pool_size": args.pool_size,
            "beam_size": args.beam_size, "finalists_per_budget": args.finalists_per_budget,
            "budgets": list(args.budgets), "complexity_penalty": args.complexity_penalty,
        },
        "J&F_percent": {
            "h28_baseline": round(100 * baseline_mean, 4),
            "raw_oracle": round(100 * raw_mean, 4),
            "raw_oracle_delta": round(100 * float(raw_deltas.mean()), 4),
            "penalized_dynamic": round(100 * selected_mean, 4),
            "penalized_dynamic_delta": round(100 * float(selected_deltas.mean()), 4),
        },
        "raw_delta_in_fraction_units": bootstrap_delta(raw_deltas, args.seed + 1),
        "selected_delta_in_fraction_units": bootstrap_delta(selected_deltas, args.seed + 2),
        "selected_K_distribution": dict(sorted(selected_counts.items())),
        "fraction_selected_K_gt_1": fraction_multilayer,
        "improved_equal_declined": {
            "improved": int(np.count_nonzero(selected_deltas > 1e-9)),
            "equal": int(np.count_nonzero(np.abs(selected_deltas) <= 1e-9)),
            "declined": int(np.count_nonzero(selected_deltas < -1e-9)),
        },
        "go_no_go": {
            "oracle_delta_at_least_0.3_points": bool(raw_deltas.mean() >= 0.003),
            "at_least_15_percent_choose_multilayer": bool(fraction_multilayer >= 0.15),
            "proceed_to_router_training": bool(raw_deltas.mean() >= 0.003 and fraction_multilayer >= 0.15),
        },
    }
    atomic_json(output / "metrics.json", metrics)
    atomic_json(status_path, {"state": "complete", "phase": "done", "metrics": metrics})
    return metrics


def main() -> None:
    args = parse_args()
    try:
        metrics = run(args)
    except Exception as error:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "status.json", {"state": "failed", "error": repr(error)})
        raise
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
