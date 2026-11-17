#!/usr/bin/env python3
"""Prospectively confirm the frozen TGIF-style SEG router on disjoint videos."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from tools.multilayer_seg.evaluate_spatial_map_affine_gate import (
    build_event,
    infer_baseline_logits,
)
from tools.multilayer_seg.extract_spatial_adapter_features import extract_states
from tools.multilayer_seg.train_eval_fusion_standard import (
    cache_video_backbones,
    preprocess_video,
)
from work.evaluate_layer_policy_oracle_feasibility import (
    atomic_json,
    atomic_torch,
    pack_binary_mask,
    paired_summary,
    unpack_binary_mask,
)
from work.evaluate_tgif_query_causality_ablation import (
    fused_state,
    preprocessor_from_checkpoint,
    validation_global_weights,
)
from work.train_evaluate_tgif_seg_router import (
    ALL_LAYERS,
    TgifSegRouter,
    decode_frame_mask,
    error_rates,
    query_embedding,
    segment_metrics,
    transform_query,
)


ACTIONS = ("correct_query", "validation_global")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", nargs="?")
    parser.add_argument("--data-root")
    parser.add_argument("--calibration-manifest")
    parser.add_argument("--parent-dir")
    parser.add_argument("--output")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--selection-salt", default="exp039")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_json(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def video_id_from_key(key: str) -> str:
    if "_" not in key:
        raise ValueError(f"cannot parse video ID from router key: {key}")
    return key.rsplit("_", 1)[0]


def select_items(
    metadata: dict[str, Any], excluded: set[str], samples: int, salt: str
) -> list[dict[str, str]]:
    eligible = [str(video_id) for video_id in metadata if str(video_id) not in excluded]
    eligible.sort(
        key=lambda video_id: (
            hashlib.sha256(f"{salt}:{video_id}".encode()).hexdigest(),
            video_id,
        )
    )
    selected: list[dict[str, str]] = []
    for video_id in eligible:
        expressions = metadata[video_id]["expressions"]
        annotated = [
            str(exp_id)
            for exp_id, expression in expressions.items()
            if expression.get("anno_id")
        ]
        annotated.sort(
            key=lambda exp_id: (
                hashlib.sha256(f"{salt}:{video_id}:{exp_id}".encode()).hexdigest(),
                exp_id,
            )
        )
        if not annotated:
            continue
        exp_id = annotated[0]
        selected.append(
            {"key": f"{video_id}_{exp_id}", "video_id": video_id, "exp_id": exp_id}
        )
        if len(selected) == samples:
            break
    if len(selected) != samples:
        raise RuntimeError(f"only {len(selected)} eligible annotated videos, need {samples}")
    if len({item["video_id"] for item in selected}) != samples:
        raise RuntimeError("selection is not one unique video per row")
    if any(item["video_id"] in excluded for item in selected):
        raise RuntimeError("selected cohort overlaps an excluded video")
    return selected


def build_selection_manifest(
    metadata: dict[str, Any],
    router_split: dict[str, Any],
    calibration: dict[str, Any],
    samples: int,
    salt: str,
) -> dict[str, Any]:
    train_videos = {video_id_from_key(str(key)) for key in router_split["train_keys"]}
    validation_videos = {
        video_id_from_key(str(key)) for key in router_split["validation_keys"]
    }
    calibration_videos = {
        str(item["video_id"]) for item in calibration["calibration"][:8]
    }
    if train_videos & validation_videos:
        raise RuntimeError("router train and validation video IDs overlap")
    router_videos = train_videos | validation_videos
    if len(router_videos) != len(router_split["train_keys"]) + len(
        router_split["validation_keys"]
    ):
        raise RuntimeError("router split is not one unique video per key")
    if router_videos & calibration_videos:
        raise RuntimeError("router cohort unexpectedly overlaps calibration cohort")
    excluded = router_videos | calibration_videos
    selected = select_items(metadata, excluded, samples, salt)
    selected_sha = sha256_json(selected)
    return {
        "state": "locked_before_model_load_and_gt_access",
        "selection_salt": salt,
        "selection_rule": {
            "video": "ascending sha256(<salt>:<video_id>)",
            "expression": "ascending sha256(<salt>:<video_id>:<expression_id>) among annotated expressions",
            "one_expression_per_video": True,
        },
        "selected": selected,
        "selected_sha256": selected_sha,
        "excluded": {
            "router_train_count": len(train_videos),
            "router_validation_count": len(validation_videos),
            "calibration_count": len(calibration_videos),
            "total_unique_count": len(excluded),
            "video_ids_sha256": sha256_json(sorted(excluded)),
        },
        "overlap_selected_excluded": [],
        "metadata_video_count": len(metadata),
        "gt_access": False,
        "valid_access": False,
        "sealed_test_access": False,
    }


def protocol_hash(
    selection: dict[str, Any], checkpoint: dict[str, Any]
) -> str:
    contract = {
        "selection_sha256": selection["selected_sha256"],
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "checkpoint_layers": list(checkpoint["layer_indices"]),
        "actions": ("native_h28",) + ACTIONS,
        "correct_query": "frozen_exp037_router",
        "validation_global": "frozen_exp037_validation32_mean_weights",
        "conditioning_frames": 5,
        "propagation": "complete_video",
        "spatial_prompts": "none",
        "candidate_phase": "all_trajectories_frozen_before_gt",
        "training": "none",
    }
    return sha256_json(contract)


def positive_contribution_fraction(values: list[float]) -> float:
    positive = [float(value) for value in values if value > 0.0]
    if not positive:
        return 1.0
    return float(max(positive) / sum(positive))


def action_statistics(rows: list[dict[str, Any]], action: str) -> dict[str, Any]:
    return {
        "mean_metrics": {
            metric: float(
                np.mean([row["actions"][action]["metrics"][metric] for row in rows])
            )
            for metric in ("J", "F", "J&F")
        },
        "delta_vs_native_jf_points": paired_summary(
            [row["actions"][action]["delta_J&F_points"] for row in rows]
        ),
        "conditioning_delta_jf_points": paired_summary(
            [
                row["actions"][action]["conditioning_delta_J&F_points"]
                for row in rows
            ]
        ),
        "suffix_delta_jf_points": paired_summary(
            [row["actions"][action]["suffix_delta_J&F_points"] for row in rows]
        ),
        "mean_conditioning_false_positive_delta_points": float(
            np.mean(
                [
                    row["actions"][action][
                        "conditioning_false_positive_delta_points"
                    ]
                    for row in rows
                ]
            )
        ),
        "mean_suffix_false_positive_delta_points": float(
            np.mean(
                [
                    row["actions"][action]["suffix_false_positive_delta_points"]
                    for row in rows
                ]
            )
        ),
        "mean_conditioning_false_negative_delta_points": float(
            np.mean(
                [
                    row["actions"][action][
                        "conditioning_false_negative_delta_points"
                    ]
                    for row in rows
                ]
            )
        ),
        "mean_suffix_false_negative_delta_points": float(
            np.mean(
                [
                    row["actions"][action]["suffix_false_negative_delta_points"]
                    for row in rows
                ]
            )
        ),
    }


def summarize(
    rows: list[dict[str, Any]], selection: dict[str, Any], protocol: str
) -> dict[str, Any]:
    if not rows:
        raise ValueError("no scored rows")
    correct_deltas = [
        float(row["actions"]["correct_query"]["delta_J&F_points"]) for row in rows
    ]
    correct_global = [
        100.0
        * (
            row["actions"]["correct_query"]["metrics"]["J&F"]
            - row["actions"]["validation_global"]["metrics"]["J&F"]
        )
        for row in rows
    ]
    correct_summary = paired_summary(correct_deltas)
    correct_global_summary = paired_summary(correct_global)
    contribution_fraction = positive_contribution_fraction(correct_deltas)
    gates = {
        "correct_mean_delta_jf_points_ge_0p5": correct_summary["mean"] >= 0.5,
        "correct_improved_videos_ge_9_of_16": correct_summary[
            "better_equal_worse"
        ][0]
        >= 9,
        "correct_minus_global_mean_jf_points_ge_0p2": correct_global_summary["mean"]
        >= 0.2,
        "no_correct_decline_at_or_below_minus_10": correct_summary[
            "catastrophic_at_or_below_minus_10"
        ]
        == 0,
        "max_positive_contribution_fraction_le_0p6": contribution_fraction <= 0.6,
        "prospective_selection_disjoint": not selection[
            "overlap_selected_excluded"
        ],
    }
    return {
        "status": "complete",
        "samples": len(rows),
        "baseline_mean": {
            metric: float(np.mean([row["baseline_h28"][metric] for row in rows]))
            for metric in ("J", "F", "J&F")
        },
        "actions": {action: action_statistics(rows, action) for action in ACTIONS},
        "comparisons": {
            "correct_minus_validation_global_jf_points": correct_global_summary
        },
        "max_positive_contribution_fraction": contribution_fraction,
        "largest_positive_key": max(
            rows,
            key=lambda row: row["actions"]["correct_query"]["delta_J&F_points"],
        )["key"],
        "selection_sha256": selection["selected_sha256"],
        "protocol_hash": protocol,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "expansion_authorized": all(gates.values()),
        "claim_scope": "prospectively locked disjoint 16-video MeViS-train confirmation",
        "training": "none; frozen exp037 epoch-3 router",
        "valid_access": False,
        "sealed_test_access": False,
    }


def score_action(
    truth: np.ndarray,
    native: np.ndarray,
    candidate: np.ndarray,
    native_metrics: dict[str, float],
    native_conditioning: dict[str, float],
    native_suffix: dict[str, float],
    native_errors: tuple[float, float, float, float],
) -> dict[str, Any]:
    metrics = segment_metrics(truth, candidate)
    conditioning = segment_metrics(truth[:5], candidate[:5])
    suffix = segment_metrics(truth[5:], candidate[5:])
    cond_fp, cond_fn = error_rates(truth[:5], candidate[:5])
    suffix_fp, suffix_fn = error_rates(truth[5:], candidate[5:])
    native_cond_fp, native_cond_fn, native_suffix_fp, native_suffix_fn = native_errors
    return {
        "metrics": metrics,
        "delta_J&F_points": 100.0 * (metrics["J&F"] - native_metrics["J&F"]),
        "conditioning_metrics": conditioning,
        "suffix_metrics": suffix,
        "conditioning_delta_J&F_points": 100.0
        * (conditioning["J&F"] - native_conditioning["J&F"]),
        "suffix_delta_J&F_points": 100.0
        * (suffix["J&F"] - native_suffix["J&F"]),
        "conditioning_false_positive_delta_points": 100.0
        * (cond_fp - native_cond_fp),
        "suffix_false_positive_delta_points": 100.0
        * (suffix_fp - native_suffix_fp),
        "conditioning_false_negative_delta_points": 100.0
        * (cond_fn - native_cond_fn),
        "suffix_false_negative_delta_points": 100.0
        * (suffix_fn - native_suffix_fn),
    }


def self_test() -> None:
    metadata = {
        f"v{index}": {
            "expressions": {
                "0": {"anno_id": [index]},
                "1": {"anno_id": []},
                "2": {"anno_id": [index + 100]},
            }
        }
        for index in range(10)
    }
    excluded = {"v1", "v7"}
    first = select_items(metadata, excluded, 4, "test")
    second = select_items(metadata, excluded, 4, "test")
    assert first == second
    assert len({item["video_id"] for item in first}) == 4
    assert not ({item["video_id"] for item in first} & excluded)
    assert abs(positive_contribution_fraction([1.0, 2.0, -1.0]) - 2.0 / 3.0) < 1e-9
    assert positive_contribution_fraction([-2.0, 0.0]) == 1.0
    states = torch.randn(28, 7)
    weights = torch.zeros(28)
    weights[-1] = 1.0
    assert torch.equal(fused_state(states, weights), states[-1].float())
    print(
        json.dumps(
            {
                "state": "self_test_passed",
                "selection": first,
                "positive_fraction": 2.0 / 3.0,
            },
            indent=2,
        )
    )


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    required = (
        args.model_path,
        args.data_root,
        args.calibration_manifest,
        args.parent_dir,
        args.output,
    )
    if not all(required):
        raise ValueError("model/data/calibration/parent/output paths are required")
    if args.samples != 16:
        raise ValueError("the locked confirmation requires exactly 16 samples")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output = Path(args.output)
    row_dir = output / "rows"
    trajectory_dir = output / "frozen_trajectories"
    output.mkdir(parents=True, exist_ok=True)
    row_dir.mkdir(exist_ok=True)
    trajectory_dir.mkdir(exist_ok=True)
    parent_dir = Path(args.parent_dir)
    data_root = Path(args.data_root)
    split = data_root / "video_datas" / "mevis" / "train"

    # The selection is committed using only split metadata and prior cohort IDs.
    # In particular, mask_dict.json has not been opened at this point.
    metadata = json.loads((split / "meta_expressions.json").read_text())["videos"]
    router_split = json.loads((parent_dir / "router_split.json").read_text())
    calibration = json.loads(Path(args.calibration_manifest).read_text())
    selection = build_selection_manifest(
        metadata, router_split, calibration, args.samples, args.selection_salt
    )
    atomic_json(output / "selection_manifest.json", selection)
    atomic_json(
        output / "status.json",
        {
            "state": "running",
            "phase": "selection_locked_before_model_load_and_gt_access",
            "selected_sha256": selection["selected_sha256"],
            "gt_access": False,
        },
    )

    checkpoint = torch.load(
        parent_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False
    )
    if tuple(checkpoint["layer_indices"]) != ALL_LAYERS:
        raise RuntimeError("parent checkpoint layer contract mismatch")
    preprocessor = preprocessor_from_checkpoint(checkpoint)
    global_weights = validation_global_weights(checkpoint)
    router = TgifSegRouter(
        int(checkpoint["query_dim"]),
        int(checkpoint["router_hidden_dim"]),
        len(ALL_LAYERS),
        float(checkpoint["h28_prior"]),
    )
    router.load_state_dict(checkpoint["state_dict"])
    router.eval().to("cuda:0")
    for parameter in router.parameters():
        parameter.requires_grad_(False)
    protocol = protocol_hash(selection, checkpoint)
    atomic_json(
        output / "status.json",
        {
            "state": "running",
            "phase": "model_load",
            "selected_sha256": selection["selected_sha256"],
            "protocol_hash": protocol,
            "gt_access": False,
        },
    )

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
    projection = model.text_hidden_fcs
    projection_dtype = next(projection.parameters()).dtype
    grounding = model.grounding_encoder

    frozen_paths: list[Path] = []
    for number, item in enumerate(selection["selected"], 1):
        event = build_event(data_root, item, metadata)
        states, _, _, _ = extract_states(model, tokenizer, event, 5, ALL_LAYERS)
        query = query_embedding(model, tokenizer, event["prompt"])
        reduced = transform_query(query[None], preprocessor).to("cuda:0")
        correct_weights = router.layer_weights(reduced)[0].float().cpu()
        native_state = states[-1].float().to("cuda:0")
        native_embedding = projection(native_state.to(dtype=projection_dtype)).reshape(-1)
        embeddings: dict[str, torch.Tensor] = {}
        for action, weights in {
            "correct_query": correct_weights,
            "validation_global": global_weights,
        }.items():
            fused = fused_state(states.float(), weights).to("cuda:0")
            embeddings[action] = projection(fused.to(dtype=projection_dtype)).reshape(-1)

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
        action_logits = {
            action: infer_baseline_logits(grounding, images, cache, embedding, size)
            for action, embedding in embeddings.items()
        }
        trajectory = {
            "key": event["key"],
            "video_id": event["video_id"],
            "exp_id": event["exp_id"],
            "annotation_ids": event["annotation_ids"],
            "frame_count": len(images),
            "output_size_wh": list(size),
            "selection_sha256": selection["selected_sha256"],
            "protocol_hash": protocol,
            "native_mask": pack_binary_mask(native_logits.numpy() > 0),
            "action_masks": {
                action: pack_binary_mask(logits.numpy() > 0)
                for action, logits in action_logits.items()
            },
            "correct_layer_weights": correct_weights,
            "global_layer_weights": global_weights,
            "correct_top8_layers": [
                int(index) + 1
                for index in torch.argsort(correct_weights, descending=True)[:8]
            ],
            "correct_h28_weight": float(correct_weights[-1]),
            "native_correct_embedding_cosine": float(
                F.cosine_similarity(
                    native_embedding.float(), embeddings["correct_query"].float(), dim=0
                )
            ),
            "native_global_embedding_cosine": float(
                F.cosine_similarity(
                    native_embedding.float(), embeddings["validation_global"].float(), dim=0
                )
            ),
            "contract": (
                "frozen exp037 TGIF-style query router over RMS-aligned raw l1-l28 SEG; "
                "native projector; no point/box/mask/map; complete-video propagation"
            ),
            "gt_access": False,
            "valid_access": False,
            "sealed_test_access": False,
        }
        path = trajectory_dir / f"{event['key']}.pt"
        atomic_torch(path, trajectory)
        frozen_paths.append(path)
        print(
            f"EXP039_FREEZE [{number}/{args.samples}] {event['key']} "
            f"h28={float(correct_weights[-1]):.6f} "
            f"top={int(correct_weights.argmax()) + 1}",
            flush=True,
        )
        del states, query, reduced, correct_weights, native_state, native_embedding
        del embeddings, images, cache, native_logits, action_logits, trajectory
        gc.collect()
        torch.cuda.empty_cache()

    if len(frozen_paths) != args.samples:
        raise RuntimeError("not all candidate trajectories were frozen")
    del router, model, tokenizer, projection, grounding
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(
        output / "status.json",
        {
            "state": "running",
            "phase": "all_trajectories_frozen_before_gt_access",
            "completed": len(frozen_paths),
            "selected_sha256": selection["selected_sha256"],
            "protocol_hash": protocol,
            "gt_access": False,
        },
    )

    # Phase boundary: GT masks are opened only after every trajectory is fixed
    # and the model/router have been unloaded.
    masks = json.loads((split / "mask_dict.json").read_text())
    rows: list[dict[str, Any]] = []
    for path in frozen_paths:
        trajectory = torch.load(path, map_location="cpu", weights_only=False)
        width, height = trajectory["output_size_wh"]
        truth = np.zeros(
            (trajectory["frame_count"], height, width), dtype=np.uint8
        )
        for frame_index in range(trajectory["frame_count"]):
            truth[frame_index] = decode_frame_mask(
                masks,
                trajectory["annotation_ids"],
                frame_index,
                (height, width),
            )
        truth = truth.astype(bool)
        native = unpack_binary_mask(trajectory["native_mask"])
        native_metrics = segment_metrics(truth, native)
        native_conditioning = segment_metrics(truth[:5], native[:5])
        native_suffix = segment_metrics(truth[5:], native[5:])
        native_cond_fp, native_cond_fn = error_rates(truth[:5], native[:5])
        native_suffix_fp, native_suffix_fn = error_rates(truth[5:], native[5:])
        native_errors = (
            native_cond_fp,
            native_cond_fn,
            native_suffix_fp,
            native_suffix_fn,
        )
        actions = {
            action: score_action(
                truth,
                native,
                unpack_binary_mask(trajectory["action_masks"][action]),
                native_metrics,
                native_conditioning,
                native_suffix,
                native_errors,
            )
            for action in ACTIONS
        }
        row = {
            "key": trajectory["key"],
            "video_id": trajectory["video_id"],
            "exp_id": trajectory["exp_id"],
            "selection_sha256": selection["selected_sha256"],
            "protocol_hash": protocol,
            "baseline_h28": native_metrics,
            "baseline_conditioning": native_conditioning,
            "baseline_suffix": native_suffix,
            "actions": actions,
            "correct_layer_weights": trajectory["correct_layer_weights"].tolist(),
            "global_layer_weights": trajectory["global_layer_weights"].tolist(),
            "correct_top8_layers": trajectory["correct_top8_layers"],
            "correct_h28_weight": trajectory["correct_h28_weight"],
            "native_correct_embedding_cosine": trajectory[
                "native_correct_embedding_cosine"
            ],
            "native_global_embedding_cosine": trajectory[
                "native_global_embedding_cosine"
            ],
            "gt_access_after_all_candidate_freeze": True,
            "valid_access": False,
            "sealed_test_access": False,
        }
        atomic_json(row_dir / f"{trajectory['key']}.json", row)
        rows.append(row)
        print(
            f"EXP039_SCORE {trajectory['key']} "
            f"correct={actions['correct_query']['delta_J&F_points']:+.3f}pt "
            f"global={actions['validation_global']['delta_J&F_points']:+.3f}pt",
            flush=True,
        )

    summary = summarize(rows, selection, protocol)
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "status.json",
        {
            "state": "complete",
            "phase": "done",
            "completed": len(rows),
            "selection_sha256": selection["selected_sha256"],
            "protocol_hash": protocol,
            "all_gates_pass": summary["all_gates_pass"],
            "expansion_authorized": summary["expansion_authorized"],
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
