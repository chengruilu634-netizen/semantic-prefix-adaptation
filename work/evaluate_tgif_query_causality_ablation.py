#!/usr/bin/env python3
"""Ablate correct-query, reversed-query and global TGIF [SEG] weights."""

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
from work.train_evaluate_tgif_seg_router import (
    ALL_LAYERS,
    TgifSegRouter,
    decode_frame_mask,
    error_rates,
    query_embedding,
    segment_metrics,
    transform_query,
)


ACTIONS = ("correct_query", "reverse_query", "validation_global")
RESCUE_KEY = "7cc4813dfe30_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", nargs="?")
    parser.add_argument("--data-root")
    parser.add_argument("--manifest")
    parser.add_argument("--reference-baseline-dir")
    parser.add_argument("--parent-dir")
    parser.add_argument("--output")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def preprocessor_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {}
    for key, value in checkpoint["query_preprocessor"].items():
        if not torch.is_tensor(value):
            raise TypeError(f"query preprocessor field {key} is not a tensor")
        result[key] = value.float().cpu().numpy()
    return result


def validation_global_weights(checkpoint: dict[str, Any]) -> torch.Tensor:
    mapping = checkpoint["training_metrics"]["router"]["mean_by_layer"]
    weights = torch.tensor(
        [float(mapping[str(layer)]) for layer in ALL_LAYERS], dtype=torch.float32
    )
    if not torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5):
        raise RuntimeError(f"validation global weights sum to {float(weights.sum())}")
    return weights / weights.sum()


def fused_state(states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if states.shape[-2] != len(ALL_LAYERS) or weights.shape[-1] != len(ALL_LAYERS):
        raise ValueError("expected 28 ordered states and weights")
    aligned = TgifSegRouter.rms_align(states.float())
    return torch.einsum("l,lh->h", weights.float(), aligned)


def reverse_sources(count: int) -> list[int]:
    if count < 2 or count % 2:
        raise ValueError("reverse-query control requires an even count >=2")
    result = list(range(count - 1, -1, -1))
    if any(index == source for index, source in enumerate(result)):
        raise RuntimeError("reverse permutation contains a fixed point")
    return result


def protocol_hash(manifest: dict[str, Any], checkpoint: dict[str, Any]) -> str:
    contract = {
        "split_hash": manifest["calibration_sha256"],
        "parent_best_epoch": checkpoint["best_epoch"],
        "parent_layers": list(checkpoint["layer_indices"]),
        "actions": ACTIONS,
        "reverse_assignment": "cohort_order_reversed_no_fixed_points",
        "global_source": "exp037_validation32_mean_weights",
        "spatial_prompts": "none",
        "training": "none",
        "reference_native": "exp034",
        "reference_correct": "exp037",
    }
    raw = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def action_statistics(rows: list[dict[str, Any]], action: str) -> dict[str, Any]:
    metrics = {
        metric: float(np.mean([row["actions"][action]["metrics"][metric] for row in rows]))
        for metric in ("J", "F", "J&F")
    }
    deltas = [row["actions"][action]["delta_J&F_points"] for row in rows]
    return {
        "mean_metrics": metrics,
        "delta_vs_native_jf_points": paired_summary(deltas),
        "conditioning_delta_jf_points": paired_summary(
            [row["actions"][action]["conditioning_delta_J&F_points"] for row in rows]
        ),
        "suffix_delta_jf_points": paired_summary(
            [row["actions"][action]["suffix_delta_J&F_points"] for row in rows]
        ),
    }


def compare_actions(
    rows: list[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    differences = [
        100.0
        * (
            row["actions"][left]["metrics"]["J&F"]
            - row["actions"][right]["metrics"]["J&F"]
        )
        for row in rows
    ]
    return paired_summary(differences)


def summarize(rows: list[dict[str, Any]], protocol: str) -> dict[str, Any]:
    by_action = {action: action_statistics(rows, action) for action in ACTIONS}
    correct_reverse = compare_actions(rows, "correct_query", "reverse_query")
    correct_global = compare_actions(rows, "correct_query", "validation_global")
    rescue = next(row for row in rows if row["key"] == RESCUE_KEY)
    rescue_correct_reverse = 100.0 * (
        rescue["actions"]["correct_query"]["metrics"]["J&F"]
        - rescue["actions"]["reverse_query"]["metrics"]["J&F"]
    )
    rescue_correct_global = 100.0 * (
        rescue["actions"]["correct_query"]["metrics"]["J&F"]
        - rescue["actions"]["validation_global"]["metrics"]["J&F"]
    )
    gates = {
        "correct_minus_reverse_mean_ge_0p2": correct_reverse["mean"] >= 0.2,
        "correct_minus_global_mean_ge_0p2": correct_global["mean"] >= 0.2,
        "rescue_correct_minus_reverse_ge_2": rescue_correct_reverse >= 2.0,
        "rescue_correct_minus_global_ge_2": rescue_correct_global >= 2.0,
        "no_correct_catastrophic_decline": by_action["correct_query"][
            "delta_vs_native_jf_points"
        ]["catastrophic_at_or_below_minus_10"]
        == 0,
        "native_reference_pixel_exact": all(
            row["native_reference_xor_fraction"] == 0.0 for row in rows
        ),
        "correct_reference_pixel_exact": all(
            row["correct_reference_xor_fraction"] == 0.0 for row in rows
        ),
    }
    return {
        "status": "complete",
        "samples": len(rows),
        "baseline_mean": {
            metric: float(np.mean([row["baseline_h28"][metric] for row in rows]))
            for metric in ("J", "F", "J&F")
        },
        "actions": by_action,
        "comparisons": {
            "correct_minus_reverse_jf_points": correct_reverse,
            "correct_minus_validation_global_jf_points": correct_global,
        },
        "rescue": {
            "key": RESCUE_KEY,
            "correct_minus_reverse_jf_points": rescue_correct_reverse,
            "correct_minus_validation_global_jf_points": rescue_correct_global,
            "actions": rescue["actions"],
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "protocol_hash": protocol,
        "claim_scope": "post-hoc mechanism diagnostic on reused calibration",
        "sealed_test_access": False,
        "training": "none; frozen exp037 checkpoint",
    }


def self_test() -> None:
    assert reverse_sources(8) == [7, 6, 5, 4, 3, 2, 1, 0]
    states = torch.randn(28, 7)
    weights = torch.zeros(28)
    weights[-1] = 1.0
    assert torch.equal(fused_state(states, weights), states[-1].float())
    weights = torch.full((28,), 1.0 / 28.0)
    assert torch.isfinite(fused_state(states, weights)).all()
    print(
        json.dumps(
            {"state": "self_test_passed", "actions": ACTIONS, "rescue": RESCUE_KEY},
            indent=2,
        )
    )


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    required = (
        args.model_path,
        args.data_root,
        args.manifest,
        args.reference_baseline_dir,
        args.parent_dir,
        args.output,
    )
    if not all(required):
        raise ValueError("all model/data/manifest/reference/parent/output paths are required")
    if args.samples != 8:
        raise ValueError("this locked diagnostic requires exactly eight samples")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = Path(args.output)
    row_dir = output / "rows"
    trajectory_dir = output / "frozen_trajectories"
    row_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    parent_dir = Path(args.parent_dir)
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
        28,
        float(checkpoint["h28_prior"]),
    )
    router.load_state_dict(checkpoint["state_dict"])
    router.eval().to("cuda:0")

    manifest = json.loads(Path(args.manifest).read_text())
    selected = manifest["calibration"][: args.samples]
    protocol = protocol_hash(manifest, checkpoint)
    atomic_json(
        output / "status.json",
        {
            "state": "running",
            "phase": "model_load",
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
    data_root = Path(args.data_root)
    split = data_root / "video_datas" / "mevis" / "train"
    metadata = json.loads((split / "meta_expressions.json").read_text())["videos"]
    events = [build_event(data_root, item, metadata) for item in selected]

    states_by_index = []
    correct_weights = []
    for number, event in enumerate(events, 1):
        states, _, _, _ = extract_states(model, tokenizer, event, 5, ALL_LAYERS)
        query = query_embedding(model, tokenizer, event["prompt"])
        reduced = transform_query(query[None], preprocessor).to("cuda:0")
        weights = router.layer_weights(reduced)[0].float().cpu()
        states_by_index.append(states.float().cpu())
        correct_weights.append(weights)
        print(
            f"QUERY [{number}/{len(events)}] {event['key']} "
            f"h28={float(weights[-1]):.6f} top={int(weights.argmax())+1}",
            flush=True,
        )
    correct_weights_tensor = torch.stack(correct_weights)
    sources = reverse_sources(len(events))
    projection = model.text_hidden_fcs
    projection_dtype = next(projection.parameters()).dtype
    grounding = model.grounding_encoder
    native_reference_dir = Path(args.reference_baseline_dir)
    correct_reference_dir = parent_dir / "frozen_trajectories"
    frozen = []

    for number, (event, states) in enumerate(zip(events, states_by_index), 1):
        weight_map = {
            "correct_query": correct_weights_tensor[number - 1],
            "reverse_query": correct_weights_tensor[sources[number - 1]],
            "validation_global": global_weights,
        }
        native_state = states[-1].to("cuda:0")
        native_embedding = projection(native_state.to(dtype=projection_dtype)).reshape(-1)
        embeddings = {}
        for action, weights in weight_map.items():
            state = fused_state(states, weights).to("cuda:0")
            embeddings[action] = projection(state.to(dtype=projection_dtype)).reshape(-1)
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
        logits = {
            action: infer_baseline_logits(grounding, images, cache, embedding, size)
            for action, embedding in embeddings.items()
        }
        native_mask = native_logits.numpy() > 0
        candidate_masks = {action: value.numpy() > 0 for action, value in logits.items()}
        native_reference = torch.load(
            native_reference_dir / f"{event['key']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        native_reference_mask = unpack_binary_mask(native_reference["baseline_mask"])
        native_xor = float(np.logical_xor(native_mask, native_reference_mask).mean())
        correct_reference = torch.load(
            correct_reference_dir / f"{event['key']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        correct_reference_mask = unpack_binary_mask(correct_reference["fused_mask"])
        correct_xor = float(
            np.logical_xor(candidate_masks["correct_query"], correct_reference_mask).mean()
        )
        if native_xor != 0.0 or correct_xor != 0.0:
            raise RuntimeError(
                f"reference reproduction failed {event['key']}: "
                f"native_xor={native_xor} correct_xor={correct_xor}"
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
            "candidate_masks": {
                action: pack_binary_mask(mask) for action, mask in candidate_masks.items()
            },
            "weights": weight_map,
            "reverse_source_key": events[sources[number - 1]]["key"],
            "native_reference_xor_fraction": native_xor,
            "correct_reference_xor_fraction": correct_xor,
            "embedding_cosine_to_native": {
                action: float(
                    F.cosine_similarity(
                        native_embedding.float(), embedding.float(), dim=0
                    )
                )
                for action, embedding in embeddings.items()
            },
            "gt_access": False,
            "sealed_test_access": False,
        }
        path = trajectory_dir / f"{event['key']}.pt"
        atomic_torch(path, trajectory)
        frozen.append(path)
        print(
            f"FREEZE [{number}/{len(events)}] {event['key']} "
            f"reverse_source={trajectory['reverse_source_key']} "
            f"correct_xor={correct_xor:.8f}",
            flush=True,
        )
        del native_state, native_embedding, embeddings, images, cache
        del native_logits, logits, native_mask, candidate_masks
        gc.collect()
        torch.cuda.empty_cache()

    del model, tokenizer, router, grounding, projection
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(
        output / "status.json",
        {
            "state": "running",
            "phase": "all_candidates_frozen_before_gt",
            "completed": len(frozen),
            "protocol_hash": protocol,
            "gt_access": False,
        },
    )

    mask_dict = json.loads((split / "mask_dict.json").read_text())
    rows = []
    for path in frozen:
        trajectory = torch.load(path, map_location="cpu", weights_only=False)
        width, height = trajectory["output_size_wh"]
        truth = np.zeros((trajectory["frame_count"], height, width), dtype=np.uint8)
        for frame_index in range(trajectory["frame_count"]):
            truth[frame_index] = decode_frame_mask(
                mask_dict,
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
        action_rows = {}
        for action in ACTIONS:
            candidate = unpack_binary_mask(trajectory["candidate_masks"][action])
            metrics = segment_metrics(truth, candidate)
            conditioning = segment_metrics(truth[:5], candidate[:5])
            suffix = segment_metrics(truth[5:], candidate[5:])
            cond_fp, cond_fn = error_rates(truth[:5], candidate[:5])
            suffix_fp, suffix_fn = error_rates(truth[5:], candidate[5:])
            action_rows[action] = {
                "metrics": metrics,
                "delta_J&F_points": 100.0 * (
                    metrics["J&F"] - native_metrics["J&F"]
                ),
                "conditioning_metrics": conditioning,
                "conditioning_delta_J&F_points": 100.0 * (
                    conditioning["J&F"] - native_conditioning["J&F"]
                ),
                "suffix_metrics": suffix,
                "suffix_delta_J&F_points": 100.0 * (
                    suffix["J&F"] - native_suffix["J&F"]
                ),
                "conditioning_false_positive_delta_points": 100.0 * (
                    cond_fp - native_cond_fp
                ),
                "conditioning_false_negative_delta_points": 100.0 * (
                    cond_fn - native_cond_fn
                ),
                "suffix_false_positive_delta_points": 100.0 * (
                    suffix_fp - native_suffix_fp
                ),
                "suffix_false_negative_delta_points": 100.0 * (
                    suffix_fn - native_suffix_fn
                ),
                "weights": trajectory["weights"][action].tolist(),
                "embedding_cosine_to_native": trajectory[
                    "embedding_cosine_to_native"
                ][action],
            }
        row = {
            "key": trajectory["key"],
            "video_id": trajectory["video_id"],
            "exp_id": trajectory["exp_id"],
            "baseline_h28": native_metrics,
            "baseline_conditioning": native_conditioning,
            "baseline_suffix": native_suffix,
            "actions": action_rows,
            "reverse_source_key": trajectory["reverse_source_key"],
            "native_reference_xor_fraction": trajectory[
                "native_reference_xor_fraction"
            ],
            "correct_reference_xor_fraction": trajectory[
                "correct_reference_xor_fraction"
            ],
            "gt_access_after_all_candidates_frozen": True,
            "sealed_test_access": False,
        }
        atomic_json(row_dir / f"{trajectory['key']}.json", row)
        rows.append(row)
        print(
            f"SCORE {trajectory['key']} "
            + " ".join(
                f"{action}={action_rows[action]['delta_J&F_points']:+.3f}"
                for action in ACTIONS
            ),
            flush=True,
        )
    summary = summarize(rows, protocol)
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "status.json",
        {
            "state": "complete",
            "phase": "done",
            "protocol_hash": protocol,
            "all_gates_pass": summary["all_gates_pass"],
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

