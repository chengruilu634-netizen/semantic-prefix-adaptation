#!/usr/bin/env python3
"""Measure end-to-end oracle headroom for fixed Sa2VA layer-map actions."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats
from transformers import AutoModel, AutoTokenizer

from tools.multilayer_seg.evaluate_raw_similarity_point_prompt import infer_points
from tools.multilayer_seg.evaluate_spatial_map_affine_gate import (
    build_event,
    infer_baseline_logits,
)
from tools.multilayer_seg.extract_spatial_adapter_features import extract_states
from tools.multilayer_seg.run_dynamic_k_oracle import exact_metrics
from tools.multilayer_seg.train_eval_fusion_standard import (
    cache_video_backbones,
    ground_truth_video,
    preprocess_video,
)


LAYERS = (2, 21, 28)
ACTION_SPECS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("point_l2", (2,)),
    ("point_l21", (21,)),
    ("point_l28", (28,)),
    ("point_rankfusion_l2_l28", (2, 28)),
    ("point_rankfusion_l21_l28", (21, 28)),
    ("point_rankfusion_l2_l21_l28", (2, 21, 28)),
)
FUSION_ACTIONS = {name for name, layers in ACTION_SPECS if len(layers) > 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", nargs="?")
    parser.add_argument("--data-root")
    parser.add_argument("--manifest")
    parser.add_argument("--raw-map-dir")
    parser.add_argument("--output")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--candidates-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def pack_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    array = np.asarray(mask, dtype=np.bool_)
    packed = np.packbits(array.reshape(-1).astype(np.uint8))
    return {"shape": list(array.shape), "packed": torch.from_numpy(packed)}


def unpack_binary_mask(payload: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(value) for value in payload["shape"])
    total = int(np.prod(shape))
    packed = payload["packed"].cpu().numpy()
    return np.unpackbits(packed, count=total).astype(bool).reshape(shape)


def rank_normalize(maps: torch.Tensor) -> torch.Tensor:
    """Map each frame to deterministic percentile ranks in [0, 1]."""
    if maps.ndim != 3:
        raise ValueError("expected [frames, height, width]")
    result = torch.empty_like(maps, dtype=torch.float32)
    for frame_index, frame in enumerate(maps.float()):
        flat = frame.reshape(-1).cpu().numpy()
        order = np.argsort(flat, kind="mergesort")
        ranks = np.empty(len(flat), dtype=np.float32)
        if len(flat) == 1:
            ranks[0] = 0.5
        else:
            ranks[order] = np.arange(len(flat), dtype=np.float32) / float(len(flat) - 1)
        result[frame_index] = torch.from_numpy(ranks.reshape(frame.shape))
    return result


def build_action_maps(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    layers = [int(value) for value in payload["layers"]]
    stack = payload["maps"].float()
    selected = {layer: stack[layers.index(layer)] for layer in LAYERS}
    ranked = {layer: rank_normalize(value) for layer, value in selected.items()}
    actions: dict[str, torch.Tensor] = {}
    for name, members in ACTION_SPECS:
        if len(members) == 1:
            actions[name] = selected[members[0]]
        else:
            actions[name] = torch.stack([ranked[layer] for layer in members]).mean(dim=0)
    return actions


def paired_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2 or np.allclose(array, array[0]):
        interval = [float(array.mean()), float(array.mean())]
    else:
        sem = stats.sem(array)
        half = float(stats.t.ppf(0.975, len(array) - 1) * sem)
        interval = [float(array.mean() - half), float(array.mean() + half)]
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "ci95_t": interval,
        "min": float(array.min()),
        "max": float(array.max()),
        "better_equal_worse": [
            int(np.sum(array > 1e-9)),
            int(np.sum(np.abs(array) <= 1e-9)),
            int(np.sum(array < -1e-9)),
        ],
        "catastrophic_at_or_below_minus_10": int(np.sum(array <= -10.0)),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no rows")
    action_names = [name for name, _ in ACTION_SPECS]
    per_action: dict[str, Any] = {}
    for name in action_names:
        deltas = [
            next(item for item in row["candidates"] if item["candidate_id"] == name)[
                "delta_J&F_points"
            ]
            for row in rows
        ]
        per_action[name] = paired_summary(deltas)

    oracle_deltas: list[float] = []
    winners: list[str] = []
    non_native_gt_one = 0
    for row in rows:
        candidates = row["candidates"]
        best = max(candidates, key=lambda item: item["delta_J&F_points"])
        best_delta = float(best["delta_J&F_points"])
        if best_delta > 1.0:
            non_native_gt_one += 1
        if best_delta > 0.0:
            oracle_deltas.append(best_delta)
            winners.append(best["candidate_id"])
        else:
            oracle_deltas.append(0.0)
            winners.append("native_h28_no_spatial_prompt")

    non_native_winners = sorted({winner for winner in winners if winner != "native_h28_no_spatial_prompt"})
    fusion_winners = sum(winner in FUSION_ACTIONS for winner in winners)
    oracle = paired_summary(oracle_deltas)
    gates = {
        "oracle_minus_native_mean_jf_points_ge_3": oracle["mean"] >= 3.0,
        "videos_with_non_native_gain_gt_1_point_ge_4": non_native_gt_one >= 4,
        "distinct_non_native_winners_ge_2": len(non_native_winners) >= 2,
        "fusion_winners_ge_1": fusion_winners >= 1,
    }
    return {
        "samples": len(rows),
        "candidate_actions": action_names,
        "per_action_vs_native_jf_points": per_action,
        "oracle_vs_native_jf_points": oracle,
        "oracle_winners": winners,
        "distinct_non_native_winners": non_native_winners,
        "videos_with_non_native_gain_gt_1_point": non_native_gt_one,
        "fusion_winners": fusion_winners,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "claim_scope": "reused-calibration non-deployable oracle feasibility only",
        "sealed_test_access": False,
    }


def protocol_hash(manifest: dict[str, Any]) -> str:
    contract = {
        "split_hash": manifest["calibration_sha256"],
        "samples": 8,
        "language_layer": 28,
        "map_layers": LAYERS,
        "actions": ACTION_SPECS,
        "normalization": "within_frame_percentile_rank",
        "fusion": "fixed_equal",
        "conditioning_frames": 5,
        "point_count": 1,
        "fallback": "native_h28_no_spatial_prompt",
        "training": "none",
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def self_test() -> None:
    maps = torch.tensor(
        [
            [[[0.0, 1.0], [2.0, 3.0]]],
            [[[3.0, 2.0], [1.0, 0.0]]],
            [[[0.0, 2.0], [1.0, 3.0]]],
        ]
    )
    payload = {"layers": [2, 21, 28], "maps": maps}
    actions = build_action_maps(payload)
    assert set(actions) == {name for name, _ in ACTION_SPECS}
    assert actions["point_l2"].shape == (1, 2, 2)
    assert math.isclose(float(actions["point_rankfusion_l2_l28"].min()), 0.0)
    rows = []
    for index in range(8):
        deltas = [float((index + offset) % 5 - 2) for offset in range(len(ACTION_SPECS))]
        rows.append(
            {
                "candidates": [
                    {"candidate_id": name, "delta_J&F_points": delta}
                    for (name, _), delta in zip(ACTION_SPECS, deltas)
                ]
            }
        )
    summary = summarize_rows(rows)
    assert summary["samples"] == 8
    assert summary["oracle_vs_native_jf_points"]["min"] >= 0.0
    original = np.asarray([[[True, False], [False, True]]], dtype=bool)
    assert np.array_equal(unpack_binary_mask(pack_binary_mask(original)), original)
    print(json.dumps({"state": "self_test_passed", "actions": list(actions)}, indent=2))


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    required = (args.model_path, args.data_root, args.manifest, args.raw_map_dir, args.output)
    if not all(required):
        raise ValueError("model_path, data-root, manifest, raw-map-dir and output are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    output = Path(args.output)
    row_dir = output / "rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    selected = manifest["calibration"][: args.max_samples]
    protocol = protocol_hash(manifest)
    atomic_json(output / "status.json", {"state": "running", "phase": "model_load", "protocol_hash": protocol})

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
    projection = model.text_hidden_fcs
    projection_dtype = next(projection.parameters()).dtype

    root = Path(args.data_root)
    split = root / "video_datas" / "mevis" / "train"
    metadata = json.loads((split / "meta_expressions.json").read_text())["videos"]
    trajectory_dir = output / "frozen_trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    frozen: list[Path] = []

    for index, item in enumerate(selected, 1):
        event = build_event(root, item, metadata)
        payload = torch.load(
            Path(args.raw_map_dir) / f"{event['key']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        action_maps = build_action_maps(payload)
        seg, _, _, _ = extract_states(model, tokenizer, event, 5, (28,))
        language_embedding = projection(
            seg[0].to("cuda:0", dtype=projection_dtype)
        ).reshape(-1)
        images, size = preprocess_video(
            grounding,
            model.extra_image_processor,
            model.torch_dtype,
            event["frame_paths"],
        )
        width, height = size
        cache = cache_video_backbones(grounding, images)
        baseline_logits = infer_baseline_logits(
            grounding, images, cache, language_embedding, size
        )
        candidate_trajectories = []
        for action_name, _ in ACTION_SPECS:
            logits, points = infer_points(
                grounding,
                images,
                cache,
                language_embedding,
                action_maps[action_name],
                size,
                1,
                0,
            )
            candidate_trajectories.append(
                {
                    "candidate_id": action_name,
                    "mask": pack_binary_mask(logits.numpy() > 0),
                    "conditioning_points": points,
                }
            )
            del logits

        trajectory = {
            "key": event["key"],
            "video_id": event["video_id"],
            "exp_id": event["exp_id"],
            "annotation_ids": event["annotation_ids"],
            "frame_count": len(images),
            "output_size_wh": [width, height],
            "protocol_hash": protocol,
            "baseline_mask": pack_binary_mask(baseline_logits.numpy() > 0),
            "candidates": candidate_trajectories,
            "contract": "frozen h28 language; fixed raw/rank-fused layer maps; one point; complete-video propagation",
            "calibration_reuse": True,
            "sealed_test_access": False,
            "gt_access": False,
        }
        trajectory_path = trajectory_dir / f"{event['key']}.pt"
        atomic_torch(trajectory_path, trajectory)
        frozen.append(trajectory_path)
        print(
            f"LAYER_ORACLE_FREEZE [{index}/{len(selected)}] {event['key']} actions={len(candidate_trajectories)}",
            flush=True,
        )
        del payload, action_maps, seg, language_embedding, images, cache
        del baseline_logits
        gc.collect()
        torch.cuda.empty_cache()

    del model, grounding, projection
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(
        output / "status.json",
        {
            "state": "running" if not args.candidates_only else "complete",
            "phase": "all_candidate_trajectories_frozen",
            "completed": len(frozen),
            "protocol_hash": protocol,
            "gt_access": False,
            "router_fit": False,
        },
    )
    if args.candidates_only:
        return

    # Phase boundary: GT is opened only after every candidate trajectory is
    # frozen and the model has been unloaded.
    masks = json.loads((split / "mask_dict.json").read_text())
    completed_rows: list[dict[str, Any]] = []
    for trajectory_path in frozen:
        trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=False)
        width, height = trajectory["output_size_wh"]
        truth = ground_truth_video(
            masks,
            trajectory["annotation_ids"],
            trajectory["frame_count"],
            (height, width),
        )
        baseline = exact_metrics(truth, unpack_binary_mask(trajectory["baseline_mask"]))
        candidates = []
        for candidate in trajectory["candidates"]:
            metrics = exact_metrics(truth, unpack_binary_mask(candidate["mask"]))
            candidates.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "metrics": metrics,
                    "delta_J&F_points": float((metrics["J&F"] - baseline["J&F"]) * 100.0),
                    "conditioning_points": candidate["conditioning_points"],
                }
            )
        row = {
            "key": trajectory["key"],
            "video_id": trajectory["video_id"],
            "exp_id": trajectory["exp_id"],
            "protocol_hash": protocol,
            "baseline_h28": baseline,
            "candidates": candidates,
            "contract": trajectory["contract"],
            "calibration_reuse": True,
            "sealed_test_access": False,
            "gt_access_after_candidate_freeze": True,
        }
        atomic_json(row_dir / f"{trajectory['key']}.json", row)
        completed_rows.append(row)
        print(
            f"LAYER_ORACLE_SCORE {trajectory['key']} "
            + " ".join(
                f"{candidate['candidate_id']}={candidate['delta_J&F_points']:+.3f}pt"
                for candidate in candidates
            ),
            flush=True,
        )

    summary = summarize_rows(completed_rows)
    summary["protocol_hash"] = protocol
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "status.json",
        {
            "state": "complete",
            "phase": "done",
            "completed": len(completed_rows),
            "protocol_hash": protocol,
            "all_gates_pass": summary["all_gates_pass"],
        },
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
