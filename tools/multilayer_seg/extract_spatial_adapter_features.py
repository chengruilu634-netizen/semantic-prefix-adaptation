#!/usr/bin/env python3
"""Cache frozen Sa2VA states and first-five-frame mask targets for map adapters."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as cocomask
from transformers import AutoModel, AutoTokenizer

from tools.multilayer_seg.hidden_state_extractor import extract_generation_seg_states
from tools.multilayer_seg.train_fusion_official_video_spsa import collect_video_events


LAYERS = (16, 20, 24, 28)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--train-seed", type=int, default=20260831)
    parser.add_argument("--validation-seed", type=int, default=20260818)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


def select_events(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    data_root = Path(args.data_root)
    validation = collect_video_events(
        data_root, args.validation_samples, 1, args.validation_seed
    )
    validation_videos = {event["video_id"] for event in validation}
    pool_count = args.train_samples + args.validation_samples * 2
    pool = collect_video_events(data_root, pool_count, 1, args.train_seed)
    train = [event for event in pool if event["video_id"] not in validation_videos]
    if len(train) < args.train_samples:
        raise RuntimeError(
            f"only found {len(train)}/{args.train_samples} video-disjoint training events"
        )
    train = train[: args.train_samples]
    train_videos = {event["video_id"] for event in train}
    if train_videos & validation_videos:
        raise AssertionError("training and validation video IDs overlap")
    return [("train", event) for event in train] + [
        ("validation", event) for event in validation
    ]


@torch.inference_mode()
def extract_states(
    model: Any,
    tokenizer: Any,
    event: dict[str, Any],
    frames: int,
    layers: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    images = [Image.open(path).convert("RGB") for path in event["frame_paths"][:frames]]
    capture: dict[str, Any] = {}
    original_generate = model.generate
    grounding = model.grounding_encoder
    original_language_inference = grounding.language_embd_inference

    def capture_generate(*generate_args: Any, **generate_kwargs: Any):
        capture["input_ids"] = generate_kwargs["input_ids"].detach().clone()
        result = original_generate(*generate_args, **generate_kwargs)
        capture["result"] = result
        return result

    def dummy_language_inference(state: Any, language: Any, return_obj_ptrs: bool = False):
        count = len(language)
        device = next(grounding.parameters()).device
        masks = torch.zeros(count, 1, 8, 8, device=device, dtype=torch.bfloat16)
        if not return_obj_ptrs:
            return masks
        pointers = torch.zeros(count, 1, 256, device=device, dtype=torch.bfloat16)
        return masks, pointers

    model.generate = capture_generate
    grounding.language_embd_inference = dummy_language_inference
    try:
        model.predict_forward(video=images, text=event["prompt"], tokenizer=tokenizer)
    finally:
        model.generate = original_generate
        grounding.language_embd_inference = original_language_inference
        for image in images:
            image.close()

    output = capture["result"]
    input_ids = capture["input_ids"]
    image_mask = input_ids[0].eq(int(model.img_context_token_id))
    frame_count = len(event["frame_paths"][:frames])
    patch_count = int(model.patch_token)
    if int(image_mask.sum()) != frame_count * patch_count:
        raise RuntimeError("prompt image-token count does not match visible video frames")
    side = int(round(math.sqrt(patch_count)))
    if side * side != patch_count:
        raise RuntimeError("patch_token is not a square grid")

    seg = extract_generation_seg_states(
        output.hidden_states,
        output.sequences,
        int(model.seg_token_idx),
        layers,
        expected_transformer_layers=28,
    ).states[0]
    first_step = output.hidden_states[0]
    image_states = torch.stack(
        [first_step[layer][0, image_mask].reshape(frame_count, patch_count, -1) for layer in layers]
    )
    if frame_count < frames:
        image_states = F.pad(image_states, (0, 0, 0, 0, 0, frames - frame_count))
    return seg.half().cpu(), image_states[:, :frames].half().cpu(), frame_count, side


def mask_target(
    event: dict[str, Any], mask_dict: dict[str, Any], frames: int, side: int
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_frames = min(frames, len(event["frame_paths"]))
    with Image.open(event["frame_paths"][0]) as image:
        width, height = image.size
    masks = np.zeros((frames, height, width), dtype=np.uint8)
    for frame_index in range(valid_frames):
        for annotation_id in event["annotation_ids"]:
            track = mask_dict.get(str(annotation_id))
            if track is None or frame_index >= len(track) or not track[frame_index]:
                continue
            decoded = cocomask.decode(track[frame_index]).astype(np.uint8)
            if decoded.ndim == 3:
                decoded = decoded.any(axis=2).astype(np.uint8)
            masks[frame_index] |= decoded
    target = torch.from_numpy(masks).float().unsqueeze(1)
    target = F.interpolate(target, size=(side, side), mode="area")
    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3) / 16.0
    target = F.conv2d(target, kernel, padding=1).squeeze(1).clamp(0.0, 1.0)
    valid = torch.zeros(frames, dtype=torch.bool)
    valid[:valid_frames] = True
    return target.half(), valid


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen Sa2VA extraction")
    if min(args.layers) < 1 or max(args.layers) > 28:
        raise ValueError("layers must be in [1,28]")
    layers = tuple(args.layers)
    output = Path(args.output)
    shards = output / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    status = output / "status.json"
    events = select_events(args)
    split_root = Path(args.data_root) / "video_datas" / "mevis" / "train"
    mask_dict = json.loads((split_root / "mask_dict.json").read_text())

    atomic_json(status, {"state": "running", "phase": "model_load"})
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

    metadata = []
    for number, (split, event) in enumerate(events, 1):
        shard_path = shards / f"{number:04d}_{event['key']}.pt"
        if not shard_path.exists():
            seg, image, valid_frames, side = extract_states(
                model, tokenizer, event, args.frames, layers
            )
            target, valid = mask_target(event, mask_dict, args.frames, side)
            torch.save(
                {
                    "key": event["key"],
                    "video_id": event["video_id"],
                    "split": split,
                    "seg_states": seg,
                    "image_states": image,
                    "targets": target,
                    "valid_frames": valid,
                },
                shard_path,
            )
        metadata.append({"key": event["key"], "video_id": event["video_id"], "split": split})
        atomic_json(status, {
            "state": "running", "phase": "feature_extraction",
            "completed": number, "total": len(events), "key": event["key"], "split": split,
        })
        print(f"ADAPTER_FEATURE [{number}/{len(events)}] {split} {event['key']}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(status, {"state": "running", "phase": "consolidation", "total": len(events)})
    loaded = [torch.load(path, map_location="cpu", weights_only=False) for path in sorted(shards.glob("*.pt"))]
    if len(loaded) != len(events):
        raise RuntimeError(f"expected {len(events)} shards, found {len(loaded)}")
    payload = {
        "keys": [item["key"] for item in loaded],
        "video_ids": [item["video_id"] for item in loaded],
        "splits": [item["split"] for item in loaded],
        "layers": layers,
        "seg_states": torch.stack([item["seg_states"] for item in loaded]),
        "image_states": torch.stack([item["image_states"] for item in loaded]),
        "targets": torch.stack([item["targets"] for item in loaded]),
        "valid_frames": torch.stack([item["valid_frames"] for item in loaded]),
        "contract": "frozen same-layer generated SEG and first-five-frame LLM image tokens; video-disjoint split",
    }
    torch.save(payload, output / "spatial_adapter_features.pt")
    atomic_json(output / "manifest.json", {
        "samples": len(events),
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "layers": layers,
        "seg_shape": list(payload["seg_states"].shape),
        "image_shape": list(payload["image_states"].shape),
        "target_shape": list(payload["targets"].shape),
        "video_disjoint": not bool(
            {m["video_id"] for m in metadata if m["split"] == "train"}
            & {m["video_id"] for m in metadata if m["split"] == "validation"}
        ),
    })
    atomic_json(status, {"state": "complete", "phase": "done", "samples": len(events)})


if __name__ == "__main__":
    main()
