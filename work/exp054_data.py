"""Frozen no-GT prefix cache construction and explicit train/development inputs."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from work.prefix5_propagation_training import PREFIX, REVISION, materialize

SAM2_SOURCE_SHA256 = "c48128f3e4283032a839ec8182cf702561028c7e5700fd32822df531eb020c3c"


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_model(model_path):
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(str(model_path), torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model.preparing_for_generation(tokenizer=tokenizer)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    source = Path(inspect.getfile(type(model.grounding_encoder.sam2_model)))
    if sha_file(source) != SAM2_SOURCE_SHA256:
        raise RuntimeError("SAM2 implementation differs from the audited source")
    return model, tokenizer


@torch.no_grad()
def prepare_clip(model, tokenizer, root, metadata, item, *, window_frames=7):
    """Generate only five auxiliary frames; retain full video-length metadata."""
    from work import evaluate_prefix5_direct_memory_fusion as core
    if window_frames <= PREFIX:
        raise ValueError("a prefix+suffix window is required")
    event = core.base.build_event(Path(root) / "data", item, metadata)
    if not event["annotation_ids"] or len(event["frame_paths"]) <= PREFIX:
        raise ValueError("an annotated video longer than five frames is required")
    row_seed = (core.SEED + int(hashlib.sha256(item["key"].encode()).hexdigest()[:8], 16)) % (2**32)
    torch.manual_seed(row_seed)
    import numpy as np
    np.random.seed(row_seed)
    states, _, _, _ = core.base.extract_states(model, tokenizer, event, PREFIX, core.base.ALL_LAYERS)
    aligned = core.base.TgifSegRouter.rms_align(states.float().cuda())
    projection = model.text_hidden_fcs
    embeddings = projection(aligned.to(next(projection.parameters()).dtype)).reshape(len(core.base.ALL_LAYERS), -1)
    parent = torch.load(Path(root) / "work/derived/exp_037_tgif_seg_layer_router/pilot8/checkpoint_best.pt",
                        map_location="cpu", weights_only=False)
    if tuple(parent["layer_indices"]) != core.base.ALL_LAYERS or parent["best_epoch"] != 3:
        raise RuntimeError("query preprocessor checkpoint contract changed")
    preprocessor = core.base.preprocessor_from_checkpoint(parent)
    query = core.base.transform_query(core.base.query_embedding(model, tokenizer, event["prompt"])[None], preprocessor)[0].half().float()
    grounding = model.grounding_encoder
    images, size = core.base.preprocess_video(grounding, model.extra_image_processor,
        model.torch_dtype, event["frame_paths"][:window_frames])
    images = images.cpu()
    high, low = [], []
    for layer in core.base.LAYERS:
        layer_high, layer_low, _ = core.native_trajectory(grounding, images[:PREFIX], embeddings[layer - 1])
        high.append(layer_high)
        low.append(layer_low)
    return materialize({"revision": REVISION, **item, "images": images,
        "query": query.cpu(), "embedding": embeddings[-1].cpu(),
        "prefix_high": torch.stack(high), "prefix_low": torch.stack(low),
        "total_video_frames": len(event["frame_paths"]), "window_frames": len(images),
        "output_size_wh": size, "annotation_ids": event["annotation_ids"],
        "auxiliary_frames": PREFIX, "gt_access": False, "sam2_source_sha256": SAM2_SOURCE_SHA256})


def historical_exclusions(root):
    path = Path(root) / "work/derived/exp_053_prefix5_disjoint_confirmation32/confirmation32/selection_manifest.json"
    selection = json.loads(path.read_text())
    excluded = set(selection["excluded_video_ids"]) | {r["video_id"] for r in selection["selected"]}
    if len(excluded) != 986:
        raise RuntimeError("unexpected historical exclusion count; review before choosing new data")
    return excluded


def validate_manifest(manifest, metadata, excluded):
    if manifest.get("schema") != "exp054_train_dev_v1" or manifest.get("purpose") != "training_development_not_final_evaluation":
        raise ValueError("an explicit exp054 train/development manifest is required")
    if set(manifest) != {"schema", "purpose", "train", "development"}:
        raise ValueError("unexpected manifest fields or test/valid split")
    if not manifest["train"] or not manifest["development"]:
        raise ValueError("both training and checkpoint-development videos are required")
    seen = set()
    for split in ("train", "development"):
        for item in manifest[split]:
            if set(item) != {"key", "video_id", "exp_id"}:
                raise ValueError("items require only key, video_id, exp_id")
            vid = item["video_id"]
            if vid in seen or vid in excluded or vid not in metadata:
                raise ValueError("reused, excluded, or unknown video")
            if item["key"] != vid + "_" + item["exp_id"]:
                raise ValueError("key mismatch")
            video = metadata[vid]
            expression = video["expressions"].get(item["exp_id"])
            if len(video["frames"]) <= PREFIX or not expression or not expression.get("anno_id"):
                raise ValueError("missing expression/annotation or insufficient frames")
            seen.add(vid)
    return manifest


def attach_training_truth(clip, mask_dict, *, loss_size=128):
    """Only an explicitly registered training/development run calls this function."""
    import numpy as np
    from work.train_evaluate_tgif_seg_router import decode_frame_mask
    width, height = clip["output_size_wh"]
    truth = np.stack([decode_frame_mask(mask_dict, clip["annotation_ids"], i, (height, width))
                      for i in range(len(clip["images"]))])
    target = F.interpolate(torch.from_numpy(truth).float()[:, None], size=(loss_size, loss_size), mode="nearest")[:, 0].bool()
    return {**clip, "truth": target, "gt_access": True}
