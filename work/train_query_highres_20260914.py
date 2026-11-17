"""Controlled D4 pilot: h28 query-conditioned SAM2 high-resolution skip gate."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("VISION_SOURCE_DEPTH", "6")
ROOT = Path(".")
EXPERIMENT = "exp059_h28_query_highres_20260914_v1"
OUT = ROOT / "work/derived" / EXPERIMENT
D4 = ROOT / "work/derived/exp053_standalone_training_20260906_v1"
MODEL = Path("pretrained/Sa2VA-InternVL3-2B")
SEED, EPOCHS, WINDOW, START = 20260914, 6, 16, 12

import numpy as np
import torch
from torch.nn import functional as F

torch.set_num_threads(1)
sys.path.insert(0, str(ROOT))

from work.exp054_data import attach_training_truth, load_frozen_model
from work import prefix5_memoryspace as memoryspace
from work.vision_single_depth_future_gradient_audit_20260914 import capture_prompt_inputs, teacher_forced_seg
from tools.multilayer_seg.evaluate_spatial_map_affine_gate import build_event
from work.query_highres_gate_20260914 import DecoderHighResPatch, QueryHighResGate


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def decode_prefix(backend, images, embedding, patch):
    state = memoryspace.materialize(backend.init_state(images[:5]))
    obj = memoryspace.register_object(state)
    native, memories = [], []
    with memoryspace.autocast_for(state):
        for frame in range(5):
            with patch.active(embedding):
                current, _ = backend._run_single_frame_inference(
                    inference_state=state, output_dict=state["output_dict_per_obj"][obj], frame_idx=frame,
                    batch_size=1, is_init_cond_frame=True, point_inputs=None, mask_inputs=None, reverse=False,
                    run_mem_encoder=False, prev_sam_mask_logits=None, language_embd=embedding,
                )
            logits = current["pred_masks"].float()
            high = F.interpolate(logits, (backend.image_size, backend.image_size), mode="bilinear", align_corners=False)
            if backend.non_overlap_masks_for_mem_enc:
                high = backend._apply_non_overlapping_constraints(high)
            memory, positions = backend._run_memory_encoder(
                inference_state=state, frame_idx=frame, batch_size=1, high_res_masks=high, is_mask_from_pts=True
            )
            memories.append(memory[0])
            native.append({"pred_masks": current["pred_masks"].detach(), "obj_ptr": current["obj_ptr"].detach(),
                           "maskmem_features": memory.detach(), "maskmem_pos_enc": [p.detach() for p in positions]})
    return native, torch.stack(memories)


def average(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in ("future", "total")}


def run() -> None:
    started = time.time()
    torch.manual_seed(SEED); random.seed(SEED); torch.cuda.init()
    audit = json.loads((OUT / "audit/identity_gradient_result.json").read_text())
    if audit["state"] != "passed":
        raise RuntimeError("identity/gradient audit did not pass")
    d4_protocol = json.loads((D4 / "protocol.json").read_text())
    gate = QueryHighResGate().cuda().float()
    protocol = {
        "revision": EXPERIMENT,
        "hypothesis": "The final h28 SEG query spatially selects target-relevant detail in SAM2 high-resolution skip features, improving later frozen-SAM2 propagation beyond an identity native h28 path.",
        "primary_variable": "zero-initialized, h28-query-conditioned spatial residual applied to SAM2 feat_s0/feat_s1 only during the five prefix decodes",
        "architecture": {"query": "standalone-projector h28 SEG [1,1,256]", "features": [[32,256,256],[64,128,128]],
                         "per_level": "query-feature cosine gate; 1x1, depthwise-3x3, zero-init 1x1 residual", "parameters": sum(p.numel() for p in gate.parameters())},
        "frozen": {"model": str(MODEL), "LLM": True, "vision_encoder": True, "text_hidden_fcs": True, "SAM2": True,
                   "MLLM_context_frames": 5, "window": WINDOW, "future_start": START, "split": "exact consumed D4 128/32"},
        "optimization": {"optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-2, "clip": 2.0, "epochs": EPOCHS, "updates": 128 * EPOCHS, "loss": "unchanged D4 future objective"},
        "controls": {"epoch0": "bit-exact native h28 full propagation", "matched_capacity": "required later: query-free high-res adapter", "query_causality": "required later: query shuffle at inference"},
        "split": {key: d4_protocol["split"][key] for key in ("train", "development")},
        "GT_policy": "reuse previously consumed D4 train/development only; no confirmation GT",
        "checkpoint_selection": "minimum mean development future loss across epochs 0..6",
        "decision_scope": "consumed-cohort mechanism pilot only; not an independent efficacy claim",
        "source_sha256": {"gate": sha256(ROOT / "work/query_highres_gate_20260914.py"), "audit": sha256(OUT / "audit/identity_gradient_result.json"), "trainer": sha256(Path(__file__)), "D4_cache_manifest": sha256(D4 / "cache_manifest.json")},
    }
    existing = OUT / "protocol.json"
    if existing.exists() and json.loads(existing.read_text()) != protocol:
        raise RuntimeError("locked protocol differs from current sources")
    atomic_json(existing, protocol)
    cache_manifest = json.loads((D4 / "cache_manifest.json").read_text())
    records = {row["key"]: row for row in cache_manifest["records"]}
    if len(records) != 160:
        raise RuntimeError("expected frozen D4 160-record cache")
    metadata = json.loads((ROOT / "data/video_datas/mevis/train/meta_expressions.json").read_text())["videos"]
    allowed = {str(a) for row in records.values() for a in row["annotation_ids"]}
    from work import run_projector_isolation_20260906 as isolation
    masks = isolation.selected_gt(ROOT / "data/video_datas/mevis/train/mask_dict.json", allowed)
    atomic_json(OUT / "training_GT_access_receipt.json", {"selected_keys": sorted(records), "annotation_ids": sorted(allowed), "purpose": "consumed D4 pilot", "confirmation_access": False, "D4_cache_manifest_sha256": sha256(D4 / "cache_manifest.json")})
    targets = {}
    for key, row in records.items():
        clip = torch.load(D4 / "cache" / f"{key}.pt", map_location="cpu", weights_only=True, mmap=True)
        targets[key] = attach_training_truth(clip, masks)["truth"].clone()
        del clip
    del masks
    model, tokenizer = load_frozen_model(MODEL); model._count = 0
    backend = model.grounding_encoder.sam2_model
    patch = DecoderHighResPatch(backend.sam_mask_decoder, gate)
    rollout = memoryspace.MemoryRollout(backend)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-3, weight_decay=1e-2)
    history = []

    def item_loss(item, train: bool):
        clip = torch.load(D4 / "cache" / f"{item['key']}.pt", map_location="cpu", weights_only=True, mmap=True)
        clip["truth"] = targets[item["key"]]
        event = build_event(ROOT / "data", item, metadata)
        embedding = teacher_forced_seg(model, capture_prompt_inputs(model, tokenizer, event))
        native, memories = decode_prefix(backend, clip["images"], embedding, patch)
        result = rollout(clip["images"], native, memories, total_video_frames=clip["total_video_frames"], native_loop=not train, backprop_start=START, cpu_output=False)
        logits = result["logits"]
        truth = clip["truth"].to(logits.device)
        resized = F.interpolate(logits[START:, None].float(), truth.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        loss = memoryspace.frame_losses(resized, truth[START:]).mean()
        return loss, {"future": float(loss.detach()), "total": float(loss.detach())}

    try:
        for epoch in range(EPOCHS + 1):
            if epoch:
                gate.train()
                order = list(protocol["split"]["train"]); random.Random(SEED + epoch).shuffle(order)
                losses = []
                for item in order:
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = item_loss(item, True)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(gate.parameters(), 2.0); optimizer.step()
                    losses.append(float(loss.detach()))
                    gc.collect(); torch.cuda.empty_cache()
            gate.eval(); dev = []
            with torch.no_grad():
                for item in protocol["split"]["development"]:
                    _, row = item_loss(item, False); dev.append(row)
                    gc.collect(); torch.cuda.empty_cache()
            record = {"epoch": epoch, "development": average(dev), "train_mean": None if not epoch else float(np.mean(losses)), "gate_stats": gate.last_stats}
            history.append(record)
            torch.save({"epoch": epoch, "gate": gate.state_dict(), "optimizer": optimizer.state_dict(), "record": record}, OUT / f"gate_epoch{epoch}.pt")
            atomic_json(OUT / "history.json", history)
        best = min(history, key=lambda r: r["development"]["total"])
        torch.save({"epoch": best["epoch"], "gate": torch.load(OUT / f"gate_epoch{best['epoch']}.pt", map_location="cpu", weights_only=False)["gate"], "record": best}, OUT / "gate_best.pt")
        atomic_json(OUT / "training_result.json", {"state": "completed", "best": best, "history": history, "seconds": time.time() - started})
    finally:
        patch.close()


if __name__ == "__main__":
    try:
        run()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "TRAINING_FAILED.txt").write_text(traceback.format_exc())
        raise
