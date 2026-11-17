"""Trace synthetic future-frame loss through SAM2, the LLM, and Vision-PKVA.

Uses one previously consumed D4 development example and no ground-truth masks.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import MethodType


ROOT = Path(".")
EXPERIMENT = os.environ.get("VISION_FUSION_EXPERIMENT", "exp057_single_depth_probe_20260914_v1")
FUSION_KIND = os.environ.get("VISION_FUSION_KIND", "single_depth")
SELECTED_DEPTH = int(os.environ["VISION_SOURCE_DEPTH"])
OUT = ROOT / "work/derived" / EXPERIMENT / "audit"
MODEL = Path("pretrained/Sa2VA-InternVL3-2B")
SEED = 20260913
DEPTHS = (SELECTED_DEPTH, 24)
WINDOW = 8

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
sys.path.insert(0, str(ROOT))

from work.vision_single_depth_audit_20260914 import atomic_json, build_fusion, gradient_table, sha256


class InputsCaptured(RuntimeError):
    pass


def capture_prompt_inputs(model, tokenizer, event):
    captured = {}
    original = model.generate

    def capture_generate(**kwargs):
        captured.update(kwargs)
        raise InputsCaptured

    model.generate = capture_generate
    images = [Image.open(path).convert("RGB") for path in event["frame_paths"][:5]]
    try:
        try:
            model.predict_forward(video=images, text=event["prompt"], tokenizer=tokenizer)
        except InputsCaptured:
            pass
    finally:
        for image in images:
            image.close()
        model.generate = original
    required = {"pixel_values", "input_ids", "attention_mask"}
    if not required.issubset(captured):
        raise RuntimeError(f"prompt capture missing {sorted(required - set(captured))}")
    return captured


def fused_extract(self, pixel_values, fusion):
    vision = self.vision_model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
    states = [vision.hidden_states[depth][:, 1:, :] for depth in DEPTHS]
    plan = fusion(states)
    embeds = plan["fused"]
    batch, tokens, width = embeds.shape
    side = int(tokens**0.5)
    embeds = embeds.reshape(batch, side, side, width)
    embeds = self.pixel_shuffle(embeds, scale_factor=self.downsample_ratio)
    embeds = embeds.reshape(batch, -1, embeds.shape[-1])
    self._vision_pkva_last_plan = plan
    return self.mlp1(embeds)


def teacher_forced_seg(model, mm_inputs):
    input_ids = mm_inputs["input_ids"]
    seg = torch.tensor([[int(model.seg_token_idx)]], device=input_ids.device, dtype=input_ids.dtype)
    input_ids = torch.cat([input_ids, seg], dim=1)
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    pixels = mm_inputs["pixel_values"].to(device="cuda", dtype=model.vision_model.dtype)
    outputs = model._llm_forward(
        pixel_values=pixels,
        input_ids=input_ids,
        attention_mask=attention,
        position_ids=None,
        image_flags=torch.ones(len(pixels), 1, device="cuda", dtype=torch.long),
        labels=None,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1][:, -1, :]
    return model.text_hidden_fcs(hidden).reshape(1, 1, -1)


def decode_live_prefix(backend, images, embedding, memoryspace):
    state = backend.init_state(images[:5])
    state = memoryspace.materialize(state)
    obj = memoryspace.register_object(state)
    native, memories = [], []
    with memoryspace.autocast_for(state):
        for frame in range(5):
            current, _ = backend._run_single_frame_inference(
                inference_state=state,
                output_dict=state["output_dict_per_obj"][obj],
                frame_idx=frame,
                batch_size=1,
                is_init_cond_frame=True,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=False,
                prev_sam_mask_logits=None,
                language_embd=embedding,
            )
            logits = current["pred_masks"].float()
            high = F.interpolate(logits, (backend.image_size, backend.image_size),
                                 mode="bilinear", align_corners=False)
            if backend.non_overlap_masks_for_mem_enc:
                high = backend._apply_non_overlapping_constraints(high)
            memory, positions = backend._run_memory_encoder(
                inference_state=state,
                frame_idx=frame,
                batch_size=1,
                high_res_masks=high,
                is_mask_from_pts=True,
            )
            memories.append(memory[0])
            native.append({
                "pred_masks": current["pred_masks"].detach(),
                "obj_ptr": current["obj_ptr"].detach(),
                "maskmem_features": memory.detach(),
                "maskmem_pos_enc": [position.detach() for position in positions],
            })
    return native, torch.stack(memories)


def run() -> None:
    started = time.time()
    torch.manual_seed(SEED)
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info()

    from work.exp054_data import load_frozen_model
    from work import prefix5_memoryspace as memoryspace
    from tools.multilayer_seg.evaluate_spatial_map_affine_gate import build_event
    from tools.multilayer_seg.train_eval_fusion_standard import preprocess_video

    protocol_path = ROOT / "work/derived/exp053_standalone_training_20260906_v1/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    item = protocol["split"]["development"][0]
    metadata = json.loads((ROOT / "data/video_datas/mevis/train/meta_expressions.json").read_text())["videos"]
    event = build_event(ROOT / "data", item, metadata)

    model, tokenizer = load_frozen_model(MODEL)
    # The training wrapper normally owns this diagnostic counter, while the
    # exported HF model's direct _llm_forward path still increments it.
    model._count = 0
    frozen_versions = {name: parameter._version for name, parameter in model.named_parameters()}
    mm_inputs = capture_prompt_inputs(model, tokenizer, event)
    images, _ = preprocess_video(model.grounding_encoder, model.extra_image_processor,
                                 model.torch_dtype, event["frame_paths"][:WINDOW])
    images = memoryspace.materialize(images.cpu())
    if len(images) != WINDOW:
        raise RuntimeError("fixture is shorter than the audit window")

    with torch.no_grad():
        native_embedding = teacher_forced_seg(model, mm_inputs).detach().cpu()

    fusion, required = build_fusion(FUSION_KIND)
    fusion = fusion.to(device="cuda", dtype=model.vision_model.dtype)
    original_extract = model.extract_feature
    model.extract_feature = MethodType(lambda self, pixels: fused_extract(self, pixels, fusion), model)
    backend = model.grounding_encoder.sam2_model
    rollout = memoryspace.MemoryRollout(backend)

    def future_pass():
        embedding = teacher_forced_seg(model, mm_inputs)
        native, memories = decode_live_prefix(backend, images, embedding, memoryspace)
        result = rollout(images, native, memories, total_video_frames=WINDOW,
                         native_loop=False, backprop_start=5, cpu_output=False)
        logits = result["logits"][5:]
        rows = torch.arange(logits.shape[-2], device=logits.device)[:, None]
        cols = torch.arange(logits.shape[-1], device=logits.device)[None, :]
        target = ((rows // 32 + cols // 32) % 2).float().expand_as(logits)
        loss = F.binary_cross_entropy_with_logits(logits.float(), target)
        return loss, embedding, result

    first_loss, first_embedding, first_result = future_pass()
    if not torch.equal(first_embedding.detach().cpu(), native_embedding):
        raise RuntimeError("zero fusion changed the teacher-forced SEG embedding")
    first_loss.backward()
    first_gradients = gradient_table(fusion)
    if first_gradients["out_proj.weight"]["l1"] <= 0 or not first_gradients["out_proj.weight"]["finite"]:
        raise RuntimeError("future loss does not reach the zero-initialized output projection")

    optimizer = torch.optim.SGD(fusion.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    del first_loss, first_embedding, first_result
    torch.cuda.empty_cache()

    second_loss, _, second_result = future_pass()
    second_loss.backward()
    second_gradients = gradient_table(fusion)
    missing = [name for name in required
               if not second_gradients[name]["finite"] or second_gradients[name]["l1"] <= 0]
    if missing:
        raise RuntimeError(f"post-update future gradient missing for {missing}")

    model.extract_feature = original_extract
    changed = [name for name, parameter in model.named_parameters()
               if parameter._version != frozen_versions[name]]
    frozen_grads = [name for name, parameter in model.named_parameters() if parameter.grad is not None]
    if changed or frozen_grads:
        raise RuntimeError(f"frozen model violation: changed={changed[:3]}, grads={frozen_grads[:3]}")

    receipt = {
        "state": "passed",
        "decision": "PASS_future_frame_gradient_reaches_all_intended_fusion_groups_after_disposable_update",
        "fusion_kind": FUSION_KIND,
        "scope": "five_real_frame_MLLM_context; five_real_prefix_frames; three_real_suffix_frames; synthetic_target",
        "item": item,
        "depths": list(DEPTHS),
        "window": WINDOW,
        "GT_access": False,
        "identity": {"teacher_forced_SEG_embedding_bit_exact_at_zero": True},
        "gradient": {"first_step": first_gradients, "post_disposable_update": second_gradients,
                     "required_nonzero": list(required), "first_loss": float(first_loss) if False else None,
                     "second_loss": float(second_loss.detach())},
        "propagation": {"indices": second_result["indices"], "future_frames": 3,
                        "real_memory_attention": True, "real_memory_encoder": True},
        "frozen_model": {"parameter_versions_unchanged": True, "parameter_grads_absent": True},
        "protocol_sha256": sha256(protocol_path),
        "source_sha256": sha256(Path(__file__)),
        "gpu": {"name": torch.cuda.get_device_name(), "free_mib_at_admission": int(free / 1024**2)},
        "seconds": time.time() - started,
    }
    atomic_json(OUT / "future_gradient_result.json", receipt)
    print(json.dumps({"state": receipt["state"], "decision": receipt["decision"],
                      "item": item["key"], "second_loss": receipt["gradient"]["second_loss"],
                      "seconds": receipt["seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "FUTURE_GRADIENT_FAILED.txt").write_text(traceback.format_exc())
        raise
