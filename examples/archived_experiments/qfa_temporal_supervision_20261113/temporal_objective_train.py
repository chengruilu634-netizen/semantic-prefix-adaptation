"""Train QFA with a matched four-frame gradient window and prefix supervision."""
from contextlib import nullcontext
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

from register import D4, ROOT, RUN, sha, write

sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from objective_terms import decode_with_prefix_loss
from work import train_query_highres_20260914 as reference
from tools.multilayer_seg.train_eval_fusion_standard import preprocess_video
class StopGradientProxy:
    def __init__(self, proxy, stop, window):
        self.proxy = proxy
        self.stop = stop
        self.window = window

    def propagate_in_video(self, state, **kwargs):
        generator = self.proxy.propagate_in_video(state, **kwargs)
        try:
            for frame in range(self.window):
                with torch.no_grad() if frame >= self.stop else nullcontext():
                    row = next(generator)
                yield row
        finally:
            generator.close()


def fit(protocol, arm, data):
    events, old, targets = data
    seed = protocol["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    gate = reference.QueryHighResGate().cuda().float()
    model, tokenizer = reference.load_frozen_model(Path(protocol["model"]))
    model._count = 0
    backend = model.grounding_encoder.sam2_model
    patch = reference.DecoderHighResPatch(backend.sam_mask_decoder, gate)
    rollout = reference.memoryspace.MemoryRollout(backend)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=protocol["lr"], weight_decay=protocol["weight_decay"])
    start = protocol["future_start"]
    window = protocol["window"]
    interval = protocol["validate_every"]
    prefix_weight = protocol["prefix_weight"]
    audit_windows = set()

    def new_images(event):
        return preprocess_video(model.grounding_encoder, model.extra_image_processor, model.torch_dtype,
                                event["frame_paths"][:window])

    def loss_for(item, training, first=start):
        key = item
        event = events[key]
        if key in old:
            clip = torch.load(D4 / "cache" / f"{key}.pt", map_location="cpu", weights_only=True, mmap=True)
            images, total_frames = clip["images"], clip["total_video_frames"]
        else:
            images, _ = new_images(event)
            images, total_frames = images.cpu(), len(event["frame_paths"])
        embedding = reference.teacher_forced_seg(model, reference.capture_prompt_inputs(model, tokenizer, event))
        if training:
            native, memories, prefix_loss = decode_with_prefix_loss(
                backend, images, embedding, patch, targets[key])
        else:
            native, memories = reference.decode_prefix(backend, images, embedding, patch)
        stop = first + 4
        proxy = rollout.proxy
        if training and stop < window:
            rollout.proxy = StopGradientProxy(proxy, stop, window)
        try:
            result = rollout(images, native, memories, total_video_frames=total_frames,
                             native_loop=not training, backprop_start=first, cpu_output=False)
        finally:
            rollout.proxy = proxy
        logits = result["logits"]
        truth = targets[key].to(logits.device)
        resized = F.interpolate(logits[first:stop, None].float(), truth.shape[-2:],
                                mode="bilinear", align_corners=False)[:, 0]
        future_loss = reference.memoryspace.frame_losses(resized, truth[first:stop]).mean()
        if training and first not in audit_windows:
            assert result["indices"] == list(range(window))
            state = result["state"]["output_dict"]["non_cond_frame_outputs"]
            assert all(state[i]["maskmem_features"].requires_grad == (first <= i < stop)
                       for i in range(5, window))
            (memory_grad,) = torch.autograd.grad(future_loss, memories, retain_graph=True)
            assert torch.isfinite(memory_grad).all() and memory_grad.abs().max() > 0
            audit_windows.add(first)
            write(RUN / f"gradient_window_{first + 1}_{stop}.json",
                  {"key": key, "forward_frames": window, "first_frame": first + 1,
                   "last_frame": stop, "prefix_memory_gradient_max_abs": float(memory_grad.abs().max())})
        return future_loss + prefix_weight * prefix_loss if training else future_loss

    def update(item, first):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(item, True, first)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Nonfinite loss for {item}")
        loss.backward()
        value = float(loss.detach())
        del loss
        grad = torch.nn.utils.clip_grad_norm_(gate.parameters(), protocol["clip"], error_if_nonfinite=True)
        optimizer.step()
        gc.collect()
        return value, float(grad)

    try:
        out = RUN / f"qfa_{arm}"
        out.mkdir()
        schedule = protocol["schedules"][arm]
        starts = protocol["supervision_starts"]
        history, losses = [], []
        updates = protocol["updates"]
        step_log = (out / "steps.jsonl").open("x", buffering=1)
        for step in range(updates + 1):
            if step:
                step_seed = seed + step
                random.seed(step_seed)
                np.random.seed(step_seed)
                torch.manual_seed(step_seed)
                key = schedule[step - 1]
                gate.train()
                first = starts[step - 1]
                loss, grad = update(key, first)
                losses.append(loss)
                step_log.write(json.dumps({"step": step, "key": key, "seed": step_seed,
                                           "supervised_frames": [first + 1, first + 4],
                                           "loss": loss, "gradient_norm": grad}) + "\n")
                if step == 1 or step % 16 == 0:
                    write(RUN / "progress.json", {"state": "training", "arm": arm, "update": step, "total": updates,
                                                   "supervised_frames": [first + 1, first + 4],
                                                   "key": key, "loss": loss, "gradient_norm": grad, "time": time.time()})
            if step % interval == 0:
                gate.eval()
                with torch.no_grad():
                    dev = [float(loss_for(item["key"], False)) for item in protocol["development"]]
                assert all(math.isfinite(value) for value in dev)
                record = {"updates": step, "development_future_loss": float(np.mean(dev)),
                          "train_mean": float(np.mean(losses)) if losses else None}
                losses.clear()
                if not history or record["development_future_loss"] < min(r["development_future_loss"] for r in history):
                    temporary = out / "gate_best.tmp"
                    torch.save({"gate": gate.state_dict(), "record": record}, temporary)
                    temporary.replace(out / "gate_best.pt")
                if step == updates:
                    fixed = out / "fixed"
                    fixed.mkdir()
                    torch.save({"gate": gate.state_dict(), "record": record}, fixed / "gate_best.pt")
                history.append(record)
                write(out / "history.json", history)
                print(arm, step, record, flush=True)
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
        write(out / "complete.json", {"best": min(history, key=lambda row: row["development_future_loss"]),
                                     "updates": updates, "checkpoint_sha256": sha(out / "gate_best.pt"),
                                     "fixed_checkpoint_sha256": sha(out / "fixed/gate_best.pt")})
    finally:
        if "step_log" in locals():
            step_log.close()
        patch.close()



