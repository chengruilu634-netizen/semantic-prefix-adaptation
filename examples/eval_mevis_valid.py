"""Matched full MeViS valid export: frozen v4b Q+M versus native Sa2VA P.

Both arms use the same model, manifest, prefix and propagation implementation.
The hidden valid split has no locally usable labels; this runner only freezes
GT-free predictions for separate official submissions.
"""
from __future__ import annotations

import gc, hashlib, json, os, sys, time
from pathlib import Path

MODE, VARIANT = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else ("", "")
if MODE not in {"lock", "infer"} or VARIANT not in {"on", "baseline"}:
    raise SystemExit("usage: eval_mevis_valid_v4b_pair_20260927.py {lock,infer} {on,baseline}")
os.environ.setdefault("VISION_SOURCE_DEPTH", "6")

ROOT = Path(".")
EXP = ROOT / "work/prefix_joint_20260918"
V4B = EXP / "improvement_v4b_exp059Q_M"
OUT = V4B / f"external_mevis_valid_{VARIANT}_20260927_v1"
MODEL = Path("pretrained/Sa2VA-InternVL3-2B")
QCKPT = ROOT / "work/derived/exp059_h28_query_highres_20260914_v1/gate_best.pt"
MCKPT = V4B / "runs/exp059Q_M/best.pt"
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from pycocotools import mask as cm
from third_parts.revos.utils.metircs import db_eval_boundary, db_eval_iou
from tools.multilayer_seg.train_eval_fusion_standard import extract_states_without_sam, preprocess_video, prompt_for
from work.exp054_data import load_frozen_model
from work import evaluate_recurrent_one_shot_fused_memory as base
from work.prefix5_propagation_training import materialize
from work.prefix_joint_20260918.joint_forward import JointMemoryFusion, JointPrefix
from work.query_highres_gate_20260914 import QueryHighResGate

def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)); tmp.replace(path)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
def encode(masks):
    out = []
    for mask in masks:
        rle = cm.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode(); out.append(rle)
    return out
def decode(rle): return cm.decode({**rle, "counts": rle["counts"].encode()})
def source():
    ds = {"meta": "data/video_datas/mevis/valid/meta_expressions.json",
          "images": "data/video_datas/mevis/valid/JPEGImages"}
    meta = read(ROOT / ds["meta"])["videos"]
    ds["items"] = [{"key": f"{vid}_{eid}", "video_id": vid, "exp_id": str(eid)}
                   for vid, video in sorted(meta.items())
                   for eid in sorted(video["expressions"], key=int)]
    return ds, meta
def manifest(ds):
    return [{"key": str(x["key"]), "video_id": str(x["video_id"]), "exp_id": str(x["exp_id"])} for x in ds["items"]]
def embedding(model, tokenizer, prompt, paths):
    states = extract_states_without_sam(model, tokenizer, paths[:5], prompt, (21, 24, 25, 28))
    states = base.TgifSegRouter.rms_align(states.float().cuda())
    return model.text_hidden_fcs(states.to(next(model.text_hidden_fcs.parameters()).dtype)).reshape(4, -1)

def lock():
    ds, meta = source(); items = manifest(ds)
    train_protocol, mdone = read(V4B / "protocol.json"), read(MCKPT.with_name("complete.json"))
    if OUT.exists(): raise RuntimeError(f"output already exists: {OUT}")
    if train_protocol["Q_checkpoint"]["sha256"] != sha(QCKPT): raise RuntimeError("Exp059 Q checkpoint receipt mismatch")
    if mdone["state"] != "complete" or mdone["best_checkpoint_sha256"] != sha(MCKPT): raise RuntimeError("v4b M checkpoint receipt mismatch")
    if len(meta) != 140 or len(items) != 2236 or len({x["key"] for x in items}) != 2236:
        raise RuntimeError("full MeViS valid manifest mismatch")
    prior = read(EXP / "baseline_P_mevis_valid_v1/protocol.json")
    if prior["manifest_sha256"] != hashlib.sha256(json.dumps(
            [{**x, "frames": len(meta[x["video_id"]]["frames"]),
              "annotation_ids": meta[x["video_id"]]["expressions"][x["exp_id"]].get("anno_id")}
             for x in items], sort_keys=True).encode()).hexdigest():
        raise RuntimeError("existing full-valid baseline manifest differs")
    p = {
        "revision": f"v4b_exp059Q_mevis_valid_{VARIANT}_20260927_v1", "dataset": "MeViS valid",
        "scope": "matched hidden-test export of v4b Q+M and native P on 140 videos / 2236 expressions",
        "GT_access": False, "items": items,
        "manifest_sha256": hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest(), "metadata_sha256": sha(ROOT / ds["meta"]),
        "Q_checkpoint_sha256": sha(QCKPT), "M_checkpoint_sha256": sha(MCKPT), "v4b_protocol_sha256": sha(V4B / "protocol.json"),
        "method": ("v4b Exp059 Q + trained plain residual M" if VARIANT == "on" else "native Sa2VA P, Q and M disabled"),
        "arm": "J" if VARIANT == "on" else "P",
        "comparison": "same frozen backbone, same item/frame list and inference code; separate official submissions required",
        "scoring": "local GT unavailable on hidden MeViS valid",
        "code_sha256": {str(Path(__file__)): sha(Path(__file__)), str(EXP / "joint_forward.py"): sha(EXP / "joint_forward.py"), str(ROOT / "work/query_highres_gate_20260914.py"): sha(ROOT / "work/query_highres_gate_20260914.py")},
    }
    write(OUT / "protocol.json", p); print("LOCKED", len(items), sha(OUT / "protocol.json"), flush=True)

def infer():
    p = read(OUT / "protocol.json"); ds, meta = source()
    if p["GT_access"] is not False or sha(ROOT / ds["meta"]) != p["metadata_sha256"]: raise RuntimeError("protocol/data mismatch")
    qstate = torch.load(QCKPT, map_location="cpu", weights_only=True)
    mstate = torch.load(MCKPT, map_location="cpu", weights_only=True)
    model, tokenizer = load_frozen_model(MODEL); model._count = 0
    joint = JointPrefix(model.grounding_encoder.sam2_model, QueryHighResGate().cuda().float(), JointMemoryFusion().cuda().float())
    joint.gate.load_state_dict(qstate["gate"]); joint.fusion.load_state_dict(mstate["fusion"]); joint.gate.eval(); joint.fusion.eval()
    versions = {n: x._version for n, x in model.named_parameters()}; records = []
    try:
        with torch.no_grad():
            for i, item in enumerate(p["items"], 1):
                path = OUT / "predictions" / (item["key"] + ".json")
                if path.exists():
                    old = read(path); records.append({"key": item["key"], "video_id": item["video_id"], "frames": old["frames"], "sha256": sha(path)}); continue
                video = meta[item["video_id"]]
                paths = [str(ROOT / ds["images"] / item["video_id"] / (f if Path(f).suffix else f + ".jpg")) for f in video["frames"]]
                images, size = preprocess_video(model.grounding_encoder, model.extra_image_processor, model.torch_dtype, paths)
                images = materialize(images.cpu()); emb = embedding(model, tokenizer, prompt_for(video["expressions"][item["exp_id"]]["exp"]), paths)
                decoded = joint.decode(images, emb, p["arm"])
                result = joint.propagate(images, decoded, total_video_frames=len(paths), backprop_start=min(12, len(images) - 1), native_loop=True, cpu_output=True)
                masks = encode(base.binary_at_output(result["logits"], size, torch.device("cuda")))
                if result["indices"] != list(range(len(paths))) or len(masks) != len(paths): raise RuntimeError("prediction coverage failure")
                record = {**item, "frames": len(paths), "output_size_wh": list(size), "protocol_sha256": sha(OUT / "protocol.json"), "GT_access": False, "prediction_masks": masks}
                write(path, record); records.append({"key": item["key"], "video_id": item["video_id"], "frames": len(paths), "sha256": sha(path)})
                write(OUT / "progress.json", {"completed": i, "total": len(p["items"]), "last_key": item["key"], "GT_access": False})
                print("DONE", i, len(p["items"]), item["key"], flush=True)
                del images, emb, decoded, result, masks, record; gc.collect(); torch.cuda.empty_cache()
        if len(records) != len(p["items"]) or not all(x.grad is None and x._version == versions[n] for n, x in model.named_parameters()): raise RuntimeError("prediction freeze check failed")
        write(OUT / "prediction_freeze.json", {"state": "complete", "files": records, "protocol_sha256": sha(OUT / "protocol.json"), "GT_access": False, "time": time.time()})
        print("FROZEN", len(records), flush=True)
    finally: joint.close()

{"lock": lock, "infer": infer}[MODE]()
