"""Vision-PKVA structural, identity, and module-gradient audit.

This is an engineering prerequisite only.  It does not train on or read GT.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(".")
EXPERIMENT = os.environ.get("VISION_FUSION_EXPERIMENT", "exp057_single_depth_probe_20260914_v1")
FUSION_KIND = os.environ.get("VISION_FUSION_KIND", "single_depth")
SELECTED_DEPTH = int(os.environ["VISION_SOURCE_DEPTH"])
OUT = ROOT / "work/derived" / EXPERIMENT / "audit"
MODEL = Path("pretrained/Sa2VA-InternVL3-2B")
DEPTHS = (SELECTED_DEPTH, 24)
SEED = 20260913

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import torch
from torch import nn

torch.set_num_threads(1)
sys.path.insert(0, str(ROOT))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VisionPKVA(nn.Module):
    """Per-patch attention across aligned InternViT depths with a native anchor."""

    def __init__(self, hidden_size: int = 1024, heads: int = 4, head_dim: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in DEPTHS])
        self.q_proj = nn.Linear(hidden_size, inner)
        self.k_proj = nn.Linear(hidden_size, inner)
        self.v_proj = nn.Linear(hidden_size, inner)
        self.depth_embedding = nn.Parameter(torch.zeros(len(DEPTHS), inner))
        self.depth_head_bias = nn.Parameter(torch.zeros(len(DEPTHS), heads))
        self.out_proj = nn.Linear(inner, hidden_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, states: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        if len(states) != len(DEPTHS):
            raise ValueError(f"expected {len(DEPTHS)} states, received {len(states)}")
        shape = states[-1].shape
        if any(x.shape != shape for x in states):
            raise ValueError("all vision depths must have the same token grid and width")
        batch, tokens, _ = shape
        compute_dtype = self.q_proj.weight.dtype
        normalized = [norm(x.to(compute_dtype)) for norm, x in zip(self.norms, states)]
        query = self.q_proj(normalized[-1]).view(batch, tokens, self.heads, self.head_dim)
        keys = torch.stack([self.k_proj(x) for x in normalized], dim=2)
        values = torch.stack([self.v_proj(x) for x in normalized], dim=2)
        keys = keys + self.depth_embedding.view(1, 1, len(DEPTHS), -1)
        keys = keys.view(batch, tokens, len(DEPTHS), self.heads, self.head_dim)
        values = values.view(batch, tokens, len(DEPTHS), self.heads, self.head_dim)
        logits = torch.einsum("bnhd,bnlhd->bnlh", query, keys) * (self.head_dim ** -0.5)
        logits = logits + self.depth_head_bias.view(1, 1, len(DEPTHS), self.heads)
        attention = logits.softmax(dim=2)
        mixed = torch.einsum("bnlh,bnlhd->bnhd", attention, values).reshape(batch, tokens, -1)
        raw = self.out_proj(mixed)
        rms = states[-1].float().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        correction = (rms.to(raw.dtype) * torch.tanh(raw)).to(states[-1].dtype)
        return {"fused": states[-1] + correction, "correction": correction,
                "attention": attention}


class H24OnlyAdapter(nn.Module):
    """Matched-budget depth-24-only residual adapter; no shallow-layer input."""

    def __init__(self, hidden_size: int = 1024, bottleneck: int = 131):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.in_proj = nn.Linear(hidden_size, bottleneck)
        self.out_proj = nn.Linear(bottleneck, hidden_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, states: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        anchor = states[-1]
        dtype = self.in_proj.weight.dtype
        raw = self.out_proj(torch.nn.functional.gelu(self.in_proj(self.norm(anchor.to(dtype)))))
        rms = anchor.float().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        correction = (rms.to(raw.dtype) * torch.tanh(raw)).to(anchor.dtype)
        return {"fused": anchor + correction, "correction": correction}


class SingleDepthAdapter(nn.Module):
    """Use one source depth to make a residual correction to the native H24 anchor."""

    def __init__(self, hidden_size: int = 1024, bottleneck: int = 131):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.in_proj = nn.Linear(hidden_size, bottleneck)
        self.out_proj = nn.Linear(bottleneck, hidden_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, states: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        if len(states) != 2:
            raise ValueError("single-depth adapter expects source and H24 anchor")
        source, anchor = states
        dtype = self.in_proj.weight.dtype
        raw = self.out_proj(torch.nn.functional.gelu(self.in_proj(self.norm(source.to(dtype)))))
        rms = anchor.float().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        correction = (rms.to(raw.dtype) * torch.tanh(raw)).to(anchor.dtype)
        return {"fused": anchor + correction, "correction": correction}


def build_fusion(kind: str) -> tuple[nn.Module, tuple[str, ...]]:
    if kind == "pkva":
        return VisionPKVA(), ("q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight",
                               "depth_embedding", "depth_head_bias")
    if kind == "h24_only":
        return H24OnlyAdapter(), ("norm.weight", "in_proj.weight", "out_proj.weight")
    if kind == "single_depth":
        return SingleDepthAdapter(), ("norm.weight", "in_proj.weight", "out_proj.weight")
    raise ValueError(f"unknown fusion kind: {kind}")


def gradient_table(module: nn.Module) -> dict[str, dict[str, float | bool]]:
    table = {}
    for name, parameter in module.named_parameters():
        grad = parameter.grad
        table[name] = {
            "finite": bool(grad is not None and torch.isfinite(grad).all()),
            "l1": float(grad.detach().float().abs().sum()) if grad is not None else 0.0,
        }
    return table


def run() -> None:
    started = time.time()
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    free, total = torch.cuda.mem_get_info()

    from work.exp054_data import load_frozen_model

    model, _ = load_frozen_model(MODEL)
    if model.config.vision_config.num_hidden_layers != 24:
        raise RuntimeError("unexpected vision depth")
    if model.config.vision_config.hidden_size != 1024:
        raise RuntimeError("unexpected vision width")
    frozen_versions = [(name, parameter._version) for name, parameter in model.named_parameters()]

    pixels = torch.randn(1, 3, 448, 448, device="cuda", dtype=model.vision_model.dtype)
    with torch.no_grad():
        official = model.extract_feature(pixels)
        vision = model.vision_model(pixel_values=pixels, output_hidden_states=True, return_dict=True)
        states = [vision.hidden_states[depth][:, 1:, :] for depth in DEPTHS]
        if not torch.equal(states[-1], vision.last_hidden_state[:, 1:, :]):
            raise RuntimeError("hidden_states[24] is not the official final vision state")

    fusion, required = build_fusion(FUSION_KIND)
    fusion = fusion.to(device="cuda", dtype=model.vision_model.dtype)
    zero = fusion(states)
    if not torch.equal(zero["fused"], states[-1]) or bool(zero["correction"].any()):
        raise RuntimeError("zero-initialized fusion is not bit-exact with the final vision state")
    batch, tokens, width = zero["fused"].shape
    side = int(tokens**0.5)
    candidate = zero["fused"].reshape(batch, side, side, width)
    candidate = model.pixel_shuffle(candidate, scale_factor=model.downsample_ratio)
    candidate = candidate.reshape(batch, -1, candidate.shape[-1])
    candidate = model.mlp1(candidate)
    if not torch.equal(candidate, official):
        raise RuntimeError("zero-initialized fusion is not bit-exact after pixel_shuffle and mlp1")

    target = torch.randn_like(states[-1])
    first = fusion(states)
    loss = (first["fused"].float() - target.float()).square().mean()
    loss.backward()
    first_gradients = gradient_table(fusion)
    if not first_gradients["out_proj.weight"]["finite"] or first_gradients["out_proj.weight"]["l1"] <= 0:
        raise RuntimeError("zero-step output projection has no usable gradient")
    expected_zero_first = [name for name, row in first_gradients.items()
                           if name not in {"out_proj.weight", "out_proj.bias"} and row["l1"] == 0]

    optimizer = torch.optim.SGD(fusion.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = fusion(states)
    second_loss = (second["fused"].float() - target.float()).square().mean()
    second_loss.backward()
    second_gradients = gradient_table(fusion)
    missing = [name for name in required
               if not second_gradients[name]["finite"] or second_gradients[name]["l1"] <= 0]
    if missing:
        raise RuntimeError(f"post-update gradient reachability failed for {missing}")

    frozen_changed = [name for name, version in frozen_versions
                      if dict(model.named_parameters())[name]._version != version]
    frozen_grads = [name for name, parameter in model.named_parameters() if parameter.grad is not None]
    if frozen_changed or frozen_grads:
        raise RuntimeError("frozen model changed or accumulated gradients")

    receipt = {
        "state": "incomplete",
        "decision": "PASS_visual_identity_and_module_gradient; future_frame_gradient_not_yet_audited",
        "fusion_kind": FUSION_KIND,
        "model": str(MODEL),
        "vision": {"depths": list(DEPTHS), "hidden_size": 1024, "tokens": tokens,
                   "input_shape": list(pixels.shape), "dtype": str(pixels.dtype)},
        "identity": {"final_state_bit_exact": True, "post_mlp1_bit_exact": True,
                     "official_shape": list(official.shape), "max_abs_error": 0.0},
        "gradient": {"first_step": first_gradients, "expected_zero_before_output_update": expected_zero_first,
                     "post_disposable_update": second_gradients, "required_nonzero": list(required)},
        "frozen_model": {"parameter_versions_unchanged": True, "parameter_grads_absent": True},
        "parameters": sum(parameter.numel() for parameter in fusion.parameters()),
        "GT_access": False,
        "full_SAM2_propagation_tested": False,
        "future_frame_loss_tested": False,
        "source_sha256": sha256(Path(__file__)),
        "gpu": {"name": torch.cuda.get_device_name(), "visible_device": 0,
                "free_mib_at_admission": int(free / 1024**2)},
        "seconds": time.time() - started,
    }
    atomic_json(OUT / "result.json", receipt)
    print(json.dumps({"state": receipt["state"], "decision": receipt["decision"],
                      "parameters": receipt["parameters"], "identity": receipt["identity"],
                      "seconds": receipt["seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "FAILED.txt").write_text(traceback.format_exc())
        raise
