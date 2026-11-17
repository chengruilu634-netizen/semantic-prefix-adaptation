# Semantic Prefix Adaptation

Adaptation-specific research code for frozen Sa2VA/SAM2 referring video object segmentation.

## Scope

The installable package contains the query-conditioned high-resolution adapter (`Q`), the anchored multi-layer memory residual (`M`), and a CPU module smoke check. The query module is copied verbatim from the archived experimental implementation. The memory class is extracted verbatim, with its fixed prefix constant supplied locally. A source manifest documents this extraction.

The native Sa2VA/SAM2 models, pretrained weights, training cache, dataset images/labels and full benchmark predictions are external dependencies. This repository does not yet provide a portable end-to-end training/evaluation launcher. Integration depends on the native decoder/memory interfaces described below; do not interpret the smoke check as benchmark reproduction.

## Experiment source

- [Joint prefix decoding and anchored memory refinement](work/prefix_joint_20260918/joint_forward.py)
- [QFA implementation and decoder integration](work/query_highres_gate_20260914.py)
- [Early/late QFA supervision](examples/archived_experiments/qfa_temporal_supervision_20261113/)
- [Joint evaluation with early/late QFA and original AMR](examples/archived_experiments/temporal_qfa_amr_20261114/evaluate.py)
- [MeViS v1 validation export reference](examples/eval_mevis_valid.py)

These files capture the research implementation and experiment structure. The archived scripts also expect external model, dataset, checkpoint, and protocol files. Obtain the base implementation from [Sa2VA](https://github.com/bytedance/Sa2VA) and the pretrained base weights from [Sa2VA-InternVL3-2B](https://huggingface.co/ByteDance/Sa2VA-InternVL3-2B). The pretrained base weights, experiment checkpoints, and benchmark datasets are external to this source release.

## Install and check

```sh
python -m pip install -e .
python tests/smoke_modules.py
```

## Integration contract

* `Q` takes two frozen feature maps with 32/64 channels and a projected SEG query `[B,1,256]`. Each map has its own 16-channel residual and cosine gate. Output projections are zero-initialized. Its added parameter count is 13,248.
* `M` takes ordinary finite tensor memories `[4,5,64,H,W]` in depth order 21/24/25/28. It concatenates the live h28 anchor and three differences, then adds a bounded residual scaled by the detached per-frame anchor RMS. Its added parameter count is 40,288.
* Q is applied during prefix decoding only. Candidate branches use the current Q-derived decoder features but the native image features for memory encoding.
* J must add the residual to the live Q-derived anchor, retain the native h28 mask prompt/logits, object pointers and positional states, and replace only the memory feature. Future propagation remains native. A stale native cache is not a valid joint anchor.
* The archived matched training used six epochs/768 updates per seeds 17/29/43. Historical checkpoints and public/server dataset versions are separate protocols.

## Evidence boundaries

Joint gains are heterogeneous. The adjusted J-minus-Q interval includes zero, and query/depth controls do not establish semantic necessity or uniform transfer. This repository reports method implementation, not new scores.

## Upstream dependencies and rights

Obtain and use Sa2VA and SAM2 under their own upstream terms. No upstream model implementation or weight license is replaced here. This initial repository does not specify an additional license grant for the adaptation code.
