"""Extract aligned [SEG] representations from selected transformer layers.

The functions in this module deliberately depend only on PyTorch. They do not
load Sa2VA, a checkpoint, SAM2, a dataset, or CUDA, which keeps the alignment
contract testable on CPU before any model evaluation is attempted.

Layer indices use transformer-layer numbering: hidden_states[0] is the token
embedding output and hidden_states[n] is transformer layer n. Negative indices
are rejected so that the layer contract remains explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


DEFAULT_LAYER_INDICES = (14, 18, 21, 24, 28)
DEFAULT_TRANSFORMER_LAYERS = 28


@dataclass(frozen=True)
class MultiLayerSegStates:
    """Aligned multi-layer representations for all [SEG] tokens.

    Attributes:
        states: Tensor with shape [N_seg, K, D].
        counts: Number of [SEG] tokens per input sample, shape [B].
        batch_indices: Source batch index for each extracted token, [N_seg].
        token_indices: Token position in the aligned token sequence, [N_seg].
        layer_indices: Transformer layers represented by axis K.
    """

    states: Tensor
    counts: Tensor
    batch_indices: Tensor
    token_indices: Tensor
    layer_indices: tuple[int, ...]

    def validate(self) -> None:
        if self.states.ndim != 3:
            raise ValueError("states must have shape [N_seg, K, D]")
        if self.counts.ndim != 1:
            raise ValueError("counts must have shape [B]")
        if self.batch_indices.ndim != 1 or self.token_indices.ndim != 1:
            raise ValueError("batch_indices and token_indices must be vectors")
        num_seg, num_layers, _ = self.states.shape
        if num_layers != len(self.layer_indices):
            raise ValueError("states layer axis does not match layer_indices")
        if self.batch_indices.numel() != num_seg:
            raise ValueError("batch_indices length does not match N_seg")
        if self.token_indices.numel() != num_seg:
            raise ValueError("token_indices length does not match N_seg")
        if int(self.counts.sum().item()) != num_seg:
            raise ValueError("sum(counts) does not match N_seg")


def _normalize_layer_indices(
    layer_indices: Sequence[int],
    expected_transformer_layers: int,
) -> tuple[int, ...]:
    normalized = tuple(layer_indices)
    if not normalized:
        raise ValueError("layer_indices cannot be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("layer_indices contains duplicates")
    for layer in normalized:
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise TypeError("each layer index must be an integer")
        if layer < 1 or layer > expected_transformer_layers:
            raise IndexError(
                "transformer layer must be in "
                f"[1, {expected_transformer_layers}], got {layer}"
            )
    return normalized


def _validate_hidden_state_tuple(
    hidden_states: Sequence[Tensor],
    *,
    expected_transformer_layers: int,
    context: str,
) -> tuple[int, int, int]:
    expected_states = expected_transformer_layers + 1
    if len(hidden_states) != expected_states:
        raise ValueError(
            f"{context}: expected {expected_states} hidden states "
            f"(embedding + {expected_transformer_layers} transformer layers), "
            f"got {len(hidden_states)}"
        )
    reference = hidden_states[0]
    if reference.ndim != 3:
        raise ValueError(f"{context}: hidden state 0 must have shape [B, T, D]")
    shape = tuple(reference.shape)
    for layer, state in enumerate(hidden_states):
        if not isinstance(state, Tensor):
            raise TypeError(f"{context}: hidden state {layer} is not a tensor")
        if state.ndim != 3:
            raise ValueError(
                f"{context}: hidden state {layer} must have shape [B, T, D]"
            )
        if tuple(state.shape) != shape:
            raise ValueError(
                f"{context}: hidden state {layer} has shape {tuple(state.shape)}, "
                f"expected {shape}"
            )
        if state.device != reference.device:
            raise ValueError(f"{context}: all hidden states must share a device")
        if state.dtype != reference.dtype:
            raise ValueError(f"{context}: all hidden states must share a dtype")
    return shape


def extract_teacher_forced_seg_states(
    hidden_states: Sequence[Tensor],
    input_ids: Tensor,
    seg_token_id: int,
    layer_indices: Sequence[int] = DEFAULT_LAYER_INDICES,
    *,
    expected_transformer_layers: int = DEFAULT_TRANSFORMER_LAYERS,
) -> MultiLayerSegStates:
    """Extract [SEG] states from a teacher-forced model forward pass."""

    layers = _normalize_layer_indices(
        layer_indices, expected_transformer_layers
    )
    batch_size, sequence_length, _ = _validate_hidden_state_tuple(
        hidden_states,
        expected_transformer_layers=expected_transformer_layers,
        context="teacher-forced",
    )
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T]")
    if tuple(input_ids.shape) != (batch_size, sequence_length):
        raise ValueError(
            "input_ids token shape does not match hidden states: "
            f"{tuple(input_ids.shape)} vs {(batch_size, sequence_length)}"
        )

    seg_mask = input_ids.eq(seg_token_id)
    selected = torch.stack([hidden_states[layer] for layer in layers], dim=2)
    states = selected[seg_mask]
    positions = seg_mask.nonzero(as_tuple=False)
    result = MultiLayerSegStates(
        states=states,
        counts=seg_mask.sum(dim=1, dtype=torch.long),
        batch_indices=positions[:, 0],
        token_indices=positions[:, 1],
        layer_indices=layers,
    )
    result.validate()
    return result


def extract_generation_seg_states(
    generation_hidden_states: Sequence[Sequence[Tensor]],
    sequences: Tensor,
    seg_token_id: int,
    layer_indices: Sequence[int] = DEFAULT_LAYER_INDICES,
    *,
    expected_transformer_layers: int = DEFAULT_TRANSFORMER_LAYERS,
) -> MultiLayerSegStates:
    """Extract [SEG] states from Hugging Face generation output.

    This intentionally reproduces Sa2VA's existing inference alignment:
    concatenate each generation step, take the tail matching sequences[0, :-1],
    then select [SEG] positions. The first implementation supports batch size 1,
    matching the audited Sa2VA predict_forward path.
    """

    layers = _normalize_layer_indices(
        layer_indices, expected_transformer_layers
    )
    if sequences.ndim != 2:
        raise ValueError("sequences must have shape [B, T]")
    if sequences.shape[0] != 1:
        raise NotImplementedError(
            "generation extraction currently supports batch size 1"
        )
    if not generation_hidden_states:
        raise ValueError("generation_hidden_states cannot be empty")

    hidden_size = None
    for step_index, step_states in enumerate(generation_hidden_states):
        step_batch, _, step_hidden = _validate_hidden_state_tuple(
            step_states,
            expected_transformer_layers=expected_transformer_layers,
            context=f"generation step {step_index}",
        )
        if step_batch != 1:
            raise ValueError("generation hidden-state batch size must be 1")
        if hidden_size is None:
            hidden_size = step_hidden
        elif step_hidden != hidden_size:
            raise ValueError("hidden size changed across generation steps")

    aligned_token_ids = sequences[0, :-1]
    aligned_length = aligned_token_ids.numel()
    seg_mask = aligned_token_ids.eq(seg_token_id)
    per_layer = []
    for layer in layers:
        stream = torch.cat(
            [step_states[layer][0] for step_states in generation_hidden_states],
            dim=0,
        )
        if stream.shape[0] < aligned_length:
            raise ValueError(
                f"layer {layer} has {stream.shape[0]} hidden tokens, "
                f"but alignment needs {aligned_length}"
            )
        aligned = stream[:0] if aligned_length == 0 else stream[-aligned_length:]
        per_layer.append(aligned[seg_mask])

    states = torch.stack(per_layer, dim=1)
    positions = seg_mask.nonzero(as_tuple=False).flatten()
    result = MultiLayerSegStates(
        states=states,
        counts=torch.tensor(
            [int(seg_mask.sum().item())],
            dtype=torch.long,
            device=sequences.device,
        ),
        batch_indices=torch.zeros_like(positions),
        token_indices=positions,
        layer_indices=layers,
    )
    result.validate()
    return result
