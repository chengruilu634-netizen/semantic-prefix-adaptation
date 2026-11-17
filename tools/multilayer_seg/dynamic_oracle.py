"""Pure utilities for bounded dynamic-layer/[SEG] oracle experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ResidualCandidate:
    """A signed, final-layer-anchored residual candidate."""

    layers: tuple[int, ...]
    signs: tuple[int, ...]
    budget: float

    @property
    def k(self) -> int:
        """Total number of layers, including the h28 anchor."""
        return 1 + len(self.layers)

    @property
    def weights(self) -> tuple[float, ...]:
        if not self.layers:
            return ()
        scale = self.budget / len(self.layers)
        return tuple(scale * sign for sign in self.signs)


def directional_signs(gradients: Sequence[float], eps: float = 1e-12) -> tuple[int, ...]:
    """Return the first-order loss-decreasing sign for each residual."""
    result = []
    for gradient in gradients:
        if abs(float(gradient)) <= eps:
            result.append(1)
        else:
            result.append(-1 if gradient > 0 else 1)
    return tuple(result)


def gradient_beam_candidates(
    layer_indices: Sequence[int],
    gradients: Sequence[float],
    *,
    residual_count: int,
    pool_size: int = 8,
    beam_size: int = 4,
    budgets: Iterable[float] = (0.05, 0.10, 0.15),
) -> list[ResidualCandidate]:
    """Build a diverse bounded beam ranked by first-order gain."""
    if len(layer_indices) != len(gradients):
        raise ValueError("layer_indices and gradients must have equal length")
    if residual_count < 1:
        raise ValueError("residual_count must be positive")
    if residual_count > pool_size:
        raise ValueError("residual_count cannot exceed pool_size")
    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError("layer_indices must be unique")
    budget_values = tuple(float(value) for value in budgets)
    if not budget_values or any(value <= 0 for value in budget_values):
        raise ValueError("budgets must contain positive values")

    order = sorted(
        range(len(layer_indices)),
        key=lambda index: (-abs(float(gradients[index])), int(layer_indices[index])),
    )[: min(pool_size, len(layer_indices))]
    if residual_count > len(order):
        return []
    signs = directional_signs(gradients)
    ranked = sorted(
        combinations(order, residual_count),
        key=lambda group: (
            -sum(abs(float(gradients[index])) for index in group),
            tuple(int(layer_indices[index]) for index in group),
        ),
    )[:beam_size]
    result = []
    for group in ranked:
        sorted_group = sorted(group, key=lambda index: int(layer_indices[index]))
        layers = tuple(int(layer_indices[index]) for index in sorted_group)
        group_signs = tuple(signs[index] for index in sorted_group)
        for budget in budget_values:
            result.append(ResidualCandidate(layers, group_signs, budget))
    return result


def select_proxy_finalists(
    scored: Sequence[tuple[ResidualCandidate, float]], finalists: int = 2
) -> list[ResidualCandidate]:
    """Select proxy finalists deterministically, without duplicate configs."""
    if finalists < 1:
        raise ValueError("finalists must be positive")
    ranked = sorted(
        scored,
        key=lambda item: (
            -float(item[1]),
            item[0].k,
            item[0].layers,
            item[0].signs,
            item[0].budget,
        ),
    )
    result = []
    seen = set()
    for candidate, _ in ranked:
        key = (candidate.layers, candidate.signs, candidate.budget)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
        if len(result) == finalists:
            break
    return result


def select_budget_diverse_finalists(
    scored: Sequence[tuple[ResidualCandidate, float]],
    finalists_per_budget: int = 2,
) -> list[ResidualCandidate]:
    """Keep the gradient-primary and proxy-best candidate for every budget."""
    if finalists_per_budget < 1:
        raise ValueError("finalists_per_budget must be positive")
    budgets = sorted({candidate.budget for candidate, _ in scored})
    result: list[ResidualCandidate] = []
    for budget in budgets:
        bucket = [(candidate, score) for candidate, score in scored if candidate.budget == budget]
        # Generation order preserves the gradient beam rank, so never prune its
        # primary candidate solely because the coarse mask metric is tied.
        chosen = [bucket[0][0]]
        for candidate in select_proxy_finalists(bucket, max(1, finalists_per_budget)):
            if candidate not in chosen:
                chosen.append(candidate)
            if len(chosen) == finalists_per_budget:
                break
        result.extend(chosen)
    return result


def choose_dynamic_k(
    scores_by_k: dict[int, float], complexity_penalty: float = 0.0005
) -> int:
    """Choose K using exact score minus a small extra-layer penalty."""
    if not scores_by_k:
        raise ValueError("scores_by_k cannot be empty")
    if complexity_penalty < 0:
        raise ValueError("complexity_penalty must be non-negative")
    return min(
        scores_by_k,
        key=lambda k: (
            -(float(scores_by_k[k]) - complexity_penalty * (int(k) - 1)),
            int(k),
        ),
    )
