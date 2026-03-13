"""Bootstrap confidence interval computation."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: list[float],
    n_samples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for the mean of values.

    Returns (point_estimate, ci_low, ci_high).
    """
    if not values:
        return (0.0, 0.0, 0.0)

    arr = np.array(values)
    rng = np.random.default_rng(seed)
    point = float(np.mean(arr))

    boot_means = np.array([
        float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
        for _ in range(n_samples)
    ])

    alpha = 1 - confidence
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (point, ci_low, ci_high)


def bootstrap_delta_ci(
    candidate: list[float],
    reference: list[float],
    n_samples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for the mean difference (candidate - reference).

    Returns (delta, ci_low, ci_high).
    """
    if not candidate or not reference:
        return (0.0, 0.0, 0.0)

    c_arr = np.array(candidate)
    r_arr = np.array(reference)
    delta = float(np.mean(c_arr)) - float(np.mean(r_arr))

    rng = np.random.default_rng(seed)
    boot_deltas = np.array([
        float(np.mean(rng.choice(c_arr, size=len(c_arr), replace=True)))
        - float(np.mean(rng.choice(r_arr, size=len(r_arr), replace=True)))
        for _ in range(n_samples)
    ])

    alpha = 1 - confidence
    ci_low = float(np.percentile(boot_deltas, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_deltas, 100 * (1 - alpha / 2)))
    return (delta, ci_low, ci_high)
