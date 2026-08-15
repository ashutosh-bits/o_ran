"""Cluster-aware statistical utilities for the confirmatory evaluation.

The dataset contains millions of autocorrelated rows, so row-wise confidence
intervals are intentionally not provided.  These helpers resample whole
caller-supplied clusters (trace blocks, RNTI leases, or attack episodes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    replicates: int
    clusters: int


def _validate_arrays(*arrays: np.ndarray) -> None:
    lengths = {len(np.asarray(a)) for a in arrays}
    if len(lengths) != 1:
        raise ValueError("all input arrays must have equal length")
    if not lengths or next(iter(lengths)) == 0:
        raise ValueError("input arrays must be non-empty")


def _cluster_sums(cluster_ids: Sequence[object], values: Sequence[float]) -> np.ndarray:
    clusters = np.asarray(cluster_ids)
    vals = np.asarray(values, dtype=float)
    _validate_arrays(clusters, vals)
    _, inverse = np.unique(clusters.astype(str), return_inverse=True)
    return np.bincount(inverse, weights=vals).astype(float)


def paired_cluster_ratio_bootstrap(
    cluster_ids: Sequence[object],
    numerator_a: Sequence[float],
    denominator_a: Sequence[float],
    numerator_b: Sequence[float],
    denominator_b: Sequence[float],
    *,
    replicates: int = 2_000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> BootstrapInterval:
    """Bootstrap the paired difference ``ratio(a) - ratio(b)`` by cluster.

    Numerator and denominator contributions must be additive within a cluster.
    A single bootstrap draw samples the same clusters for both policies, which
    retains pairing and substantially improves precision over independent CIs.
    """

    arrays = tuple(
        np.asarray(x)
        for x in (cluster_ids, numerator_a, denominator_a, numerator_b, denominator_b)
    )
    _validate_arrays(*arrays)
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    a_num = _cluster_sums(arrays[0], arrays[1])
    a_den = _cluster_sums(arrays[0], arrays[2])
    b_num = _cluster_sums(arrays[0], arrays[3])
    b_den = _cluster_sums(arrays[0], arrays[4])
    cluster_count = len(a_num)
    if cluster_count < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")
    if a_den.sum() <= 0 or b_den.sum() <= 0:
        raise ValueError("aggregate denominators must be positive")

    estimate = a_num.sum() / a_den.sum() - b_num.sum() / b_den.sum()
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        den_a = a_den[sampled].sum()
        den_b = b_den[sampled].sum()
        draws[index] = (
            np.nan
            if den_a <= 0 or den_b <= 0
            else a_num[sampled].sum() / den_a - b_num[sampled].sum() / den_b
        )
    draws = draws[np.isfinite(draws)]
    if len(draws) < max(100, replicates // 2):
        raise ValueError("too many invalid bootstrap draws")
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])
    return BootstrapInterval(
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
        replicates=int(len(draws)),
        clusters=cluster_count,
    )


def upper_confidence_bound(interval: BootstrapInterval) -> float:
    """Return the upper endpoint used for conservative budget admission."""

    return interval.upper


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm step-down family-wise-error adjusted p-values."""

    p = np.asarray(list(p_values), dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence")
    if np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate((len(p) - np.arange(len(p))) * ranked)
    adjusted_ranked = np.minimum(adjusted_ranked, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted

