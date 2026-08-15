"""Tune-only, budget-matched policy selection against a stateless reference.

The search grid in :mod:`oran.policy_search` is intentionally left untouched.
This module provides a confirmatory selection layer with two safeguards:

* the stateless raw-score reference is recalibrated from every benign tuning
  epoch, rather than selected from a coarse threshold grid; and
* a temporal candidate must operate near the same declared friction budget
  before security, delay, or churn outcomes are compared.

Ground-truth attack indicators are used only for tuning-set calibration and
metric evaluation.  They never define controller resets.  The locked test split
is rejected by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Sequence

import numpy as np
import polars as pl

from .controller import AccessState, timestamps_seconds
from .metrics import (
    attack_episode_metrics,
    summarize_attack_episodes,
    time_weighted_access_metrics,
)


DEFAULT_TUNING_SPLIT = "controller_tune"


@dataclass(frozen=True, slots=True)
class SelectionTolerances:
    """Predeclared practical-equivalence and success margins.

    ``budget_undershoot_relative=0.05`` requires at least 95% nominal budget
    utilization, apart from one indivisible benign epoch.  The remaining margins
    match the locked study protocol: two percentage points of ALLOW exposure,
    five percentage points each of not-ISOLATED exposure and episode coverage,
    one decision epoch of delay, and a minimum 25% reduction in transition rate.
    """

    budget_undershoot_relative: float = 0.05
    security_noninferiority_abs: float = 0.02
    isolation_noninferiority_abs: float = 0.05
    coverage_noninferiority_abs: float = 0.05
    delay_noninferiority_s: float = 1.0
    minimum_transition_reduction: float = 0.25
    numeric_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        bounded = {
            "budget_undershoot_relative": self.budget_undershoot_relative,
            "security_noninferiority_abs": self.security_noninferiority_abs,
            "isolation_noninferiority_abs": self.isolation_noninferiority_abs,
            "coverage_noninferiority_abs": self.coverage_noninferiority_abs,
            "minimum_transition_reduction": self.minimum_transition_reduction,
        }
        for name, value in bounded.items():
            if not np.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if (
            not np.isfinite(self.delay_noninferiority_s)
            or self.delay_noninferiority_s < 0
        ):
            raise ValueError("delay_noninferiority_s must be finite and non-negative")
        if not np.isfinite(self.numeric_tolerance) or self.numeric_tolerance < 0:
            raise ValueError("numeric_tolerance must be finite and non-negative")


def _array(name: str, values: Sequence[Any] | np.ndarray, length: int | None = None):
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must have length {length}")
    return result


def _assert_tuning_split(
    split: Sequence[str] | np.ndarray,
    length: int,
    tuning_split: str,
) -> None:
    if not isinstance(tuning_split, str) or not tuning_split:
        raise ValueError("tuning_split must be a non-empty string")
    if isinstance(split, str):
        raise TypeError(
            "split must contain one auditable split marker per row, not a scalar"
        )
    split_values = _array("split", split, length)
    observed = {str(value) for value in split_values}
    if observed != {tuning_split}:
        raise ValueError(
            "policy calibration and selection may use the tuning split only; "
            f"expected {tuning_split!r}, found {sorted(observed)}"
        )


def _identifiers(
    name: str, values: Sequence[Hashable] | np.ndarray, length: int
) -> np.ndarray:
    result = np.asarray(values, dtype=object)
    if result.ndim != 1 or len(result) != length:
        raise ValueError(f"{name} must be one-dimensional with length {length}")
    for value in result:
        if value is None or (
            isinstance(value, (float, np.floating)) and np.isnan(value)
        ):
            raise ValueError(f"{name} must not contain missing values")
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(f"{name} values must be hashable") from exc
    return result


def _binary(values: Sequence[bool] | np.ndarray, length: int) -> np.ndarray:
    result = _array("is_attack", values, length)
    if result.dtype.kind == "b":
        return result.astype(bool, copy=False)
    if result.dtype.kind in "iuf" and np.isfinite(result).all() and np.isin(
        result, (0, 1)
    ).all():
        return result.astype(bool)
    raise TypeError("is_attack must contain only booleans or 0/1 values")


def _validate_chronology(
    times: np.ndarray, subjects: np.ndarray, leases: np.ndarray
) -> None:
    previous: dict[Hashable, tuple[Hashable, float]] = {}
    for time_s, subject, lease in zip(times, subjects, leases, strict=True):
        prior = previous.get(subject)
        if prior is not None and prior[0] == lease and time_s < prior[1]:
            raise ValueError(
                "timestamps must be nondecreasing within each subject/lease"
            )
        previous[subject] = (lease, float(time_s))


def effective_previous_values(
    values: Sequence[float] | np.ndarray,
    subject_ids: Sequence[Hashable] | np.ndarray,
    lease_ids: Sequence[Hashable] | np.ndarray,
    *,
    initial_value: float = np.nan,
) -> np.ndarray:
    """Shift a stream by one report within each explicit subject lease.

    This implements the experiment's one-epoch actuation delay without looking
    at attack labels or inferring lifecycle boundaries from time gaps.
    """

    supplied = _array("values", values)
    subjects = _identifiers("subject_ids", subject_ids, len(supplied))
    leases = _identifiers("lease_ids", lease_ids, len(supplied))
    output = np.full(len(supplied), initial_value, dtype=float)
    previous: dict[Hashable, tuple[Hashable, float]] = {}
    for index, (value, subject, lease) in enumerate(
        zip(supplied, subjects, leases, strict=True)
    ):
        prior = previous.get(subject)
        if prior is not None and prior[0] == lease:
            output[index] = prior[1]
        previous[subject] = (lease, float(value))
    return output


def _threshold_from_below(
    effective_risk: np.ndarray,
    benign: np.ndarray,
    durations: np.ndarray,
    budget: float,
    numeric_tolerance: float,
) -> tuple[float, float, float]:
    """Return threshold, attainable friction, and next boundary jump."""

    valid = benign & np.isfinite(effective_risk) & (durations > 0)
    values = effective_risk[valid]
    weights = durations[valid]
    total_benign = float(durations[benign].sum())
    if total_benign <= 0 or not len(values):
        raise ValueError("stateless calibration requires positive benign entity-time")

    order = np.argsort(-values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    starts = np.r_[0, np.flatnonzero(ordered_values[1:] != ordered_values[:-1]) + 1]
    scores = ordered_values[starts]
    masses = np.add.reduceat(ordered_weights, starts)
    cumulative = np.cumsum(masses) / total_benign
    feasible = np.flatnonzero(cumulative <= budget + numeric_tolerance)
    if not len(feasible):
        threshold = float(np.nextafter(scores[0], np.inf))
        return threshold, 0.0, float(masses[0] / total_benign)
    index = int(feasible[-1])
    next_jump = (
        float(masses[index + 1] / total_benign)
        if index + 1 < len(masses)
        else 0.0
    )
    return float(scores[index]), float(cumulative[index]), next_jump


def _effective_binary_actions(
    risk: np.ndarray,
    threshold: float,
    subjects: np.ndarray,
    leases: np.ndarray,
) -> np.ndarray:
    decisions = np.where(
        risk >= threshold,
        # The raw reference jumps directly to the strongest action.  Its ALLOW
        # exposure and not-ISOLATED exposure are therefore identical, providing
        # an auditable hard-containment comparator for the three-state policy.
        int(AccessState.ISOLATE),
        int(AccessState.ALLOW),
    )
    return effective_previous_values(
        decisions,
        subjects,
        leases,
        initial_value=float(AccessState.ALLOW),
    ).astype(np.int8)


def calibrate_stateless_references(
    risk: Sequence[float] | np.ndarray,
    timestamps: Sequence[Any] | np.ndarray,
    subject_ids: Sequence[Hashable] | np.ndarray,
    lease_ids: Sequence[Hashable] | np.ndarray,
    is_attack: Sequence[bool] | np.ndarray,
    durations_s: Sequence[float] | np.ndarray,
    budgets: Sequence[float] | np.ndarray,
    *,
    split: Sequence[str] | np.ndarray,
    tuning_split: str = DEFAULT_TUNING_SPLIT,
    tolerances: SelectionTolerances = SelectionTolerances(),
) -> pl.DataFrame:
    """Calibrate one time-weighted stateless reference per tuning budget.

    Thresholds are chosen from the full one-epoch-lagged benign score stream and
    approach each budget from below.  No test row may be supplied.  The returned
    ``reference_calibrated`` flag is false when score ties or insufficient data
    prevent at least 95% budget utilization (under the default tolerance).
    """

    scores = np.asarray(risk, dtype=float)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("risk must be a non-empty one-dimensional sequence")
    if not np.isfinite(scores).all():
        raise ValueError("risk scores must be finite")
    length = len(scores)
    times = timestamps_seconds(_array("timestamps", timestamps, length))
    subjects = _identifiers("subject_ids", subject_ids, length)
    leases = _identifiers("lease_ids", lease_ids, length)
    attacks = _binary(is_attack, length)
    durations = np.asarray(_array("durations_s", durations_s, length), dtype=float)
    if not np.isfinite(durations).all() or (durations < 0).any():
        raise ValueError("durations_s must be finite and non-negative")
    _assert_tuning_split(split, length, tuning_split)
    _validate_chronology(times, subjects, leases)
    if not attacks.any() or attacks.all():
        raise ValueError("reference calibration requires benign and attack epochs")

    requested = np.asarray(budgets, dtype=float)
    if requested.ndim != 1 or not len(requested):
        raise ValueError("budgets must be a non-empty one-dimensional sequence")
    if (
        not np.isfinite(requested).all()
        or (requested <= 0).any()
        or (requested >= 1).any()
    ):
        raise ValueError("budgets must be finite and strictly between zero and one")

    benign = ~attacks
    benign_time = float(durations[benign].sum())
    if benign_time <= 0:
        raise ValueError("positive benign entity-time is required")
    positive_benign = durations[benign & (durations > 0)]
    if not len(positive_benign):
        raise ValueError("positive benign epoch durations are required")
    atomic_resolution = float(positive_benign.max() / benign_time)
    lagged_risk = effective_previous_values(scores, subjects, leases)
    rows: list[dict[str, Any]] = []

    for budget in np.unique(np.sort(requested)):
        threshold, attainable, next_jump = _threshold_from_below(
            lagged_risk,
            benign,
            durations,
            float(budget),
            tolerances.numeric_tolerance,
        )
        effective_states = _effective_binary_actions(
            scores, threshold, subjects, leases
        )
        metrics = time_weighted_access_metrics(
            times,
            effective_states,
            attacks,
            subject_ids=subjects,
            lease_ids=leases,
            durations_s=durations,
            containment_state=AccessState.RESTRICT,
        )
        isolation_metrics = time_weighted_access_metrics(
            times,
            effective_states,
            attacks,
            subject_ids=subjects,
            lease_ids=leases,
            durations_s=durations,
            containment_state=AccessState.ISOLATE,
        )
        if not np.isclose(
            metrics.benign_friction,
            attainable,
            atol=max(tolerances.numeric_tolerance, np.finfo(float).eps * 16),
            rtol=0.0,
        ):
            raise RuntimeError("internal stateless friction calibration mismatch")
        episodes = summarize_attack_episodes(
            attack_episode_metrics(
                times,
                effective_states,
                attacks,
                subject_ids=subjects,
                lease_ids=leases,
                durations_s=durations,
                delay_cap_s=30.0,
                containment_state=AccessState.RESTRICT,
            )
        )
        if episodes.height != 1:
            raise RuntimeError("aggregate attack episode summary must contain one row")
        episode = episodes.row(0, named=True)
        minimum_friction = (
            float(budget) * (1.0 - tolerances.budget_undershoot_relative)
            - atomic_resolution
        )
        rows.append(
            {
                "candidate": f"stateless-reference-{budget:.6g}",
                "family": "stateless_reference",
                "split": tuning_split,
                "friction_budget": float(budget),
                "restrict_threshold": threshold,
                "benign_friction": metrics.benign_friction,
                "budget_utilization": metrics.benign_friction / float(budget),
                "budget_gap": float(budget) - metrics.benign_friction,
                "friction_resolution": atomic_resolution,
                "next_boundary_jump": next_jump,
                "reference_calibrated": bool(
                    metrics.benign_friction
                    >= minimum_friction - tolerances.numeric_tolerance
                ),
                "malicious_allow": metrics.malicious_exposure,
                "malicious_not_isolated": isolation_metrics.malicious_exposure,
                "transitions": metrics.transitions,
                "transitions_per_minute": metrics.transitions_per_minute,
                "severity_transitions_per_minute": (
                    metrics.severity_transitions_per_minute
                ),
                "false_restriction_episodes": metrics.false_restriction_episodes,
                "attack_episodes": int(episode["episodes"]),
                "episode_coverage": float(episode["episode_coverage"]),
                "median_capped_delay_s": float(episode["median_capped_delay_s"]),
            }
        )
    return pl.DataFrame(rows, strict=False).sort("friction_budget")


def _require_tune_table(
    frame: pl.DataFrame,
    required: set[str],
    tuning_split: str,
    name: str,
) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    observed = {str(value) for value in frame["split"].unique().to_list()}
    if observed != {tuning_split}:
        raise ValueError(
            f"{name} may contain {tuning_split!r} rows only; found {sorted(observed)}"
        )


def _finite_columns(frame: pl.DataFrame, columns: Sequence[str], name: str) -> None:
    for column in columns:
        try:
            values = np.asarray(frame[column].to_list(), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}.{column} must be numeric") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{name}.{column} must be finite")


def _diagnostic_choice(frame: pl.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    return frame.sort(list(columns)).row(0, named=True)


def select_against_stateless_reference(
    candidate_metrics: pl.DataFrame,
    stateless_references: pl.DataFrame,
    *,
    tuning_split: str = DEFAULT_TUNING_SPLIT,
    tolerances: SelectionTolerances = SelectionTolerances(),
    families: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Apply a tune-only, stateless-referenced confirmatory selection rule.

    For each family and budget, the rule is:

    1. require friction within ``[95% of budget - one epoch, budget + one epoch]``;
    2. require malicious exposure no more than 0.02 above the stateless reference;
    3. require not-ISOLATED exposure no more than 0.05 above the reference;
    4. require episode coverage no more than 0.05 below the reference;
    5. require capped median delay no more than one second above the reference;
    6. among the closest-to-budget stratum (within one epoch of the closest
       candidate), minimize transition rate; and
    7. declare success only for at least a 25% transition-rate reduction.

    Every family/budget produces an audit row.  Failures are explicit rather than
    silently replaced with a severely under-budget candidate.
    """

    if not isinstance(candidate_metrics, pl.DataFrame) or not isinstance(
        stateless_references, pl.DataFrame
    ):
        raise TypeError("candidate_metrics and stateless_references must be Polars frames")
    candidate_required = {
        "candidate",
        "family",
        "split",
        "benign_friction",
        "malicious_allow",
        "malicious_not_isolated",
        "episode_coverage",
        "median_capped_delay_s",
        "transitions_per_minute",
        "false_restriction_episodes",
    }
    reference_required = {
        "split",
        "friction_budget",
        "benign_friction",
        "friction_resolution",
        "reference_calibrated",
        "malicious_allow",
        "malicious_not_isolated",
        "episode_coverage",
        "median_capped_delay_s",
        "transitions_per_minute",
    }
    _require_tune_table(
        candidate_metrics, candidate_required, tuning_split, "candidate_metrics"
    )
    _require_tune_table(
        stateless_references,
        reference_required,
        tuning_split,
        "stateless_references",
    )
    _finite_columns(
        candidate_metrics,
        [
            "benign_friction",
            "malicious_allow",
            "malicious_not_isolated",
            "episode_coverage",
            "median_capped_delay_s",
            "transitions_per_minute",
            "false_restriction_episodes",
        ],
        "candidate_metrics",
    )
    has_declared_budget = "friction_budget" in candidate_metrics.columns
    if has_declared_budget:
        _finite_columns(
            candidate_metrics,
            ["friction_budget"],
            "candidate_metrics",
        )
    _finite_columns(
        stateless_references,
        [
            "friction_budget",
            "benign_friction",
            "friction_resolution",
            "malicious_allow",
            "malicious_not_isolated",
            "episode_coverage",
            "median_capped_delay_s",
            "transitions_per_minute",
        ],
        "stateless_references",
    )
    if candidate_metrics["candidate"].n_unique() != candidate_metrics.height:
        raise ValueError("candidate identifiers must be unique")
    if stateless_references["friction_budget"].n_unique() != stateless_references.height:
        raise ValueError("each friction budget must have exactly one stateless reference")
    if not stateless_references["reference_calibrated"].all():
        failed = stateless_references.filter(~pl.col("reference_calibrated"))[
            "friction_budget"
        ].to_list()
        raise ValueError(f"stateless reference could not calibrate budgets: {failed}")

    if families is None:
        selected_families = sorted(
            set(candidate_metrics["family"].to_list())
            - {"stateless", "stateless_reference"}
        )
    else:
        selected_families = list(dict.fromkeys(str(family) for family in families))
        unknown = set(selected_families).difference(candidate_metrics["family"].to_list())
        if unknown:
            raise ValueError(f"unknown candidate families: {sorted(unknown)}")
    if not selected_families:
        raise ValueError("at least one non-stateless candidate family is required")

    rows: list[dict[str, Any]] = []
    eps = tolerances.numeric_tolerance
    for reference in stateless_references.sort("friction_budget").iter_rows(named=True):
        budget = float(reference["friction_budget"])
        resolution = float(reference["friction_resolution"])
        lower = budget * (1.0 - tolerances.budget_undershoot_relative) - resolution
        upper = budget + resolution
        for family in selected_families:
            family_frame = candidate_metrics.filter(pl.col("family") == family)
            # Second-stage matched-search rows are calibrated for a declared
            # budget.  Do not silently reuse a configuration calibrated for a
            # different budget merely because its realized friction overlaps.
            if has_declared_budget:
                family_frame = family_frame.filter(
                    (pl.col("friction_budget") - budget).abs() <= eps
                )
            window = family_frame.filter(
                (pl.col("benign_friction") >= lower - eps)
                & (pl.col("benign_friction") <= upper + eps)
            )
            security = window.filter(
                pl.col("malicious_allow")
                <= float(reference["malicious_allow"])
                + tolerances.security_noninferiority_abs
                + eps
            )
            isolation = security.filter(
                pl.col("malicious_not_isolated")
                <= float(reference["malicious_not_isolated"])
                + tolerances.isolation_noninferiority_abs
                + eps
            )
            coverage = isolation.filter(
                pl.col("episode_coverage")
                >= float(reference["episode_coverage"])
                - tolerances.coverage_noninferiority_abs
                - eps
            )
            delay = coverage.filter(
                pl.col("median_capped_delay_s")
                <= float(reference["median_capped_delay_s"])
                + tolerances.delay_noninferiority_s
                + eps
            )

            status: str
            chosen: dict[str, Any] | None
            closest_count = 0
            if window.is_empty():
                status = "no_budget_matched_candidate"
                chosen = None
            elif security.is_empty():
                status = "security_inferior"
                chosen = _diagnostic_choice(
                    window, ["malicious_allow", "transitions_per_minute", "candidate"]
                )
            elif isolation.is_empty():
                status = "isolation_inferior"
                chosen = _diagnostic_choice(
                    security,
                    [
                        "malicious_not_isolated",
                        "transitions_per_minute",
                        "candidate",
                    ],
                )
            elif coverage.is_empty():
                status = "coverage_inferior"
                chosen = isolation.sort(
                    ["episode_coverage", "transitions_per_minute", "candidate"],
                    descending=[True, False, False],
                ).row(0, named=True)
            elif delay.is_empty():
                status = "delay_inferior"
                chosen = _diagnostic_choice(
                    coverage,
                    ["median_capped_delay_s", "transitions_per_minute", "candidate"],
                )
            else:
                distances = (delay["benign_friction"] - budget).abs()
                closest_distance = float(distances.min())
                closest = delay.filter(
                    (pl.col("benign_friction") - budget).abs()
                    <= closest_distance + resolution + eps
                )
                closest_count = closest.height
                chosen = _diagnostic_choice(
                    closest,
                    [
                        "transitions_per_minute",
                        "false_restriction_episodes",
                        "malicious_allow",
                        "candidate",
                    ],
                )
                reference_churn = float(reference["transitions_per_minute"])
                if reference_churn <= eps:
                    status = "reference_churn_zero"
                else:
                    reduction = 1.0 - float(chosen["transitions_per_minute"]) / reference_churn
                    status = (
                        "selected"
                        if reduction + eps >= tolerances.minimum_transition_reduction
                        else "insufficient_transition_reduction"
                    )

            candidate_churn = (
                float(chosen["transitions_per_minute"])
                if chosen is not None
                else np.nan
            )
            reference_churn = float(reference["transitions_per_minute"])
            reduction = (
                1.0 - candidate_churn / reference_churn
                if chosen is not None and reference_churn > eps
                else np.nan
            )
            rows.append(
                {
                    "friction_budget": budget,
                    "family": family,
                    "status": status,
                    "candidate": None if chosen is None else chosen["candidate"],
                    "benign_friction": (
                        np.nan if chosen is None else float(chosen["benign_friction"])
                    ),
                    "budget_utilization": (
                        np.nan
                        if chosen is None
                        else float(chosen["benign_friction"]) / budget
                    ),
                    "malicious_allow": (
                        np.nan if chosen is None else float(chosen["malicious_allow"])
                    ),
                    "malicious_not_isolated": (
                        np.nan
                        if chosen is None
                        else float(chosen["malicious_not_isolated"])
                    ),
                    "episode_coverage": (
                        np.nan if chosen is None else float(chosen["episode_coverage"])
                    ),
                    "median_capped_delay_s": (
                        np.nan
                        if chosen is None
                        else float(chosen["median_capped_delay_s"])
                    ),
                    "transitions_per_minute": candidate_churn,
                    "transition_reduction": reduction,
                    "reference_candidate": reference.get("candidate"),
                    "reference_benign_friction": float(reference["benign_friction"]),
                    "reference_malicious_allow": float(reference["malicious_allow"]),
                    "reference_malicious_not_isolated": float(
                        reference["malicious_not_isolated"]
                    ),
                    "reference_episode_coverage": float(reference["episode_coverage"]),
                    "reference_median_capped_delay_s": float(
                        reference["median_capped_delay_s"]
                    ),
                    "reference_transitions_per_minute": reference_churn,
                    "budget_lower_bound": lower,
                    "budget_upper_bound": upper,
                    "budget_matched_count": window.height,
                    "security_noninferior_count": security.height,
                    "isolation_noninferior_count": isolation.height,
                    "coverage_noninferior_count": coverage.height,
                    "delay_noninferior_count": delay.height,
                    "closest_budget_stratum_count": closest_count,
                    "budget_undershoot_relative": (
                        tolerances.budget_undershoot_relative
                    ),
                    "security_margin": tolerances.security_noninferiority_abs,
                    "isolation_margin": tolerances.isolation_noninferiority_abs,
                    "coverage_margin": tolerances.coverage_noninferiority_abs,
                    "delay_margin_s": tolerances.delay_noninferiority_s,
                    "minimum_transition_reduction": (
                        tolerances.minimum_transition_reduction
                    ),
                }
            )
    return pl.DataFrame(rows, strict=False).sort(["friction_budget", "family"])


__all__ = [
    "DEFAULT_TUNING_SPLIT",
    "SelectionTolerances",
    "effective_previous_values",
    "calibrate_stateless_references",
    "select_against_stateless_reference",
]
