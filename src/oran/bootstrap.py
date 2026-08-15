"""Paired complete-block bootstrap for locked controller comparisons.

Inputs are the additive action and attack tables emitted by
``oran.evaluation``.  This module does not read artifacts, select candidates, or
print evaluation outcomes.  Every replicate samples one shared vector of whole
``trace_block_id`` clusters and applies it to both proposed and reference
policies and to every estimand.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from .data import TRACE_BLOCK_COLUMN, DataValidationError
from .evaluation import ANY_ATTACK_TYPE


DEFAULT_REPLICATES = 5_000
DEFAULT_SEED = 1_729

ACTION_ADDITIVE_COLUMNS: tuple[str, ...] = (
    "observed_time_s",
    "benign_time_s",
    "benign_contained_time_s",
    "attack_time_s",
    "attack_exposed_time_s",
    "effective_transition_count",
)
ATTACK_ADDITIVE_COLUMNS: tuple[str, ...] = (
    "episode_count",
    "covered_episode_count",
    "capped_delay_sum_s",
)


@dataclass(frozen=True)
class CompleteBlockGrid:
    """Candidate-by-block tables with explicit zeros for absent rows."""

    proposed_candidate: str
    reference_candidate: str
    attack_type: str
    block_ids: tuple[Any, ...]
    missing_action_blocks: Mapping[str, tuple[Any, ...]]
    missing_attack_blocks: Mapping[str, tuple[Any, ...]]
    action: pl.DataFrame
    attack: pl.DataFrame


@dataclass(frozen=True)
class ContrastInterval:
    metric: str
    contrast: str
    estimate: float | None
    ci_lower: float | None
    ci_upper: float | None
    one_sided_lower: float | None
    one_sided_upper: float | None
    gate_direction: str
    gate_endpoint: float | None
    confidence: float
    one_sided_confidence: float
    requested_replicates: int
    valid_replicates: int
    invalid_replicates: int
    clusters: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedBootstrapReport:
    proposed_candidate: str
    reference_candidate: str
    attack_type: str
    seed: int
    requested_replicates: int
    confidence: float
    one_sided_confidence: float
    clusters: int
    completed_action_rows: int
    completed_attack_rows: int
    missing_action_block_counts: Mapping[str, int]
    missing_attack_block_counts: Mapping[str, int]
    shared_draw_fingerprint: str
    metrics: tuple[ContrastInterval, ...]

    def metric(self, name: str) -> ContrastInterval:
        matches = [item for item in self.metrics if item.metric == name]
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedMedianBootstrapReport:
    proposed_candidate: str
    reference_candidate: str
    attack_type: str
    seed: int
    requested_replicates: int
    clusters: int
    proposed_episodes: int
    reference_episodes: int
    empty_episode_blocks: int
    shared_draw_fingerprint: str
    interval: ContrastInterval

    @property
    def one_sided_gate_endpoint(self) -> float | None:
        return self.interval.gate_endpoint

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SinglePolicyRatioReport:
    candidate: str
    seed: int
    requested_replicates: int
    clusters: int
    missing_block_count: int
    draw_fingerprint: str
    interval: ContrastInterval

    @property
    def upper_confidence_bound(self) -> float | None:
        return self.interval.one_sided_upper

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedAttackExposureReport:
    """One attack-class exposure contrast under shared complete-block draws."""

    proposed_candidate: str
    reference_candidate: str
    attack_type: str
    seed: int
    requested_replicates: int
    clusters: int
    missing_block_counts: Mapping[str, int]
    shared_draw_fingerprint: str
    interval: ContrastInterval

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateRule:
    metric: str
    direction: str
    threshold: float

    def __post_init__(self) -> None:
        if self.direction not in {"max", "min"}:
            raise ValueError("gate direction must be 'max' or 'min'")
        if not math.isfinite(self.threshold):
            raise ValueError("gate threshold must be finite")


@dataclass(frozen=True)
class GateDecision:
    metric: str
    direction: str
    threshold: float
    endpoint: float | None
    passed: bool | None
    status: str


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], table: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise DataValidationError(f"{table} contribution table missing columns: {missing}")


def _ordered_blocks(values: Sequence[Any]) -> tuple[Any, ...]:
    unique: dict[tuple[str, str], Any] = {}
    for value in values:
        if value is None:
            raise DataValidationError("trace-block identifiers cannot be null")
        token = (type(value).__name__, repr(value))
        unique[token] = value
    return tuple(unique[token] for token in sorted(unique))


def _validate_additive_values(
    frame: pl.DataFrame,
    columns: Sequence[str],
    table_name: str,
) -> None:
    for column in columns:
        try:
            values = np.asarray(frame[column].to_numpy(), dtype=float)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"{table_name}.{column} must be numeric"
            ) from exc
        if not np.isfinite(values).all() or (values < 0).any():
            raise DataValidationError(
                f"{table_name}.{column} must be finite and non-negative"
            )


def _aggregate_contributions(
    frame: pl.DataFrame,
    *,
    candidates: tuple[str, str],
    block_column: str,
    value_columns: Sequence[str],
    attack_type: str | None = None,
) -> pl.DataFrame:
    selected = frame.filter(pl.col("candidate").cast(pl.String).is_in(list(candidates)))
    if attack_type is not None:
        selected = selected.filter(pl.col("attack_type") == attack_type)
    if selected.is_empty():
        return pl.DataFrame(
            schema={
                "candidate": pl.String,
                block_column: frame.schema[block_column],
                **{column: pl.Float64 for column in value_columns},
            }
        )
    return (
        selected.select(["candidate", block_column, *value_columns])
        .with_columns(pl.col("candidate").cast(pl.String))
        .group_by(["candidate", block_column])
        .agg([pl.col(column).sum().cast(pl.Float64).alias(column) for column in value_columns])
    )


def _complete_table(
    aggregated: pl.DataFrame,
    *,
    candidates: tuple[str, str],
    blocks: tuple[Any, ...],
    block_column: str,
    value_columns: Sequence[str],
) -> pl.DataFrame:
    observed = {
        (str(row["candidate"]), row[block_column]): row
        for row in aggregated.iter_rows(named=True)
    }
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        for block in blocks:
            source = observed.get((candidate, block))
            record: dict[str, Any] = {
                "candidate": candidate,
                block_column: block,
            }
            for column in value_columns:
                record[column] = 0.0 if source is None else float(source[column])
            records.append(record)
    return pl.DataFrame(records, strict=False)


def complete_paired_block_grid(
    action_contributions: pl.DataFrame,
    attack_contributions: pl.DataFrame,
    *,
    proposed_candidate: str,
    reference_candidate: str,
    attack_type: str = ANY_ATTACK_TYPE,
    block_column: str = TRACE_BLOCK_COLUMN,
    trace_block_ids: Sequence[Any] | None = None,
) -> CompleteBlockGrid:
    """Aggregate duplicates and zero-fill the union of policy trace blocks."""

    proposed = str(proposed_candidate)
    reference = str(reference_candidate)
    if not proposed or not reference or proposed == reference:
        raise ValueError("proposed and reference candidates must be distinct and non-empty")
    _require_columns(
        action_contributions,
        ["candidate", block_column, *ACTION_ADDITIVE_COLUMNS],
        "action",
    )
    _require_columns(
        attack_contributions,
        ["candidate", block_column, "attack_type", *ATTACK_ADDITIVE_COLUMNS],
        "attack",
    )
    _validate_additive_values(action_contributions, ACTION_ADDITIVE_COLUMNS, "action")
    _validate_additive_values(attack_contributions, ATTACK_ADDITIVE_COLUMNS, "attack")
    observed_candidates = {
        str(value)
        for value in action_contributions["candidate"].drop_nulls().unique().to_list()
    } | {
        str(value)
        for value in attack_contributions["candidate"].drop_nulls().unique().to_list()
    }
    absent = sorted({proposed, reference} - observed_candidates)
    if absent:
        raise DataValidationError(f"candidate absent from both contribution tables: {absent}")

    candidates = (proposed, reference)
    action = _aggregate_contributions(
        action_contributions,
        candidates=candidates,
        block_column=block_column,
        value_columns=ACTION_ADDITIVE_COLUMNS,
    )
    attack = _aggregate_contributions(
        attack_contributions,
        candidates=candidates,
        block_column=block_column,
        value_columns=ATTACK_ADDITIVE_COLUMNS,
        attack_type=attack_type,
    )
    observed_blocks = action[block_column].to_list() + attack[block_column].to_list()
    blocks = _ordered_blocks(
        observed_blocks if trace_block_ids is None else list(trace_block_ids)
    )
    if len(blocks) < 2:
        raise DataValidationError("paired block bootstrap requires at least two blocks")
    unknown_blocks = [value for value in _ordered_blocks(observed_blocks) if value not in set(blocks)]
    if unknown_blocks:
        raise DataValidationError(
            f"contribution blocks absent from trace_block_ids: {unknown_blocks}"
        )
    action_observed = {
        candidate: set(
            action.filter(pl.col("candidate") == candidate)[block_column].to_list()
        )
        for candidate in candidates
    }
    attack_observed = {
        candidate: set(
            attack.filter(pl.col("candidate") == candidate)[block_column].to_list()
        )
        for candidate in candidates
    }
    missing_action = {
        candidate: tuple(block for block in blocks if block not in action_observed[candidate])
        for candidate in candidates
    }
    missing_attack = {
        candidate: tuple(block for block in blocks if block not in attack_observed[candidate])
        for candidate in candidates
    }
    completed_action = _complete_table(
        action,
        candidates=candidates,
        blocks=blocks,
        block_column=block_column,
        value_columns=ACTION_ADDITIVE_COLUMNS,
    )
    completed_attack = _complete_table(
        attack,
        candidates=candidates,
        blocks=blocks,
        block_column=block_column,
        value_columns=ATTACK_ADDITIVE_COLUMNS,
    )
    return CompleteBlockGrid(
        proposed_candidate=proposed,
        reference_candidate=reference,
        attack_type=attack_type,
        block_ids=blocks,
        missing_action_blocks=missing_action,
        missing_attack_blocks=missing_attack,
        action=completed_action,
        attack=completed_attack,
    )


def _candidate_arrays(
    frame: pl.DataFrame,
    candidate: str,
    blocks: tuple[Any, ...],
    value_columns: Sequence[str],
    block_column: str,
) -> dict[str, np.ndarray]:
    rows = {
        row[block_column]: row
        for row in frame.filter(pl.col("candidate") == candidate).iter_rows(named=True)
    }
    if set(rows) != set(blocks):
        raise AssertionError("completed candidate grid does not cover the common blocks")
    return {
        column: np.asarray([float(rows[block][column]) for block in blocks], dtype=float)
        for column in value_columns
    }


def _draw_sums(values: np.ndarray, sampled_indices: np.ndarray) -> np.ndarray:
    return values[sampled_indices].sum(axis=1)


def _interval(
    *,
    metric: str,
    contrast: str,
    estimate: float | None,
    draws: np.ndarray,
    requested_replicates: int,
    clusters: int,
    confidence: float,
    one_sided_confidence: float,
    gate_direction: str,
    undefined_status: str | None = None,
) -> ContrastInterval:
    finite = np.asarray(draws, dtype=float)
    finite = finite[np.isfinite(finite)]
    valid = len(finite)
    invalid = requested_replicates - valid
    minimum_valid = max(100, math.ceil(0.5 * requested_replicates))
    if undefined_status is not None:
        status = undefined_status
        lower = upper = one_lower = one_upper = endpoint = None
    elif valid < minimum_valid:
        status = "insufficient_valid_replicates"
        lower = upper = one_lower = one_upper = endpoint = None
    else:
        status = "ok"
        alpha = 1.0 - confidence
        lower, upper = (
            float(value)
            for value in np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
        )
        one_alpha = 1.0 - one_sided_confidence
        one_lower, one_upper = (
            float(value)
            for value in np.quantile(finite, [one_alpha, 1.0 - one_alpha])
        )
        endpoint = one_upper if gate_direction == "upper" else one_lower
    return ContrastInterval(
        metric=metric,
        contrast=contrast,
        estimate=estimate,
        ci_lower=lower,
        ci_upper=upper,
        one_sided_lower=one_lower,
        one_sided_upper=one_upper,
        gate_direction=gate_direction,
        gate_endpoint=endpoint,
        confidence=confidence,
        one_sided_confidence=one_sided_confidence,
        requested_replicates=requested_replicates,
        valid_replicates=valid,
        invalid_replicates=invalid,
        clusters=clusters,
        status=status,
    )


def _difference_of_ratios(
    *,
    metric: str,
    proposed_num: np.ndarray,
    proposed_den: np.ndarray,
    reference_num: np.ndarray,
    reference_den: np.ndarray,
    sampled_indices: np.ndarray,
    confidence: float,
    one_sided_confidence: float,
    gate_direction: str,
) -> ContrastInterval:
    proposed_den_total = float(proposed_den.sum())
    reference_den_total = float(reference_den.sum())
    undefined = None
    estimate: float | None
    if proposed_den_total <= 0 or reference_den_total <= 0:
        estimate = None
        undefined = "undefined_zero_aggregate_denominator"
    else:
        estimate = float(
            proposed_num.sum() / proposed_den_total
            - reference_num.sum() / reference_den_total
        )
    p_num = _draw_sums(proposed_num, sampled_indices)
    p_den = _draw_sums(proposed_den, sampled_indices)
    r_num = _draw_sums(reference_num, sampled_indices)
    r_den = _draw_sums(reference_den, sampled_indices)
    draws = np.full(len(sampled_indices), np.nan, dtype=float)
    valid = (p_den > 0) & (r_den > 0)
    draws[valid] = p_num[valid] / p_den[valid] - r_num[valid] / r_den[valid]
    return _interval(
        metric=metric,
        contrast="proposed_ratio_minus_reference_ratio",
        estimate=estimate,
        draws=draws,
        requested_replicates=len(sampled_indices),
        clusters=sampled_indices.shape[1],
        confidence=confidence,
        one_sided_confidence=one_sided_confidence,
        gate_direction=gate_direction,
        undefined_status=undefined,
    )


def _transition_rate_ratio(
    *,
    proposed_transitions: np.ndarray,
    proposed_time: np.ndarray,
    reference_transitions: np.ndarray,
    reference_time: np.ndarray,
    sampled_indices: np.ndarray,
    confidence: float,
    one_sided_confidence: float,
) -> ContrastInterval:
    p_transition_total = float(proposed_transitions.sum())
    p_time_total = float(proposed_time.sum())
    r_transition_total = float(reference_transitions.sum())
    r_time_total = float(reference_time.sum())
    undefined = None
    estimate: float | None
    if p_time_total <= 0 or r_time_total <= 0:
        estimate = None
        undefined = "undefined_zero_observed_time"
    elif r_transition_total <= 0:
        estimate = None
        undefined = "undefined_zero_reference_transition_rate"
    else:
        estimate = float(
            (p_transition_total / p_time_total)
            / (r_transition_total / r_time_total)
        )
    p_t = _draw_sums(proposed_transitions, sampled_indices)
    p_time = _draw_sums(proposed_time, sampled_indices)
    r_t = _draw_sums(reference_transitions, sampled_indices)
    r_time = _draw_sums(reference_time, sampled_indices)
    draws = np.full(len(sampled_indices), np.nan, dtype=float)
    valid = (p_time > 0) & (r_time > 0) & (r_t > 0)
    draws[valid] = (p_t[valid] / p_time[valid]) / (r_t[valid] / r_time[valid])
    return _interval(
        metric="transition_rate_ratio",
        contrast="proposed_transition_rate_divided_by_reference_transition_rate",
        estimate=estimate,
        draws=draws,
        requested_replicates=len(sampled_indices),
        clusters=sampled_indices.shape[1],
        confidence=confidence,
        one_sided_confidence=one_sided_confidence,
        gate_direction="upper",
        undefined_status=undefined,
    )


def paired_policy_bootstrap(
    action_contributions: pl.DataFrame,
    attack_contributions: pl.DataFrame,
    *,
    proposed_candidate: str,
    reference_candidate: str,
    attack_type: str = ANY_ATTACK_TYPE,
    block_column: str = TRACE_BLOCK_COLUMN,
    trace_block_ids: Sequence[Any] | None = None,
    replicates: int = DEFAULT_REPLICATES,
    confidence: float = 0.95,
    one_sided_confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> PairedBootstrapReport:
    """Estimate five predeclared proposed-reference paired contrasts."""

    if isinstance(replicates, bool) or replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if not 0 < confidence < 1 or not 0 < one_sided_confidence < 1:
        raise ValueError("confidence levels must lie strictly between zero and one")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    grid = complete_paired_block_grid(
        action_contributions,
        attack_contributions,
        proposed_candidate=proposed_candidate,
        reference_candidate=reference_candidate,
        attack_type=attack_type,
        block_column=block_column,
        trace_block_ids=trace_block_ids,
    )
    p_action = _candidate_arrays(
        grid.action,
        grid.proposed_candidate,
        grid.block_ids,
        ACTION_ADDITIVE_COLUMNS,
        block_column,
    )
    r_action = _candidate_arrays(
        grid.action,
        grid.reference_candidate,
        grid.block_ids,
        ACTION_ADDITIVE_COLUMNS,
        block_column,
    )
    p_attack = _candidate_arrays(
        grid.attack,
        grid.proposed_candidate,
        grid.block_ids,
        ATTACK_ADDITIVE_COLUMNS,
        block_column,
    )
    r_attack = _candidate_arrays(
        grid.attack,
        grid.reference_candidate,
        grid.block_ids,
        ATTACK_ADDITIVE_COLUMNS,
        block_column,
    )

    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(
        0,
        len(grid.block_ids),
        size=(int(replicates), len(grid.block_ids)),
    )
    fingerprint = hashlib.sha256(sampled.tobytes(order="C")).hexdigest()
    metrics = (
        _difference_of_ratios(
            metric="benign_friction_difference",
            proposed_num=p_action["benign_contained_time_s"],
            proposed_den=p_action["benign_time_s"],
            reference_num=r_action["benign_contained_time_s"],
            reference_den=r_action["benign_time_s"],
            sampled_indices=sampled,
            confidence=confidence,
            one_sided_confidence=one_sided_confidence,
            gate_direction="upper",
        ),
        _difference_of_ratios(
            metric="attack_exposure_difference",
            proposed_num=p_action["attack_exposed_time_s"],
            proposed_den=p_action["attack_time_s"],
            reference_num=r_action["attack_exposed_time_s"],
            reference_den=r_action["attack_time_s"],
            sampled_indices=sampled,
            confidence=confidence,
            one_sided_confidence=one_sided_confidence,
            gate_direction="upper",
        ),
        _difference_of_ratios(
            metric="episode_coverage_difference",
            proposed_num=p_attack["covered_episode_count"],
            proposed_den=p_attack["episode_count"],
            reference_num=r_attack["covered_episode_count"],
            reference_den=r_attack["episode_count"],
            sampled_indices=sampled,
            confidence=confidence,
            one_sided_confidence=one_sided_confidence,
            gate_direction="lower",
        ),
        _difference_of_ratios(
            metric="mean_capped_delay_difference_s",
            proposed_num=p_attack["capped_delay_sum_s"],
            proposed_den=p_attack["episode_count"],
            reference_num=r_attack["capped_delay_sum_s"],
            reference_den=r_attack["episode_count"],
            sampled_indices=sampled,
            confidence=confidence,
            one_sided_confidence=one_sided_confidence,
            gate_direction="upper",
        ),
        _transition_rate_ratio(
            proposed_transitions=p_action["effective_transition_count"],
            proposed_time=p_action["observed_time_s"],
            reference_transitions=r_action["effective_transition_count"],
            reference_time=r_action["observed_time_s"],
            sampled_indices=sampled,
            confidence=confidence,
            one_sided_confidence=one_sided_confidence,
        ),
    )
    return PairedBootstrapReport(
        proposed_candidate=grid.proposed_candidate,
        reference_candidate=grid.reference_candidate,
        attack_type=attack_type,
        seed=int(seed),
        requested_replicates=int(replicates),
        confidence=float(confidence),
        one_sided_confidence=float(one_sided_confidence),
        clusters=len(grid.block_ids),
        completed_action_rows=grid.action.height,
        completed_attack_rows=grid.attack.height,
        missing_action_block_counts={
            candidate: len(blocks)
            for candidate, blocks in grid.missing_action_blocks.items()
        },
        missing_attack_block_counts={
            candidate: len(blocks)
            for candidate, blocks in grid.missing_attack_blocks.items()
        },
        shared_draw_fingerprint=fingerprint,
        metrics=metrics,
    )


def _weighted_median_sorted(values: np.ndarray, weights: np.ndarray) -> float:
    """Exact median of integer-weighted observations without materializing repeats."""

    total = int(weights.sum())
    if total <= 0:
        return float("nan")
    cumulative = np.cumsum(weights, dtype=np.int64)
    lower_position = (total - 1) // 2
    upper_position = total // 2
    lower_index = int(np.searchsorted(cumulative, lower_position, side="right"))
    upper_index = int(np.searchsorted(cumulative, upper_position, side="right"))
    return float((values[lower_index] + values[upper_index]) / 2.0)


def paired_median_capped_delay_bootstrap(
    episodes: pl.DataFrame,
    *,
    proposed_candidate: str,
    reference_candidate: str,
    trace_block_ids: Sequence[Any],
    attack_type: str = ANY_ATTACK_TYPE,
    candidate_column: str = "candidate",
    episode_key_column: str = "episode_key",
    block_column: str = "onset_block_id",
    attack_type_column: str = "attack_type",
    delay_column: str = "capped_delay_s",
    replicates: int = DEFAULT_REPLICATES,
    confidence: float = 0.95,
    one_sided_confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> PairedMedianBootstrapReport:
    """Bootstrap proposed-reference median capped delay by complete trace block.

    Episode identities and onset blocks must match across candidates.  This is a
    deliberate integrity check: controller actions may change delay values but
    cannot change the ground-truth episode set.  ``trace_block_ids`` is required
    so blocks with no episode onset remain in the bootstrap sampling universe.
    """

    if isinstance(replicates, bool) or replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if not 0 < confidence < 1 or not 0 < one_sided_confidence < 1:
        raise ValueError("confidence levels must lie strictly between zero and one")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    proposed = str(proposed_candidate)
    reference = str(reference_candidate)
    if not proposed or not reference or proposed == reference:
        raise ValueError("proposed and reference candidates must be distinct and non-empty")
    required = [
        candidate_column,
        episode_key_column,
        block_column,
        attack_type_column,
        delay_column,
    ]
    _require_columns(episodes, required, "episode")
    observed_candidates = {
        str(value)
        for value in episodes[candidate_column].drop_nulls().unique().to_list()
    }
    absent = sorted({proposed, reference} - observed_candidates)
    if absent:
        raise DataValidationError(f"candidate absent from episode table: {absent}")
    blocks = _ordered_blocks(list(trace_block_ids))
    if len(blocks) < 2:
        raise DataValidationError("paired block bootstrap requires at least two blocks")
    block_index = {block: index for index, block in enumerate(blocks)}
    subset = episodes.filter(
        pl.col(candidate_column).cast(pl.String).is_in([proposed, reference])
        & (pl.col(attack_type_column) == attack_type)
    ).select(required).with_columns(pl.col(candidate_column).cast(pl.String))
    if not subset.is_empty():
        values = np.asarray(subset[delay_column].to_numpy(), dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise DataValidationError("episode capped delays must be finite and non-negative")
        if subset[episode_key_column].has_nulls() or subset[block_column].has_nulls():
            raise DataValidationError("episode keys and onset blocks cannot be null")

    by_candidate: dict[str, dict[Any, Mapping[str, Any]]] = {}
    for candidate in (proposed, reference):
        candidate_rows = subset.filter(pl.col(candidate_column) == candidate)
        records: dict[Any, Mapping[str, Any]] = {}
        for row in candidate_rows.iter_rows(named=True):
            key = row[episode_key_column]
            try:
                hash(key)
            except TypeError as exc:
                raise DataValidationError("episode keys must be hashable") from exc
            if key in records:
                raise DataValidationError(
                    f"duplicate episode key for candidate {candidate!r}: {key!r}"
                )
            if row[block_column] not in block_index:
                raise DataValidationError(
                    f"episode onset block absent from trace_block_ids: {row[block_column]!r}"
                )
            records[key] = row
        by_candidate[candidate] = records
    proposed_keys = set(by_candidate[proposed])
    reference_keys = set(by_candidate[reference])
    if proposed_keys != reference_keys:
        raise DataValidationError(
            "paired policies must contain the same ground-truth episode keys"
        )
    for key in proposed_keys:
        if (
            by_candidate[proposed][key][block_column]
            != by_candidate[reference][key][block_column]
        ):
            raise DataValidationError(
                f"episode onset block differs across policies for key {key!r}"
            )

    ordered_keys = sorted(proposed_keys, key=lambda value: (type(value).__name__, repr(value)))
    proposed_delay = np.asarray(
        [float(by_candidate[proposed][key][delay_column]) for key in ordered_keys],
        dtype=float,
    )
    reference_delay = np.asarray(
        [float(by_candidate[reference][key][delay_column]) for key in ordered_keys],
        dtype=float,
    )
    episode_blocks = np.asarray(
        [block_index[by_candidate[proposed][key][block_column]] for key in ordered_keys],
        dtype=np.int64,
    )
    p_order = np.argsort(proposed_delay, kind="stable")
    r_order = np.argsort(reference_delay, kind="stable")
    p_sorted = proposed_delay[p_order]
    r_sorted = reference_delay[r_order]
    p_block_sorted = episode_blocks[p_order]
    r_block_sorted = episode_blocks[r_order]

    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(0, len(blocks), size=(int(replicates), len(blocks)))
    fingerprint = hashlib.sha256(sampled.tobytes(order="C")).hexdigest()
    draws = np.full(int(replicates), np.nan, dtype=float)
    for index, sample in enumerate(sampled):
        multiplicity = np.bincount(sample, minlength=len(blocks))
        p_weights = multiplicity[p_block_sorted]
        r_weights = multiplicity[r_block_sorted]
        p_median = _weighted_median_sorted(p_sorted, p_weights)
        r_median = _weighted_median_sorted(r_sorted, r_weights)
        if math.isfinite(p_median) and math.isfinite(r_median):
            draws[index] = p_median - r_median
    if len(proposed_delay):
        estimate: float | None = float(
            np.median(proposed_delay) - np.median(reference_delay)
        )
        undefined = None
    else:
        estimate = None
        undefined = "undefined_no_episodes"
    interval = _interval(
        metric="median_capped_delay_difference_s",
        contrast="proposed_median_minus_reference_median",
        estimate=estimate,
        draws=draws,
        requested_replicates=int(replicates),
        clusters=len(blocks),
        confidence=confidence,
        one_sided_confidence=one_sided_confidence,
        gate_direction="upper",
        undefined_status=undefined,
    )
    blocks_with_episodes = set(episode_blocks.tolist())
    return PairedMedianBootstrapReport(
        proposed_candidate=proposed,
        reference_candidate=reference,
        attack_type=attack_type,
        seed=int(seed),
        requested_replicates=int(replicates),
        clusters=len(blocks),
        proposed_episodes=len(proposed_delay),
        reference_episodes=len(reference_delay),
        empty_episode_blocks=len(blocks) - len(blocks_with_episodes),
        shared_draw_fingerprint=fingerprint,
        interval=interval,
    )


def single_policy_benign_friction_bootstrap(
    action_contributions: pl.DataFrame,
    *,
    candidate: str,
    trace_block_ids: Sequence[Any] | None = None,
    block_column: str = TRACE_BLOCK_COLUMN,
    numerator_column: str = "benign_contained_time_s",
    denominator_column: str = "benign_time_s",
    replicates: int = DEFAULT_REPLICATES,
    confidence: float = 0.95,
    one_sided_confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> SinglePolicyRatioReport:
    """Return a block-bootstrap CI and one-sided UCB for benign friction."""

    if isinstance(replicates, bool) or replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if not 0 < confidence < 1 or not 0 < one_sided_confidence < 1:
        raise ValueError("confidence levels must lie strictly between zero and one")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    candidate_name = str(candidate)
    _require_columns(
        action_contributions,
        ["candidate", block_column, numerator_column, denominator_column],
        "action",
    )
    _validate_additive_values(
        action_contributions,
        [numerator_column, denominator_column],
        "action",
    )
    observed_candidates = {
        str(value)
        for value in action_contributions["candidate"].drop_nulls().unique().to_list()
    }
    if candidate_name not in observed_candidates:
        raise DataValidationError(f"candidate absent from action table: {candidate_name}")
    if trace_block_ids is None:
        blocks = _ordered_blocks(action_contributions[block_column].to_list())
    else:
        blocks = _ordered_blocks(list(trace_block_ids))
    if len(blocks) < 2:
        raise DataValidationError("block bootstrap requires at least two blocks")
    selected = (
        action_contributions.filter(
            pl.col("candidate").cast(pl.String) == candidate_name
        )
        .select(["candidate", block_column, numerator_column, denominator_column])
        .group_by(["candidate", block_column])
        .agg(
            pl.col(numerator_column).sum().cast(pl.Float64).alias(numerator_column),
            pl.col(denominator_column).sum().cast(pl.Float64).alias(denominator_column),
        )
    )
    observed = {
        row[block_column]: row for row in selected.iter_rows(named=True)
    }
    unknown = sorted(
        (block for block in observed if block not in set(blocks)),
        key=lambda value: (type(value).__name__, repr(value)),
    )
    if unknown:
        raise DataValidationError(
            f"candidate blocks absent from trace_block_ids: {unknown}"
        )
    numerator = np.asarray(
        [float(observed[block][numerator_column]) if block in observed else 0.0 for block in blocks],
        dtype=float,
    )
    denominator = np.asarray(
        [float(observed[block][denominator_column]) if block in observed else 0.0 for block in blocks],
        dtype=float,
    )
    denominator_total = float(denominator.sum())
    if denominator_total > 0:
        estimate: float | None = float(numerator.sum() / denominator_total)
        undefined = None
    else:
        estimate = None
        undefined = "undefined_zero_aggregate_denominator"
    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(0, len(blocks), size=(int(replicates), len(blocks)))
    fingerprint = hashlib.sha256(sampled.tobytes(order="C")).hexdigest()
    sampled_num = _draw_sums(numerator, sampled)
    sampled_den = _draw_sums(denominator, sampled)
    draws = np.full(int(replicates), np.nan, dtype=float)
    valid = sampled_den > 0
    draws[valid] = sampled_num[valid] / sampled_den[valid]
    interval = _interval(
        metric="benign_friction",
        contrast="single_policy_ratio",
        estimate=estimate,
        draws=draws,
        requested_replicates=int(replicates),
        clusters=len(blocks),
        confidence=confidence,
        one_sided_confidence=one_sided_confidence,
        gate_direction="upper",
        undefined_status=undefined,
    )
    return SinglePolicyRatioReport(
        candidate=candidate_name,
        seed=int(seed),
        requested_replicates=int(replicates),
        clusters=len(blocks),
        missing_block_count=sum(block not in observed for block in blocks),
        draw_fingerprint=fingerprint,
        interval=interval,
    )


def paired_attack_exposure_bootstrap(
    attack_contributions: pl.DataFrame,
    *,
    proposed_candidate: str,
    reference_candidate: str,
    attack_type: str,
    trace_block_ids: Sequence[Any] | None = None,
    block_column: str = TRACE_BLOCK_COLUMN,
    replicates: int = DEFAULT_REPLICATES,
    confidence: float = 0.95,
    one_sided_confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> PairedAttackExposureReport:
    """Bootstrap a class-specific malicious-ALLOW-time difference.

    Unlike :func:`paired_policy_bootstrap`, whose action-time exposure is the
    pooled all-attack estimand, this helper takes ``attack_type``-specific time
    numerators and denominators from the attack contribution table.  Supplying
    the complete block universe keeps blocks with zero instances of that class
    in the resampling distribution.
    """

    if isinstance(replicates, bool) or replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if not 0 < confidence < 1 or not 0 < one_sided_confidence < 1:
        raise ValueError("confidence levels must lie strictly between zero and one")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    proposed = str(proposed_candidate)
    reference = str(reference_candidate)
    if not proposed or not reference or proposed == reference:
        raise ValueError("proposed and reference candidates must be distinct and non-empty")
    required = [
        "candidate",
        block_column,
        "attack_type",
        "attack_time_s",
        "exposed_time_s",
    ]
    _require_columns(attack_contributions, required, "attack")
    _validate_additive_values(
        attack_contributions, ["attack_time_s", "exposed_time_s"], "attack"
    )
    selected = (
        attack_contributions.filter(
            pl.col("candidate").cast(pl.String).is_in([proposed, reference])
            & (pl.col("attack_type") == attack_type)
        )
        .select(required)
        .with_columns(pl.col("candidate").cast(pl.String))
        .group_by(["candidate", block_column])
        .agg(
            pl.col("attack_time_s").sum().cast(pl.Float64),
            pl.col("exposed_time_s").sum().cast(pl.Float64),
        )
    )
    observed_candidates = set(selected["candidate"].unique().to_list())
    absent = sorted({proposed, reference} - observed_candidates)
    if absent:
        raise DataValidationError(
            f"candidate absent for attack type {attack_type!r}: {absent}"
        )
    blocks = (
        _ordered_blocks(selected[block_column].to_list())
        if trace_block_ids is None
        else _ordered_blocks(list(trace_block_ids))
    )
    if len(blocks) < 2:
        raise DataValidationError("paired block bootstrap requires at least two blocks")
    block_set = set(blocks)
    unknown = [value for value in selected[block_column].unique().to_list() if value not in block_set]
    if unknown:
        raise DataValidationError(f"attack contribution blocks absent from block universe: {unknown}")
    completed = _complete_table(
        selected,
        candidates=(proposed, reference),
        blocks=blocks,
        block_column=block_column,
        value_columns=("attack_time_s", "exposed_time_s"),
    )
    proposed_values = _candidate_arrays(
        completed,
        proposed,
        blocks,
        ("attack_time_s", "exposed_time_s"),
        block_column,
    )
    reference_values = _candidate_arrays(
        completed,
        reference,
        blocks,
        ("attack_time_s", "exposed_time_s"),
        block_column,
    )
    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(0, len(blocks), size=(int(replicates), len(blocks)))
    fingerprint = hashlib.sha256(sampled.tobytes(order="C")).hexdigest()
    interval = _difference_of_ratios(
        metric="attack_exposure_difference",
        proposed_num=proposed_values["exposed_time_s"],
        proposed_den=proposed_values["attack_time_s"],
        reference_num=reference_values["exposed_time_s"],
        reference_den=reference_values["attack_time_s"],
        sampled_indices=sampled,
        confidence=confidence,
        one_sided_confidence=one_sided_confidence,
        gate_direction="upper",
    )
    observed_by_candidate = {
        candidate: set(
            selected.filter(pl.col("candidate") == candidate)[block_column].to_list()
        )
        for candidate in (proposed, reference)
    }
    return PairedAttackExposureReport(
        proposed_candidate=proposed,
        reference_candidate=reference,
        attack_type=str(attack_type),
        seed=int(seed),
        requested_replicates=int(replicates),
        clusters=len(blocks),
        missing_block_counts={
            candidate: sum(block not in observed_by_candidate[candidate] for block in blocks)
            for candidate in (proposed, reference)
        },
        shared_draw_fingerprint=fingerprint,
        interval=interval,
    )


def evaluate_gate_rules(
    report: PairedBootstrapReport,
    rules: Sequence[GateRule],
) -> tuple[GateDecision, ...]:
    """Apply thresholds to one-sided endpoints; unavailable CIs remain unknown."""

    decisions: list[GateDecision] = []
    seen: set[str] = set()
    for rule in rules:
        if rule.metric in seen:
            raise ValueError(f"duplicate gate metric: {rule.metric}")
        seen.add(rule.metric)
        interval = report.metric(rule.metric)
        expected_direction = "upper" if rule.direction == "max" else "lower"
        if interval.gate_direction != expected_direction:
            raise ValueError(
                f"gate rule direction for {rule.metric} conflicts with estimand"
            )
        endpoint = interval.gate_endpoint
        if interval.status != "ok" or endpoint is None:
            passed: bool | None = None
            status = interval.status
        elif rule.direction == "max":
            passed = endpoint <= rule.threshold
            status = "pass" if passed else "fail"
        else:
            passed = endpoint >= rule.threshold
            status = "pass" if passed else "fail"
        decisions.append(
            GateDecision(
                metric=rule.metric,
                direction=rule.direction,
                threshold=float(rule.threshold),
                endpoint=endpoint,
                passed=passed,
                status=status,
            )
        )
    return tuple(decisions)


__all__ = [
    "DEFAULT_REPLICATES",
    "DEFAULT_SEED",
    "CompleteBlockGrid",
    "ContrastInterval",
    "PairedBootstrapReport",
    "PairedMedianBootstrapReport",
    "SinglePolicyRatioReport",
    "PairedAttackExposureReport",
    "GateRule",
    "GateDecision",
    "complete_paired_block_grid",
    "paired_policy_bootstrap",
    "paired_median_capped_delay_bootstrap",
    "single_policy_benign_friction_bootstrap",
    "paired_attack_exposure_bootstrap",
    "evaluate_gate_rules",
]
