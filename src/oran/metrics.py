"""Action-level metrics for causal O-RAN containment replays.

Rows are interpreted as sample-and-hold observations: the state and ground truth
at row ``i`` apply until the next report for the same subject and lease.  This
makes friction and exposure robust to irregular sampling.  The final report in a
lease has zero duration unless an explicit observation horizon is supplied.

Ground-truth labels are used only here, after controller replay.  They never enter
the controller lifecycle or persistence logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Hashable

import numpy as np

from .controller import AccessState, timestamp_seconds, timestamps_seconds


def _one_dimensional(name: str, values: Sequence[Any] | np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _matching_array(
    name: str,
    values: Sequence[Any] | np.ndarray | None,
    length: int,
    default: Hashable,
) -> np.ndarray:
    if values is None:
        return np.full(length, default, dtype=object)
    result = np.asarray(values, dtype=object)
    if result.ndim != 1 or len(result) != length:
        raise ValueError(f"{name} must be 1-D with length {length}")
    for item in result:
        if item is None or (
            isinstance(item, (float, np.floating)) and np.isnan(item)
        ):
            raise ValueError(f"{name} must not contain missing identifiers")
        try:
            hash(item)
        except TypeError as exc:
            raise TypeError(f"{name} values must be hashable") from exc
    return result


def _boolean_array(
    name: str, values: Sequence[bool] | np.ndarray, length: int
) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != length:
        raise ValueError(f"{name} must be 1-D with length {length}")
    if result.dtype.kind == "b":
        return result.astype(bool, copy=False)
    if result.dtype.kind in "iu" and np.isin(result, (0, 1)).all():
        return result.astype(bool)
    if result.dtype.kind == "f" and np.isfinite(result).all() and np.isin(
        result, (0.0, 1.0)
    ).all():
        return result.astype(bool)
    raise TypeError(f"{name} must contain only booleans or 0/1 values")


def state_values(states: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Normalize enum, integer, or state-name values to ``int8`` severity."""

    raw = _one_dimensional("states", states)
    normalized = np.empty(len(raw), dtype=np.int8)
    for index, value in enumerate(raw):
        if isinstance(value, str):
            try:
                normalized[index] = int(AccessState[value.strip().upper()])
            except KeyError as exc:
                raise ValueError(f"invalid access state {value!r}") from exc
        else:
            try:
                normalized[index] = int(AccessState(int(value)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid access state {value!r}") from exc
    return normalized


def _horizon_for_key(
    observation_end: Any,
    subject: Hashable,
    lease: Hashable,
) -> float | None:
    if observation_end is None:
        return None
    if isinstance(observation_end, Mapping):
        if (subject, lease) in observation_end:
            return timestamp_seconds(observation_end[(subject, lease)])
        if subject in observation_end:
            return timestamp_seconds(observation_end[subject])
        if lease in observation_end:
            return timestamp_seconds(observation_end[lease])
        return None
    return timestamp_seconds(observation_end)


def observation_durations(
    timestamps: Sequence[Any] | np.ndarray,
    *,
    subject_ids: Sequence[Hashable] | np.ndarray | None = None,
    lease_ids: Sequence[Hashable] | np.ndarray | None = None,
    observation_end: Any | Mapping[Any, Any] | None = None,
    interval_cap_s: float | None = None,
) -> np.ndarray:
    """Compute forward sample-and-hold durations in entity-time seconds.

    Durations never bridge a subject or lease boundary.  Interleaved subjects are
    supported and the input is not sorted implicitly.  ``interval_cap_s`` can be
    used to prevent sparse telemetry gaps from dominating time-weighted metrics;
    it caps intervals but does not reset a controller lifecycle.
    """

    times = timestamps_seconds(timestamps)
    length = len(times)
    subjects = _matching_array("subject_ids", subject_ids, length, "__subject__")
    leases = _matching_array("lease_ids", lease_ids, length, "__lease__")
    if interval_cap_s is not None:
        if not np.isfinite(interval_cap_s) or interval_cap_s <= 0:
            raise ValueError("interval_cap_s must be finite and positive")

    durations = np.zeros(length, dtype=float)
    # A lease identifier should be unique, but treating every observed lease
    # change as a new lifecycle also prevents accidental A -> B -> A reuse from
    # bridging two controller lifecycles.
    last_lease: dict[Hashable, Hashable] = {}
    generation: dict[Hashable, int] = {}
    previous: dict[tuple[Hashable, int], tuple[int, float, Hashable]] = {}
    for index, (time_s, subject, lease) in enumerate(
        zip(times, subjects, leases, strict=True)
    ):
        if subject not in last_lease:
            generation[subject] = 0
        elif last_lease[subject] != lease:
            generation[subject] += 1
        last_lease[subject] = lease
        key = (subject, generation[subject])
        if key in previous:
            previous_index, previous_time, _ = previous[key]
            elapsed = time_s - previous_time
            if elapsed < 0:
                raise ValueError(
                    "timestamps must be nondecreasing within each subject/lease"
                )
            durations[previous_index] = elapsed
        previous[key] = (index, time_s, lease)

    for (subject, _), (index, time_s, lease) in previous.items():
        horizon = _horizon_for_key(observation_end, subject, lease)
        if horizon is not None:
            elapsed = horizon - time_s
            if elapsed < 0:
                raise ValueError("observation_end precedes a lease's final report")
            durations[index] = elapsed

    if interval_cap_s is not None:
        np.minimum(durations, interval_cap_s, out=durations)
    return durations


def _metric_durations(
    times: np.ndarray,
    subjects: np.ndarray,
    leases: np.ndarray,
    durations_s: Sequence[float] | np.ndarray | None,
    observation_end: Any | Mapping[Any, Any] | None,
    interval_cap_s: float | None,
) -> np.ndarray:
    if durations_s is None:
        return observation_durations(
            times,
            subject_ids=subjects,
            lease_ids=leases,
            observation_end=observation_end,
            interval_cap_s=interval_cap_s,
        )
    if observation_end is not None:
        raise ValueError("durations_s and observation_end are mutually exclusive")
    durations = np.asarray(durations_s, dtype=float)
    if durations.ndim != 1 or len(durations) != len(times):
        raise ValueError("durations_s must be 1-D and match timestamps")
    if not np.isfinite(durations).all() or (durations < 0).any():
        raise ValueError("durations_s must be finite and non-negative")
    durations = durations.copy()
    if interval_cap_s is not None:
        if not np.isfinite(interval_cap_s) or interval_cap_s <= 0:
            raise ValueError("interval_cap_s must be finite and positive")
        np.minimum(durations, interval_cap_s, out=durations)
    return durations


@dataclass(frozen=True, slots=True)
class TimeWeightedMetrics:
    """Time-weighted service and containment outcomes."""

    observed_time_s: float
    benign_time_s: float
    attack_time_s: float
    benign_restricted_time_s: float
    benign_isolated_time_s: float
    malicious_exposure_time_s: float
    malicious_contained_time_s: float
    benign_friction: float
    benign_isolation: float
    malicious_exposure: float
    malicious_containment: float
    transitions: int
    severity_transitions: int
    transitions_per_minute: float
    severity_transitions_per_minute: float
    false_restriction_episodes: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _transition_counts(
    states: np.ndarray, subjects: np.ndarray, leases: np.ndarray
) -> tuple[int, int]:
    previous: dict[Hashable, tuple[Hashable, int]] = {}
    transitions = 0
    severity = 0
    for state, subject, lease in zip(states, subjects, leases, strict=True):
        last = previous.get(subject)
        if last is not None and last[0] == lease and last[1] != state:
            transitions += 1
            severity += abs(int(state) - int(last[1]))
        previous[subject] = (lease, int(state))
    return transitions, severity


def _false_episode_count(
    states: np.ndarray,
    attacks: np.ndarray,
    subjects: np.ndarray,
    leases: np.ndarray,
    containment_state: int,
) -> int:
    previous: dict[Hashable, tuple[Hashable, bool]] = {}
    episodes = 0
    for state, attack, subject, lease in zip(
        states, attacks, subjects, leases, strict=True
    ):
        active = not attack and state >= containment_state
        last = previous.get(subject)
        prior_active = bool(last is not None and last[0] == lease and last[1])
        if active and not prior_active:
            episodes += 1
        previous[subject] = (lease, bool(active))
    return episodes


def time_weighted_access_metrics(
    timestamps: Sequence[Any] | np.ndarray,
    states: Sequence[Any] | np.ndarray,
    is_attack: Sequence[bool] | np.ndarray,
    *,
    subject_ids: Sequence[Hashable] | np.ndarray | None = None,
    lease_ids: Sequence[Hashable] | np.ndarray | None = None,
    durations_s: Sequence[float] | np.ndarray | None = None,
    observation_end: Any | Mapping[Any, Any] | None = None,
    interval_cap_s: float | None = None,
    containment_state: AccessState = AccessState.RESTRICT,
) -> TimeWeightedMetrics:
    """Compute time-weighted friction, exposure, and action churn.

    ``benign_friction`` is the fraction of benign entity-time at or above
    ``containment_state``.  ``malicious_exposure`` is the fraction of attack
    entity-time below that state.  Churn is normalized by total observed
    entity-time, so unequal sampling frequencies do not change its denominator.
    """

    times = timestamps_seconds(timestamps)
    length = len(times)
    normalized_states = state_values(states)
    if len(normalized_states) != length:
        raise ValueError("states and timestamps must have equal length")
    attacks = _boolean_array("is_attack", is_attack, length)
    subjects = _matching_array("subject_ids", subject_ids, length, "__subject__")
    leases = _matching_array("lease_ids", lease_ids, length, "__lease__")
    threshold = int(AccessState(containment_state))
    durations = _metric_durations(
        times,
        subjects,
        leases,
        durations_s,
        observation_end,
        interval_cap_s,
    )

    benign = ~attacks
    contained = normalized_states >= threshold
    isolated = normalized_states == int(AccessState.ISOLATE)
    observed_time = float(durations.sum())
    benign_time = float(durations[benign].sum())
    attack_time = float(durations[attacks].sum())
    benign_restricted = float(durations[benign & contained].sum())
    benign_isolated = float(durations[benign & isolated].sum())
    attack_contained = float(durations[attacks & contained].sum())
    attack_exposed = float(durations[attacks & ~contained].sum())
    transitions, severity_transitions = _transition_counts(
        normalized_states, subjects, leases
    )
    false_episodes = _false_episode_count(
        normalized_states, attacks, subjects, leases, threshold
    )
    return TimeWeightedMetrics(
        observed_time_s=observed_time,
        benign_time_s=benign_time,
        attack_time_s=attack_time,
        benign_restricted_time_s=benign_restricted,
        benign_isolated_time_s=benign_isolated,
        malicious_exposure_time_s=attack_exposed,
        malicious_contained_time_s=attack_contained,
        benign_friction=_ratio(benign_restricted, benign_time),
        benign_isolation=_ratio(benign_isolated, benign_time),
        malicious_exposure=_ratio(attack_exposed, attack_time),
        malicious_containment=_ratio(attack_contained, attack_time),
        transitions=transitions,
        severity_transitions=severity_transitions,
        transitions_per_minute=_ratio(60.0 * transitions, observed_time),
        severity_transitions_per_minute=_ratio(
            60.0 * severity_transitions, observed_time
        ),
        false_restriction_episodes=false_episodes,
    )


# Short discoverable alias for notebooks and experiment drivers.
compute_time_weighted_metrics = time_weighted_access_metrics


def _missing(value: Any) -> bool:
    if value is None:
        return True
    return bool(isinstance(value, (float, np.floating)) and np.isnan(value))


def attack_episode_metrics(
    timestamps: Sequence[Any] | np.ndarray,
    states: Sequence[Any] | np.ndarray,
    is_attack: Sequence[bool] | np.ndarray,
    *,
    attack_types: Sequence[Any] | np.ndarray | None = None,
    episode_ids: Sequence[Any] | np.ndarray | None = None,
    subject_ids: Sequence[Hashable] | np.ndarray | None = None,
    lease_ids: Sequence[Hashable] | np.ndarray | None = None,
    durations_s: Sequence[float] | np.ndarray | None = None,
    observation_end: Any | Mapping[Any, Any] | None = None,
    interval_cap_s: float | None = None,
    delay_cap_s: float | None = None,
    containment_state: AccessState = AccessState.RESTRICT,
):
    """Return one row per contiguous ground-truth attack episode.

    A detected episode has an uncensored ``delay_s`` from attack onset to the
    first contained report.  For a miss, ``delay_s`` is NaN, ``censored`` is
    true, and ``capped_delay_s`` is the lesser of episode duration and the
    optional delay cap.  Thus misses remain represented in delay summaries
    instead of disappearing from detected-only statistics.

    Attack labels and episode identifiers define evaluation episodes only; they
    are never used to reset controller state.
    """

    import polars as pl

    times = timestamps_seconds(timestamps)
    length = len(times)
    normalized_states = state_values(states)
    if len(normalized_states) != length:
        raise ValueError("states and timestamps must have equal length")
    attacks = _boolean_array("is_attack", is_attack, length)
    subjects = _matching_array("subject_ids", subject_ids, length, "__subject__")
    leases = _matching_array("lease_ids", lease_ids, length, "__lease__")
    if attack_types is None:
        types = np.full(length, "attack", dtype=object)
    else:
        types = _one_dimensional("attack_types", attack_types).astype(object)
        if len(types) != length:
            raise ValueError("attack_types and timestamps must have equal length")
    if episode_ids is None:
        supplied_episode_ids = np.full(length, None, dtype=object)
        has_supplied_ids = False
    else:
        supplied_episode_ids = _one_dimensional(
            "episode_ids", episode_ids
        ).astype(object)
        if len(supplied_episode_ids) != length:
            raise ValueError("episode_ids and timestamps must have equal length")
        has_supplied_ids = True
    if delay_cap_s is not None:
        if not np.isfinite(delay_cap_s) or delay_cap_s <= 0:
            raise ValueError("delay_cap_s must be finite and positive")

    durations = _metric_durations(
        times,
        subjects,
        leases,
        durations_s,
        observation_end,
        interval_cap_s,
    )
    threshold = int(AccessState(containment_state))
    active: dict[Hashable, dict[str, Any]] = {}
    previous_lease: dict[Hashable, Hashable] = {}
    records: list[dict[str, Any]] = []
    next_episode = 0

    def finish(subject: Hashable) -> None:
        episode = active.pop(subject, None)
        if episode is None:
            return
        duration = float(episode["duration_s"])
        first_action = episode["first_action_s"]
        covered = first_action is not None
        delay = float(first_action - episode["start_s"]) if covered else np.nan
        censor_time = duration
        capped = delay if covered else censor_time
        capped = min(float(capped), duration)
        if delay_cap_s is not None:
            capped = min(capped, delay_cap_s)
        records.append(
            {
                "episode": episode["episode"],
                "source_episode_id": episode["source_episode_id"],
                "subject_id": subject,
                "lease_id": episode["lease_id"],
                "attack_type": episode["attack_type"],
                "start_s": episode["start_s"],
                "end_s": episode["start_s"] + duration,
                "duration_s": duration,
                "reports": episode["reports"],
                "covered": covered,
                "first_action_s": first_action if covered else np.nan,
                "delay_s": delay,
                "censored": not covered,
                "censor_time_s": censor_time,
                "capped_delay_s": capped,
                "contained_time_s": episode["contained_time_s"],
                "contained_fraction": _ratio(
                    episode["contained_time_s"], duration
                ),
            }
        )

    for index in range(length):
        subject = subjects[index]
        lease = leases[index]
        attack = bool(attacks[index])
        attack_type = "attack" if _missing(types[index]) else str(types[index])
        supplied_id = supplied_episode_ids[index]
        token = supplied_id if has_supplied_ids and not _missing(supplied_id) else None

        if subject in previous_lease and previous_lease[subject] != lease:
            finish(subject)
        previous_lease[subject] = lease
        current = active.get(subject)

        boundary = False
        if current is not None:
            if not attack or current["lease_id"] != lease:
                boundary = True
            elif current["attack_type"] != attack_type:
                boundary = True
            elif has_supplied_ids and current["source_episode_id"] != token:
                boundary = True
        if boundary:
            finish(subject)
            current = None

        if not attack:
            continue
        if current is None:
            current = {
                "episode": next_episode,
                "source_episode_id": token,
                "lease_id": lease,
                "attack_type": attack_type,
                "start_s": float(times[index]),
                "duration_s": 0.0,
                "reports": 0,
                "first_action_s": None,
                "contained_time_s": 0.0,
            }
            next_episode += 1
            active[subject] = current

        current["reports"] += 1
        current["duration_s"] += float(durations[index])
        if normalized_states[index] >= threshold:
            current["contained_time_s"] += float(durations[index])
            if current["first_action_s"] is None:
                current["first_action_s"] = float(times[index])

    for subject in list(active):
        finish(subject)

    columns = [
        "episode",
        "source_episode_id",
        "subject_id",
        "lease_id",
        "attack_type",
        "start_s",
        "end_s",
        "duration_s",
        "reports",
        "covered",
        "first_action_s",
        "delay_s",
        "censored",
        "censor_time_s",
        "capped_delay_s",
        "contained_time_s",
        "contained_fraction",
    ]
    if not records:
        return pl.DataFrame({column: [] for column in columns})
    return pl.DataFrame(records, strict=False).select(columns)


def summarize_attack_episodes(episodes):
    """Aggregate episode coverage and censored-delay summaries by attack type."""

    import polars as pl

    required = {
        "attack_type",
        "covered",
        "censored",
        "duration_s",
        "contained_time_s",
        "delay_s",
        "capped_delay_s",
    }
    frame = _as_polars_table(episodes)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"episode table is missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for group in frame.partition_by("attack_type", maintain_order=False):
        attack_type = group["attack_type"][0]
        covered = group["covered"].to_numpy().astype(bool)
        delay_values = group["delay_s"].to_numpy().astype(float)
        detected_delays = delay_values[covered]
        duration_values = group["duration_s"].to_numpy().astype(float)
        contained_values = group["contained_time_s"].to_numpy().astype(float)
        capped_values = group["capped_delay_s"].to_numpy().astype(float)
        total_duration = float(duration_values.sum())
        contained_duration = float(contained_values.sum())
        rows.append(
            {
                "attack_type": attack_type,
                "episodes": int(len(group)),
                "covered_episodes": int(covered.sum()),
                "censored_episodes": int((~covered).sum()),
                "episode_coverage": float(covered.mean()),
                "attack_time_s": total_duration,
                "contained_time_s": contained_duration,
                "time_coverage": _ratio(contained_duration, total_duration),
                "median_detected_delay_s": (
                    float(np.median(detected_delays))
                    if len(detected_delays)
                    else np.nan
                ),
                "median_capped_delay_s": float(np.median(capped_values)),
            }
        )
    if not rows:
        return pl.DataFrame(
            {
                "attack_type": [],
                "episodes": [],
                "covered_episodes": [],
                "censored_episodes": [],
                "episode_coverage": [],
                "attack_time_s": [],
                "contained_time_s": [],
                "time_coverage": [],
                "median_detected_delay_s": [],
                "median_capped_delay_s": [],
            }
        )
    return pl.DataFrame(rows, strict=False).sort("attack_type")


def _as_polars_table(table):
    """Accept Polars, pandas-like, mappings, or records without pandas import."""

    import polars as pl

    if isinstance(table, pl.DataFrame):
        return table.clone()
    if hasattr(table, "collect") and hasattr(table, "schema"):
        collected = table.collect()
        if isinstance(collected, pl.DataFrame):
            return collected
    if hasattr(table, "to_dicts"):
        return pl.DataFrame(table.to_dicts(), strict=False)
    if hasattr(table, "to_dict") and hasattr(table, "columns"):
        # pandas-compatible path, without importing pandas in this package.
        try:
            return pl.DataFrame(table.to_dict(orient="records"), strict=False)
        except TypeError:
            pass
    return pl.DataFrame(table, strict=False)


def _calibration_table(table, split_col: str, calibration_label: str):
    frame = _as_polars_table(table)
    if split_col not in frame.columns:
        raise ValueError(
            f"candidate metrics must include {split_col!r} to prove calibration-only selection"
        )
    roles = [str(value).casefold() for value in frame[split_col].to_list()]
    expected = calibration_label.casefold()
    if any(role != expected for role in roles):
        observed = sorted(
            {str(value) for value, role in zip(frame[split_col], roles) if role != expected}
        )
        raise ValueError(
            "budget candidates may be selected from calibration data only; "
            f"found {observed}"
        )
    return frame


def calibration_budget_frontier(
    candidate_metrics,
    budgets: Sequence[float] | np.ndarray,
    *,
    candidate_col: str = "candidate",
    friction_col: str = "benign_friction",
    objective_cols: Sequence[str] = (
        "malicious_exposure",
        "transitions_per_minute",
        "median_capped_delay_s",
    ),
    split_col: str = "split",
    calibration_label: str = "calibration",
):
    """Return nondominated calibration candidates feasible at each budget.

    All objectives are minimized; convert maximization measures such as attack
    coverage to a loss before calling.  The split marker is mandatory and any
    non-calibration row is rejected, preventing accidental test-set tuning.
    """

    import polars as pl

    frame = _calibration_table(candidate_metrics, split_col, calibration_label)
    required = {candidate_col, friction_col, *objective_cols}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"candidate table is missing columns: {sorted(missing)}")
    candidate_values = frame[candidate_col].to_list()
    if len(set(candidate_values)) != len(candidate_values):
        raise ValueError("candidate identifiers must be unique on calibration data")
    numeric_cols = [friction_col, *objective_cols]
    try:
        numeric = np.column_stack(
            [np.asarray(frame[column].to_list(), dtype=float) for column in numeric_cols]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "friction and objective columns must be finite numeric values"
        ) from exc
    if not np.isfinite(numeric).all():
        raise ValueError("friction and objective columns must be finite numeric values")
    records = frame.to_dicts()
    for row_index, record in enumerate(records):
        for column_index, column in enumerate(numeric_cols):
            record[column] = float(numeric[row_index, column_index])

    budget_array = np.asarray(budgets, dtype=float)
    if budget_array.ndim != 1 or not len(budget_array):
        raise ValueError("budgets must be a non-empty one-dimensional sequence")
    if not np.isfinite(budget_array).all() or (budget_array < 0).any():
        raise ValueError("budgets must be finite and non-negative")

    frontier_rows = []
    for budget in np.unique(np.sort(budget_array)):
        feasible = [row for row in records if row[friction_col] <= budget]
        if not feasible:
            continue
        values = np.asarray(
            [[row[column] for column in objective_cols] for row in feasible],
            dtype=float,
        )
        keep = np.ones(len(feasible), dtype=bool)
        for index in range(len(feasible)):
            dominated = np.all(values <= values[index], axis=1) & np.any(
                values < values[index], axis=1
            )
            dominated[index] = False
            if dominated.any():
                keep[index] = False
        for selected, retained in zip(feasible, keep, strict=True):
            if retained:
                frontier_rows.append(
                    {"friction_budget": float(budget), **selected}
                )
    if not frontier_rows:
        return pl.DataFrame(
            {column: [] for column in ["friction_budget", *frame.columns]}
        )
    return pl.DataFrame(frontier_rows, strict=False)


def select_budget_candidates(
    candidate_metrics,
    budgets: Sequence[float] | np.ndarray,
    *,
    candidate_col: str = "candidate",
    friction_col: str = "benign_friction",
    objective_priority: Sequence[str] = (
        "malicious_exposure",
        "transitions_per_minute",
        "median_capped_delay_s",
    ),
    split_col: str = "split",
    calibration_label: str = "calibration",
):
    """Select one deterministic Pareto candidate per calibration budget.

    Selection is lexicographic in ``objective_priority`` after Pareto filtering.
    Keeping the priority explicit avoids an undocumented weighted sum.  Test data
    cannot be passed through this API because non-calibration rows are rejected.
    """

    frontier = calibration_budget_frontier(
        candidate_metrics,
        budgets,
        candidate_col=candidate_col,
        friction_col=friction_col,
        objective_cols=objective_priority,
        split_col=split_col,
        calibration_label=calibration_label,
    )
    if frontier.is_empty():
        import polars as pl

        return frontier.with_columns(pl.Series("selected", [], dtype=pl.Boolean))
    rows = frontier.to_dicts()
    selected_rows = []
    for budget in sorted({float(row["friction_budget"]) for row in rows}):
        feasible = [row for row in rows if float(row["friction_budget"]) == budget]
        chosen = min(
            feasible,
            key=lambda row: (
                *(float(row[column]) for column in objective_priority),
                float(row[friction_col]),
                str(row[candidate_col]),
            ),
        )
        selected_rows.append({**chosen, "selected": True})
    import polars as pl

    return pl.DataFrame(selected_rows, strict=False).sort("friction_budget")


# Naming aliases used in experiment plans.
budget_frontier = calibration_budget_frontier
budget_frontier_candidate_selection = select_budget_candidates


__all__ = [
    "TimeWeightedMetrics",
    "state_values",
    "observation_durations",
    "time_weighted_access_metrics",
    "compute_time_weighted_metrics",
    "attack_episode_metrics",
    "summarize_attack_episodes",
    "calibration_budget_frontier",
    "budget_frontier",
    "select_budget_candidates",
    "budget_frontier_candidate_selection",
]
