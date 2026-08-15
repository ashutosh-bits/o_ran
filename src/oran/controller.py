"""Causal score-to-action controllers for O-RAN containment experiments.

The module deliberately knows nothing about attack labels.  A controller consumes
only a risk score, an observation time, a subject identifier, and an explicit
``lease_id`` supplied by the data preparation layer.  Per-subject state is reset
when, and only when, that lease changes (apart from an explicit ``clear`` made by
the caller before starting a new replay).

All controllers use the same three ordered actions::

    ALLOW < RESTRICT < ISOLATE

``TemporalAccessController`` is the configurable policy used by the experiments.
The remaining classes are intentionally small baselines that make the temporal
assumptions in a comparison explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Hashable, Sequence

import numpy as np


class AccessState(IntEnum):
    """Ordered access/containment actions."""

    ALLOW = 0
    RESTRICT = 1
    ISOLATE = 2


# Convenient public constants, while retaining an enum for ordering and typing.
ALLOW = AccessState.ALLOW
RESTRICT = AccessState.RESTRICT
ISOLATE = AccessState.ISOLATE


def _finite_nonnegative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative; got {value!r}")


def timestamp_seconds(value: Any) -> float:
    """Convert a numeric or datetime-like scalar to floating-point seconds.

    Datetimes are represented as seconds since the Unix epoch.  Numeric inputs
    are assumed already to use seconds.  The absolute origin is irrelevant to
    controller behavior; only within-lease differences are used.
    """

    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("timestamp must not be NaT")
        return float(value.astype("datetime64[ns]").astype(np.int64)) / 1e9
    if isinstance(value, datetime):
        normalized = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
        return float(normalized.timestamp())
    # pandas.Timestamp and similar objects expose ``to_datetime64``.
    if hasattr(value, "to_datetime64"):
        return timestamp_seconds(value.to_datetime64())
    if isinstance(value, str):
        try:
            return timestamp_seconds(np.datetime64(value, "ns"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid timestamp string {value!r}") from exc
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"unsupported timestamp {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError("timestamp must be finite")
    return result


def timestamps_seconds(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Vector form of :func:`timestamp_seconds`."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if array.dtype.kind == "M":
        if np.isnat(array).any():
            raise ValueError("timestamps must not contain NaT")
        return array.astype("datetime64[ns]").astype(np.int64) / 1e9
    if array.dtype.kind in "iuf":
        result = array.astype(float, copy=False)
        if not np.isfinite(result).all():
            raise ValueError("timestamps must be finite")
        return result
    return np.fromiter(
        (timestamp_seconds(item) for item in array), dtype=float, count=len(array)
    )


def _valid_identifier(name: str, value: Any) -> Hashable:
    if value is None:
        raise ValueError(f"{name} must be supplied")
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        raise ValueError(f"{name} must not be NaN")
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be hashable; got {value!r}") from exc
    return value


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    """Configuration for a causal, three-state temporal controller.

    ``*_duration_s`` and ``*_reports`` are conjunctive: both the elapsed time
    and report count must be satisfied.  A duration of zero and one report gives
    an immediate transition.  Recovery holds never block escalation.
    """

    restrict_enter: float
    restrict_exit: float
    isolate_enter: float
    isolate_exit: float
    entry_reports: int = 1
    recovery_reports: int = 1
    entry_duration_s: float = 0.0
    recovery_duration_s: float = 0.0
    min_restrict_hold_s: float = 0.0
    min_isolate_hold_s: float = 0.0
    ewma_tau_s: float | None = None
    initial_state: AccessState = AccessState.ALLOW

    def __post_init__(self) -> None:
        thresholds = (
            self.restrict_exit,
            self.restrict_enter,
            self.isolate_exit,
            self.isolate_enter,
        )
        if not all(np.isfinite(x) for x in thresholds):
            raise ValueError("all thresholds must be finite")
        if self.restrict_exit > self.restrict_enter:
            raise ValueError("restrict_exit must not exceed restrict_enter")
        if self.isolate_exit > self.isolate_enter:
            raise ValueError("isolate_exit must not exceed isolate_enter")
        if self.restrict_enter > self.isolate_enter:
            raise ValueError("restrict_enter must not exceed isolate_enter")
        if self.restrict_exit > self.isolate_exit:
            raise ValueError("restrict_exit must not exceed isolate_exit")
        if isinstance(self.entry_reports, bool) or self.entry_reports < 1:
            raise ValueError("entry_reports must be a positive integer")
        if isinstance(self.recovery_reports, bool) or self.recovery_reports < 1:
            raise ValueError("recovery_reports must be a positive integer")
        if int(self.entry_reports) != self.entry_reports:
            raise ValueError("entry_reports must be an integer")
        if int(self.recovery_reports) != self.recovery_reports:
            raise ValueError("recovery_reports must be an integer")
        _finite_nonnegative("entry_duration_s", self.entry_duration_s)
        _finite_nonnegative("recovery_duration_s", self.recovery_duration_s)
        _finite_nonnegative("min_restrict_hold_s", self.min_restrict_hold_s)
        _finite_nonnegative("min_isolate_hold_s", self.min_isolate_hold_s)
        if self.ewma_tau_s is not None:
            if not np.isfinite(self.ewma_tau_s) or self.ewma_tau_s <= 0:
                raise ValueError("ewma_tau_s must be finite and strictly positive")
        try:
            AccessState(self.initial_state)
        except ValueError as exc:
            raise ValueError(f"invalid initial_state {self.initial_state!r}") from exc


@dataclass(slots=True)
class _Runtime:
    lease_id: Hashable
    state: AccessState
    state_since_s: float
    last_time_s: float
    evidence: float | None = None
    last_evidence_time_s: float | None = None
    candidate: AccessState | None = None
    candidate_since_s: float | None = None
    candidate_reports: int = 0


@dataclass(frozen=True, slots=True)
class Decision:
    """One causal controller decision."""

    timestamp_s: float
    subject_id: Hashable
    lease_id: Hashable
    raw_score: float
    evidence_score: float
    previous_state: AccessState
    state: AccessState
    transitioned: bool
    lifecycle_start: bool


@dataclass(frozen=True, slots=True)
class ControllerTrace:
    """Array-first result returned by :meth:`TemporalAccessController.run`."""

    timestamp_s: np.ndarray
    subject_id: np.ndarray
    lease_id: np.ndarray
    raw_score: np.ndarray
    evidence_score: np.ndarray
    state: np.ndarray
    previous_state: np.ndarray
    transitioned: np.ndarray
    lifecycle_start: np.ndarray

    def __len__(self) -> int:
        return len(self.state)

    @property
    def state_name(self) -> np.ndarray:
        names = np.asarray([item.name for item in AccessState], dtype=object)
        return names[self.state]

    def to_pandas(self):
        """Return a pandas DataFrame without making pandas a core dependency."""

        import pandas as pd

        return pd.DataFrame(
            {
                "timestamp_s": self.timestamp_s,
                "subject_id": self.subject_id,
                "lease_id": self.lease_id,
                "raw_score": self.raw_score,
                "evidence_score": self.evidence_score,
                "state": self.state,
                "state_name": self.state_name,
                "previous_state": self.previous_state,
                "transitioned": self.transitioned,
                "lifecycle_start": self.lifecycle_start,
            }
        )

    def to_polars(self):
        """Return the trace in the project's native Polars table format."""

        import polars as pl

        return pl.DataFrame(
            {
                "timestamp_s": self.timestamp_s,
                "subject_id": self.subject_id,
                "lease_id": self.lease_id,
                "raw_score": self.raw_score,
                "evidence_score": self.evidence_score,
                "state": self.state,
                "state_name": self.state_name,
                "previous_state": self.previous_state,
                "transitioned": self.transitioned,
                "lifecycle_start": self.lifecycle_start,
            },
            strict=False,
        )


class TemporalAccessController:
    """Causal EWMA/hysteresis/persistence controller.

    State is kept separately for every ``subject_id``.  A subject's runtime is
    replaced only when its explicit ``lease_id`` changes.  In particular, long
    time gaps, score changes, and ground-truth labels (which are not accepted by
    this API) cannot create an implicit lifecycle reset.
    """

    def __init__(self, config: ControllerConfig):
        self.config = config
        self._runtime: dict[Hashable, _Runtime] = {}

    def clear(self) -> None:
        """Explicitly clear runtime before a separate replay."""

        self._runtime.clear()

    def _new_runtime(self, lease_id: Hashable, timestamp_s: float) -> _Runtime:
        return _Runtime(
            lease_id=lease_id,
            state=AccessState(self.config.initial_state),
            state_since_s=timestamp_s,
            last_time_s=timestamp_s,
        )

    def _target(self, state: AccessState, evidence: float) -> AccessState:
        config = self.config
        if state == AccessState.ALLOW:
            if evidence >= config.isolate_enter:
                return AccessState.ISOLATE
            if evidence >= config.restrict_enter:
                return AccessState.RESTRICT
            return state
        if state == AccessState.RESTRICT:
            if evidence >= config.isolate_enter:
                return AccessState.ISOLATE
            if evidence < config.restrict_exit:
                return AccessState.ALLOW
            return state
        # Recovery is deliberately staged: ISOLATE -> RESTRICT -> ALLOW.
        if evidence < config.isolate_exit:
            return AccessState.RESTRICT
        return state

    def _filtered_score(
        self, runtime: _Runtime, raw_score: float, timestamp_s: float
    ) -> float:
        if not np.isfinite(raw_score):
            return float("nan")
        tau = self.config.ewma_tau_s
        if tau is None or runtime.evidence is None:
            runtime.evidence = raw_score
        else:
            elapsed = timestamp_s - float(runtime.last_evidence_time_s)
            alpha = -np.expm1(-elapsed / tau)  # stable 1-exp(-dt/tau)
            runtime.evidence += alpha * (raw_score - runtime.evidence)
        runtime.last_evidence_time_s = timestamp_s
        return float(runtime.evidence)

    def _hold_satisfied(
        self, runtime: _Runtime, target: AccessState, timestamp_s: float
    ) -> bool:
        if target > runtime.state:
            return True
        held_for = timestamp_s - runtime.state_since_s
        if runtime.state == AccessState.ISOLATE:
            return held_for >= self.config.min_isolate_hold_s
        if runtime.state == AccessState.RESTRICT:
            return held_for >= self.config.min_restrict_hold_s
        return True

    def update(
        self,
        score: float,
        timestamp: Any,
        lease_id: Hashable,
        subject_id: Hashable = "__single_subject__",
    ) -> Decision:
        """Consume one report and return its action.

        Reports for each subject/lease must be nondecreasing in time.  Missing
        scores hold the current state and break consecutive persistence evidence.
        """

        subject_id = _valid_identifier("subject_id", subject_id)
        lease_id = _valid_identifier("lease_id", lease_id)
        timestamp_s = timestamp_seconds(timestamp)
        try:
            raw_score = float(score)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"score must be numeric; got {score!r}") from exc

        runtime = self._runtime.get(subject_id)
        lifecycle_start = runtime is None or runtime.lease_id != lease_id
        if lifecycle_start:
            runtime = self._new_runtime(lease_id, timestamp_s)
            self._runtime[subject_id] = runtime
        elif timestamp_s < runtime.last_time_s:
            raise ValueError(
                "timestamps must be nondecreasing within each subject/lease"
            )

        previous = runtime.state
        evidence = self._filtered_score(runtime, raw_score, timestamp_s)
        if not np.isfinite(evidence):
            runtime.candidate = None
            runtime.candidate_since_s = None
            runtime.candidate_reports = 0
        else:
            target = self._target(runtime.state, evidence)
            if target == runtime.state:
                runtime.candidate = None
                runtime.candidate_since_s = None
                runtime.candidate_reports = 0
            else:
                if runtime.candidate != target:
                    runtime.candidate = target
                    runtime.candidate_since_s = timestamp_s
                    runtime.candidate_reports = 1
                else:
                    runtime.candidate_reports += 1

                upward = target > runtime.state
                required_reports = (
                    self.config.entry_reports
                    if upward
                    else self.config.recovery_reports
                )
                required_duration = (
                    self.config.entry_duration_s
                    if upward
                    else self.config.recovery_duration_s
                )
                candidate_elapsed = timestamp_s - float(runtime.candidate_since_s)
                if (
                    runtime.candidate_reports >= required_reports
                    and candidate_elapsed >= required_duration
                    and self._hold_satisfied(runtime, target, timestamp_s)
                ):
                    runtime.state = target
                    runtime.state_since_s = timestamp_s
                    runtime.candidate = None
                    runtime.candidate_since_s = None
                    runtime.candidate_reports = 0

        runtime.last_time_s = timestamp_s
        return Decision(
            timestamp_s=timestamp_s,
            subject_id=subject_id,
            lease_id=lease_id,
            raw_score=raw_score,
            evidence_score=evidence,
            previous_state=previous,
            state=runtime.state,
            transitioned=runtime.state != previous,
            lifecycle_start=lifecycle_start,
        )

    def run(
        self,
        scores: Sequence[float] | np.ndarray,
        timestamps: Sequence[Any] | np.ndarray,
        lease_ids: Sequence[Hashable] | np.ndarray,
        subject_ids: Sequence[Hashable] | np.ndarray | None = None,
        *,
        clear: bool = True,
    ) -> ControllerTrace:
        """Replay reports in input order and return an array-first trace.

        Interleaved subjects are supported; causality is checked independently
        for each subject and lease.  Input order is never sorted implicitly.
        """

        score_array = np.asarray(scores, dtype=float)
        time_array = np.asarray(timestamps)
        lease_array = np.asarray(lease_ids, dtype=object)
        if score_array.ndim != 1 or time_array.ndim != 1 or lease_array.ndim != 1:
            raise ValueError("scores, timestamps, and lease_ids must be 1-D")
        length = len(score_array)
        if len(time_array) != length or len(lease_array) != length:
            raise ValueError("all controller inputs must have equal length")
        if subject_ids is None:
            subject_array = np.full(length, "__single_subject__", dtype=object)
        else:
            subject_array = np.asarray(subject_ids, dtype=object)
            if subject_array.ndim != 1 or len(subject_array) != length:
                raise ValueError("subject_ids must be 1-D and match scores")

        if clear:
            self.clear()
        converted_times = timestamps_seconds(time_array)
        decisions = [
            self.update(score, time, lease, subject)
            for score, time, lease, subject in zip(
                score_array,
                converted_times,
                lease_array,
                subject_array,
                strict=True,
            )
        ]
        return ControllerTrace(
            timestamp_s=converted_times,
            subject_id=subject_array.copy(),
            lease_id=lease_array.copy(),
            raw_score=score_array.copy(),
            evidence_score=np.fromiter(
                (item.evidence_score for item in decisions), dtype=float, count=length
            ),
            state=np.fromiter(
                (int(item.state) for item in decisions), dtype=np.int8, count=length
            ),
            previous_state=np.fromiter(
                (int(item.previous_state) for item in decisions),
                dtype=np.int8,
                count=length,
            ),
            transitioned=np.fromiter(
                (item.transitioned for item in decisions), dtype=bool, count=length
            ),
            lifecycle_start=np.fromiter(
                (item.lifecycle_start for item in decisions), dtype=bool, count=length
            ),
        )


class StatelessController(TemporalAccessController):
    """Score-only baseline with no smoothing, persistence, or hysteresis."""

    def __init__(self, restrict_threshold: float, isolate_threshold: float):
        super().__init__(
            ControllerConfig(
                restrict_enter=restrict_threshold,
                restrict_exit=restrict_threshold,
                isolate_enter=isolate_threshold,
                isolate_exit=isolate_threshold,
            )
        )

    def _target(self, state: AccessState, evidence: float) -> AccessState:
        del state
        if evidence >= self.config.isolate_enter:
            return AccessState.ISOLATE
        if evidence >= self.config.restrict_enter:
            return AccessState.RESTRICT
        return AccessState.ALLOW


class NReportController(TemporalAccessController):
    """Consecutive-report persistence baseline with symmetric thresholds."""

    def __init__(
        self,
        restrict_threshold: float,
        isolate_threshold: float,
        *,
        entry_reports: int,
        recovery_reports: int | None = None,
        min_restrict_hold_s: float = 0.0,
        min_isolate_hold_s: float = 0.0,
    ):
        super().__init__(
            ControllerConfig(
                restrict_enter=restrict_threshold,
                restrict_exit=restrict_threshold,
                isolate_enter=isolate_threshold,
                isolate_exit=isolate_threshold,
                entry_reports=entry_reports,
                recovery_reports=(
                    entry_reports if recovery_reports is None else recovery_reports
                ),
                min_restrict_hold_s=min_restrict_hold_s,
                min_isolate_hold_s=min_isolate_hold_s,
            )
        )


class EWMAController(TemporalAccessController):
    """Time-aware EWMA baseline followed by symmetric score thresholds."""

    def __init__(
        self,
        restrict_threshold: float,
        isolate_threshold: float,
        *,
        tau_s: float,
    ):
        super().__init__(
            ControllerConfig(
                restrict_enter=restrict_threshold,
                restrict_exit=restrict_threshold,
                isolate_enter=isolate_threshold,
                isolate_exit=isolate_threshold,
                ewma_tau_s=tau_s,
            )
        )

    def _target(self, state: AccessState, evidence: float) -> AccessState:
        # EWMA is the only memory in this baseline; action thresholds themselves
        # have neither hysteresis nor staged recovery.
        del state
        if evidence >= self.config.isolate_enter:
            return AccessState.ISOLATE
        if evidence >= self.config.restrict_enter:
            return AccessState.RESTRICT
        return AccessState.ALLOW


class SymmetricHysteresisController(TemporalAccessController):
    """Symmetric Schmitt-trigger baseline for both action boundaries."""

    def __init__(
        self,
        restrict_center: float,
        isolate_center: float,
        *,
        half_width: float,
    ):
        _finite_nonnegative("half_width", half_width)
        super().__init__(
            ControllerConfig(
                restrict_enter=restrict_center + half_width,
                restrict_exit=restrict_center - half_width,
                isolate_enter=isolate_center + half_width,
                isolate_exit=isolate_center - half_width,
            )
        )


# Explicit alias used in paper/config terminology.
NReportPersistenceController = NReportController


def run_controller(
    controller: TemporalAccessController,
    scores: Sequence[float] | np.ndarray,
    timestamps: Sequence[Any] | np.ndarray,
    lease_ids: Sequence[Hashable] | np.ndarray,
    subject_ids: Sequence[Hashable] | np.ndarray | None = None,
    *,
    clear: bool = True,
) -> ControllerTrace:
    """Functional wrapper around :meth:`TemporalAccessController.run`."""

    return controller.run(
        scores,
        timestamps,
        lease_ids,
        subject_ids,
        clear=clear,
    )


__all__ = [
    "ALLOW",
    "RESTRICT",
    "ISOLATE",
    "AccessState",
    "ControllerConfig",
    "ControllerTrace",
    "Decision",
    "TemporalAccessController",
    "StatelessController",
    "NReportController",
    "NReportPersistenceController",
    "EWMAController",
    "SymmetricHysteresisController",
    "run_controller",
    "timestamp_seconds",
    "timestamps_seconds",
]
