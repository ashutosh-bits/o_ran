"""Locked, causal controller evaluation helpers.

This module contains no policy selection, artifact discovery, or reporting CLI.
It evaluates only explicitly supplied epoch frames and canonical, locked
``CandidateSpec`` objects.  Controller inputs are restricted to score, decision
time, subject, and lease.  Labels are parsed and joined only after replay.

An action decided at epoch ``t`` becomes effective in epoch ``t+1``.  Every new
RNTI lease therefore starts in ``ALLOW``.  Mixed attack epochs use multi-hot
class masks: they contribute once to the all-attack stream and once to each
present attack-type stream; class-specific totals must not be summed together.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from .controller import AccessState
from .data import ATTACK_LABELS, LABEL_ONTOLOGY, TRACE_BLOCK_COLUMN, DataValidationError
from .policy_search import CandidateSpec, effective_states_one_epoch_later, make_controller


EVALUATION_ROW_COLUMN = "_evaluation_row"
ANY_ATTACK_TYPE = "__any_attack__"
MIXED_ATTACK_TYPE = "__mixed_attack__"
ATTACK_MASK_PREFIX = "attack__"
LOCK_VERSION = 1


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("candidate parameters cannot contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"candidate parameter is not JSON-serializable: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class LockedCandidateSpecs:
    """Immutable canonical controller specifications and provenance partition."""

    canonical_json: str
    sha256: str
    selection_partition: str

    def __post_init__(self) -> None:
        observed = hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()
        if observed != self.sha256:
            raise ValueError("candidate lock digest does not match canonical JSON")
        payload = json.loads(self.canonical_json)
        if payload.get("lock_version") != LOCK_VERSION:
            raise ValueError("unsupported candidate lock version")
        if payload.get("selection_partition") != self.selection_partition:
            raise ValueError("selection partition differs from canonical lock")

    @property
    def specs(self) -> tuple[CandidateSpec, ...]:
        payload = json.loads(self.canonical_json)
        return tuple(
            CandidateSpec(
                candidate=item["candidate"],
                family=item["family"],
                controller=item["controller"],
                parameters=dict(item["parameters"]),
            )
            for item in payload["candidate_specs"]
        )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "lock_version": LOCK_VERSION,
            "selection_partition": self.selection_partition,
            "sha256": self.sha256,
            "canonical_json": self.canonical_json,
        }


def lock_candidate_specs(
    specs: Sequence[CandidateSpec | Mapping[str, Any]],
    *,
    selection_partition: str = "controller_tune",
) -> LockedCandidateSpecs:
    """Canonicalize specs selected outside the held-out evaluation partition."""

    normalized_partition = selection_partition.strip()
    if not normalized_partition:
        raise ValueError("selection_partition is required")
    casefolded = normalized_partition.casefold()
    if any(token in casefolded for token in ("test", "holdout", "held_out", "evaluation")):
        raise ValueError("candidate specifications cannot be selected on held-out data")
    if not specs:
        raise ValueError("at least one candidate specification is required")

    records: list[dict[str, Any]] = []
    for value in specs:
        if isinstance(value, CandidateSpec):
            spec = value
        elif isinstance(value, Mapping):
            spec = CandidateSpec(
                candidate=str(value["candidate"]),
                family=str(value["family"]),
                controller=str(value["controller"]),
                parameters=dict(value["parameters"]),
            )
        else:
            raise TypeError("specs must contain CandidateSpec objects or mappings")
        # Constructor validation happens here, before anything is locked.
        make_controller(spec)
        records.append(
            {
                "candidate": str(spec.candidate),
                "family": str(spec.family),
                "controller": str(spec.controller),
                "parameters": _json_safe(spec.parameters),
            }
        )
    names = [record["candidate"] for record in records]
    if len(set(names)) != len(names):
        raise ValueError("locked candidate identifiers must be unique")
    records.sort(key=lambda item: item["candidate"])
    payload = {
        "lock_version": LOCK_VERSION,
        "selection_partition": normalized_partition,
        "candidate_specs": records,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return LockedCandidateSpecs(
        canonical_json=canonical,
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        selection_partition=normalized_partition,
    )


def attack_mask_column(attack_label: str) -> str:
    if attack_label not in ATTACK_LABELS:
        raise ValueError(f"unknown attack label: {attack_label}")
    slug = attack_label.casefold().replace("-", "_").replace(" ", "_")
    return f"{ATTACK_MASK_PREFIX}{slug}"


def _with_evaluation_row(frame: pl.DataFrame) -> pl.DataFrame:
    if EVALUATION_ROW_COLUMN not in frame.columns:
        return frame.with_row_index(EVALUATION_ROW_COLUMN)
    if frame[EVALUATION_ROW_COLUMN].has_nulls():
        raise DataValidationError("evaluation row identifier cannot be null")
    if frame[EVALUATION_ROW_COLUMN].n_unique() != frame.height:
        raise DataValidationError("evaluation row identifier must be unique")
    return frame.clone()


def _validate_replay_frame(
    frame: pl.DataFrame,
    *,
    score_column: str,
    time_column: str,
    subject_column: str,
    lease_column: str,
    duration_column: str,
    block_column: str,
) -> None:
    required = {
        score_column,
        time_column,
        subject_column,
        lease_column,
        duration_column,
        block_column,
        EVALUATION_ROW_COLUMN,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"evaluation frame missing columns: {missing}")
    for column in (score_column, time_column, subject_column, lease_column, block_column):
        if frame[column].has_nulls():
            raise DataValidationError(f"evaluation column cannot contain nulls: {column}")
    scores = np.asarray(frame[score_column].to_numpy(), dtype=float)
    times = np.asarray(frame[time_column].to_numpy(), dtype=float)
    durations = np.asarray(frame[duration_column].to_numpy(), dtype=float)
    if not np.isfinite(scores).all():
        raise DataValidationError("locked replay requires finite scores")
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise DataValidationError("epoch rows must be globally nondecreasing in time")
    if not np.isfinite(durations).all() or (durations <= 0).any():
        raise DataValidationError("epoch durations must be finite and positive")
    duplicates = (
        frame.group_by([subject_column, lease_column, time_column])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicates:
        raise DataValidationError("duplicate subject/lease/decision-time epochs")


def _previous_effective_states(
    effective: np.ndarray,
    subjects: np.ndarray,
    leases: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    previous = np.full(len(effective), int(AccessState.ALLOW), dtype=np.int8)
    transitioned = np.zeros(len(effective), dtype=bool)
    severity = np.zeros(len(effective), dtype=np.int8)
    prior: dict[Any, tuple[Any, int]] = {}
    for index, (state, subject, lease) in enumerate(
        zip(effective, subjects, leases, strict=True)
    ):
        last = prior.get(subject)
        if last is not None and last[0] == lease:
            previous[index] = int(last[1])
            transitioned[index] = int(state) != int(last[1])
            severity[index] = abs(int(state) - int(last[1]))
        prior[subject] = (lease, int(state))
    return previous, transitioned, severity


def replay_locked_candidates(
    epochs: pl.DataFrame,
    locked: LockedCandidateSpecs,
    *,
    score_column: str = "risk_score",
    time_column: str = "decision_time_s",
    subject_column: str = "mac_rnti",
    lease_column: str = "rnti_lease_id",
    duration_column: str = "epoch_seconds",
    block_column: str = TRACE_BLOCK_COLUMN,
) -> pl.DataFrame:
    """Replay only score/time/subject/lease and apply a strict one-epoch lag.

    The returned action table contains no label or target columns.  Its row ID is
    the sole key used to join evaluation targets later.
    """

    frame = _with_evaluation_row(epochs)
    _validate_replay_frame(
        frame,
        score_column=score_column,
        time_column=time_column,
        subject_column=subject_column,
        lease_column=lease_column,
        duration_column=duration_column,
        block_column=block_column,
    )
    # This explicit projection is the controller's complete information set.
    controller_input = frame.select(
        [
            EVALUATION_ROW_COLUMN,
            block_column,
            score_column,
            time_column,
            subject_column,
            lease_column,
            duration_column,
        ]
    )
    scores = np.asarray(controller_input[score_column].to_numpy(), dtype=float)
    times = np.asarray(controller_input[time_column].to_numpy(), dtype=float)
    subjects = np.asarray(controller_input[subject_column].to_numpy(), dtype=object)
    leases = np.asarray(controller_input[lease_column].to_numpy(), dtype=object)
    rows: list[pl.DataFrame] = []
    for spec in locked.specs:
        trace = make_controller(spec).run(
            scores,
            times,
            leases,
            subjects,
            clear=True,
        )
        effective = effective_states_one_epoch_later(trace.state, subjects, leases)
        previous, transitioned, severity = _previous_effective_states(
            effective, subjects, leases
        )
        rows.append(
            pl.DataFrame(
                {
                    EVALUATION_ROW_COLUMN: controller_input[EVALUATION_ROW_COLUMN],
                    block_column: controller_input[block_column],
                    "candidate": [spec.candidate] * frame.height,
                    "family": [spec.family] * frame.height,
                    "controller_kind": [spec.controller] * frame.height,
                    "candidate_lock_sha256": [locked.sha256] * frame.height,
                    time_column: times,
                    # Preserve native table dtypes; object arrays are used only
                    # for the controller's hashable identifier interface.
                    subject_column: controller_input[subject_column],
                    lease_column: controller_input[lease_column],
                    duration_column: controller_input[duration_column],
                    score_column: scores,
                    "evidence_score": trace.evidence_score,
                    "decision_state": trace.state,
                    "decision_transitioned": trace.transitioned,
                    "lifecycle_start": trace.lifecycle_start,
                    "effective_previous_state": previous,
                    "effective_state": effective,
                    "effective_transitioned": transitioned,
                    "effective_severity_delta": severity,
                },
                strict=False,
            )
        )
    result = pl.concat(rows, how="vertical")
    target_columns = {
        "label",
        "labels_in_epoch",
        "label_id",
        "is_attack",
        "is_attack_epoch",
        "mixed_attack_epoch",
    }
    leaked = target_columns & set(result.columns)
    if leaked:
        raise AssertionError(f"target columns leaked into controller replay: {sorted(leaked)}")
    return result


def derive_attack_annotations(
    epochs: pl.DataFrame,
    *,
    labels_column: str = "labels_in_epoch",
) -> pl.DataFrame:
    """Return deterministic multi-hot and explicit mixed-label annotations."""

    frame = _with_evaluation_row(epochs)
    if labels_column not in frame.columns:
        raise DataValidationError(f"missing epoch label-list column: {labels_column}")
    known_labels = set(LABEL_ONTOLOGY)
    records: list[dict[str, Any]] = []
    computed_attack: list[bool] = []
    for row_id, raw_labels in zip(
        frame[EVALUATION_ROW_COLUMN].to_list(),
        frame[labels_column].to_list(),
        strict=True,
    ):
        if raw_labels is None or isinstance(raw_labels, (str, bytes)):
            raise DataValidationError("labels_in_epoch must contain non-empty label lists")
        labels = list(raw_labels)
        if not labels or any(value is None for value in labels):
            raise DataValidationError("labels_in_epoch must contain non-empty known labels")
        normalized = tuple(
            label
            for label in LABEL_ONTOLOGY
            if label in {str(value) for value in labels}
        )
        unknown = sorted({str(value) for value in labels} - known_labels)
        if unknown:
            raise DataValidationError(f"unknown labels in epoch: {unknown}")
        attacks = tuple(label for label in ATTACK_LABELS if label in normalized)
        record: dict[str, Any] = {
            EVALUATION_ROW_COLUMN: row_id,
            "labels_normalized": list(normalized),
            "attack_types": list(attacks),
            "label_type_count": len(normalized),
            "attack_type_count": len(attacks),
            "is_attack_epoch": bool(attacks),
            "mixed_label_epoch": len(normalized) > 1,
            "mixed_attack_epoch": len(attacks) > 1,
            "exclusive_attack_type": (
                None
                if not attacks
                else attacks[0]
                if len(attacks) == 1
                else MIXED_ATTACK_TYPE
            ),
            "attack_type_key": "|".join(attacks) if attacks else None,
        }
        for label in ATTACK_LABELS:
            record[attack_mask_column(label)] = label in attacks
        records.append(record)
        computed_attack.append(bool(attacks))

    if "is_attack_epoch" in frame.columns:
        observed = frame["is_attack_epoch"].to_list()
        if any(value is None for value in observed) or [bool(value) for value in observed] != computed_attack:
            raise DataValidationError(
                "existing is_attack_epoch disagrees with labels_in_epoch"
            )
    return pl.DataFrame(records, strict=False)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def aggregate_action_metrics(
    evaluated: pl.DataFrame,
    *,
    containment_state: AccessState = AccessState.RESTRICT,
    duration_column: str = "epoch_seconds",
) -> pl.DataFrame:
    """Aggregate action, friction, exposure, and churn by locked candidate."""

    required = {
        "candidate",
        "family",
        duration_column,
        "effective_state",
        "effective_previous_state",
        "effective_transitioned",
        "effective_severity_delta",
        "is_attack_epoch",
    }
    missing = sorted(required - set(evaluated.columns))
    if missing:
        raise DataValidationError(f"action aggregation missing columns: {missing}")
    threshold = int(AccessState(containment_state))
    records: list[dict[str, Any]] = []
    for group in evaluated.partition_by("candidate", maintain_order=True):
        duration = np.asarray(group[duration_column].to_numpy(), dtype=float)
        attack = np.asarray(group["is_attack_epoch"].to_numpy(), dtype=bool)
        state = np.asarray(group["effective_state"].to_numpy(), dtype=np.int8)
        previous = np.asarray(group["effective_previous_state"].to_numpy(), dtype=np.int8)
        contained = state >= threshold
        starts = (previous < threshold) & contained
        benign = ~attack
        observed_time = float(duration.sum())
        benign_time = float(duration[benign].sum())
        attack_time = float(duration[attack].sum())
        benign_contained = float(duration[benign & contained].sum())
        attack_contained = float(duration[attack & contained].sum())
        isolated = state == int(AccessState.ISOLATE)
        records.append(
            {
                "candidate": group["candidate"][0],
                "family": group["family"][0],
                "epochs": group.height,
                "observed_time_s": observed_time,
                "benign_time_s": benign_time,
                "attack_time_s": attack_time,
                "benign_contained_time_s": benign_contained,
                "benign_isolated_time_s": float(duration[benign & isolated].sum()),
                "attack_contained_time_s": attack_contained,
                "attack_exposed_time_s": float(duration[attack & ~contained].sum()),
                "benign_friction": _ratio(benign_contained, benign_time),
                "attack_time_containment": _ratio(attack_contained, attack_time),
                "effective_transitions": int(group["effective_transitioned"].sum()),
                "severity_transition_total": int(group["effective_severity_delta"].sum()),
                "containment_starts": int(starts.sum()),
                "benign_containment_starts": int((starts & benign).sum()),
                "transitions_per_lease_hour": _ratio(
                    3_600.0 * int(group["effective_transitioned"].sum()),
                    observed_time,
                ),
            }
        )
    return pl.DataFrame(records, strict=False).sort("candidate")


def _attack_streams(evaluated: pl.DataFrame) -> tuple[tuple[str, str], ...]:
    streams: list[tuple[str, str]] = [(ANY_ATTACK_TYPE, "is_attack_epoch")]
    for label in ATTACK_LABELS:
        column = attack_mask_column(label)
        if column not in evaluated.columns:
            raise DataValidationError(f"missing per-attack mask: {column}")
        streams.append((label, column))
    streams.append((MIXED_ATTACK_TYPE, "mixed_attack_epoch"))
    return tuple(streams)


def attack_episode_table(
    evaluated: pl.DataFrame,
    *,
    merge_gap_s: float = 0.0,
    delay_cap_s: float | None = None,
    containment_state: AccessState = AccessState.RESTRICT,
    time_column: str = "decision_time_s",
    duration_column: str = "epoch_seconds",
    subject_column: str = "mac_rnti",
    lease_column: str = "rnti_lease_id",
    block_column: str = TRACE_BLOCK_COLUMN,
) -> pl.DataFrame:
    """Create candidate-specific, per-attack episodes with onset strata.

    Type episodes are independent multi-hot streams.  A mixed epoch can therefore
    belong to two type episodes, the mixed stream, and the all-attack stream.
    ``pre_contained_at_onset`` requires containment in the immediately preceding
    effective epoch; a transition effective exactly at onset is a separate stratum.
    """

    if not math.isfinite(merge_gap_s) or merge_gap_s < 0:
        raise ValueError("merge_gap_s must be finite and non-negative")
    if delay_cap_s is not None and (not math.isfinite(delay_cap_s) or delay_cap_s <= 0):
        raise ValueError("delay_cap_s must be finite and positive")
    required = {
        "candidate",
        "family",
        time_column,
        duration_column,
        subject_column,
        lease_column,
        block_column,
        "effective_state",
        "effective_previous_state",
        "mixed_attack_epoch",
    }
    missing = sorted(required - set(evaluated.columns))
    if missing:
        raise DataValidationError(f"episode evaluation missing columns: {missing}")
    threshold = int(AccessState(containment_state))
    records: list[dict[str, Any]] = []

    def finalize(active: dict[str, Any]) -> None:
        span = float(active["last_attack_end_s"] - active["start_s"])
        first = active["first_containment_start_s"]
        covered = first is not None
        delay = float(first - active["start_s"]) if covered else float("nan")
        capped = delay if covered else span
        if delay_cap_s is not None:
            capped = min(capped, float(delay_cap_s))
        records.append(
            {
                "candidate": active["candidate"],
                "family": active["family"],
                "episode_key": active["episode_key"],
                "subject_id": active["subject_id"],
                "lease_id": active["lease_id"],
                "attack_type": active["attack_type"],
                "episode_sequence": active["episode_sequence"],
                "onset_block_id": active["onset_block_id"],
                "start_s": active["start_s"],
                "end_s": active["last_attack_end_s"],
                "span_s": span,
                "attack_time_s": active["attack_time_s"],
                "contained_time_s": active["contained_time_s"],
                "exposed_time_s": active["exposed_time_s"],
                "attack_epochs": active["attack_epochs"],
                "blocks_touched": len(active["blocks_touched"]),
                "mixed_attack_time_s": active["mixed_attack_time_s"],
                "contained_at_onset": active["contained_at_onset"],
                "pre_contained_at_onset": active["pre_contained_at_onset"],
                "onset_transition_to_containment": active[
                    "onset_transition_to_containment"
                ],
                "onset_stratum": active["onset_stratum"],
                "mixed_attack_at_onset": active["mixed_attack_at_onset"],
                "covered": covered,
                "reactively_covered": covered
                and not active["pre_contained_at_onset"],
                "first_containment_start_s": first if covered else float("nan"),
                "delay_s": delay,
                "censored": not covered,
                "capped_delay_s": float(capped),
                "exposure_before_containment_s": active[
                    "exposure_before_containment_s"
                ],
            }
        )

    group_columns = ["candidate", subject_column, lease_column]
    for group in evaluated.partition_by(group_columns, maintain_order=False):
        ordered = group.sort(time_column, maintain_order=True)
        times = np.asarray(ordered[time_column].to_numpy(), dtype=float)
        durations = np.asarray(ordered[duration_column].to_numpy(), dtype=float)
        starts = times - durations
        if len(starts) > 1 and np.any(starts[1:] < times[:-1] - 1e-9):
            raise DataValidationError("epochs overlap within a subject/lease")
        candidate = str(ordered["candidate"][0])
        family = str(ordered["family"][0])
        subject = ordered[subject_column][0]
        lease = ordered[lease_column][0]
        states = np.asarray(ordered["effective_state"].to_numpy(), dtype=np.int8)
        previous_states = np.asarray(
            ordered["effective_previous_state"].to_numpy(), dtype=np.int8
        )
        mixed = np.asarray(ordered["mixed_attack_epoch"].to_numpy(), dtype=bool)
        blocks = ordered[block_column].to_list()

        for attack_type, mask_column in _attack_streams(ordered):
            mask = np.asarray(ordered[mask_column].to_numpy(), dtype=bool)
            active: dict[str, Any] | None = None
            sequence = 0
            # Iterate over every observed epoch.  A genuinely observed false
            # label terminates the episode immediately; ``merge_gap_s`` applies
            # only when telemetry is absent between two positive observations.
            # Iterating only over true indices would incorrectly merge an
            # attack across intervening benign or different-class observations.
            for index in range(len(mask)):
                if not mask[index]:
                    if active is not None:
                        finalize(active)
                        active = None
                    continue
                start = float(starts[index])
                end = float(times[index])
                if active is None or start - active["last_attack_end_s"] > merge_gap_s + 1e-12:
                    if active is not None:
                        finalize(active)
                    contained_at_onset = bool(states[index] >= threshold)
                    pre_contained = bool(previous_states[index] >= threshold)
                    onset_transition = bool(
                        contained_at_onset and previous_states[index] < threshold
                    )
                    if pre_contained:
                        onset_stratum = "pre_contained"
                    elif onset_transition:
                        onset_stratum = "contained_onset_transition"
                    else:
                        onset_stratum = "initially_exposed"
                    episode_key = json.dumps(
                        [subject, lease, attack_type, sequence],
                        separators=(",", ":"),
                        default=str,
                    )
                    active = {
                        "candidate": candidate,
                        "family": family,
                        "episode_key": episode_key,
                        "subject_id": subject,
                        "lease_id": lease,
                        "attack_type": attack_type,
                        "episode_sequence": sequence,
                        "onset_block_id": blocks[index],
                        "start_s": start,
                        "last_attack_end_s": end,
                        "attack_time_s": 0.0,
                        "contained_time_s": 0.0,
                        "exposed_time_s": 0.0,
                        "attack_epochs": 0,
                        "blocks_touched": set(),
                        "mixed_attack_time_s": 0.0,
                        "contained_at_onset": contained_at_onset,
                        "pre_contained_at_onset": pre_contained,
                        "onset_transition_to_containment": onset_transition,
                        "onset_stratum": onset_stratum,
                        "mixed_attack_at_onset": bool(mixed[index]),
                        "first_containment_start_s": None,
                        "exposure_before_containment_s": 0.0,
                    }
                    sequence += 1
                duration = float(durations[index])
                contained = bool(states[index] >= threshold)
                active["last_attack_end_s"] = end
                active["attack_time_s"] += duration
                active["attack_epochs"] += 1
                active["blocks_touched"].add(blocks[index])
                if mixed[index]:
                    active["mixed_attack_time_s"] += duration
                if contained:
                    active["contained_time_s"] += duration
                    if active["first_containment_start_s"] is None:
                        active["first_containment_start_s"] = start
                else:
                    active["exposed_time_s"] += duration
                    if active["first_containment_start_s"] is None:
                        active["exposure_before_containment_s"] += duration
            if active is not None:
                finalize(active)

    if not records:
        return pl.DataFrame(
            schema={
                "candidate": pl.String,
                "family": pl.String,
                "episode_key": pl.String,
                "attack_type": pl.String,
            }
        )
    return pl.DataFrame(records, strict=False).sort(
        ["candidate", "attack_type", "start_s", "subject_id"]
    )


def summarize_episode_strata(episodes: pl.DataFrame) -> pl.DataFrame:
    """Aggregate episodes overall and by pre-contained/onset-transition stratum."""

    if episodes.is_empty():
        return pl.DataFrame(
            schema={
                "candidate": pl.String,
                "attack_type": pl.String,
                "onset_stratum": pl.String,
                "episode_count": pl.Int64,
            }
        )
    required = {
        "candidate",
        "family",
        "attack_type",
        "onset_stratum",
        "covered",
        "reactively_covered",
        "pre_contained_at_onset",
        "onset_transition_to_containment",
        "attack_time_s",
        "contained_time_s",
        "exposed_time_s",
        "capped_delay_s",
        "exposure_before_containment_s",
    }
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise DataValidationError(f"episode summary missing columns: {missing}")

    records: list[dict[str, Any]] = []
    for keys in (("candidate", "family", "attack_type"), (
        "candidate",
        "family",
        "attack_type",
        "onset_stratum",
    )):
        for group in episodes.partition_by(list(keys), maintain_order=False):
            attack_time = float(group["attack_time_s"].sum())
            contained_time = float(group["contained_time_s"].sum())
            stratum = group["onset_stratum"][0] if "onset_stratum" in keys else "__all__"
            records.append(
                {
                    "candidate": group["candidate"][0],
                    "family": group["family"][0],
                    "attack_type": group["attack_type"][0],
                    "onset_stratum": stratum,
                    "episode_count": group.height,
                    "covered_episode_count": int(group["covered"].sum()),
                    "reactively_covered_episode_count": int(
                        group["reactively_covered"].sum()
                    ),
                    "pre_contained_episode_count": int(
                        group["pre_contained_at_onset"].sum()
                    ),
                    "onset_transition_episode_count": int(
                        group["onset_transition_to_containment"].sum()
                    ),
                    "attack_time_s": attack_time,
                    "contained_time_s": contained_time,
                    "exposed_time_s": float(group["exposed_time_s"].sum()),
                    "capped_delay_sum_s": float(group["capped_delay_s"].sum()),
                    "exposure_before_containment_sum_s": float(
                        group["exposure_before_containment_s"].sum()
                    ),
                    "episode_coverage": float(group["covered"].mean()),
                    "time_containment": _ratio(contained_time, attack_time),
                    "mean_capped_delay_s": float(group["capped_delay_s"].mean()),
                }
            )
    return pl.DataFrame(records, strict=False).sort(
        ["candidate", "attack_type", "onset_stratum"]
    )


def action_block_contributions(
    evaluated: pl.DataFrame,
    *,
    containment_state: AccessState = AccessState.RESTRICT,
    block_column: str = TRACE_BLOCK_COLUMN,
    duration_column: str = "epoch_seconds",
) -> pl.DataFrame:
    """Return additive action numerators/denominators per candidate and block."""

    threshold = int(AccessState(containment_state))
    records: list[dict[str, Any]] = []
    for group in evaluated.partition_by(["candidate", block_column], maintain_order=False):
        duration = np.asarray(group[duration_column].to_numpy(), dtype=float)
        attack = np.asarray(group["is_attack_epoch"].to_numpy(), dtype=bool)
        state = np.asarray(group["effective_state"].to_numpy(), dtype=np.int8)
        previous = np.asarray(group["effective_previous_state"].to_numpy(), dtype=np.int8)
        contained = state >= threshold
        starts = (previous < threshold) & contained
        benign = ~attack
        records.append(
            {
                "candidate": group["candidate"][0],
                "family": group["family"][0],
                block_column: group[block_column][0],
                "epoch_count": group.height,
                "observed_time_s": float(duration.sum()),
                "benign_time_s": float(duration[benign].sum()),
                "attack_time_s": float(duration[attack].sum()),
                "benign_contained_time_s": float(duration[benign & contained].sum()),
                "benign_isolated_time_s": float(
                    duration[benign & (state == int(AccessState.ISOLATE))].sum()
                ),
                "attack_contained_time_s": float(duration[attack & contained].sum()),
                "attack_exposed_time_s": float(duration[attack & ~contained].sum()),
                "effective_transition_count": int(group["effective_transitioned"].sum()),
                "severity_transition_total": int(group["effective_severity_delta"].sum()),
                "containment_start_count": int(starts.sum()),
                "benign_containment_start_count": int((starts & benign).sum()),
            }
        )
    return pl.DataFrame(records, strict=False).sort(["candidate", block_column])


def attack_block_contributions(
    evaluated: pl.DataFrame,
    episodes: pl.DataFrame,
    *,
    containment_state: AccessState = AccessState.RESTRICT,
    block_column: str = TRACE_BLOCK_COLUMN,
    duration_column: str = "epoch_seconds",
) -> pl.DataFrame:
    """Return additive class-time and onset-assigned episode contributions.

    Episode counts and delay totals are assigned to the onset block.  Attack-time
    contributions remain in the block where time was observed.  Ratios and
    medians are intentionally absent so a paired bootstrap can resample blocks,
    sum contributions, and only then form estimands.
    """

    threshold = int(AccessState(containment_state))
    candidates = sorted(str(value) for value in evaluated["candidate"].unique())
    blocks = sorted(evaluated[block_column].unique().to_list())
    streams = _attack_streams(evaluated)
    records: dict[tuple[str, Any, str], dict[str, Any]] = {}
    family_by_candidate = {
        str(group["candidate"][0]): str(group["family"][0])
        for group in evaluated.partition_by("candidate", maintain_order=False)
    }
    for candidate in candidates:
        for block in blocks:
            for attack_type, _ in streams:
                records[(candidate, block, attack_type)] = {
                    "candidate": candidate,
                    "family": family_by_candidate[candidate],
                    block_column: block,
                    "attack_type": attack_type,
                    "attack_epoch_count": 0,
                    "attack_time_s": 0.0,
                    "contained_time_s": 0.0,
                    "exposed_time_s": 0.0,
                    "episode_count": 0,
                    "covered_episode_count": 0,
                    "reactively_covered_episode_count": 0,
                    "pre_contained_episode_count": 0,
                    "onset_transition_episode_count": 0,
                    "censored_episode_count": 0,
                    "capped_delay_sum_s": 0.0,
                    "exposure_before_containment_sum_s": 0.0,
                }

    for group in evaluated.partition_by(["candidate", block_column], maintain_order=False):
        candidate = str(group["candidate"][0])
        block = group[block_column][0]
        durations = np.asarray(group[duration_column].to_numpy(), dtype=float)
        states = np.asarray(group["effective_state"].to_numpy(), dtype=np.int8)
        contained = states >= threshold
        for attack_type, mask_column in streams:
            mask = np.asarray(group[mask_column].to_numpy(), dtype=bool)
            record = records[(candidate, block, attack_type)]
            record["attack_epoch_count"] = int(mask.sum())
            record["attack_time_s"] = float(durations[mask].sum())
            record["contained_time_s"] = float(durations[mask & contained].sum())
            record["exposed_time_s"] = float(durations[mask & ~contained].sum())

    if not episodes.is_empty():
        for episode in episodes.iter_rows(named=True):
            key = (
                str(episode["candidate"]),
                episode["onset_block_id"],
                str(episode["attack_type"]),
            )
            record = records[key]
            record["episode_count"] += 1
            record["covered_episode_count"] += int(bool(episode["covered"]))
            record["reactively_covered_episode_count"] += int(
                bool(episode["reactively_covered"])
            )
            record["pre_contained_episode_count"] += int(
                bool(episode["pre_contained_at_onset"])
            )
            record["onset_transition_episode_count"] += int(
                bool(episode["onset_transition_to_containment"])
            )
            record["censored_episode_count"] += int(bool(episode["censored"]))
            record["capped_delay_sum_s"] += float(episode["capped_delay_s"])
            record["exposure_before_containment_sum_s"] += float(
                episode["exposure_before_containment_s"]
            )
    return pl.DataFrame(list(records.values()), strict=False).sort(
        ["candidate", "attack_type", block_column]
    )


@dataclass(frozen=True)
class EvaluationBundle:
    candidate_lock_sha256: str
    action_trace: pl.DataFrame
    aggregate_metrics: pl.DataFrame
    episodes: pl.DataFrame
    episode_strata: pl.DataFrame
    action_block_contributions: pl.DataFrame
    attack_block_contributions: pl.DataFrame


def evaluate_locked_candidates(
    epochs: pl.DataFrame,
    locked: LockedCandidateSpecs,
    *,
    merge_gap_s: float = 0.0,
    delay_cap_s: float | None = None,
    containment_state: AccessState = AccessState.RESTRICT,
    score_column: str = "risk_score",
) -> EvaluationBundle:
    """Run the complete, non-selecting evaluation pipeline on an in-memory frame."""

    indexed = _with_evaluation_row(epochs)
    actions = replay_locked_candidates(indexed, locked, score_column=score_column)
    annotations = derive_attack_annotations(indexed)
    evaluated = actions.join(annotations, on=EVALUATION_ROW_COLUMN, how="left")
    if evaluated["is_attack_epoch"].has_nulls():
        raise AssertionError("target annotation did not cover every replay row")
    aggregate = aggregate_action_metrics(
        evaluated, containment_state=containment_state
    )
    episodes = attack_episode_table(
        evaluated,
        merge_gap_s=merge_gap_s,
        delay_cap_s=delay_cap_s,
        containment_state=containment_state,
    )
    episode_strata = summarize_episode_strata(episodes)
    action_contributions = action_block_contributions(
        evaluated, containment_state=containment_state
    )
    attack_contributions = attack_block_contributions(
        evaluated,
        episodes,
        containment_state=containment_state,
    )
    return EvaluationBundle(
        candidate_lock_sha256=locked.sha256,
        action_trace=evaluated,
        aggregate_metrics=aggregate,
        episodes=episodes,
        episode_strata=episode_strata,
        action_block_contributions=action_contributions,
        attack_block_contributions=attack_contributions,
    )


__all__ = [
    "ANY_ATTACK_TYPE",
    "MIXED_ATTACK_TYPE",
    "LockedCandidateSpecs",
    "EvaluationBundle",
    "lock_candidate_specs",
    "attack_mask_column",
    "replay_locked_candidates",
    "derive_attack_annotations",
    "aggregate_action_metrics",
    "attack_episode_table",
    "summarize_episode_strata",
    "action_block_contributions",
    "attack_block_contributions",
    "evaluate_locked_candidates",
]
