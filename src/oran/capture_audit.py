"""Label-blind robustness audit for observable O-RAN trace grouping.

The dataset has no trustworthy capture or experiment-run identifier.  This
module therefore compares several *candidate clustering units* without calling
any of them a true capture.  Every boundary is based on corrected time,
mobility metadata, or adjacent numeric-RNTI population overlap.  Traffic labels
and derived targets are never boundary inputs.

The primary output is one row per existing observable trace block.  Larger
candidate IDs can be joined to row- or epoch-level metrics for paired cluster
bootstrap sensitivity analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from .data import (
    DEFAULT_LEASE_GAP_SECONDS,
    DEFAULT_MAX_BLOCK_SECONDS,
    DEFAULT_TIMESTAMP_SCALE,
    DEFAULT_TRACE_GAP_SECONDS,
    EVENT_TIME_COLUMN,
    LEASE_ID_COLUMN,
    TRACE_BLOCK_COLUMN,
    DataValidationError,
    load_semicolon_csv,
    prepare_trace,
)


FORBIDDEN_BOUNDARY_COLUMNS: frozenset[str] = frozenset(
    {
        "label",
        "label_id",
        "is_attack",
        "labels_in_epoch",
        "is_attack_epoch",
        "target_observed",
        "risk",
        "risk_score",
        "prediction",
    }
)
APPROVED_BOUNDARY_COLUMNS: Mapping[str, frozenset[str]] = {
    "block identity": frozenset({TRACE_BLOCK_COLUMN}),
    "event time": frozenset({EVENT_TIME_COLUMN, "decision_time_s"}),
    "RNTI population": frozenset({"mac_rnti"}),
    "mobility context": frozenset({"mob_pattern"}),
    "lease identity": frozenset({LEASE_ID_COLUMN}),
}


@dataclass(frozen=True)
class GroupingConfig:
    """Predeclared, label-blind candidate grouping thresholds."""

    fixed_window_seconds: tuple[int, ...] = (3_600, 10_800, 21_600)
    gap_threshold_seconds: tuple[int, ...] = (300, 1_800, 3_600)
    mobility_tv_thresholds: tuple[float, ...] = (0.25, 0.50)
    rnti_jaccard_thresholds: tuple[float, ...] = (0.05, 0.25)
    campaign_gap_seconds: float = 1_800.0
    campaign_mobility_tv: float = 0.25
    campaign_min_rnti_jaccard: float = 0.05
    split_column: str | None = "split"

    def validate(self) -> None:
        for value in self.fixed_window_seconds:
            if value <= 0:
                raise ValueError("fixed windows must be positive")
        for value in self.gap_threshold_seconds:
            if value <= 0:
                raise ValueError("gap thresholds must be positive")
        for value in self.mobility_tv_thresholds:
            if not 0 <= value <= 1:
                raise ValueError("mobility-TV thresholds must be in [0, 1]")
        for value in self.rnti_jaccard_thresholds:
            if not 0 <= value <= 1:
                raise ValueError("RNTI-Jaccard thresholds must be in [0, 1]")
        if self.campaign_gap_seconds <= 0:
            raise ValueError("campaign gap must be positive")
        if not 0 <= self.campaign_mobility_tv <= 1:
            raise ValueError("campaign mobility-TV threshold must be in [0, 1]")
        if not 0 <= self.campaign_min_rnti_jaccard <= 1:
            raise ValueError("campaign RNTI-Jaccard threshold must be in [0, 1]")
        if self.split_column not in {None, "split"}:
            raise ValueError("only the frozen 'split' column may reset candidate groups")


@dataclass(frozen=True)
class CandidateSummary:
    candidate: str
    interpretation: str
    n_groups: int
    n_observable_blocks: int
    n_rows: int
    singleton_group_fraction: float
    blocks_per_group_q25: float
    blocks_per_group_median: float
    blocks_per_group_q75: float
    blocks_per_group_max: int
    group_span_hours_median: float
    group_span_hours_p90: float
    group_span_hours_max: float
    groups_by_split: Mapping[str, int]
    at_least_30_groups: bool
    minimum_groups_across_splits: int | None
    at_least_30_groups_in_every_split: bool


@dataclass(frozen=True)
class CaptureRobustnessAudit:
    """Compact report that contains no labels or row-level RNTI values."""

    audit_version: int
    epistemic_status: str
    label_blind: bool
    n_rows: int
    n_observable_blocks: int
    trace_start_s: float
    trace_end_s: float
    observable_block_span_seconds_median: float
    grouping_config: Mapping[str, Any]
    adjacent_gap_seconds_quantiles: Mapping[str, float]
    adjacent_rnti_jaccard_quantiles: Mapping[str, float]
    adjacent_mobility_tv_quantiles: Mapping[str, float]
    mobility_signature_changes: int
    zero_rnti_overlap_transitions: int
    candidate_summaries: tuple[CandidateSummary, ...]
    candidate_mapping_sha256: str
    cautions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_boundary_column(name: str, purpose: str) -> None:
    if name in FORBIDDEN_BOUNDARY_COLUMNS:
        raise DataValidationError(
            f"{purpose} cannot use forbidden target-derived column {name!r}"
        )
    approved = APPROVED_BOUNDARY_COLUMNS[purpose]
    if name not in approved:
        raise DataValidationError(
            f"{purpose} must use an approved observable column {sorted(approved)}, "
            f"not {name!r}"
        )


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _quantile_map(values: Sequence[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "q00": _linear_quantile(finite, 0.00),
        "q25": _linear_quantile(finite, 0.25),
        "q50": _linear_quantile(finite, 0.50),
        "q75": _linear_quantile(finite, 0.75),
        "q90": _linear_quantile(finite, 0.90),
        "q100": _linear_quantile(finite, 1.00),
    }


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _total_variation(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def summarize_observable_blocks(
    frame: pl.DataFrame,
    *,
    block_column: str = TRACE_BLOCK_COLUMN,
    event_time_column: str = EVENT_TIME_COLUMN,
    rnti_column: str = "mac_rnti",
    mobility_column: str = "mob_pattern",
    lease_column: str = LEASE_ID_COLUMN,
    split_column: str | None = "split",
) -> pl.DataFrame:
    """Build a label-free table of adjacent observable-block characteristics.

    ``mob_pattern`` is retained only as a metadata sensitivity signal.  It is not
    assumed to be deployable, causal, or a true campaign identifier.
    """

    for name, purpose in (
        (block_column, "block identity"),
        (event_time_column, "event time"),
        (rnti_column, "RNTI population"),
        (mobility_column, "mobility context"),
        (lease_column, "lease identity"),
    ):
        _check_boundary_column(name, purpose)
    required = {block_column, event_time_column, rnti_column, mobility_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"capture audit missing columns: {missing}")
    if frame.is_empty():
        raise DataValidationError("capture audit requires at least one row")

    aggregation: list[pl.Expr] = [
        pl.col(event_time_column).min().alias("start_s"),
        pl.col(event_time_column).max().alias("end_s"),
        pl.len().alias("n_rows"),
        pl.col(rnti_column).drop_nulls().unique().sort().alias("_rnti_values"),
    ]
    if lease_column in frame.columns:
        aggregation.append(
            pl.col(lease_column).drop_nulls().n_unique().alias("n_rnti_leases")
        )
    if split_column is not None and split_column in frame.columns:
        split_check = frame.group_by(block_column).agg(
            pl.col(split_column).n_unique().alias("_n_splits")
        )
        if split_check.select((pl.col("_n_splits") > 1).any()).item():
            raise DataValidationError("an observable block spans multiple splits")
        aggregation.append(pl.col(split_column).first().alias(split_column))

    base = (
        frame.select(
            [
                block_column,
                event_time_column,
                rnti_column,
                mobility_column,
                *([lease_column] if lease_column in frame.columns else []),
                *(
                    [split_column]
                    if split_column is not None and split_column in frame.columns
                    else []
                ),
            ]
        )
        .group_by(block_column)
        .agg(aggregation)
        .sort(["start_s", block_column])
    )
    mobility_counts = (
        frame.select([block_column, mobility_column])
        .drop_nulls(mobility_column)
        .group_by([block_column, mobility_column])
        .len()
        .sort([block_column, mobility_column])
    )
    profiles: dict[int, dict[str, float]] = {}
    raw_counts: dict[int, dict[str, int]] = {}
    for row in mobility_counts.iter_rows(named=True):
        block_id = int(row[block_column])
        raw_counts.setdefault(block_id, {})[str(row[mobility_column])] = int(row["len"])
    for block_id, counts in raw_counts.items():
        total = sum(counts.values())
        profiles[block_id] = {
            key: value / total for key, value in sorted(counts.items())
        }

    output_rows: list[dict[str, Any]] = []
    previous_end: float | None = None
    previous_rntis: set[int] | None = None
    previous_profile: Mapping[str, float] | None = None
    previous_split: str | None = None
    for raw in base.iter_rows(named=True):
        block_id = int(raw[block_column])
        start = float(raw["start_s"])
        end = float(raw["end_s"])
        if end < start:
            raise DataValidationError(f"negative block span for block {block_id}")
        if previous_end is not None and start < previous_end:
            raise DataValidationError("observable block time intervals overlap")
        rntis = {int(value) for value in (raw["_rnti_values"] or [])}
        profile = profiles.get(block_id, {})
        split = (
            str(raw[split_column])
            if split_column is not None
            and split_column in raw
            and raw[split_column] is not None
            else None
        )
        is_first = previous_end is None
        split_changed = (
            not is_first
            and split_column is not None
            and split is not None
            and previous_split is not None
            and split != previous_split
        )
        dominant = (
            min(profile, key=lambda key: (-profile[key], key)) if profile else None
        )
        signature = json.dumps(profile, sort_keys=True, separators=(",", ":"))
        output: dict[str, Any] = {
            block_column: block_id,
            "start_s": start,
            "end_s": end,
            "block_span_s": end - start,
            "n_rows": int(raw["n_rows"]),
            "n_numeric_rntis": len(rntis),
            "n_rnti_leases": int(raw.get("n_rnti_leases", 0)),
            "mobility_profile": signature,
            "dominant_mobility": dominant,
            "utc_day_index": int(math.floor(start / 86_400.0)),
            "gap_from_previous_s": None if is_first else start - float(previous_end),
            "rnti_jaccard_previous": (
                None if previous_rntis is None else _jaccard(previous_rntis, rntis)
            ),
            "mobility_tv_previous": (
                None
                if previous_profile is None
                else _total_variation(previous_profile, profile)
            ),
            "dominant_mobility_changed": (
                None
                if previous_profile is None
                else dominant
                != (
                    min(
                        previous_profile,
                        key=lambda key: (-previous_profile[key], key),
                    )
                    if previous_profile
                    else None
                )
            ),
            "split_changed": bool(split_changed),
        }
        if split_column is not None and split_column in raw:
            output[split_column] = raw[split_column]
        output_rows.append(output)
        previous_end = end
        previous_rntis = rntis
        previous_profile = profile
        previous_split = split

    result = pl.DataFrame(output_rows)
    leaked = FORBIDDEN_BOUNDARY_COLUMNS & set(result.columns)
    if leaked:
        raise AssertionError(f"target-derived columns leaked into block audit: {leaked}")
    return result


def _threshold_token(value: float | int) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return format(numeric, "g").replace(".", "p")


def _sequential_group_ids(boundaries: Sequence[bool]) -> list[int]:
    group_id = -1
    result: list[int] = []
    for boundary in boundaries:
        if boundary:
            group_id += 1
        result.append(group_id)
    return result


def build_grouping_candidates(
    block_table: pl.DataFrame,
    *,
    config: GroupingConfig = GroupingConfig(),
    block_column: str = TRACE_BLOCK_COLUMN,
) -> pl.DataFrame:
    """Add intact-block grouping candidates for bootstrap sensitivity.

    Group IDs are sequential and deterministic.  If a split column is present,
    every candidate is forcibly reset at a split boundary.
    """

    config.validate()
    required = {
        block_column,
        "start_s",
        "end_s",
        "gap_from_previous_s",
        "rnti_jaccard_previous",
        "mobility_tv_previous",
    }
    missing = sorted(required - set(block_table.columns))
    if missing:
        raise DataValidationError(f"candidate grouping missing columns: {missing}")
    if block_table.is_empty():
        raise DataValidationError("candidate grouping requires observable blocks")

    rows = list(block_table.sort(["start_s", block_column]).iter_rows(named=True))
    base_boundary = [
        index == 0
        or bool(row.get("split_changed", False))
        for index, row in enumerate(rows)
    ]
    candidate_values: dict[str, list[int]] = {
        "observable_block_group": list(range(len(rows)))
    }

    for seconds in config.fixed_window_seconds:
        buckets = [int(math.floor(float(row["end_s"]) / seconds)) for row in rows]
        boundaries = [
            base_boundary[index]
            or index == 0
            or buckets[index] != buckets[index - 1]
            for index in range(len(rows))
        ]
        candidate_values[f"fixed_{_threshold_token(seconds)}s_group"] = (
            _sequential_group_ids(boundaries)
        )

    day_buckets = [int(math.floor(float(row["end_s"]) / 86_400.0)) for row in rows]
    day_boundaries = [
        base_boundary[index]
        or index == 0
        or day_buckets[index] != day_buckets[index - 1]
        for index in range(len(rows))
    ]
    candidate_values["utc_day_group"] = _sequential_group_ids(day_boundaries)

    for seconds in config.gap_threshold_seconds:
        boundaries = [
            base_boundary[index]
            or row["gap_from_previous_s"] is None
            or float(row["gap_from_previous_s"]) > seconds
            for index, row in enumerate(rows)
        ]
        candidate_values[f"gap_{_threshold_token(seconds)}s_group"] = (
            _sequential_group_ids(boundaries)
        )

    for threshold in config.mobility_tv_thresholds:
        boundaries = [
            base_boundary[index]
            or row["mobility_tv_previous"] is None
            or float(row["mobility_tv_previous"]) > threshold
            for index, row in enumerate(rows)
        ]
        candidate_values[f"mobility_tv_{_threshold_token(threshold)}_group"] = (
            _sequential_group_ids(boundaries)
        )

    for threshold in config.rnti_jaccard_thresholds:
        boundaries = [
            base_boundary[index]
            or row["rnti_jaccard_previous"] is None
            or float(row["rnti_jaccard_previous"]) < threshold
            for index, row in enumerate(rows)
        ]
        candidate_values[
            f"rnti_jaccard_{_threshold_token(threshold)}_group"
        ] = _sequential_group_ids(boundaries)

    campaign_boundaries = [
        base_boundary[index]
        or row["gap_from_previous_s"] is None
        or float(row["gap_from_previous_s"]) > config.campaign_gap_seconds
        or row["mobility_tv_previous"] is None
        or float(row["mobility_tv_previous"]) > config.campaign_mobility_tv
        or row["rnti_jaccard_previous"] is None
        or float(row["rnti_jaccard_previous"])
        < config.campaign_min_rnti_jaccard
        for index, row in enumerate(rows)
    ]
    candidate_values["campaign_proxy_group"] = _sequential_group_ids(
        campaign_boundaries
    )
    return block_table.sort(["start_s", block_column]).with_columns(
        [pl.Series(name, values, dtype=pl.Int64) for name, values in candidate_values.items()]
    )


def candidate_group_columns(frame: pl.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in frame.columns
        if column == "observable_block_group" or column.endswith("_group")
    )


def _candidate_interpretation(name: str) -> str:
    if name == "observable_block_group":
        return "existing observable fixed-window/gap block"
    if name.startswith("fixed_"):
        return "UTC-aligned time-only clustering sensitivity"
    if name == "utc_day_group":
        return "UTC-day sensitivity; too few groups for standalone inference"
    if name.startswith("gap_"):
        return "inter-event-gap heuristic; not a verified capture"
    if name.startswith("mobility_tv_"):
        return "mobility-metadata composition run; metadata-only sensitivity"
    if name.startswith("rnti_jaccard_"):
        return "adjacent numeric-RNTI population-continuity heuristic"
    if name == "campaign_proxy_group":
        return "composite gap/mobility/population proxy; not a true campaign"
    return "unclassified robustness grouping"


def summarize_candidate(
    grouped_blocks: pl.DataFrame,
    candidate: str,
    *,
    split_column: str | None = "split",
) -> CandidateSummary:
    if candidate not in grouped_blocks.columns:
        raise DataValidationError(f"unknown grouping candidate: {candidate}")
    group_by = [candidate]
    if split_column is not None and split_column in grouped_blocks.columns:
        group_by.insert(0, split_column)
    groups = grouped_blocks.group_by(group_by).agg(
        pl.len().alias("n_blocks"),
        pl.col("n_rows").sum().alias("n_rows"),
        pl.col("start_s").min().alias("start_s"),
        pl.col("end_s").max().alias("end_s"),
    ).with_columns((pl.col("end_s") - pl.col("start_s")).alias("span_s"))
    blocks_per_group = [int(value) for value in groups["n_blocks"].to_list()]
    spans = [float(value) / 3_600.0 for value in groups["span_s"].to_list()]
    groups_by_split: dict[str, int] = {}
    if split_column is not None and split_column in groups.columns:
        groups_by_split = {
            str(row[split_column]): int(row["len"])
            for row in groups.group_by(split_column).len().sort(split_column).iter_rows(
                named=True
            )
        }
    return CandidateSummary(
        candidate=candidate,
        interpretation=_candidate_interpretation(candidate),
        n_groups=groups.height,
        n_observable_blocks=grouped_blocks.height,
        n_rows=int(grouped_blocks["n_rows"].sum()),
        singleton_group_fraction=(
            sum(value == 1 for value in blocks_per_group) / len(blocks_per_group)
        ),
        blocks_per_group_q25=_linear_quantile(blocks_per_group, 0.25),
        blocks_per_group_median=_linear_quantile(blocks_per_group, 0.50),
        blocks_per_group_q75=_linear_quantile(blocks_per_group, 0.75),
        blocks_per_group_max=max(blocks_per_group),
        group_span_hours_median=_linear_quantile(spans, 0.50),
        group_span_hours_p90=_linear_quantile(spans, 0.90),
        group_span_hours_max=max(spans),
        groups_by_split=groups_by_split,
        at_least_30_groups=groups.height >= 30,
        minimum_groups_across_splits=(
            min(groups_by_split.values()) if groups_by_split else None
        ),
        at_least_30_groups_in_every_split=(
            bool(groups_by_split)
            and min(groups_by_split.values()) >= 30
        ),
    )


def candidate_mapping(
    grouped_blocks: pl.DataFrame,
    *,
    block_column: str = TRACE_BLOCK_COLUMN,
    split_column: str | None = "split",
) -> pl.DataFrame:
    """Return the minimal label-free block-to-cluster artifact."""

    columns = [block_column, "start_s", "end_s"]
    if split_column is not None and split_column in grouped_blocks.columns:
        columns.append(split_column)
    columns.extend(candidate_group_columns(grouped_blocks))
    mapping = grouped_blocks.select(columns).sort(["start_s", block_column])
    leaked = FORBIDDEN_BOUNDARY_COLUMNS & set(mapping.columns)
    if leaked:
        raise AssertionError(f"target-derived columns leaked into mapping: {leaked}")
    return mapping


def attach_grouping_candidate(
    frame: pl.DataFrame,
    grouped_blocks: pl.DataFrame,
    candidate: str,
    *,
    block_column: str = TRACE_BLOCK_COLUMN,
) -> pl.DataFrame:
    """Attach one full-block cluster ID to row-, epoch-, or metric-level data."""

    if candidate not in candidate_group_columns(grouped_blocks):
        raise DataValidationError(f"unknown grouping candidate: {candidate}")
    if block_column not in frame.columns:
        raise DataValidationError(f"frame missing block column: {block_column}")
    if candidate in frame.columns:
        raise DataValidationError(f"candidate column already exists: {candidate}")
    lookup = grouped_blocks.select([block_column, candidate]).unique()
    if lookup[block_column].n_unique() != lookup.height:
        raise DataValidationError("one observable block maps to multiple candidate groups")
    attached = frame.join(lookup, on=block_column, how="left")
    if attached.select(pl.col(candidate).is_null().any()).item():
        raise DataValidationError("candidate mapping does not cover every input block")
    return attached


def _mapping_sha256(mapping: pl.DataFrame) -> str:
    canonical = mapping.write_csv(separator=";", include_header=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_capture_robustness_audit(
    frame: pl.DataFrame,
    *,
    config: GroupingConfig = GroupingConfig(),
    block_column: str = TRACE_BLOCK_COLUMN,
    event_time_column: str = EVENT_TIME_COLUMN,
    rnti_column: str = "mac_rnti",
    mobility_column: str = "mob_pattern",
    lease_column: str = LEASE_ID_COLUMN,
) -> tuple[pl.DataFrame, CaptureRobustnessAudit]:
    """Return candidate block mappings and a compact robustness report."""

    block_table = summarize_observable_blocks(
        frame,
        block_column=block_column,
        event_time_column=event_time_column,
        rnti_column=rnti_column,
        mobility_column=mobility_column,
        lease_column=lease_column,
        split_column=config.split_column,
    )
    grouped = build_grouping_candidates(
        block_table,
        config=config,
        block_column=block_column,
    )
    mapping = candidate_mapping(
        grouped,
        block_column=block_column,
        split_column=config.split_column,
    )
    adjacent = grouped.slice(1)
    summaries = tuple(
        summarize_candidate(
            grouped,
            candidate,
            split_column=config.split_column,
        )
        for candidate in candidate_group_columns(grouped)
    )
    mobility_changes = int(
        adjacent.select(
            pl.col("dominant_mobility_changed").fill_null(False).sum()
        ).item()
    )
    zero_overlap = int(
        adjacent.select((pl.col("rnti_jaccard_previous") == 0).sum()).item()
    )
    audit = CaptureRobustnessAudit(
        audit_version=1,
        epistemic_status=(
            "candidate clustering units only; no true capture/campaign identifier exists"
        ),
        label_blind=True,
        n_rows=int(grouped["n_rows"].sum()),
        n_observable_blocks=grouped.height,
        trace_start_s=float(grouped["start_s"].min()),
        trace_end_s=float(grouped["end_s"].max()),
        observable_block_span_seconds_median=_linear_quantile(
            [float(value) for value in grouped["block_span_s"].to_list()], 0.50
        ),
        grouping_config=asdict(config),
        adjacent_gap_seconds_quantiles=_quantile_map(
            [
                float(value)
                for value in adjacent["gap_from_previous_s"].drop_nulls().to_list()
            ]
        ),
        adjacent_rnti_jaccard_quantiles=_quantile_map(
            [
                float(value)
                for value in adjacent["rnti_jaccard_previous"].drop_nulls().to_list()
            ]
        ),
        adjacent_mobility_tv_quantiles=_quantile_map(
            [
                float(value)
                for value in adjacent["mobility_tv_previous"].drop_nulls().to_list()
            ]
        ),
        mobility_signature_changes=mobility_changes,
        zero_rnti_overlap_transitions=zero_overlap,
        candidate_summaries=summaries,
        candidate_mapping_sha256=_mapping_sha256(mapping),
        cautions=(
            "mob_pattern is scenario metadata and may not be deployable context",
            "numeric RNTI overlap is not subscriber overlap because RNTIs are reused",
            "gap and composite groups are sensitivity units, not verified captures",
            "UTC-day and very long groups provide too few clusters for inference",
            "group boundaries must be frozen without inspecting labels or outcomes",
        ),
    )
    return grouped, audit


def write_audit_artifacts(
    grouped_blocks: pl.DataFrame,
    audit: CaptureRobustnessAudit,
    *,
    report_path: str | Path,
    mapping_path: str | Path,
    source_path: str | Path | None = None,
) -> None:
    """Write only derived, label-free artifacts and refuse the source path."""

    report = Path(report_path).resolve()
    mapping = Path(mapping_path).resolve()
    source = Path(source_path).resolve() if source_path is not None else None
    if source is not None and source in {report, mapping}:
        raise DataValidationError("refusing to overwrite the source CSV")
    if report == mapping:
        raise DataValidationError("report and mapping paths must differ")
    report.parent.mkdir(parents=True, exist_ok=True)
    mapping.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    candidate_mapping(grouped_blocks).write_csv(mapping, separator=";")


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--mapping-output", type=Path)
    parser.add_argument("--timestamp-scale", type=float, default=DEFAULT_TIMESTAMP_SCALE)
    parser.add_argument("--trace-gap-seconds", type=float, default=DEFAULT_TRACE_GAP_SECONDS)
    parser.add_argument("--max-block-seconds", type=float, default=DEFAULT_MAX_BLOCK_SECONDS)
    parser.add_argument("--lease-gap-seconds", type=float, default=DEFAULT_LEASE_GAP_SECONDS)
    args = parser.parse_args(argv)
    if (args.report_output is None) != (args.mapping_output is None):
        parser.error("--report-output and --mapping-output must be supplied together")

    raw = load_semicolon_csv(args.source_csv)
    prepared = prepare_trace(
        raw,
        timestamp_scale=args.timestamp_scale,
        trace_gap_seconds=args.trace_gap_seconds,
        max_block_seconds=args.max_block_seconds,
        lease_gap_seconds=args.lease_gap_seconds,
    )
    grouped, audit = run_capture_robustness_audit(
        prepared,
        config=GroupingConfig(split_column=None),
    )
    if args.report_output is not None and args.mapping_output is not None:
        write_audit_artifacts(
            grouped,
            audit,
            report_path=args.report_output,
            mapping_path=args.mapping_output,
            source_path=args.source_csv,
        )
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
    return 0


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
