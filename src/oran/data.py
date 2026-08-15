"""Deterministic, leakage-resistant data preparation for the O-RAN trace.

The functions in this module never modify the source CSV.  They return new Polars
frames and deliberately keep labels out of chronology, trace-block, and RNTI
lease segmentation.  A decision made from an epoch produced here is timestamped
at the *end* of that epoch and therefore applies no earlier than the next epoch.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import polars as pl


CSV_SEPARATOR = ";"
DEFAULT_TIMESTAMP_SCALE = 100_000.0
DEFAULT_TRACE_GAP_SECONDS = 300.0
DEFAULT_MAX_BLOCK_SECONDS = 900.0
DEFAULT_LEASE_GAP_SECONDS = 30.0

SOURCE_ROW_COLUMN = "_source_row"
EVENT_TIME_COLUMN = "event_time_s"
TRACE_BLOCK_COLUMN = "trace_block_id"
LEASE_SEQUENCE_COLUMN = "rnti_lease_sequence"
LEASE_ID_COLUMN = "rnti_lease_id"

EXPECTED_COLUMNS: tuple[str, ...] = (
    "mac_rnti",
    "mac_dl_cqi",
    "mac_dl_mcs",
    "mac_dl_brate",
    "mac_dl_ok",
    "mac_dl_nok",
    "phy_ul_pusch_sinr",
    "phy_ul_pucch_sinr",
    "phy_ul_mcs",
    "mac_ul_brate",
    "mac_ul_ok",
    "mac_ul_nok",
    "mac_ul_bsr",
    "mac_pci",
    "mac_nof_tti",
    "mac_cc_idx",
    "mac_dl_buffer",
    "mac_dl_ri",
    "mac_dl_pmi",
    "mac_phr",
    "mac_dl_cqi_offset",
    "mac_ul_snr_offset",
    "mac_ul_rssi",
    "mac_fec_iters",
    "mac_dl_mcs_samples",
    "mac_ul_mcs",
    "mac_ul_mcs_samples",
    "phy_ul_n",
    "phy_ul_pusch_rssi",
    "phy_ul_pusch_tpc",
    "phy_ul_pucch_rssi",
    "phy_ul_pucch_ni",
    "phy_ul_turbo_iters",
    "phy_ul_n_samples",
    "phy_ul_n_samples_pucch",
    "phy_dl_mcs",
    "phy_dl_pucch_tpc",
    "phy_dl_n_samples",
    "rf_o",
    "rf_u",
    "rf_l",
    "rf_error",
    "label",
    "ue_ident",
    "timestamp",
    "id_ue",
    "mob_pattern",
)

# The ordering is part of the experiment contract; never derive IDs from row order.
LABEL_ONTOLOGY: Mapping[str, int] = {
    "Web Browsing": 0,
    "SIPP": 1,
    "youtube": 2,
    "iot": 3,
    "portscan": 4,
    "ddos-ripper-C": 5,
    "dos-hulk-C": 6,
    "slowloris-C": 7,
}
NON_ATTACK_TRAFFIC_LABELS: tuple[str, ...] = (
    "Web Browsing",
    "SIPP",
    "youtube",
    "iot",
)
ATTACK_LABELS: tuple[str, ...] = (
    "portscan",
    "ddos-ripper-C",
    "dos-hulk-C",
    "slowloris-C",
)

# These values were observed to be constant in the audited source.  They are
# checked rather than silently discarded so a changed export cannot pass unnoticed.
EXPECTED_CONSTANT_VALUES: Mapping[str, int | float] = {
    "mac_pci": 1,
    "mac_cc_idx": 0,
    "mac_dl_ri": 0.0,
    "mac_dl_pmi": 0.0,
    "mac_ul_rssi": 0.0,
    "mac_fec_iters": 0.0,
    "mac_dl_mcs_samples": 0,
    "mac_ul_mcs": 0.0,
    "mac_ul_mcs_samples": 0,
    "phy_ul_n": 0.0,
    "phy_ul_pusch_tpc": 0,
    "phy_dl_pucch_tpc": 0,
    "rf_o": 0,
    "rf_u": 0,
    "rf_l": 0,
}

IDENTIFIER_COLUMNS: tuple[str, ...] = ("mac_rnti", "ue_ident", "id_ue")
TARGET_COLUMNS: tuple[str, ...] = ("label", "label_id", "is_attack")
DEFAULT_CONTEXT_COLUMNS: tuple[str, ...] = ("id_ue", "mob_pattern")
INTERNAL_COLUMNS: frozenset[str] = frozenset(
    {
        SOURCE_ROW_COLUMN,
        EVENT_TIME_COLUMN,
        TRACE_BLOCK_COLUMN,
        LEASE_SEQUENCE_COLUMN,
        LEASE_ID_COLUMN,
    }
)


class DataValidationError(ValueError):
    """Raised when an input trace violates the frozen data contract."""


@dataclass(frozen=True)
class TraceAudit:
    """JSON-serializable audit of the raw trace contract."""

    n_rows: int
    n_source_columns: int
    source_columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    unexpected_columns: tuple[str, ...]
    column_order_matches: bool
    label_counts: Mapping[str, int]
    unknown_labels: tuple[str, ...]
    null_label_rows: int
    exact_duplicate_row_members: int
    identifier_mismatch_rows: int
    timestamp_null_rows: int
    timestamp_nonfinite_rows: int
    original_time_reversals: int
    constant_value_violations: Mapping[str, int]

    @property
    def is_valid(self) -> bool:
        return not (
            self.missing_columns
            or self.unexpected_columns
            or not self.column_order_matches
            or self.unknown_labels
            or self.null_label_rows
            or self.exact_duplicate_row_members
            or self.identifier_mismatch_rows
            or self.timestamp_null_rows
            or self.timestamp_nonfinite_rows
            or any(self.constant_value_violations.values())
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _source_columns(frame: pl.DataFrame) -> list[str]:
    """Return CSV columns while ignoring columns created by this module."""

    return [name for name in frame.columns if name not in INTERNAL_COLUMNS]


def scan_semicolon_csv(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    infer_schema_length: int = 10_000,
) -> pl.LazyFrame:
    """Lazily scan a semicolon-delimited trace and add a stable source-row index."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    lazy = pl.scan_csv(
        source,
        separator=CSV_SEPARATOR,
        has_header=True,
        infer_schema_length=infer_schema_length,
        schema_overrides={
            "mac_rnti": pl.Int64,
            "ue_ident": pl.Int64,
            "id_ue": pl.Int64,
            "timestamp": pl.Float64,
            "label": pl.String,
            "mob_pattern": pl.String,
        },
        truncate_ragged_lines=False,
        low_memory=True,
        rechunk=False,
    ).with_row_index(SOURCE_ROW_COLUMN)
    if columns is not None:
        selected = [SOURCE_ROW_COLUMN, *columns]
        lazy = lazy.select(selected)
    return lazy


def load_semicolon_csv(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    infer_schema_length: int = 10_000,
) -> pl.DataFrame:
    """Read the trace without writing to or otherwise modifying ``path``."""

    return scan_semicolon_csv(
        path,
        columns=columns,
        infer_schema_length=infer_schema_length,
    ).collect(engine="streaming")


def audit_trace(frame: pl.DataFrame) -> TraceAudit:
    """Check schema, ontology, constants, duplicate IDs, and raw chronology."""

    source_columns = _source_columns(frame)
    expected_set = set(EXPECTED_COLUMNS)
    source_set = set(source_columns)
    missing = tuple(name for name in EXPECTED_COLUMNS if name not in source_set)
    unexpected = tuple(name for name in source_columns if name not in expected_set)
    order_matches = tuple(source_columns) == EXPECTED_COLUMNS

    label_counts: dict[str, int] = {}
    unknown_labels: tuple[str, ...] = ()
    null_label_rows = 0
    if "label" in frame.columns:
        label_counts = {
            str(row["label"]): int(row["len"])
            for row in frame.group_by("label").len().sort("label").iter_rows(named=True)
            if row["label"] is not None
        }
        null_label_rows = int(frame.select(pl.col("label").is_null().sum()).item())
        unknown_labels = tuple(sorted(set(label_counts) - set(LABEL_ONTOLOGY)))

    identifier_mismatch_rows = 0
    if {"mac_rnti", "ue_ident"}.issubset(frame.columns):
        identifier_mismatch_rows = int(
            frame.select(
                (
                    (pl.col("mac_rnti").is_null() != pl.col("ue_ident").is_null())
                    | (pl.col("mac_rnti") != pl.col("ue_ident")).fill_null(False)
                ).sum()
            ).item()
        )

    # Exclude the synthetic source-row index: two records are duplicates only
    # when every field exported by the dataset is equal. ``is_duplicated``
    # marks every member of an exact duplicate group, which makes a zero count
    # an unambiguous no-duplicate audit result without discarding a first copy.
    exact_duplicate_row_members = (
        int(frame.select(source_columns).is_duplicated().sum())
        if source_columns
        else 0
    )

    timestamp_null_rows = 0
    timestamp_nonfinite_rows = 0
    original_time_reversals = 0
    if "timestamp" in frame.columns:
        in_source_order = (
            frame.sort(SOURCE_ROW_COLUMN)
            if SOURCE_ROW_COLUMN in frame.columns
            else frame
        )
        timestamp_null_rows, timestamp_nonfinite_rows, original_time_reversals = (
            int(value)
            for value in in_source_order.select(
                pl.col("timestamp").is_null().sum().alias("null_rows"),
                (
                    pl.col("timestamp").is_not_null()
                    & ~pl.col("timestamp").cast(pl.Float64, strict=False).is_finite()
                ).sum().alias("nonfinite_rows"),
                (pl.col("timestamp").cast(pl.Float64, strict=False).diff() < 0)
                .fill_null(False)
                .sum()
                .alias("time_reversals"),
            ).row(0)
        )

    constant_violations: dict[str, int] = {}
    for column, expected in EXPECTED_CONSTANT_VALUES.items():
        if column not in frame.columns:
            continue
        # Treat null as a violation; these fields are expected to have an exact,
        # observed constant value in this specific export.
        constant_violations[column] = int(
            frame.select(
                (pl.col(column).is_null() | (pl.col(column) != expected)).sum()
            ).item()
        )

    return TraceAudit(
        n_rows=frame.height,
        n_source_columns=len(source_columns),
        source_columns=tuple(source_columns),
        missing_columns=missing,
        unexpected_columns=unexpected,
        column_order_matches=order_matches,
        label_counts=label_counts,
        unknown_labels=unknown_labels,
        null_label_rows=null_label_rows,
        exact_duplicate_row_members=exact_duplicate_row_members,
        identifier_mismatch_rows=identifier_mismatch_rows,
        timestamp_null_rows=timestamp_null_rows,
        timestamp_nonfinite_rows=timestamp_nonfinite_rows,
        original_time_reversals=original_time_reversals,
        constant_value_violations=constant_violations,
    )


def validate_trace(frame: pl.DataFrame) -> TraceAudit:
    """Return the audit or raise with all hard contract violations."""

    audit = audit_trace(frame)
    errors: list[str] = []
    if audit.missing_columns:
        errors.append(f"missing columns: {list(audit.missing_columns)}")
    if audit.unexpected_columns:
        errors.append(f"unexpected columns: {list(audit.unexpected_columns)}")
    if not audit.column_order_matches:
        errors.append("source column order differs from the frozen schema")
    if audit.unknown_labels:
        errors.append(f"unknown labels: {list(audit.unknown_labels)}")
    if audit.null_label_rows:
        errors.append(f"null labels: {audit.null_label_rows}")
    if audit.exact_duplicate_row_members:
        errors.append(
            "rows participating in exact duplicate groups: "
            f"{audit.exact_duplicate_row_members}"
        )
    if audit.identifier_mismatch_rows:
        errors.append(
            "mac_rnti and ue_ident differ on "
            f"{audit.identifier_mismatch_rows} rows"
        )
    if audit.timestamp_null_rows or audit.timestamp_nonfinite_rows:
        errors.append(
            "invalid timestamps: "
            f"{audit.timestamp_null_rows} null, "
            f"{audit.timestamp_nonfinite_rows} non-finite"
        )
    bad_constants = {
        key: value
        for key, value in audit.constant_value_violations.items()
        if value
    }
    if bad_constants:
        errors.append(f"declared constant values changed: {bad_constants}")
    if errors:
        raise DataValidationError("; ".join(errors))
    return audit


def add_label_ontology(frame: pl.DataFrame) -> pl.DataFrame:
    """Add frozen integer and binary targets after rejecting unknown labels."""

    if "label" not in frame.columns:
        raise DataValidationError("label column is required")
    unknown = (
        frame.filter(
            pl.col("label").is_null()
            | ~pl.col("label").is_in(list(LABEL_ONTOLOGY))
        )
        .select("label")
        .unique()
        .to_series()
        .to_list()
    )
    if unknown:
        raise DataValidationError(f"unknown or null labels: {unknown}")
    label_id = pl.col("label").replace_strict(
        LABEL_ONTOLOGY,
        return_dtype=pl.Int8,
    )
    return frame.with_columns(
        label_id.alias("label_id"),
        pl.col("label").is_in(list(ATTACK_LABELS)).alias("is_attack"),
    )


def correct_chronology(
    frame: pl.DataFrame,
    *,
    timestamp_scale: float = DEFAULT_TIMESTAMP_SCALE,
    timestamp_column: str = "timestamp",
    event_time_column: str = EVENT_TIME_COLUMN,
) -> pl.DataFrame:
    """Add corrected Unix seconds and stably order events without changing raw time."""

    if not math.isfinite(timestamp_scale) or timestamp_scale <= 0:
        raise ValueError("timestamp_scale must be finite and positive")
    if timestamp_column not in frame.columns:
        raise DataValidationError(f"missing timestamp column: {timestamp_column}")
    if SOURCE_ROW_COLUMN not in frame.columns:
        frame = frame.with_row_index(SOURCE_ROW_COLUMN)
    corrected = frame.with_columns(
        (
            pl.col(timestamp_column).cast(pl.Float64, strict=True)
            / float(timestamp_scale)
        ).alias(event_time_column)
    )
    invalid = corrected.select(
        (
            pl.col(event_time_column).is_null()
            | ~pl.col(event_time_column).is_finite()
        ).sum()
    ).item()
    if invalid:
        raise DataValidationError(f"{invalid} corrected timestamps are invalid")
    return corrected.sort(
        [event_time_column, SOURCE_ROW_COLUMN],
        maintain_order=True,
    )


def segment_trace_blocks(
    frame: pl.DataFrame,
    *,
    gap_seconds: float | None = DEFAULT_TRACE_GAP_SECONDS,
    max_block_seconds: float | None = DEFAULT_MAX_BLOCK_SECONDS,
    event_time_column: str = EVENT_TIME_COLUMN,
    block_column: str = TRACE_BLOCK_COLUMN,
) -> pl.DataFrame:
    """Create global chronological blocks using only observable event time.

    A block boundary is placed at an observable time gap or a deterministic
    wall-clock bin change.  Labels, user IDs, mobility, and traffic values are not
    consulted.  The finite maximum duration guarantees usable full-block splits
    even when the capture has no large gaps.
    """

    if event_time_column not in frame.columns:
        raise DataValidationError(
            f"{event_time_column} is required; call correct_chronology first"
        )
    if gap_seconds is not None and (
        not math.isfinite(gap_seconds) or gap_seconds <= 0
    ):
        raise ValueError("gap_seconds must be positive or None")
    if max_block_seconds is not None and (
        not math.isfinite(max_block_seconds) or max_block_seconds <= 0
    ):
        raise ValueError("max_block_seconds must be positive or None")
    if gap_seconds is None and max_block_seconds is None:
        raise ValueError("at least one observable block boundary rule is required")
    if frame.is_empty():
        return frame.with_columns(pl.lit(None, dtype=pl.Int64).alias(block_column))

    ordered = frame.sort(
        [event_time_column, SOURCE_ROW_COLUMN]
        if SOURCE_ROW_COLUMN in frame.columns
        else event_time_column,
        maintain_order=True,
    )
    previous = pl.col(event_time_column).shift(1)
    boundary = previous.is_null()
    if gap_seconds is not None:
        boundary = boundary | (
            pl.col(event_time_column) - previous > float(gap_seconds)
        ).fill_null(False)
    if max_block_seconds is not None:
        # Anchor to the Unix epoch, not to the first capture row.  Calendar cuts
        # such as midnight/noon then coincide with 15-minute block boundaries.
        wall_bin = (
            pl.col(event_time_column) / float(max_block_seconds)
        ).floor().cast(pl.Int64)
        boundary = boundary | (wall_bin != wall_bin.shift(1)).fill_null(False)

    return ordered.with_columns(
        (boundary.cast(pl.Int64).cum_sum() - 1).alias(block_column)
    )


def segment_rnti_leases(
    frame: pl.DataFrame,
    *,
    lease_gap_seconds: float = DEFAULT_LEASE_GAP_SECONDS,
    rnti_column: str = "mac_rnti",
    event_time_column: str = EVENT_TIME_COLUMN,
    block_column: str = TRACE_BLOCK_COLUMN,
    lease_sequence_column: str = LEASE_SEQUENCE_COLUMN,
    lease_id_column: str = LEASE_ID_COLUMN,
) -> pl.DataFrame:
    """Infer RNTI lifecycles from block boundaries and per-RNTI time gaps only."""

    if not math.isfinite(lease_gap_seconds) or lease_gap_seconds <= 0:
        raise ValueError("lease_gap_seconds must be finite and positive")
    required = {rnti_column, event_time_column, block_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"lease segmentation missing columns: {missing}")
    if frame.select(pl.col(rnti_column).is_null().sum()).item():
        raise DataValidationError("RNTI cannot be null for lease segmentation")
    if frame.is_empty():
        return frame.with_columns(
            pl.lit(None, dtype=pl.Int64).alias(lease_sequence_column),
            pl.lit(None, dtype=pl.String).alias(lease_id_column),
        )

    group = [block_column, rnti_column]
    sort_columns = [*group, event_time_column]
    if SOURCE_ROW_COLUMN in frame.columns:
        sort_columns.append(SOURCE_ROW_COLUMN)
    by_rnti = frame.sort(sort_columns, maintain_order=True)
    previous = pl.col(event_time_column).shift(1).over(group)
    starts_lease = previous.is_null() | (
        pl.col(event_time_column) - previous > float(lease_gap_seconds)
    ).fill_null(False)
    by_rnti = by_rnti.with_columns(
        (
            starts_lease.cast(pl.Int64).cum_sum().over(group) - 1
        ).alias(lease_sequence_column)
    ).with_columns(
        pl.concat_str(
            [
                pl.col(block_column).cast(pl.String),
                pl.col(rnti_column).cast(pl.String),
                pl.col(lease_sequence_column).cast(pl.String),
            ],
            separator=":",
        ).alias(lease_id_column)
    )
    chronological_sort = [event_time_column]
    if SOURCE_ROW_COLUMN in by_rnti.columns:
        chronological_sort.append(SOURCE_ROW_COLUMN)
    return by_rnti.sort(chronological_sort, maintain_order=True)


def prepare_trace(
    frame: pl.DataFrame,
    *,
    validate: bool = True,
    timestamp_scale: float = DEFAULT_TIMESTAMP_SCALE,
    trace_gap_seconds: float | None = DEFAULT_TRACE_GAP_SECONDS,
    max_block_seconds: float | None = DEFAULT_MAX_BLOCK_SECONDS,
    lease_gap_seconds: float = DEFAULT_LEASE_GAP_SECONDS,
    add_targets: bool = True,
) -> pl.DataFrame:
    """Run the frozen validation, chronology, block, and lifecycle pipeline."""

    if validate:
        validate_trace(frame)
    prepared = correct_chronology(frame, timestamp_scale=timestamp_scale)
    prepared = segment_trace_blocks(
        prepared,
        gap_seconds=trace_gap_seconds,
        max_block_seconds=max_block_seconds,
    )
    prepared = segment_rnti_leases(
        prepared,
        lease_gap_seconds=lease_gap_seconds,
    )
    return add_label_ontology(prepared) if add_targets else prepared


def _default_epoch_features(frame: pl.DataFrame) -> list[str]:
    excluded = (
        set(IDENTIFIER_COLUMNS)
        | set(TARGET_COLUMNS)
        | set(EXPECTED_CONSTANT_VALUES)
        | INTERNAL_COLUMNS
        | {"timestamp", "mob_pattern"}
    )
    schema = frame.schema
    return [
        name
        for name, dtype in schema.items()
        if name not in excluded and dtype.is_numeric()
    ]


def to_causal_epochs(
    frame: pl.DataFrame,
    *,
    epoch_seconds: float = 1.0,
    feature_columns: Sequence[str] | None = None,
    context_columns: Sequence[str] = DEFAULT_CONTEXT_COLUMNS,
    include_targets: bool = True,
    event_time_column: str = EVENT_TIME_COLUMN,
    block_column: str = TRACE_BLOCK_COLUMN,
    rnti_column: str = "mac_rnti",
    lease_id_column: str = LEASE_ID_COLUMN,
) -> pl.DataFrame:
    """Aggregate observed samples into causal fixed-width lifecycle epochs.

    Feature values are the last value observed within the epoch.  ``decision_time_s``
    is its right boundary, so downstream code must apply an action no earlier than
    that time.  Empty seconds are not fabricated: absence is not relabeled benign.
    """

    if not math.isfinite(epoch_seconds) or epoch_seconds <= 0:
        raise ValueError("epoch_seconds must be finite and positive")
    required = {event_time_column, block_column, rnti_column, lease_id_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"epoching missing columns: {missing}")
    chosen_features = list(
        _default_epoch_features(frame)
        if feature_columns is None
        else feature_columns
    )
    absent_features = sorted(set(chosen_features) - set(frame.columns))
    if absent_features:
        raise DataValidationError(f"unknown epoch features: {absent_features}")
    forbidden = (
        set(IDENTIFIER_COLUMNS)
        | set(TARGET_COLUMNS)
        | {"timestamp", "mob_pattern"}
    )
    bad_features = sorted(set(chosen_features) & forbidden)
    if bad_features:
        raise DataValidationError(
            f"identifier, context, raw-time, or target features forbidden: {bad_features}"
        )
    chosen_context = [name for name in context_columns if name in frame.columns]

    epoch_start_column = "epoch_start_s"
    ordered = frame.sort(
        [event_time_column, SOURCE_ROW_COLUMN]
        if SOURCE_ROW_COLUMN in frame.columns
        else event_time_column,
        maintain_order=True,
    ).with_columns(
        (
            (pl.col(event_time_column) / float(epoch_seconds)).floor()
            * float(epoch_seconds)
        ).alias(epoch_start_column)
    )
    group_columns = [
        block_column,
        lease_id_column,
        rnti_column,
        epoch_start_column,
    ]
    aggregations: list[pl.Expr] = [
        pl.len().alias("samples_in_epoch"),
        pl.col(event_time_column).min().alias("first_event_time_s"),
        pl.col(event_time_column).max().alias("last_event_time_s"),
    ]
    aggregations.extend(pl.col(name).last().alias(name) for name in chosen_features)
    for name in chosen_context:
        aggregations.extend(
            [
                pl.col(name).first().alias(name),
                pl.col(name).n_unique().alias(f"{name}_n_unique"),
            ]
        )
    if include_targets:
        if "label" not in frame.columns:
            raise DataValidationError("label is required when include_targets=True")
        aggregations.extend(
            [
                pl.col("label").drop_nulls().unique().sort().alias("labels_in_epoch"),
                pl.col("label").is_not_null().any().alias("target_observed"),
                pl.col("label")
                .is_in(list(ATTACK_LABELS))
                .any()
                .alias("is_attack_epoch"),
            ]
        )

    return (
        ordered.group_by(group_columns, maintain_order=True)
        .agg(aggregations)
        .with_columns(
            (
                pl.col(epoch_start_column) + float(epoch_seconds)
            ).alias("decision_time_s"),
            pl.lit(float(epoch_seconds)).alias("epoch_seconds"),
        )
        .sort(["decision_time_s", block_column, lease_id_column], maintain_order=True)
    )


def _count_summary(frame: pl.DataFrame) -> dict[str, int]:
    result = {"rows": frame.height}
    for column, key in (
        (TRACE_BLOCK_COLUMN, "trace_blocks"),
        (LEASE_ID_COLUMN, "rnti_leases"),
        ("mac_rnti", "numeric_rntis"),
    ):
        if column in frame.columns:
            result[key] = int(frame.select(pl.col(column).n_unique()).item())
    return result


def cli(argv: Sequence[str] | None = None) -> int:
    """Audit/prepare a trace and print a JSON summary; never write the source."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv", type=Path)
    parser.add_argument("--timestamp-scale", type=float, default=DEFAULT_TIMESTAMP_SCALE)
    parser.add_argument("--trace-gap-seconds", type=float, default=DEFAULT_TRACE_GAP_SECONDS)
    parser.add_argument("--max-block-seconds", type=float, default=DEFAULT_MAX_BLOCK_SECONDS)
    parser.add_argument("--lease-gap-seconds", type=float, default=DEFAULT_LEASE_GAP_SECONDS)
    parser.add_argument("--epochs", action="store_true", help="also count causal 1 s epochs")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="validate and report without preparing blocks or leases",
    )
    args = parser.parse_args(argv)

    raw = load_semicolon_csv(args.source_csv)
    audit = validate_trace(raw)
    output: dict[str, Any] = {"audit": audit.to_dict()}
    if not args.audit_only:
        prepared = prepare_trace(
            raw,
            validate=False,
            timestamp_scale=args.timestamp_scale,
            trace_gap_seconds=args.trace_gap_seconds,
            max_block_seconds=args.max_block_seconds,
            lease_gap_seconds=args.lease_gap_seconds,
        )
        output["prepared"] = _count_summary(prepared)
        if args.epochs:
            output["epochs"] = _count_summary(to_causal_epochs(prepared))
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
