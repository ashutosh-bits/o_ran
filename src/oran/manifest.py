"""Deterministic full-block chronological split manifests for the O-RAN trace."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from .data import (
    CSV_SEPARATOR,
    DEFAULT_LEASE_GAP_SECONDS,
    DEFAULT_MAX_BLOCK_SECONDS,
    DEFAULT_TIMESTAMP_SCALE,
    DEFAULT_TRACE_GAP_SECONDS,
    EVENT_TIME_COLUMN,
    LABEL_ONTOLOGY,
    TRACE_BLOCK_COLUMN,
    DataValidationError,
    load_semicolon_csv,
    prepare_trace,
)


MANIFEST_VERSION = 1
DEFAULT_SPLIT_NAMES: tuple[str, ...] = (
    "train",
    "calibration",
    "controller_tune",
    "test",
)
# Train through Oct 25, calibrate on the first half of Oct 26, tune on its
# second half, and leave Oct 27 untouched.  A block crossing a cut is moved in
# full to the later split.
DEFAULT_CUT_TIMES_UTC: tuple[str, ...] = (
    "2022-10-26T00:00:00Z",
    "2022-10-26T12:00:00Z",
    "2022-10-27T00:00:00Z",
)


@dataclass(frozen=True)
class BlockRecord:
    trace_block_id: int
    split: str
    start_s: float
    end_s: float
    n_rows: int
    n_numeric_rntis: int


@dataclass(frozen=True)
class SplitRecord:
    name: str
    block_ids: tuple[int, ...]
    start_s: float
    end_s: float
    n_rows: int
    n_numeric_rntis: int


@dataclass(frozen=True)
class SplitManifest:
    """Portable, JSON-serializable record of one immutable split assignment."""

    manifest_version: int
    source_path: str | None
    source_size_bytes: int | None
    source_sha256: str | None
    csv_separator: str
    timestamp_scale: float
    trace_gap_seconds: float | None
    max_block_seconds: float | None
    lease_gap_seconds: float
    boundary_policy: str
    split_names: tuple[str, ...]
    cut_times_s: tuple[float, ...]
    blocks: tuple[BlockRecord, ...]
    splits: tuple[SplitRecord, ...]
    novelty_reference_splits: tuple[str, ...]
    evaluation_split: str
    seen_numeric_rntis: tuple[int, ...]
    unseen_numeric_rntis: tuple[int, ...]
    label_ontology: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SplitManifest":
        values = dict(raw)
        values["split_names"] = tuple(values["split_names"])
        values["cut_times_s"] = tuple(float(v) for v in values["cut_times_s"])
        values["blocks"] = tuple(BlockRecord(**item) for item in values["blocks"])
        values["splits"] = tuple(
            SplitRecord(
                **{
                    **item,
                    "block_ids": tuple(item["block_ids"]),
                }
            )
            for item in values["splits"]
        )
        values["novelty_reference_splits"] = tuple(
            values["novelty_reference_splits"]
        )
        values["seen_numeric_rntis"] = tuple(values["seen_numeric_rntis"])
        values["unseen_numeric_rntis"] = tuple(values["unseen_numeric_rntis"])
        return cls(**values)


def parse_utc_seconds(value: str | float | int) -> float:
    """Parse Unix seconds or an ISO-8601 timestamp, treating naive time as UTC."""

    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except ValueError:
        pass
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a source file read-only in bounded chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_split_spec(
    split_names: Sequence[str], cut_times_s: Sequence[float]
) -> None:
    if len(split_names) < 2:
        raise ValueError("at least two chronological splits are required")
    if len(set(split_names)) != len(split_names):
        raise ValueError("split names must be unique")
    if len(cut_times_s) != len(split_names) - 1:
        raise ValueError("the number of cuts must be one less than split names")
    if any(right <= left for left, right in zip(cut_times_s, cut_times_s[1:])):
        raise ValueError("cut times must be strictly increasing")


def attach_split_labels(
    frame: pl.DataFrame,
    manifest: SplitManifest,
    *,
    block_column: str = TRACE_BLOCK_COLUMN,
    split_column: str = "split",
) -> pl.DataFrame:
    """Attach the one manifest split assigned to each complete trace block."""

    if block_column not in frame.columns:
        raise DataValidationError(f"missing trace-block column: {block_column}")
    if split_column in frame.columns:
        raise DataValidationError(f"split column already exists: {split_column}")
    assignments = pl.DataFrame(
        {
            block_column: [record.trace_block_id for record in manifest.blocks],
            split_column: [record.split for record in manifest.blocks],
        },
        schema={block_column: pl.Int64, split_column: pl.String},
    )
    attached = frame.join(assignments, on=block_column, how="left")
    missing = attached.select(pl.col(split_column).is_null().sum()).item()
    if missing:
        unknown_blocks = (
            attached.filter(pl.col(split_column).is_null())
            .select(block_column)
            .unique()
            .sort(block_column)
            .to_series()
            .to_list()
        )
        raise DataValidationError(
            f"manifest has no assignment for trace blocks: {unknown_blocks}"
        )
    split_counts_per_block = attached.group_by(block_column).agg(
        pl.col(split_column).n_unique().alias("n_splits")
    )
    if split_counts_per_block.select((pl.col("n_splits") != 1).any()).item():
        raise DataValidationError("a trace block was divided across splits")
    return attached


def annotate_rnti_novelty(
    frame: pl.DataFrame,
    *,
    reference_splits: Sequence[str] = DEFAULT_SPLIT_NAMES[:-1],
    evaluation_split: str = DEFAULT_SPLIT_NAMES[-1],
    split_column: str = "split",
    rnti_column: str = "mac_rnti",
) -> pl.DataFrame:
    """Flag numeric RNTIs in the evaluation split as seen or unseen previously."""

    required = {split_column, rnti_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"RNTI novelty missing columns: {missing}")
    reference_rntis = (
        frame.filter(pl.col(split_column).is_in(list(reference_splits)))
        .select(rnti_column)
        .drop_nulls()
        .unique()
        .to_series()
        .to_list()
    )
    is_evaluation = pl.col(split_column) == evaluation_split
    is_seen = pl.col(rnti_column).is_in(reference_rntis)
    return frame.with_columns(
        pl.when(is_evaluation)
        .then(is_seen)
        .otherwise(pl.lit(None, dtype=pl.Boolean))
        .alias("rnti_seen_pretest"),
        pl.when(pl.col(split_column).is_in(list(reference_splits)))
        .then(pl.lit("reference"))
        .when(is_evaluation & is_seen)
        .then(pl.lit("seen"))
        .when(is_evaluation & ~is_seen)
        .then(pl.lit("unseen"))
        .otherwise(pl.lit("not_evaluated"))
        .alias("rnti_novelty"),
    )


def build_split_manifest(
    frame: pl.DataFrame,
    *,
    split_names: Sequence[str] = DEFAULT_SPLIT_NAMES,
    cut_times: Sequence[str | float | int] = DEFAULT_CUT_TIMES_UTC,
    source_path: str | Path | None = None,
    include_source_sha256: bool = True,
    timestamp_scale: float = DEFAULT_TIMESTAMP_SCALE,
    trace_gap_seconds: float | None = DEFAULT_TRACE_GAP_SECONDS,
    max_block_seconds: float | None = DEFAULT_MAX_BLOCK_SECONDS,
    lease_gap_seconds: float = DEFAULT_LEASE_GAP_SECONDS,
    block_column: str = TRACE_BLOCK_COLUMN,
    event_time_column: str = EVENT_TIME_COLUMN,
    rnti_column: str = "mac_rnti",
    allow_empty_splits: bool = False,
) -> SplitManifest:
    """Assign every observable block, intact, to one chronological split.

    Assignment uses block *end* time.  Consequently, a block crossing a nominal
    cut is placed wholly in the later split and can never leak across that cut.
    """

    cuts = tuple(parse_utc_seconds(value) for value in cut_times)
    names = tuple(split_names)
    _validate_split_spec(names, cuts)
    required = {block_column, event_time_column, rnti_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"manifest missing columns: {missing}")
    if frame.is_empty():
        raise DataValidationError("cannot build a split manifest for an empty trace")

    summary = (
        frame.group_by(block_column)
        .agg(
            pl.col(event_time_column).min().alias("start_s"),
            pl.col(event_time_column).max().alias("end_s"),
            pl.len().alias("n_rows"),
            pl.col(rnti_column).n_unique().alias("n_numeric_rntis"),
        )
        .sort(["start_s", block_column])
    )
    raw_blocks = list(summary.iter_rows(named=True))
    for earlier, later in zip(raw_blocks, raw_blocks[1:]):
        if float(later["start_s"]) < float(earlier["end_s"]):
            raise DataValidationError("trace-block time intervals overlap")

    blocks: list[BlockRecord] = []
    for raw in raw_blocks:
        # bisect_left gives the earlier split when end_s == cut and the later
        # split whenever a block extends past the nominal cut.
        split_index = bisect.bisect_left(cuts, float(raw["end_s"]))
        blocks.append(
            BlockRecord(
                trace_block_id=int(raw[block_column]),
                split=names[split_index],
                start_s=float(raw["start_s"]),
                end_s=float(raw["end_s"]),
                n_rows=int(raw["n_rows"]),
                n_numeric_rntis=int(raw["n_numeric_rntis"]),
            )
        )

    provisional = SplitManifest(
        manifest_version=MANIFEST_VERSION,
        source_path=None,
        source_size_bytes=None,
        source_sha256=None,
        csv_separator=CSV_SEPARATOR,
        timestamp_scale=float(timestamp_scale),
        trace_gap_seconds=trace_gap_seconds,
        max_block_seconds=max_block_seconds,
        lease_gap_seconds=float(lease_gap_seconds),
        boundary_policy="assign_block_by_end_time; crossing_block_moves_later",
        split_names=names,
        cut_times_s=cuts,
        blocks=tuple(blocks),
        splits=(),
        novelty_reference_splits=names[:-1],
        evaluation_split=names[-1],
        seen_numeric_rntis=(),
        unseen_numeric_rntis=(),
        label_ontology=dict(LABEL_ONTOLOGY),
    )
    attached = attach_split_labels(frame, provisional)

    splits: list[SplitRecord] = []
    for name in names:
        subset = attached.filter(pl.col("split") == name)
        if subset.is_empty():
            if allow_empty_splits:
                continue
            raise DataValidationError(f"chronological split is empty: {name}")
        block_ids = tuple(
            int(value)
            for value in subset.select(block_column)
            .unique()
            .sort(block_column)
            .to_series()
            .to_list()
        )
        splits.append(
            SplitRecord(
                name=name,
                block_ids=block_ids,
                start_s=float(subset.select(pl.col(event_time_column).min()).item()),
                end_s=float(subset.select(pl.col(event_time_column).max()).item()),
                n_rows=subset.height,
                n_numeric_rntis=int(
                    subset.select(pl.col(rnti_column).n_unique()).item()
                ),
            )
        )
    for earlier, later in zip(splits, splits[1:]):
        if later.start_s < earlier.end_s:
            raise DataValidationError("chronological split intervals overlap")

    reference = attached.filter(pl.col("split").is_in(list(names[:-1])))
    evaluation = attached.filter(pl.col("split") == names[-1])
    reference_rntis = set(
        int(value)
        for value in reference.select(rnti_column)
        .drop_nulls()
        .unique()
        .to_series()
        .to_list()
    )
    evaluation_rntis = set(
        int(value)
        for value in evaluation.select(rnti_column)
        .drop_nulls()
        .unique()
        .to_series()
        .to_list()
    )

    resolved_source: str | None = None
    source_size: int | None = None
    source_digest: str | None = None
    if source_path is not None:
        source = Path(source_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        resolved_source = str(source)
        source_size = source.stat().st_size
        if include_source_sha256:
            source_digest = sha256_file(source)

    return SplitManifest(
        **{
            **provisional.to_dict(),
            "source_path": resolved_source,
            "source_size_bytes": source_size,
            "source_sha256": source_digest,
            "blocks": tuple(blocks),
            "splits": tuple(splits),
            "seen_numeric_rntis": tuple(sorted(evaluation_rntis & reference_rntis)),
            "unseen_numeric_rntis": tuple(sorted(evaluation_rntis - reference_rntis)),
        }
    )


def write_manifest(manifest: SplitManifest, output_path: str | Path) -> None:
    """Write derived JSON while explicitly refusing to overwrite the source CSV."""

    output = Path(output_path).resolve()
    if manifest.source_path is not None and output == Path(manifest.source_path).resolve():
        raise DataValidationError("refusing to overwrite the source CSV")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_manifest(path: str | Path) -> SplitManifest:
    return SplitManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def cli(argv: Sequence[str] | None = None) -> int:
    """Build a full-block manifest from a source CSV."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split-names", nargs="+", default=list(DEFAULT_SPLIT_NAMES))
    parser.add_argument("--cuts", nargs="+", default=list(DEFAULT_CUT_TIMES_UTC))
    parser.add_argument("--timestamp-scale", type=float, default=DEFAULT_TIMESTAMP_SCALE)
    parser.add_argument("--trace-gap-seconds", type=float, default=DEFAULT_TRACE_GAP_SECONDS)
    parser.add_argument("--max-block-seconds", type=float, default=DEFAULT_MAX_BLOCK_SECONDS)
    parser.add_argument("--lease-gap-seconds", type=float, default=DEFAULT_LEASE_GAP_SECONDS)
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args(argv)

    raw = load_semicolon_csv(args.source_csv)
    prepared = prepare_trace(
        raw,
        timestamp_scale=args.timestamp_scale,
        trace_gap_seconds=args.trace_gap_seconds,
        max_block_seconds=args.max_block_seconds,
        lease_gap_seconds=args.lease_gap_seconds,
    )
    manifest = build_split_manifest(
        prepared,
        split_names=args.split_names,
        cut_times=args.cuts,
        source_path=args.source_csv,
        include_source_sha256=not args.skip_sha256,
        timestamp_scale=args.timestamp_scale,
        trace_gap_seconds=args.trace_gap_seconds,
        max_block_seconds=args.max_block_seconds,
        lease_gap_seconds=args.lease_gap_seconds,
    )
    if args.output is None:
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    else:
        write_manifest(manifest, args.output)
        print(str(args.output.resolve()))
    return 0


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
