"""End-to-end reproducible experiment driver.

The driver is deliberately staged.  ``prepare`` derives immutable epochs and a
split manifest; ``fit`` trains/calibrates a fixed nuisance risk model and emits
scores.  Controller search and evaluation are separate commands so a test split
cannot accidentally influence model or policy development.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from .data import (
    ATTACK_LABELS,
    DEFAULT_LEASE_GAP_SECONDS,
    DEFAULT_MAX_BLOCK_SECONDS,
    DEFAULT_TIMESTAMP_SCALE,
    DEFAULT_TRACE_GAP_SECONDS,
    TRACE_BLOCK_COLUMN,
    audit_trace,
    load_semicolon_csv,
    prepare_trace,
    to_causal_epochs,
    validate_trace,
)
from .manifest import (
    DEFAULT_CUT_TIMES_UTC,
    DEFAULT_SPLIT_NAMES,
    annotate_rnti_novelty,
    attach_split_labels,
    build_split_manifest,
    read_manifest,
    write_manifest,
)
from .model import (
    DEFAULT_KPI_FEATURES,
    diagnose_model,
    fit_platt_calibrator,
    fit_primary_logistic,
    fit_shallow_hgb_sensitivity,
    save_model_bundle,
    save_score_frame,
    score_frame,
)


SOURCE_DEFAULT = Path("/nobackup/ashukuma/o_ran/dtst.csv")
ARTIFACT_ROOT_DEFAULT = Path("artifacts")
PRIMARY_FEATURES = tuple(
    feature for feature in DEFAULT_KPI_FEATURES if feature != "samples_in_epoch"
)
SCORE_PASSTHROUGH = (
    TRACE_BLOCK_COLUMN,
    "rnti_lease_id",
    "mac_rnti",
    "decision_time_s",
    "epoch_seconds",
    "samples_in_epoch",
    "is_attack_epoch",
    "labels_in_epoch",
    "split",
    "rnti_novelty",
    "rnti_seen_pretest",
    "mob_pattern",
    "mob_pattern_n_unique",
    "id_ue",
    "id_ue_n_unique",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
            mode="w",
            encoding="utf-8",
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def _attack_columns(frame: pl.DataFrame) -> pl.DataFrame:
    expressions = []
    for label in ATTACK_LABELS:
        slug = label.lower().replace("-", "_")
        expressions.append(
            pl.col("labels_in_epoch").list.contains(label).alias(f"attack_{slug}")
        )
    attack_count = pl.sum_horizontal(expressions).cast(pl.Int8)
    return frame.with_columns(expressions).with_columns(
        attack_count.alias("attack_types_in_epoch"),
        (attack_count > 1).alias("mixed_attack_epoch"),
    )


def prepare_artifacts(
    *,
    source: Path = SOURCE_DEFAULT,
    artifact_root: Path = ARTIFACT_ROOT_DEFAULT,
    timestamp_scale: float = DEFAULT_TIMESTAMP_SCALE,
    trace_gap_seconds: float = DEFAULT_TRACE_GAP_SECONDS,
    max_block_seconds: float = DEFAULT_MAX_BLOCK_SECONDS,
    lease_gap_seconds: float = DEFAULT_LEASE_GAP_SECONDS,
    epoch_seconds: float = 1.0,
    cut_times: Sequence[str | float | int] = DEFAULT_CUT_TIMES_UTC,
) -> dict[str, Any]:
    raw = load_semicolon_csv(source)
    audit = validate_trace(raw)
    prepared = prepare_trace(
        raw,
        validate=False,
        timestamp_scale=timestamp_scale,
        trace_gap_seconds=trace_gap_seconds,
        max_block_seconds=max_block_seconds,
        lease_gap_seconds=lease_gap_seconds,
    )
    manifest = build_split_manifest(
        prepared,
        split_names=DEFAULT_SPLIT_NAMES,
        cut_times=cut_times,
        source_path=source,
        timestamp_scale=timestamp_scale,
        trace_gap_seconds=trace_gap_seconds,
        max_block_seconds=max_block_seconds,
        lease_gap_seconds=lease_gap_seconds,
    )
    manifest_path = artifact_root / "manifests" / "split_manifest_v1.json"
    write_manifest(manifest, manifest_path)

    epochs = to_causal_epochs(prepared, epoch_seconds=epoch_seconds)
    epochs = attach_split_labels(epochs, manifest)
    epochs = annotate_rnti_novelty(
        epochs,
        reference_splits=manifest.novelty_reference_splits,
        evaluation_split=manifest.evaluation_split,
    )
    epochs = _attack_columns(epochs)
    epochs_path = artifact_root / "epochs" / "epochs_1s_v1.parquet"
    epochs_path.parent.mkdir(parents=True, exist_ok=True)
    epochs.write_parquet(epochs_path, compression="zstd", statistics=True)

    split_rows = {
        str(row["split"]): int(row["len"])
        for row in epochs.group_by("split").len().sort("split").iter_rows(named=True)
    }
    split_attack_prevalence = {
        str(row["split"]): float(row["attack_prevalence"])
        for row in epochs.group_by("split")
        .agg(pl.col("is_attack_epoch").mean().alias("attack_prevalence"))
        .sort("split")
        .iter_rows(named=True)
    }
    summary = {
        "source": str(source.resolve()),
        "audit": audit.to_dict(),
        "timestamp_scale": timestamp_scale,
        "trace_gap_seconds": trace_gap_seconds,
        "max_block_seconds": max_block_seconds,
        "lease_gap_seconds": lease_gap_seconds,
        "epoch_seconds": epoch_seconds,
        "prepared_rows": prepared.height,
        "trace_blocks": int(prepared.select(pl.col(TRACE_BLOCK_COLUMN).n_unique()).item()),
        "rnti_leases": int(prepared.select(pl.col("rnti_lease_id").n_unique()).item()),
        "epochs": epochs.height,
        "split_epochs": split_rows,
        "split_attack_prevalence": split_attack_prevalence,
        "mixed_attack_epochs": int(epochs.select(pl.col("mixed_attack_epoch").sum()).item()),
        "test_numeric_rntis": int(
            epochs.filter(pl.col("split") == "test")
            .select(pl.col("mac_rnti").n_unique())
            .item()
        ),
        "test_unseen_numeric_rntis": int(
            epochs.filter(
                (pl.col("split") == "test") & (pl.col("rnti_novelty") == "unseen")
            )
            .select(pl.col("mac_rnti").n_unique())
            .item()
        ),
        "epochs_path": str(epochs_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
    }
    write_json_atomic(artifact_root / "audits" / "data_summary_v1.json", summary)
    return summary


def _diagnostics_by_split(bundle, epochs: pl.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in DEFAULT_SPLIT_NAMES:
        subset = epochs.filter(pl.col("split") == split)
        output[split] = diagnose_model(bundle, subset, "is_attack_epoch")
    for novelty in ("seen", "unseen"):
        subset = epochs.filter(
            (pl.col("split") == "test") & (pl.col("rnti_novelty") == novelty)
        )
        if subset.height:
            output[f"test_{novelty}_rnti"] = diagnose_model(
                bundle, subset, "is_attack_epoch"
            )
    return output


def fit_and_score(
    *,
    artifact_root: Path = ARTIFACT_ROOT_DEFAULT,
    model_kind: str = "logistic",
    seed: int = 1729,
) -> dict[str, Any]:
    epochs_path = artifact_root / "epochs" / "epochs_1s_v1.parquet"
    epochs = pl.read_parquet(epochs_path)
    train = epochs.filter(pl.col("split") == "train")
    calibration = epochs.filter(pl.col("split") == "calibration")
    if model_kind == "logistic":
        bundle = fit_primary_logistic(
            train,
            "is_attack_epoch",
            feature_columns=PRIMARY_FEATURES,
            training_partition="train",
            balance_classes=True,
            seed=seed,
            regularization_c=1.0,
        )
        stem = "logistic"
    elif model_kind == "hgb":
        bundle = fit_shallow_hgb_sensitivity(
            train,
            "is_attack_epoch",
            feature_columns=PRIMARY_FEATURES,
            training_partition="train",
            balance_classes=True,
            seed=seed,
            hgb_max_depth=3,
            hgb_max_leaf_nodes=7,
            hgb_learning_rate=0.05,
            hgb_max_iter=150,
            hgb_l2_regularization=1.0,
        )
        stem = "hgb"
    else:
        raise ValueError("model_kind must be 'logistic' or 'hgb'")
    bundle = fit_platt_calibrator(
        bundle,
        calibration,
        "is_attack_epoch",
        calibration_partition="calibration",
        controller_tuning_partition="controller_tune",
        balance_classes=False,
        seed=seed,
    )
    model_path = save_model_bundle(
        bundle, artifact_root / "models" / f"{stem}_seed{seed}_v1.joblib"
    )
    scores = score_frame(bundle, epochs, passthrough_columns=SCORE_PASSTHROUGH)
    score_path = save_score_frame(
        scores, artifact_root / "results" / f"scores_{stem}_seed{seed}_v1.parquet"
    )
    diagnostics = {
        "model_kind": model_kind,
        "seed": seed,
        "feature_columns": list(PRIMARY_FEATURES),
        "training_rows": train.height,
        "calibration_rows": calibration.height,
        "partitions": _diagnostics_by_split(bundle, epochs),
        "model_path": str(model_path.resolve()),
        "score_path": str(score_path.resolve()),
    }
    write_json_atomic(
        artifact_root / "audits" / f"risk_diagnostics_{stem}_seed{seed}_v1.json",
        diagnostics,
    )
    return diagnostics


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    prepare_parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT_DEFAULT)
    prepare_parser.add_argument("--timestamp-scale", type=float, default=DEFAULT_TIMESTAMP_SCALE)
    prepare_parser.add_argument("--epoch-seconds", type=float, default=1.0)
    prepare_parser.add_argument(
        "--cuts",
        nargs=3,
        default=list(DEFAULT_CUT_TIMES_UTC),
        help="three chronological split cuts as ISO timestamps or Unix seconds",
    )

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT_DEFAULT)
    fit_parser.add_argument("--model-kind", choices=("logistic", "hgb"), default="logistic")
    fit_parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        output = prepare_artifacts(
            source=args.source,
            artifact_root=args.artifact_root,
            timestamp_scale=args.timestamp_scale,
            epoch_seconds=args.epoch_seconds,
            cut_times=args.cuts,
        )
    else:
        output = fit_and_score(
            artifact_root=args.artifact_root,
            model_kind=args.model_kind,
            seed=args.seed,
        )
    print(json.dumps(output, indent=2, sort_keys=True, default=_json_default))
    return 0


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
