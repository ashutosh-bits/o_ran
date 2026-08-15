"""Post-lock robustness analyses that never retune a controller threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from .confirmatory import (
    PROTOCOL_DEFAULT,
    SCORES_DEFAULT,
    _augment_action_metrics,
    _episode_summary,
    _load_protocol,
    _locked_specs,
)
from .data import (
    DEFAULT_MAX_BLOCK_SECONDS,
    DEFAULT_TIMESTAMP_SCALE,
    DEFAULT_TRACE_GAP_SECONDS,
    TRACE_BLOCK_COLUMN,
    load_semicolon_csv,
    prepare_trace,
    to_causal_epochs,
    validate_trace,
)
from .evaluation import ANY_ATTACK_TYPE, evaluate_locked_candidates, lock_candidate_specs
from .experiment import SOURCE_DEFAULT, write_json_atomic


LEASE_TIMEOUTS_DEFAULT = (5.0, 10.0, 30.0, 60.0, 300.0)


def _lease_epoch_mapping(prepared: pl.DataFrame) -> pl.DataFrame:
    return to_causal_epochs(
        prepared,
        epoch_seconds=1.0,
        feature_columns=[],
        context_columns=[],
        include_targets=False,
    ).select(
        [TRACE_BLOCK_COLUMN, "mac_rnti", "decision_time_s", "rnti_lease_id"]
    )


def run_lease_timeout_sensitivity(
    *,
    source: Path = SOURCE_DEFAULT,
    scores_path: Path = SCORES_DEFAULT,
    protocol_path: Path = PROTOCOL_DEFAULT,
    timeouts_s: Sequence[float] = LEASE_TIMEOUTS_DEFAULT,
    output_root: Path = Path("artifacts/sensitivities"),
) -> dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    locked = lock_candidate_specs(
        _locked_specs(protocol), selection_partition="controller_tune"
    )
    scores = pl.read_parquet(scores_path)
    key_columns = [TRACE_BLOCK_COLUMN, "mac_rnti", "decision_time_s"]
    if scores.select(key_columns).n_unique() != scores.height:
        raise ValueError("score-frame epoch keys are not unique")
    raw = load_semicolon_csv(source)
    validate_trace(raw)

    aggregate_frames: list[pl.DataFrame] = []
    episode_frames: list[pl.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for timeout in timeouts_s:
        timeout = float(timeout)
        if timeout <= 0:
            raise ValueError("lease timeouts must be positive")
        prepared = prepare_trace(
            raw,
            validate=False,
            timestamp_scale=DEFAULT_TIMESTAMP_SCALE,
            trace_gap_seconds=DEFAULT_TRACE_GAP_SECONDS,
            max_block_seconds=DEFAULT_MAX_BLOCK_SECONDS,
            lease_gap_seconds=timeout,
        )
        mapping = _lease_epoch_mapping(prepared)
        if mapping.select(key_columns).n_unique() != mapping.height:
            raise ValueError(f"alternative lease mapping is not one-to-one: {timeout}")
        replaced = (
            scores.drop("rnti_lease_id")
            .join(mapping, on=key_columns, how="inner", validate="1:1")
            .sort(["decision_time_s", TRACE_BLOCK_COLUMN, "rnti_lease_id"])
        )
        if replaced.height != scores.height:
            raise ValueError(
                f"lease timeout {timeout:g}s changed the observed epoch key set"
            )
        evaluation = replaced.filter(pl.col("split") == "test")
        bundle = evaluate_locked_candidates(
            evaluation,
            locked,
            merge_gap_s=timeout,
            delay_cap_s=30.0,
        )
        aggregate = _augment_action_metrics(
            bundle.action_trace, bundle.aggregate_metrics
        ).with_columns(
            pl.lit(timeout).alias("lease_timeout_s"),
            (
                1000.0 * pl.col("effective_transitions") / pl.col("epochs")
            ).alias("transitions_per_1000_observed_epochs"),
        )
        episodes = _episode_summary(bundle.episodes).filter(
            (pl.col("attack_type") == ANY_ATTACK_TYPE)
            & (pl.col("onset_stratum") == "__all__")
        ).with_columns(pl.lit(timeout).alias("lease_timeout_s"))
        aggregate_frames.append(aggregate)
        episode_frames.append(episodes)
        diagnostics.append(
            {
                "lease_timeout_s": timeout,
                "rnti_leases_all_splits": int(mapping["rnti_lease_id"].n_unique()),
                "test_rnti_leases": int(evaluation["rnti_lease_id"].n_unique()),
                "test_epochs": evaluation.height,
            }
        )

    aggregate_output = pl.concat(aggregate_frames, how="diagonal_relaxed").sort(
        ["lease_timeout_s", "candidate"]
    )
    episode_output = pl.concat(episode_frames, how="diagonal_relaxed").sort(
        ["lease_timeout_s", "candidate"]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_root / "lease_timeout_action_metrics_v1.parquet"
    episode_path = output_root / "lease_timeout_episode_metrics_v1.parquet"
    manifest_path = output_root / "lease_timeout_manifest_v1.json"
    aggregate_output.write_parquet(aggregate_path, compression="zstd")
    episode_output.write_parquet(episode_path, compression="zstd")
    manifest = {
        "candidate_lock_sha256": locked.sha256,
        "timeouts_s": [float(value) for value in timeouts_s],
        "thresholds_retuned": False,
        "risk_scores_refit": False,
        "diagnostics": diagnostics,
        "aggregate_path": str(aggregate_path.resolve()),
        "episode_path": str(episode_path.resolve()),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--scores", type=Path, default=SCORES_DEFAULT)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
    parser.add_argument(
        "--timeouts-s", type=float, nargs="+", default=list(LEASE_TIMEOUTS_DEFAULT)
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/sensitivities")
    )
    args = parser.parse_args(argv)
    output = run_lease_timeout_sensitivity(
        source=args.source,
        scores_path=args.scores,
        protocol_path=args.protocol,
        timeouts_s=args.timeouts_s,
        output_root=args.output_root,
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
