"""Reproduce the frozen block-bootstrap inference and robustness summaries.

This module performs no model fitting, threshold calibration, or policy
selection.  It consumes only the locked confirmatory contribution tables and
episode table, then resamples complete observable trace blocks with one shared
draw per comparison.  The primary delay gate uses the predeclared *median*
capped delay; mean delay is retained as a descriptive safeguard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from .bootstrap import (
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    GateRule,
    evaluate_gate_rules,
    paired_attack_exposure_bootstrap,
    paired_median_capped_delay_bootstrap,
    paired_policy_bootstrap,
    single_policy_benign_friction_bootstrap,
)
from .data import ATTACK_LABELS, TRACE_BLOCK_COLUMN, DataValidationError
from .evaluation import (
    ANY_ATTACK_TYPE,
    action_block_contributions,
    attack_block_contributions,
    attack_episode_table,
)
from .experiment import write_json_atomic


PROPOSED_DEFAULT = "proposed-template-047-B0.01"
REFERENCE_DEFAULT = "stateless-reference-0.01"
CONFIRMATORY_ROOT_DEFAULT = Path("artifacts/confirmatory")
CLUSTER_MAP_DEFAULT = Path("artifacts/block_cluster_candidates.csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _gate_record(
    *, metric: str, direction: str, threshold: float, endpoint: float | None
) -> dict[str, Any]:
    if endpoint is None:
        passed: bool | None = None
        status = "unavailable"
    elif direction == "max":
        passed = endpoint <= threshold
        status = "pass" if passed else "fail"
    elif direction == "min":
        passed = endpoint >= threshold
        status = "pass" if passed else "fail"
    else:
        raise ValueError("direction must be 'max' or 'min'")
    return {
        "metric": metric,
        "direction": direction,
        "threshold": float(threshold),
        "endpoint": endpoint,
        "passed": passed,
        "status": status,
    }


def _primary_report(
    action: pl.DataFrame,
    attack: pl.DataFrame,
    episodes: pl.DataFrame,
    *,
    block_ids: Sequence[Any],
    proposed: str,
    reference: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    paired = paired_policy_bootstrap(
        action,
        attack,
        proposed_candidate=proposed,
        reference_candidate=reference,
        attack_type=ANY_ATTACK_TYPE,
        trace_block_ids=block_ids,
        replicates=replicates,
        seed=seed,
    )
    median = paired_median_capped_delay_bootstrap(
        episodes,
        proposed_candidate=proposed,
        reference_candidate=reference,
        trace_block_ids=block_ids,
        attack_type=ANY_ATTACK_TYPE,
        replicates=replicates,
        seed=seed,
    )
    friction = single_policy_benign_friction_bootstrap(
        action,
        candidate=proposed,
        trace_block_ids=block_ids,
        replicates=replicates,
        seed=seed,
    )
    paired_gates = evaluate_gate_rules(
        paired,
        (
            GateRule("attack_exposure_difference", "max", 0.02),
            GateRule("episode_coverage_difference", "min", -0.05),
            GateRule("transition_rate_ratio", "max", 0.75),
        ),
    )
    gates = [item.__dict__ for item in paired_gates]
    gates.insert(
        2,
        _gate_record(
            metric="median_capped_delay_difference_s",
            direction="max",
            threshold=1.0,
            endpoint=median.interval.gate_endpoint,
        ),
    )
    return {
        "primary_gates": gates,
        "paired_additive_bootstrap": paired.to_dict(),
        "paired_median_delay_bootstrap": median.to_dict(),
        "proposed_friction_bootstrap": friction.to_dict(),
        "descriptive_safeguards": {
            "mean_capped_delay_difference_s": paired.metric(
                "mean_capped_delay_difference_s"
            ).to_dict(),
            "friction_budget": 0.01,
            "friction_point_within_budget": (
                friction.interval.estimate is not None
                and friction.interval.estimate <= 0.01
            ),
            "friction_one_sided_ucb_within_budget": (
                friction.upper_confidence_bound is not None
                and friction.upper_confidence_bound <= 0.01
            ),
        },
    }


def _cluster_contributions(
    frame: pl.DataFrame,
    mapping: pl.DataFrame,
    *,
    cluster_column: str,
) -> pl.DataFrame:
    """Aggregate additive contribution rows onto a label-blind cluster map."""

    required = {TRACE_BLOCK_COLUMN, cluster_column}
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise DataValidationError(f"cluster mapping missing columns: {missing}")
    if mapping.select(TRACE_BLOCK_COLUMN).n_unique() != mapping.height:
        raise DataValidationError("cluster mapping must have one row per trace block")
    joined = frame.join(
        mapping.select([TRACE_BLOCK_COLUMN, cluster_column]),
        on=TRACE_BLOCK_COLUMN,
        how="left",
        validate="m:1",
    )
    if joined[cluster_column].has_nulls():
        raise DataValidationError("a contribution block has no cluster mapping")
    keys = ["candidate"]
    for optional in ("family", "attack_type"):
        if optional in joined.columns:
            keys.append(optional)
    numeric = [
        name
        for name, dtype in joined.schema.items()
        if name not in {*keys, TRACE_BLOCK_COLUMN, cluster_column}
        and dtype.is_numeric()
    ]
    return (
        joined.group_by([*keys, cluster_column])
        .agg(pl.col(name).sum().alias(name) for name in numeric)
        .rename({cluster_column: TRACE_BLOCK_COLUMN})
        .sort(["candidate", TRACE_BLOCK_COLUMN])
    )


def _cluster_episodes(
    episodes: pl.DataFrame,
    mapping: pl.DataFrame,
    *,
    cluster_column: str,
) -> pl.DataFrame:
    clustered = episodes.join(
        mapping.select(
            pl.col(TRACE_BLOCK_COLUMN).alias("onset_block_id"), cluster_column
        ),
        on="onset_block_id",
        how="left",
        validate="m:1",
    )
    if clustered[cluster_column].has_nulls():
        raise DataValidationError("an episode onset has no cluster mapping")
    return clustered.drop("onset_block_id").rename(
        {cluster_column: "onset_block_id"}
    )


def _episode_point_summary(episodes: pl.DataFrame) -> list[dict[str, Any]]:
    subset = episodes.filter(pl.col("attack_type") == ANY_ATTACK_TYPE)
    if subset.is_empty():
        return []
    return (
        subset.group_by(["candidate", "family"])
        .agg(
            pl.len().alias("episode_count"),
            pl.col("covered").mean().alias("episode_coverage"),
            pl.col("capped_delay_s").mean().alias("mean_capped_delay_s"),
            pl.col("capped_delay_s").median().alias("median_capped_delay_s"),
            pl.col("attack_time_s").sum(),
            pl.col("exposed_time_s").sum(),
        )
        .with_columns(
            (pl.col("exposed_time_s") / pl.col("attack_time_s")).alias(
                "malicious_allow"
            )
        )
        .sort("candidate")
        .to_dicts()
    )


def run_inference(
    *,
    confirmatory_root: Path = CONFIRMATORY_ROOT_DEFAULT,
    cluster_mapping_path: Path = CLUSTER_MAP_DEFAULT,
    output_root: Path | None = None,
    proposed: str = PROPOSED_DEFAULT,
    reference: str = REFERENCE_DEFAULT,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
    hgb_root: Path | None = Path("artifacts/hgb_sensitivity"),
    alternate_timebase_root: Path | None = Path("artifacts/timebase_1e6"),
) -> dict[str, Any]:
    """Write primary, class-specific, and one-hour-cluster inference artifacts."""

    output_root = confirmatory_root if output_root is None else output_root
    paths = {
        "action": confirmatory_root / "action_block_contributions_v2.parquet",
        "attack": confirmatory_root / "attack_block_contributions_v2.parquet",
        "episodes": confirmatory_root / "attack_episodes_v2.parquet",
        "action_trace": confirmatory_root / "action_trace_v2.parquet",
        "strict_gap_episodes": confirmatory_root / "attack_episodes_v1.parquet",
        "run_manifest": confirmatory_root / "run_manifest_v2.json",
    }
    for path in (*paths.values(), cluster_mapping_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    action = pl.read_parquet(paths["action"])
    attack = pl.read_parquet(paths["attack"])
    episodes = pl.read_parquet(paths["episodes"])
    action_trace = pl.read_parquet(paths["action_trace"])
    block_ids = sorted(action[TRACE_BLOCK_COLUMN].unique().to_list())
    primary = _primary_report(
        action,
        attack,
        episodes,
        block_ids=block_ids,
        proposed=proposed,
        reference=reference,
        replicates=replicates,
        seed=seed,
    )

    per_attack: dict[str, Any] = {}
    for label in ATTACK_LABELS:
        report = paired_policy_bootstrap(
            action,
            attack,
            proposed_candidate=proposed,
            reference_candidate=reference,
            attack_type=label,
            trace_block_ids=block_ids,
            replicates=replicates,
            seed=seed,
        )
        class_exposure = paired_attack_exposure_bootstrap(
            attack,
            proposed_candidate=proposed,
            reference_candidate=reference,
            attack_type=label,
            trace_block_ids=block_ids,
            replicates=replicates,
            seed=seed,
        )
        per_attack[label] = {
            "attack_exposure_difference": class_exposure.interval.to_dict(),
            "episode_coverage_difference": report.metric(
                "episode_coverage_difference"
            ).to_dict(),
            "mean_capped_delay_difference_s": report.metric(
                "mean_capped_delay_difference_s"
            ).to_dict(),
            "shared_draw_fingerprint": class_exposure.shared_draw_fingerprint,
        }

    mapping = pl.read_csv(cluster_mapping_path, separator=";")
    cluster_column = "fixed_3600s_group"
    test_mapping = mapping.filter(pl.col(TRACE_BLOCK_COLUMN).is_in(block_ids))
    cluster_ids = sorted(test_mapping[cluster_column].unique().to_list())
    action_1h = _cluster_contributions(
        action, test_mapping, cluster_column=cluster_column
    )
    attack_1h = _cluster_contributions(
        attack, test_mapping, cluster_column=cluster_column
    )
    episodes_1h = _cluster_episodes(
        episodes, test_mapping, cluster_column=cluster_column
    )
    one_hour = _primary_report(
        action_1h,
        attack_1h,
        episodes_1h,
        block_ids=cluster_ids,
        proposed=proposed,
        reference=reference,
        replicates=replicates,
        seed=seed,
    )

    # Mandatory outcome-definition sensitivities.  Controller actions remain
    # untouched: only target rows/episode construction change here.
    mixed_attack_rows = int(
        action_trace.filter(pl.col("candidate") == proposed)
        .select(pl.col("mixed_attack_epoch").sum())
        .item()
    )
    exclusive_trace = action_trace.filter(~pl.col("mixed_attack_epoch"))
    exclusive_episodes = attack_episode_table(
        exclusive_trace,
        merge_gap_s=30.0,
        delay_cap_s=30.0,
    )
    strict_gap_episodes = pl.read_parquet(paths["strict_gap_episodes"])

    novelty_reports: dict[str, Any] = {}
    for novelty in ("seen", "unseen"):
        novelty_trace = action_trace.filter(pl.col("rnti_novelty") == novelty)
        novelty_episodes = attack_episode_table(
            novelty_trace,
            merge_gap_s=30.0,
            delay_cap_s=30.0,
        )
        novelty_action = action_block_contributions(novelty_trace)
        novelty_attack = attack_block_contributions(
            novelty_trace, novelty_episodes
        )
        novelty_reports[novelty] = {
            "numeric_rnti_count": int(novelty_trace["mac_rnti"].n_unique()),
            "epoch_count": int(
                novelty_trace.filter(pl.col("candidate") == proposed).height
            ),
            "inference": _primary_report(
                novelty_action,
                novelty_attack,
                novelty_episodes,
                block_ids=block_ids,
                proposed=proposed,
                reference=reference,
                replicates=replicates,
                seed=seed,
            ),
        }

    model_sensitivity: dict[str, Any] | None = None
    if hgb_root is not None:
        hgb_confirmatory = hgb_root / "confirmatory"
        hgb_paths = {
            "action": hgb_confirmatory / "action_block_contributions_v2.parquet",
            "attack": hgb_confirmatory / "attack_block_contributions_v2.parquet",
            "episodes": hgb_confirmatory / "attack_episodes_v2.parquet",
            "selection": hgb_root / "results/controller_strict_selection_tune_v1.parquet",
            "diagnostics": hgb_root.parent
            / "audits/risk_diagnostics_hgb_seed1729_v1.json",
        }
        if all(path.is_file() for path in hgb_paths.values()):
            hgb_action = pl.read_parquet(hgb_paths["action"])
            hgb_attack = pl.read_parquet(hgb_paths["attack"])
            hgb_episodes = pl.read_parquet(hgb_paths["episodes"])
            hgb_blocks = sorted(hgb_action[TRACE_BLOCK_COLUMN].unique().to_list())
            hgb_selection = pl.read_parquet(hgb_paths["selection"]).filter(
                pl.col("friction_budget") == 0.01
            )
            selected_ewma = "hgb-ewma-template-003-B0.01"
            hgb_reference = "hgb-stateless-reference-0.01"
            diagnostic_proposed = "hgb-proposed-template-010-B0.01-diagnostic"
            model_sensitivity = {
                "interpretation": (
                    "The constrained tune-only procedure selected EWMA for the "
                    "nonlinear scorer; the asymmetric proposed-family point was "
                    "coverage-inferior and remains diagnostic."
                ),
                "selection_rows_at_1pct": hgb_selection.to_dicts(),
                "selected_ewma_vs_stateless": _primary_report(
                    hgb_action,
                    hgb_attack,
                    hgb_episodes,
                    block_ids=hgb_blocks,
                    proposed=selected_ewma,
                    reference=hgb_reference,
                    replicates=replicates,
                    seed=seed,
                ),
                "diagnostic_asymmetric_vs_stateless": _primary_report(
                    hgb_action,
                    hgb_attack,
                    hgb_episodes,
                    block_ids=hgb_blocks,
                    proposed=diagnostic_proposed,
                    reference=hgb_reference,
                    replicates=replicates,
                    seed=seed,
                ),
                "input_sha256": {
                    name: _sha256(path) for name, path in hgb_paths.items()
                },
            }

    timebase_sensitivity: dict[str, Any] | None = None
    if alternate_timebase_root is not None:
        timebase_paths = {
            "data_summary": alternate_timebase_root / "audits/data_summary_v1.json",
            "risk_diagnostics": alternate_timebase_root
            / "audits/risk_diagnostics_logistic_seed1729_v1.json",
            "strict_selection": alternate_timebase_root
            / "results/controller_strict_selection_tune_v1.parquet",
            "matched_candidates": alternate_timebase_root
            / "results/controller_matched_candidates_tune_v1.parquet",
            "stateless_references": alternate_timebase_root
            / "results/stateless_references_tune_v1.parquet",
        }
        if all(path.is_file() for path in timebase_paths.values()):
            timebase_summary = _read_json_file(timebase_paths["data_summary"])
            timebase_diagnostics = _read_json_file(timebase_paths["risk_diagnostics"])
            timebase_selection = pl.read_parquet(
                timebase_paths["strict_selection"]
            ).filter(pl.col("friction_budget") == 0.01)
            timebase_candidates = pl.read_parquet(
                timebase_paths["matched_candidates"]
            ).filter(pl.col("friction_budget") == 0.01)
            timebase_reference = pl.read_parquet(
                timebase_paths["stateless_references"]
            ).filter(pl.col("friction_budget") == 0.01)
            timebase_sensitivity = {
                "interpretation": (
                    "A literal 10^6 divisor changes the inferred epoching and "
                    "leaves no budget-matched asymmetric proposal at 1%; no "
                    "held-out controller claim is made for this unsupported timebase."
                ),
                "timestamp_scale": timebase_summary["timestamp_scale"],
                "epochs": timebase_summary["epochs"],
                "trace_blocks": timebase_summary["trace_blocks"],
                "split_epochs": timebase_summary["split_epochs"],
                "test_auroc": timebase_diagnostics["partitions"]["test"]["auroc"],
                "feasible_candidate_rows_at_1pct": timebase_candidates.height,
                "feasible_proposed_rows_at_1pct": timebase_candidates.filter(
                    pl.col("family") == "proposed"
                ).height,
                "strict_selection_rows_at_1pct": timebase_selection.to_dicts(),
                "stateless_reference_at_1pct": timebase_reference.to_dicts(),
                "input_sha256": {
                    name: _sha256(path) for name, path in timebase_paths.items()
                },
            }

    payload = {
        "inference_version": 3,
        "selection_or_refitting_performed": False,
        "proposed_candidate": proposed,
        "reference_candidate": reference,
        "replicates": int(replicates),
        "seed": int(seed),
        "primary_cluster_unit": "label-blind observable 15-minute trace block",
        "primary_cluster_count": len(block_ids),
        "one_hour_sensitivity_cluster_count": len(cluster_ids),
        "input_sha256": {name: _sha256(path) for name, path in paths.items()},
        "cluster_mapping_sha256": _sha256(cluster_mapping_path),
        "primary": primary,
        "per_attack": per_attack,
        "one_hour_cluster_sensitivity": one_hour,
        "episode_definition_sensitivities": {
            "mixed_attack_epoch_exclusion": {
                "excluded_test_epochs": mixed_attack_rows,
                "controller_actions_replayed_or_changed": False,
                "summary": _episode_point_summary(exclusive_episodes),
            },
            "strict_zero_missing_gap": {
                "primary_missing_gap_s": 30.0,
                "sensitivity_missing_gap_s": 0.0,
                "summary": _episode_point_summary(strict_gap_episodes),
            },
        },
        "rnti_novelty_strata": novelty_reports,
        "risk_model_sensitivity": model_sensitivity,
        "timestamp_scale_sensitivity": timebase_sensitivity,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "inference_report_v3.json"
    write_json_atomic(output, payload)
    return {"output": str(output.resolve()), **payload}


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmatory-root", type=Path, default=CONFIRMATORY_ROOT_DEFAULT)
    parser.add_argument("--cluster-mapping", type=Path, default=CLUSTER_MAP_DEFAULT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--proposed", default=PROPOSED_DEFAULT)
    parser.add_argument("--reference", default=REFERENCE_DEFAULT)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--hgb-root",
        type=Path,
        default=Path("artifacts/hgb_sensitivity"),
        help="set to an absent path to omit nonlinear-model sensitivity",
    )
    parser.add_argument(
        "--alternate-timebase-root",
        type=Path,
        default=Path("artifacts/timebase_1e6"),
    )
    args = parser.parse_args(argv)
    result = run_inference(
        confirmatory_root=args.confirmatory_root,
        cluster_mapping_path=args.cluster_mapping,
        output_root=args.output_root,
        proposed=args.proposed,
        reference=args.reference,
        replicates=args.replicates,
        seed=args.seed,
        hgb_root=args.hgb_root,
        alternate_timebase_root=args.alternate_timebase_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
