"""Execute the locked chronological controller-policy evaluation exactly once.

This driver performs no policy search and never recalibrates a threshold.  It
loads the controller specifications from the versioned protocol, canonicalizes
and hashes them, filters the frozen score table to the declared evaluation
partition, and emits machine-readable action, episode, and block-contribution
artifacts.  Interpretation and statistical inference are deliberately separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from .controller import AccessState
from .evaluation import (
    ANY_ATTACK_TYPE,
    EVALUATION_ROW_COLUMN,
    aggregate_action_metrics,
    evaluate_locked_candidates,
    lock_candidate_specs,
)
from .experiment import ARTIFACT_ROOT_DEFAULT, write_json_atomic
from .policy_search import CandidateSpec


PROTOCOL_DEFAULT = Path("configs/study_protocol_v1_locked.json")
SCORES_DEFAULT = Path("artifacts/results/scores_logistic_seed1729_v1.parquet")
EVALUATION_SPLIT = "test"
PRIMARY_EPISODE_MERGE_GAP_S = 30.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol.get("protocol_status") != "locked_before_formal_controller_holdout_replay":
        raise ValueError("confirmatory execution requires the locked v1 protocol")
    if protocol["policy_selection"]["primary_friction_budget"] != 0.01:
        raise ValueError("unexpected primary friction budget in locked protocol")
    return protocol


def _locked_specs(protocol: dict[str, Any]) -> list[CandidateSpec]:
    records: list[CandidateSpec] = []
    family_aliases = {
        "stateless_reference": "stateless_reference",
        "proposed": "proposed",
        "ewma": "ewma",
        "n_report": "n_report",
        "symmetric_hysteresis": "symmetric_hysteresis",
    }
    policies = protocol["locked_primary_policies"]
    for key in (
        "stateless_reference",
        "proposed",
        "ewma",
        "n_report",
        "symmetric_hysteresis",
    ):
        item = policies[key]
        records.append(
            CandidateSpec(
                candidate=str(item["candidate"]),
                family=family_aliases[key],
                controller=str(item["controller"]),
                parameters=dict(item["parameters"]),
            )
        )
    return records


def _augment_action_metrics(
    action_trace: pl.DataFrame,
    aggregate: pl.DataFrame,
) -> pl.DataFrame:
    hard_rows: list[dict[str, Any]] = []
    for group in action_trace.partition_by("candidate", maintain_order=False):
        duration = group["epoch_seconds"].to_numpy().astype(float)
        attack = group["is_attack_epoch"].to_numpy().astype(bool)
        state = group["effective_state"].to_numpy().astype(np.int8)
        benign = ~attack
        benign_time = float(duration[benign].sum())
        attack_time = float(duration[attack].sum())
        hard_rows.append(
            {
                "candidate": group["candidate"][0],
                "benign_isolation": (
                    float(duration[benign & (state == int(AccessState.ISOLATE))].sum())
                    / benign_time
                ),
                "malicious_allow": (
                    float(duration[attack & (state == int(AccessState.ALLOW))].sum())
                    / attack_time
                ),
                "malicious_not_isolated": (
                    float(duration[attack & (state != int(AccessState.ISOLATE))].sum())
                    / attack_time
                ),
            }
        )
    hard = pl.DataFrame(hard_rows, strict=False)
    return (
        aggregate.join(hard, on="candidate", how="left")
        .with_columns(
            (pl.col("transitions_per_lease_hour") / 60.0).alias(
                "transitions_per_lease_minute"
            ),
            (1.0 - pl.col("attack_time_containment")).alias("malicious_allow_check"),
        )
        .sort("candidate")
    )


def _episode_summary(episodes: pl.DataFrame) -> pl.DataFrame:
    if episodes.is_empty():
        return pl.DataFrame()
    def summarize(group_columns: list[str], *, all_strata: bool) -> pl.DataFrame:
        frame = episodes.group_by(group_columns).agg(
            pl.len().alias("episode_count"),
            pl.col("covered").sum().alias("covered_episode_count"),
            pl.col("covered").mean().alias("episode_coverage"),
            pl.col("reactively_covered").mean().alias("reactive_coverage"),
            pl.col("pre_contained_at_onset").sum().alias(
                "pre_contained_episode_count"
            ),
            pl.col("attack_time_s").sum(),
            pl.col("contained_time_s").sum(),
            pl.col("exposed_time_s").sum(),
            pl.col("capped_delay_s").mean().alias("mean_capped_delay_s"),
            pl.col("capped_delay_s").median().alias("median_capped_delay_s"),
            pl.col("delay_s").drop_nans().median().alias("median_detected_delay_s"),
            pl.col("exposure_before_containment_s").mean().alias(
                "mean_exposure_before_containment_s"
            ),
        )
        if all_strata:
            frame = frame.with_columns(pl.lit("__all__").alias("onset_stratum"))
        return frame.with_columns(
            (pl.col("contained_time_s") / pl.col("attack_time_s")).alias(
                "attack_time_containment"
            ),
            (pl.col("exposed_time_s") / pl.col("attack_time_s")).alias(
                "malicious_allow"
            ),
        )
    return pl.concat(
        [
            summarize(["candidate", "family", "attack_type"], all_strata=True),
            summarize(
                ["candidate", "family", "attack_type", "onset_stratum"],
                all_strata=False,
            ),
        ],
        how="diagonal_relaxed",
    ).sort(["candidate", "attack_type", "onset_stratum"])


def _stratified_action_metrics(action_trace: pl.DataFrame) -> pl.DataFrame:
    contexts: list[tuple[str, pl.Expr]] = [("all", pl.lit(True))]
    if "rnti_novelty" in action_trace.columns:
        for novelty in ("seen", "unseen"):
            contexts.append((f"rnti_{novelty}", pl.col("rnti_novelty") == novelty))
    if "mob_pattern" in action_trace.columns:
        for mobility in sorted(action_trace["mob_pattern"].drop_nulls().unique().to_list()):
            contexts.append((f"mobility_{mobility}", pl.col("mob_pattern") == mobility))
    contexts.append(("exclude_mixed_labels", ~pl.col("mixed_label_epoch")))

    frames: list[pl.DataFrame] = []
    for stratum, condition in contexts:
        subset = action_trace.filter(condition)
        if subset.is_empty() or subset["is_attack_epoch"].all() or not subset[
            "is_attack_epoch"
        ].any():
            continue
        aggregate = aggregate_action_metrics(subset)
        frames.append(
            _augment_action_metrics(subset, aggregate).with_columns(
                pl.lit(stratum).alias("stratum")
            )
        )
    return pl.concat(frames, how="diagonal_relaxed").sort(["stratum", "candidate"])


def run_locked_evaluation(
    *,
    protocol_path: Path = PROTOCOL_DEFAULT,
    score_path: Path = SCORES_DEFAULT,
    artifact_root: Path = ARTIFACT_ROOT_DEFAULT,
    episode_merge_gap_s: float = PRIMARY_EPISODE_MERGE_GAP_S,
    artifact_version: str = "v2",
) -> dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    expected_score_hash = protocol["risk_model"]["input_scores_sha256"]
    observed_score_hash = _sha256_file(score_path)
    if observed_score_hash != expected_score_hash:
        raise ValueError("score-frame digest differs from the locked protocol")

    full = pl.read_parquet(score_path)
    if EVALUATION_SPLIT not in full["split"].unique().to_list():
        raise ValueError("locked evaluation partition is absent")
    evaluation = full.filter(pl.col("split") == EVALUATION_SPLIT)
    if set(evaluation["split"].unique().to_list()) != {EVALUATION_SPLIT}:
        raise AssertionError("evaluation filter admitted another partition")

    locked = lock_candidate_specs(
        _locked_specs(protocol), selection_partition="controller_tune"
    )
    bundle = evaluate_locked_candidates(
        evaluation,
        locked,
        merge_gap_s=episode_merge_gap_s,
        delay_cap_s=30.0,
        containment_state=AccessState.RESTRICT,
    )

    # Context is joined only after replay and target annotation.  It cannot
    # influence any controller state or action.
    context_columns = [
        EVALUATION_ROW_COLUMN,
        "rnti_novelty",
        "rnti_seen_pretest",
        "mob_pattern",
        "mob_pattern_n_unique",
        "id_ue",
        "id_ue_n_unique",
    ]
    indexed_context = evaluation.with_row_index(EVALUATION_ROW_COLUMN).select(
        [column for column in context_columns if column in evaluation.columns or column == EVALUATION_ROW_COLUMN]
    )
    action_trace = bundle.action_trace.join(
        indexed_context, on=EVALUATION_ROW_COLUMN, how="left"
    )
    aggregate = _augment_action_metrics(action_trace, bundle.aggregate_metrics)
    episode_summary = _episode_summary(bundle.episodes)
    stratified = _stratified_action_metrics(action_trace)

    results_dir = artifact_root / "confirmatory"
    results_dir.mkdir(parents=True, exist_ok=True)
    if not artifact_version or any(character in artifact_version for character in "/\\"):
        raise ValueError("artifact_version must be a non-empty filename token")
    paths = {
        "candidate_lock": results_dir / f"candidate_lock_{artifact_version}.json",
        "action_trace": results_dir / f"action_trace_{artifact_version}.parquet",
        "aggregate": results_dir / f"aggregate_metrics_{artifact_version}.parquet",
        "stratified": results_dir / f"stratified_action_metrics_{artifact_version}.parquet",
        "episodes": results_dir / f"attack_episodes_{artifact_version}.parquet",
        "episode_summary": results_dir / f"attack_episode_summary_{artifact_version}.parquet",
        "episode_strata": results_dir / f"attack_episode_strata_{artifact_version}.parquet",
        "action_blocks": results_dir / f"action_block_contributions_{artifact_version}.parquet",
        "attack_blocks": results_dir / f"attack_block_contributions_{artifact_version}.parquet",
        "run_manifest": results_dir / f"run_manifest_{artifact_version}.json",
    }
    write_json_atomic(paths["candidate_lock"], locked.to_dict())
    action_trace.write_parquet(paths["action_trace"], compression="zstd")
    aggregate.write_parquet(paths["aggregate"], compression="zstd")
    stratified.write_parquet(paths["stratified"], compression="zstd")
    bundle.episodes.write_parquet(paths["episodes"], compression="zstd")
    episode_summary.write_parquet(paths["episode_summary"], compression="zstd")
    bundle.episode_strata.write_parquet(paths["episode_strata"], compression="zstd")
    bundle.action_block_contributions.write_parquet(
        paths["action_blocks"], compression="zstd"
    )
    bundle.attack_block_contributions.write_parquet(
        paths["attack_blocks"], compression="zstd"
    )
    manifest = {
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256_file(protocol_path),
        "score_path": str(score_path.resolve()),
        "score_sha256": observed_score_hash,
        "evaluation_split": EVALUATION_SPLIT,
        "evaluation_epochs": evaluation.height,
        "evaluation_trace_blocks": int(evaluation["trace_block_id"].n_unique()),
        "evaluation_rnti_leases": int(evaluation["rnti_lease_id"].n_unique()),
        "candidate_count": len(locked.specs),
        "candidate_lock_sha256": locked.sha256,
        "one_epoch_action_lag": True,
        "episode_merge_gap_s": episode_merge_gap_s,
        "artifact_version": artifact_version,
        "delay_cap_s": 30.0,
        "output_paths": {key: str(value.resolve()) for key, value in paths.items()},
    }
    write_json_atomic(paths["run_manifest"], manifest)
    return manifest


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
    parser.add_argument("--scores", type=Path, default=SCORES_DEFAULT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT_DEFAULT)
    parser.add_argument(
        "--episode-merge-gap-s",
        type=float,
        default=PRIMARY_EPISODE_MERGE_GAP_S,
    )
    parser.add_argument("--artifact-version", default="v2")
    args = parser.parse_args(argv)
    output = run_locked_evaluation(
        protocol_path=args.protocol,
        score_path=args.scores,
        artifact_root=args.artifact_root,
        episode_merge_gap_s=args.episode_merge_gap_s,
        artifact_version=args.artifact_version,
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
