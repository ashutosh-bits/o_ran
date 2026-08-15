"""Fail-closed reproducibility audit for the frozen publication artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from .confirmatory import _augment_action_metrics
from .evaluation import aggregate_action_metrics
from .experiment import write_json_atomic
from .manifest import SplitManifest, read_manifest, sha256_file
from .model import FORBIDDEN_FEATURE_COLUMNS
from .policy_search import effective_states_one_epoch_later


ROOT = Path("artifacts")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _check(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def _audit_action_lag(action: pl.DataFrame) -> None:
    for candidate in action["candidate"].unique().sort().to_list():
        frame = action.filter(pl.col("candidate") == candidate)
        expected = effective_states_one_epoch_later(
            frame["decision_state"].to_numpy(),
            frame["mac_rnti"].to_numpy(),
            frame["rnti_lease_id"].to_numpy(),
        )
        observed = frame["effective_state"].to_numpy()
        if not np.array_equal(expected, observed):
            raise AssertionError(f"one-epoch lag mismatch for {candidate}")


def _audit_split_manifest(manifest: SplitManifest) -> None:
    block_sets = [set(item.block_ids) for item in manifest.splits]
    if any(left & right for i, left in enumerate(block_sets) for right in block_sets[i + 1 :]):
        raise AssertionError("split manifest contains overlapping block IDs")
    if set().union(*block_sets) != {item.trace_block_id for item in manifest.blocks}:
        raise AssertionError("split manifest does not cover every block exactly once")
    for earlier, later in zip(manifest.splits, manifest.splits[1:]):
        if later.start_s < earlier.end_s:
            raise AssertionError("split intervals overlap chronologically")


def run_reproducibility_audit(
    *,
    source: Path = Path("/nobackup/ashukuma/o_ran/dtst.csv"),
    artifact_root: Path = ROOT,
    protocol_path: Path = Path("configs/study_protocol_v1_locked.json"),
    output_path: Path = Path("reports/reproducibility_audit.json"),
) -> dict[str, Any]:
    """Verify provenance, split separation, action lag, and derived metrics."""

    paths = {
        "split_manifest": artifact_root / "manifests/split_manifest_v1.json",
        "scores": artifact_root / "results/scores_logistic_seed1729_v1.parquet",
        "risk_diagnostics": artifact_root / "audits/risk_diagnostics_logistic_seed1729_v1.json",
        "strict_selection": artifact_root / "results/controller_strict_selection_tune_v1.parquet",
        "candidate_lock": artifact_root / "confirmatory/candidate_lock_v2.json",
        "run_manifest": artifact_root / "confirmatory/run_manifest_v2.json",
        "action_trace": artifact_root / "confirmatory/action_trace_v2.parquet",
        "aggregate": artifact_root / "confirmatory/aggregate_metrics_v2.parquet",
        "episodes": artifact_root / "confirmatory/attack_episodes_v2.parquet",
        "inference": artifact_root / "confirmatory/inference_report_v3.json",
    }
    for path in (source, protocol_path, *paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    checks: list[str] = []
    protocol = _read_json(protocol_path)
    split_manifest = read_manifest(paths["split_manifest"])
    run_manifest = _read_json(paths["run_manifest"])
    risk = _read_json(paths["risk_diagnostics"])
    lock = _read_json(paths["candidate_lock"])
    inference = _read_json(paths["inference"])

    source_digest = sha256_file(source)
    _check(source_digest == split_manifest.source_sha256, "source SHA-256 matches split manifest", checks)
    _check(source_digest == protocol["data"]["sha256"], "source SHA-256 matches locked protocol", checks)
    _check(_sha(protocol_path) == run_manifest["protocol_sha256"], "protocol SHA-256 matches confirmatory run", checks)
    score_digest = _sha(paths["scores"])
    _check(score_digest == protocol["risk_model"]["input_scores_sha256"], "score SHA-256 matches locked protocol", checks)
    _check(score_digest == run_manifest["score_sha256"], "score SHA-256 matches confirmatory run", checks)
    _check(lock["sha256"] == run_manifest["candidate_lock_sha256"], "candidate lock SHA-256 matches confirmatory run", checks)
    canonical_digest = hashlib.sha256(lock["canonical_json"].encode("utf-8")).hexdigest()
    _check(canonical_digest == lock["sha256"], "candidate canonical JSON hashes to its lock", checks)

    _audit_split_manifest(split_manifest)
    checks.append("trace-block split sets are disjoint, complete, and chronological")
    forbidden = set(risk["feature_columns"]) & set(FORBIDDEN_FEATURE_COLUMNS)
    _check(not forbidden, "risk feature list contains no explicitly forbidden metadata", checks)

    scores = pl.read_parquet(paths["scores"])
    key = ["trace_block_id", "rnti_lease_id", "mac_rnti", "decision_time_s"]
    _check(scores.select(key).n_unique() == scores.height, "score epoch keys are unique", checks)
    split_blocks = {
        name: set(scores.filter(pl.col("split") == name)["trace_block_id"].unique().to_list())
        for name in scores["split"].unique().to_list()
    }
    _check(
        not any(left & right for i, left in enumerate(split_blocks.values()) for right in list(split_blocks.values())[i + 1 :]),
        "score partitions share no trace blocks",
        checks,
    )

    action = pl.read_parquet(paths["action_trace"])
    candidates = action["candidate"].unique().sort().to_list()
    evaluation_epochs = int(run_manifest["evaluation_epochs"])
    _check(action.height == evaluation_epochs * len(candidates), "action table is complete for every candidate and evaluation epoch", checks)
    _check(
        action.group_by(["candidate", "_evaluation_row"]).len().select((pl.col("len") == 1).all()).item(),
        "action candidate/evaluation-row keys are unique",
        checks,
    )
    _check(set(action["effective_state"].unique().to_list()) <= {0, 1, 2}, "all actions are ALLOW/RESTRICT/ISOLATE", checks)
    _check(action["candidate_lock_sha256"].n_unique() == 1 and action["candidate_lock_sha256"][0] == lock["sha256"], "every action row carries the candidate lock", checks)
    _audit_action_lag(action)
    checks.append("effective actions exactly reproduce the one-epoch causal lag")

    recomputed = _augment_action_metrics(action, aggregate_action_metrics(action)).sort("candidate")
    saved = pl.read_parquet(paths["aggregate"]).sort("candidate")
    _check(recomputed.columns == saved.columns and recomputed.equals(saved, null_equal=True), "saved aggregate metrics exactly recompute from action rows", checks)

    episodes = pl.read_parquet(paths["episodes"])
    all_attack = episodes.filter(pl.col("attack_type") == "__any_attack__")
    episode_sets = {
        candidate: set(all_attack.filter(pl.col("candidate") == candidate)["episode_key"].to_list())
        for candidate in candidates
    }
    first_set = episode_sets[candidates[0]]
    _check(all(values == first_set for values in episode_sets.values()), "ground-truth episode keys are identical across policies", checks)

    selected = pl.read_parquet(paths["strict_selection"])
    primary = selected.filter(
        (pl.col("friction_budget") == 0.01) & (pl.col("family") == "proposed")
    )
    _check(
        primary.height == 1
        and primary["status"][0] == "selected"
        and float(primary["transition_reduction"][0]) >= 0.25,
        "exactly one successful proposed policy was locked at the 1% tuning budget",
        checks,
    )
    gates = inference["primary"]["primary_gates"]
    _check(len(gates) == 4 and all(item["passed"] is True for item in gates), "all four predeclared held-out controller gates pass", checks)

    artifact_hashes = {name: _sha(path) for name, path in paths.items()}
    payload = {
        "audit_version": 1,
        "status": "pass",
        "checks": checks,
        "check_count": len(checks),
        "source_sha256": source_digest,
        "artifact_sha256": artifact_hashes,
        "evaluation_epochs": evaluation_epochs,
        "evaluation_candidates": candidates,
        "evaluation_trace_blocks": int(run_manifest["evaluation_trace_blocks"]),
        "evaluation_rnti_leases": int(run_manifest["evaluation_rnti_leases"]),
        "note": "A passing integrity audit does not validate causal attack prevention, durable identity, or external generalization.",
    }
    write_json_atomic(output_path, payload)
    return payload


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/nobackup/ashukuma/o_ran/dtst.csv"))
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, default=Path("configs/study_protocol_v1_locked.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/reproducibility_audit.json"))
    args = parser.parse_args(argv)
    payload = run_reproducibility_audit(
        source=args.source,
        artifact_root=args.artifact_root,
        protocol_path=args.protocol,
        output_path=args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
