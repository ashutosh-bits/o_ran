"""Reproduce exact stateless references and tune-only strict policy selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from .experiment import ARTIFACT_ROOT_DEFAULT
from .matched_search import TUNING_SPLIT
from .selection import (
    SelectionTolerances,
    calibrate_stateless_references,
    select_against_stateless_reference,
)


def run_strict_selection(
    *,
    score_path: Path = Path("artifacts/results/scores_logistic_seed1729_v1.parquet"),
    candidate_metrics_path: Path = Path(
        "artifacts/results/controller_matched_candidates_tune_v1.parquet"
    ),
    artifact_root: Path = ARTIFACT_ROOT_DEFAULT,
    budgets: Sequence[float] = (0.001, 0.005, 0.01, 0.02, 0.05),
    tolerances: SelectionTolerances = SelectionTolerances(),
) -> dict[str, Any]:
    """Calibrate references and apply the frozen constrained selection rule."""

    scores = pl.read_parquet(score_path)
    tune = scores.filter(pl.col("split") == TUNING_SPLIT)
    if tune.is_empty() or set(tune["split"].unique().to_list()) != {TUNING_SPLIT}:
        raise ValueError("score table has no isolated controller_tune partition")
    references = calibrate_stateless_references(
        tune["risk_score"].to_numpy(),
        tune["decision_time_s"].to_numpy(),
        tune["mac_rnti"].to_numpy(),
        tune["rnti_lease_id"].to_numpy(),
        tune["is_attack_epoch"].to_numpy(),
        tune["epoch_seconds"].to_numpy(),
        budgets,
        split=tune["split"].to_numpy(),
        tolerances=tolerances,
    )
    candidates = pl.read_parquet(candidate_metrics_path)
    strict = select_against_stateless_reference(
        candidates,
        references,
        tolerances=tolerances,
    )
    output = artifact_root / "results"
    output.mkdir(parents=True, exist_ok=True)
    reference_path = output / "stateless_references_tune_v1.parquet"
    strict_path = output / "controller_strict_selection_tune_v1.parquet"
    references.write_parquet(reference_path, compression="zstd")
    strict.write_parquet(strict_path, compression="zstd")
    return {
        "selection_partition": TUNING_SPLIT,
        "tuning_epochs": tune.height,
        "budgets": [float(value) for value in budgets],
        "reference_rows": references.height,
        "selection_rows": strict.height,
        "selected_rows": strict.filter(pl.col("status") == "selected").height,
        "reference_path": str(reference_path.resolve()),
        "strict_selection_path": str(strict_path.resolve()),
    }


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("artifacts/results/scores_logistic_seed1729_v1.parquet"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("artifacts/results/controller_matched_candidates_tune_v1.parquet"),
    )
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT_DEFAULT)
    parser.add_argument(
        "--budgets", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.02, 0.05]
    )
    args = parser.parse_args(argv)
    output = run_strict_selection(
        score_path=args.scores,
        candidate_metrics_path=args.candidates,
        artifact_root=args.artifact_root,
        budgets=args.budgets,
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
