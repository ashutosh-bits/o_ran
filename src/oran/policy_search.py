"""Budget-matched controller search and causal score-to-action alignment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import polars as pl

from .controller import (
    AccessState,
    ControllerConfig,
    EWMAController,
    NReportController,
    StatelessController,
    SymmetricHysteresisController,
    TemporalAccessController,
)
from .experiment import ARTIFACT_ROOT_DEFAULT, write_json_atomic
from .metrics import (
    attack_episode_metrics,
    summarize_attack_episodes,
    time_weighted_access_metrics,
)


FRICTION_BUDGETS = (0.001, 0.005, 0.01, 0.02, 0.05)
SECURITY_NONINFERIORITY = 0.02
DELAY_NONINFERIORITY_S = 1.0
EPISODE_COVERAGE_NONINFERIORITY = 0.05


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate: str
    family: str
    controller: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_name(family: str, index: int) -> str:
    return f"{family}-{index:04d}"


def make_controller(spec: CandidateSpec) -> TemporalAccessController:
    params = dict(spec.parameters)
    if spec.controller == "stateless":
        return StatelessController(**params)
    if spec.controller == "n_report":
        return NReportController(**params)
    if spec.controller == "ewma":
        return EWMAController(**params)
    if spec.controller == "symmetric_hysteresis":
        return SymmetricHysteresisController(**params)
    if spec.controller == "asymmetric_sequential":
        return TemporalAccessController(ControllerConfig(**params))
    raise ValueError(f"unknown controller kind: {spec.controller}")


def effective_states_one_epoch_later(
    decision_states: Sequence[int] | np.ndarray,
    subject_ids: Sequence[Any] | np.ndarray,
    lease_ids: Sequence[Any] | np.ndarray,
) -> np.ndarray:
    """Align a decision at epoch ``t`` with the action effective in ``t+1``.

    A new lifecycle starts in ALLOW.  The function does not inspect labels or
    timestamps and therefore cannot create an oracle reset.
    """

    decisions = np.asarray(decision_states, dtype=np.int8)
    subjects = np.asarray(subject_ids, dtype=object)
    leases = np.asarray(lease_ids, dtype=object)
    if decisions.ndim != 1 or subjects.ndim != 1 or leases.ndim != 1:
        raise ValueError("decision states, subjects, and leases must be one-dimensional")
    if len(decisions) != len(subjects) or len(decisions) != len(leases):
        raise ValueError("decision states, subjects, and leases must have equal length")
    effective = np.full(len(decisions), int(AccessState.ALLOW), dtype=np.int8)
    prior: dict[Any, tuple[Any, int]] = {}
    for index, (subject, lease, decision) in enumerate(
        zip(subjects, leases, decisions, strict=True)
    ):
        previous = prior.get(subject)
        if previous is not None and previous[0] == lease:
            effective[index] = previous[1]
        prior[subject] = (lease, int(decision))
    return effective


def _threshold_pairs(risk: np.ndarray, attack: np.ndarray) -> list[tuple[float, float]]:
    benign = np.asarray(risk, dtype=float)[~np.asarray(attack, dtype=bool)]
    if len(benign) < 100:
        raise ValueError("too few benign tuning epochs for threshold calibration")
    restrict_quantiles = (
        0.90,
        0.95,
        0.975,
        0.99,
        0.995,
        0.999,
        0.9995,
        0.9999,
    )
    isolate_quantiles = (0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995)
    restrict_values = np.quantile(benign, restrict_quantiles)
    isolate_values = np.quantile(benign, isolate_quantiles)
    pairs = {
        (float(restrict), float(isolate))
        for restrict in restrict_values
        for isolate in isolate_values
        if isolate + 1e-12 >= restrict
    }
    # Equal thresholds represent an immediate isolation policy and bound the
    # unknown benefit of the intermediate RESTRICT action.
    pairs.update((float(value), float(value)) for value in restrict_values)
    return sorted(pairs)


def generate_candidate_specs(risk: np.ndarray, attack: np.ndarray) -> list[CandidateSpec]:
    pairs = _threshold_pairs(risk, attack)
    specs: list[CandidateSpec] = []

    def add(family: str, controller: str, parameters: dict[str, Any]) -> None:
        index = sum(item.family == family for item in specs)
        specs.append(
            CandidateSpec(
                candidate=_candidate_name(family, index),
                family=family,
                controller=controller,
                parameters=parameters,
            )
        )

    add(
        "stateless",
        "stateless",
        {"restrict_threshold": 1.1, "isolate_threshold": 1.2},
    )
    for restrict, isolate in pairs:
        add(
            "stateless",
            "stateless",
            {"restrict_threshold": restrict, "isolate_threshold": isolate},
        )
        for reports in (2, 3, 5):
            add(
                "n_report",
                "n_report",
                {
                    "restrict_threshold": restrict,
                    "isolate_threshold": isolate,
                    "entry_reports": reports,
                    "recovery_reports": reports,
                },
            )
        for tau_s in (0.5, 1.0, 2.0, 5.0):
            add(
                "ewma",
                "ewma",
                {
                    "restrict_threshold": restrict,
                    "isolate_threshold": isolate,
                    "tau_s": tau_s,
                },
            )
        for half_width in (0.01, 0.025, 0.05):
            if restrict - 2.0 * half_width < 0.0:
                continue
            add(
                "symmetric_hysteresis",
                "symmetric_hysteresis",
                {
                    "restrict_center": restrict - half_width,
                    "isolate_center": isolate - half_width,
                    "half_width": half_width,
                },
            )
        # ``None`` gives immediate score evidence.  This variant isolates the
        # value of asymmetric recovery/hold from smoothing and avoids forcing a
        # short-attack delay into the proposed family.
        for tau_s in (None, 0.5, 2.0, 5.0):
            for entry_reports in (1, 2):
                for recovery_reports in (3, 5):
                    for hold_s in (2.0, 5.0):
                        exit_margin = 0.05
                        add(
                            "proposed",
                            "asymmetric_sequential",
                            {
                                "restrict_enter": restrict,
                                "restrict_exit": max(0.0, restrict - exit_margin),
                                "isolate_enter": isolate,
                                "isolate_exit": max(
                                    max(0.0, restrict - exit_margin),
                                    isolate - exit_margin,
                                ),
                                "entry_reports": entry_reports,
                                "recovery_reports": recovery_reports,
                                "min_restrict_hold_s": hold_s,
                                "min_isolate_hold_s": hold_s,
                                "ewma_tau_s": tau_s,
                            },
                        )
        # Fast-entry variants search the part of the frontier that preserves
        # short-episode coverage.  They use the raw score for escalation and
        # vary only recovery evidence, a short hold, and a narrow asymmetric
        # release margin.  This was added after the first development-only
        # loop exposed excessive short-episode censoring; no test rows were
        # consulted.
        for recovery_reports in (1, 2, 3):
            for hold_s in (0.0, 1.0, 2.0):
                for exit_margin in (0.005, 0.01, 0.025):
                    add(
                        "proposed",
                        "asymmetric_sequential",
                        {
                            "restrict_enter": restrict,
                            "restrict_exit": max(0.0, restrict - exit_margin),
                            "isolate_enter": isolate,
                            "isolate_exit": max(
                                max(0.0, restrict - exit_margin),
                                isolate - exit_margin,
                            ),
                            "entry_reports": 1,
                            "recovery_reports": recovery_reports,
                            "min_restrict_hold_s": hold_s,
                            "min_isolate_hold_s": hold_s,
                            "ewma_tau_s": None,
                        },
                    )
    return specs


def evaluate_candidate(
    spec: CandidateSpec,
    risk: np.ndarray,
    timestamps: np.ndarray,
    subjects: np.ndarray,
    leases: np.ndarray,
    attack: np.ndarray,
    durations: np.ndarray,
    split: str,
) -> dict[str, Any]:
    trace = make_controller(spec).run(risk, timestamps, leases, subjects)
    effective = effective_states_one_epoch_later(trace.state, subjects, leases)
    restrict_metrics = time_weighted_access_metrics(
        timestamps,
        effective,
        attack,
        subject_ids=subjects,
        lease_ids=leases,
        durations_s=durations,
        containment_state=AccessState.RESTRICT,
    )
    isolate_metrics = time_weighted_access_metrics(
        timestamps,
        effective,
        attack,
        subject_ids=subjects,
        lease_ids=leases,
        durations_s=durations,
        containment_state=AccessState.ISOLATE,
    )
    episodes = attack_episode_metrics(
        timestamps,
        effective,
        attack,
        subject_ids=subjects,
        lease_ids=leases,
        durations_s=durations,
        delay_cap_s=30.0,
        containment_state=AccessState.RESTRICT,
    )
    episode_summary = summarize_attack_episodes(episodes)
    if episode_summary.height:
        episode_row = episode_summary.row(0, named=True)
        coverage = float(episode_row["episode_coverage"])
        capped_delay = float(episode_row["median_capped_delay_s"])
        episodes_n = int(episode_row["episodes"])
    else:
        coverage = float("nan")
        capped_delay = float("nan")
        episodes_n = 0
    return {
        "candidate": spec.candidate,
        "family": spec.family,
        "split": split,
        "benign_friction": restrict_metrics.benign_friction,
        "benign_isolation": isolate_metrics.benign_friction,
        "malicious_allow": restrict_metrics.malicious_exposure,
        "malicious_not_isolated": isolate_metrics.malicious_exposure,
        "transitions": restrict_metrics.transitions,
        "transitions_per_minute": restrict_metrics.transitions_per_minute,
        "severity_transitions_per_minute": restrict_metrics.severity_transitions_per_minute,
        "false_restriction_episodes": restrict_metrics.false_restriction_episodes,
        "attack_episodes": episodes_n,
        "episode_coverage": coverage,
        "median_capped_delay_s": capped_delay,
    }


def select_within_family(
    metrics: pl.DataFrame,
    *,
    budgets: Sequence[float] = FRICTION_BUDGETS,
    security_margin: float = SECURITY_NONINFERIORITY,
    delay_margin_s: float = DELAY_NONINFERIORITY_S,
    coverage_margin: float = EPISODE_COVERAGE_NONINFERIORITY,
) -> pl.DataFrame:
    """Select the least-churning security-noninferior policy per family/budget."""

    rows: list[dict[str, Any]] = []
    for family in metrics["family"].unique().sort().to_list():
        family_frame = metrics.filter(pl.col("family") == family)
        for budget in budgets:
            feasible = family_frame.filter(pl.col("benign_friction") <= budget)
            if feasible.is_empty():
                continue
            best_security = float(feasible["malicious_allow"].min())
            security_set = feasible.filter(
                pl.col("malicious_allow") <= best_security + security_margin
            )
            best_coverage = float(security_set["episode_coverage"].max())
            coverage_set = security_set.filter(
                pl.col("episode_coverage") >= best_coverage - coverage_margin
            )
            best_delay = float(coverage_set["median_capped_delay_s"].min())
            delay_set = coverage_set.filter(
                pl.col("median_capped_delay_s") <= best_delay + delay_margin_s
            )
            chosen = delay_set.sort(
                [
                    "transitions_per_minute",
                    "malicious_allow",
                    "malicious_not_isolated",
                    "benign_friction",
                    "candidate",
                ]
            ).row(0, named=True)
            rows.append(
                {
                    **chosen,
                    "friction_budget": float(budget),
                    "best_family_malicious_allow": best_security,
                    "security_margin": security_margin,
                    "best_security_set_episode_coverage": best_coverage,
                    "coverage_margin": coverage_margin,
                    "delay_margin_s": delay_margin_s,
                }
            )
    return pl.DataFrame(rows, strict=False).sort(["friction_budget", "family"])


def search_controllers(
    *,
    score_path: Path,
    artifact_root: Path = ARTIFACT_ROOT_DEFAULT,
    n_jobs: int = 1,
) -> dict[str, Any]:
    frame = pl.read_parquet(score_path).filter(pl.col("split") == "controller_tune")
    arrays = {
        "risk": frame["risk_score"].to_numpy(),
        "timestamps": frame["decision_time_s"].to_numpy(),
        "subjects": frame["mac_rnti"].to_numpy(),
        "leases": frame["rnti_lease_id"].to_numpy(),
        "attack": frame["is_attack_epoch"].to_numpy(),
        "durations": frame["epoch_seconds"].to_numpy(),
    }
    specs = generate_candidate_specs(arrays["risk"], arrays["attack"])
    if n_jobs == 1:
        records = [
            evaluate_candidate(spec, **arrays, split="controller_tune")
            for spec in specs
        ]
    else:
        records = joblib.Parallel(n_jobs=n_jobs, verbose=10, batch_size=1)(
            joblib.delayed(evaluate_candidate)(
                spec, **arrays, split="controller_tune"
            )
            for spec in specs
        )
    metrics = pl.DataFrame(records, strict=False)
    selected = select_within_family(metrics)
    results_dir = artifact_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "controller_candidates_tune_v1.parquet"
    selected_path = results_dir / "controller_selected_tune_v1.parquet"
    metrics.write_parquet(metrics_path, compression="zstd")
    selected.write_parquet(selected_path, compression="zstd")
    spec_map = {spec.candidate: spec.to_dict() for spec in specs}
    specs_path = results_dir / "controller_candidate_specs_v1.json"
    write_json_atomic(specs_path, spec_map)
    return {
        "candidate_count": len(specs),
        "family_counts": metrics.group_by("family").len().sort("family").to_dicts(),
        "selected_count": selected.height,
        "metrics_path": str(metrics_path.resolve()),
        "selected_path": str(selected_path.resolve()),
        "specs_path": str(specs_path.resolve()),
    }


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-path",
        type=Path,
        default=Path("artifacts/results/scores_logistic_seed1729_v1.parquet"),
    )
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT_DEFAULT)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    output = search_controllers(
        score_path=args.score_path,
        artifact_root=args.artifact_root,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
