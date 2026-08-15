"""Tune controller families at genuinely matched benign-friction budgets.

The earlier broad development sweep used a coarse set of score quantiles.  It
was useful for eliminating slow-entry designs, but it left some controller
families far below their nominal friction budget.  This module is the frozen
second-stage tuner: for every structural template and budget, it calibrates the
entry threshold by deterministic bisection on ``controller_tune`` and only then
computes the security and stability outcomes.

The test partition is rejected by the public search API.  Ground-truth labels
are used only by this offline tuning procedure; controller instances themselves
never receive a label.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import polars as pl

from .controller import AccessState
from .experiment import ARTIFACT_ROOT_DEFAULT, write_json_atomic
from .policy_search import (
    CandidateSpec,
    FRICTION_BUDGETS,
    effective_states_one_epoch_later,
    evaluate_candidate,
    make_controller,
)


TUNING_SPLIT = "controller_tune"
ALLOW_EXPOSURE_MARGIN = 0.02
ISOLATION_EXPOSURE_MARGIN = 0.05
EPISODE_COVERAGE_MARGIN = 0.03
DELAY_MARGIN_S = 1.0
MINIMUM_CHURN_REDUCTION = 0.25


@dataclass(frozen=True, slots=True)
class PolicyTemplate:
    """A controller structure whose common entry threshold is calibrated."""

    template: str
    family: str
    controller: str
    calibration: str
    fixed: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _logit_quantile(values: np.ndarray, quantile: float) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    return float(np.quantile(values, quantile))


def generate_templates(risk: np.ndarray, attack: np.ndarray) -> list[PolicyTemplate]:
    """Return the frozen structural grid used after the exploratory sweep."""

    risk = np.asarray(risk, dtype=float)
    attack = np.asarray(attack, dtype=bool)
    if risk.ndim != 1 or attack.ndim != 1 or len(risk) != len(attack):
        raise ValueError("risk and attack must be equal-length one-dimensional arrays")
    benign = risk[~attack]
    if len(benign) < 100 or not np.isfinite(benign).all():
        raise ValueError("at least 100 finite benign scores are required")

    templates: list[PolicyTemplate] = []

    def add(
        family: str,
        controller: str,
        calibration: str,
        fixed: dict[str, Any],
    ) -> None:
        index = sum(item.family == family for item in templates)
        templates.append(
            PolicyTemplate(
                template=f"{family}-template-{index:03d}",
                family=family,
                controller=controller,
                calibration=calibration,
                fixed=fixed,
            )
        )

    add("stateless", "stateless", "equal_entry", {})
    for reports in (2, 3, 5):
        add(
            "n_report",
            "n_report",
            "equal_entry",
            {"entry_reports": reports, "recovery_reports": reports},
        )
    for tau_s in (0.5, 1.0, 2.0, 5.0):
        add("ewma", "ewma", "equal_entry", {"tau_s": tau_s})
    for half_width in (0.005, 0.01, 0.025, 0.05):
        add(
            "symmetric_hysteresis",
            "symmetric_hysteresis",
            "symmetric_entry",
            {"half_width": half_width},
        )

    # The exploratory sweep, conducted without consulting controller test
    # outcomes, showed that smoothed or multi-report escalation censored many
    # short attacks.  The confirmatory family therefore fixes immediate raw-score
    # escalation and searches only asymmetric release evidence, hold time, and
    # the separation of RESTRICT from ISOLATE.  These ingredients define the
    # operational frontier; none is presented as a novel primitive by itself.
    for isolate_quantile in (0.999, 0.9995, 0.9999):
        isolate_enter = _logit_quantile(benign, isolate_quantile)
        for recovery_reports in (1, 2, 3, 5):
            for hold_s in (0.0, 1.0, 2.0):
                for exit_margin in (0.005, 0.025, 0.05):
                    add(
                        "proposed",
                        "asymmetric_sequential",
                        "proposed_restrict_entry",
                        {
                            "isolate_enter": isolate_enter,
                            "isolate_quantile": isolate_quantile,
                            "exit_margin": exit_margin,
                            "entry_reports": 1,
                            "recovery_reports": recovery_reports,
                            "min_restrict_hold_s": hold_s,
                            "min_isolate_hold_s": hold_s,
                            "ewma_tau_s": None,
                        },
                    )
    return templates


def instantiate_at_threshold(
    template: PolicyTemplate,
    threshold: float,
    *,
    candidate: str,
) -> CandidateSpec:
    """Materialize a template at a common upward-entry score threshold."""

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    fixed = dict(template.fixed)
    if template.calibration == "equal_entry":
        parameters = {
            "restrict_threshold": float(threshold),
            "isolate_threshold": float(threshold),
            **fixed,
        }
    elif template.calibration == "symmetric_entry":
        half_width = float(fixed.pop("half_width"))
        parameters = {
            "restrict_center": float(threshold - half_width),
            "isolate_center": float(threshold - half_width),
            "half_width": half_width,
            **fixed,
        }
    elif template.calibration == "proposed_restrict_entry":
        isolate_enter = float(fixed.pop("isolate_enter"))
        fixed.pop("isolate_quantile")
        exit_margin = float(fixed.pop("exit_margin"))
        if threshold > isolate_enter + 1e-12:
            raise ValueError("restrict entry cannot exceed isolate entry")
        restrict_exit = max(0.0, float(threshold) - exit_margin)
        parameters = {
            "restrict_enter": float(threshold),
            "restrict_exit": restrict_exit,
            "isolate_enter": isolate_enter,
            "isolate_exit": max(restrict_exit, isolate_enter - exit_margin),
            **fixed,
        }
    else:
        raise ValueError(f"unknown calibration kind {template.calibration!r}")
    return CandidateSpec(
        candidate=candidate,
        family=template.family,
        controller=template.controller,
        parameters=parameters,
    )


def _threshold_bounds(template: PolicyTemplate) -> tuple[float, float]:
    if template.calibration == "symmetric_entry":
        return 2.0 * float(template.fixed["half_width"]), 1.0
    if template.calibration == "proposed_restrict_entry":
        return 0.0, float(template.fixed["isolate_enter"])
    return 0.0, 1.0


def _benign_friction(
    spec: CandidateSpec,
    *,
    risk: np.ndarray,
    timestamps: np.ndarray,
    subjects: np.ndarray,
    leases: np.ndarray,
    attack: np.ndarray,
    durations: np.ndarray,
) -> float:
    decisions = make_controller(spec).run(risk, timestamps, leases, subjects).state
    effective = effective_states_one_epoch_later(decisions, subjects, leases)
    benign = ~attack
    denominator = float(durations[benign].sum())
    if denominator <= 0:
        raise ValueError("benign entity-time must be positive")
    numerator = float(
        durations[benign & (effective >= int(AccessState.RESTRICT))].sum()
    )
    return numerator / denominator


def calibrate_template(
    template: PolicyTemplate,
    budget: float,
    *,
    risk: np.ndarray,
    timestamps: np.ndarray,
    subjects: np.ndarray,
    leases: np.ndarray,
    attack: np.ndarray,
    durations: np.ndarray,
    iterations: int = 18,
) -> tuple[CandidateSpec, dict[str, Any]] | None:
    """Find the smallest feasible threshold, maximizing use of the budget."""

    if not 0.0 < budget < 1.0:
        raise ValueError("budget must lie strictly between zero and one")
    if iterations < 8:
        raise ValueError("at least eight bisection iterations are required")
    arrays = {
        "risk": np.asarray(risk, dtype=float),
        "timestamps": np.asarray(timestamps),
        "subjects": np.asarray(subjects, dtype=object),
        "leases": np.asarray(leases, dtype=object),
        "attack": np.asarray(attack, dtype=bool),
        "durations": np.asarray(durations, dtype=float),
    }
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("all tuning arrays must have equal length")

    low, high = _threshold_bounds(template)

    def at(value: float, suffix: str) -> tuple[CandidateSpec, float]:
        spec = instantiate_at_threshold(
            template,
            value,
            candidate=f"{template.template}-B{budget:g}-{suffix}",
        )
        return spec, _benign_friction(spec, **arrays)

    high_spec, high_friction = at(high, "high")
    if high_friction > budget:
        # Irreducible isolation/hold friction makes this structure infeasible.
        return None
    low_spec, low_friction = at(low, "low")
    if low_friction <= budget:
        best_spec, best_friction = low_spec, low_friction
    else:
        best_spec, best_friction = high_spec, high_friction
        for iteration in range(iterations):
            midpoint = (low + high) / 2.0
            mid_spec, mid_friction = at(midpoint, f"iter{iteration}")
            if mid_friction <= budget:
                high = midpoint
                best_spec, best_friction = mid_spec, mid_friction
            else:
                low = midpoint

    final_spec = CandidateSpec(
        candidate=f"{template.template}-B{budget:g}",
        family=best_spec.family,
        controller=best_spec.controller,
        parameters=best_spec.parameters,
    )
    return final_spec, {
        "template": template.template,
        "friction_budget": float(budget),
        "calibrated_threshold": float(
            final_spec.parameters.get(
                "restrict_threshold",
                final_spec.parameters.get(
                    "restrict_enter",
                    final_spec.parameters.get("restrict_center", np.nan)
                    + final_spec.parameters.get("half_width", 0.0),
                ),
            )
        ),
        "calibration_friction": float(best_friction),
        "budget_slack": float(budget - best_friction),
        "bisection_iterations": int(iterations),
    }


def _load_tuning_arrays(score_path: Path) -> dict[str, np.ndarray]:
    frame = pl.read_parquet(score_path)
    observed = set(frame["split"].unique().to_list())
    if TUNING_SPLIT not in observed:
        raise ValueError(f"score frame has no {TUNING_SPLIT!r} partition")
    frame = frame.filter(pl.col("split") == TUNING_SPLIT)
    return {
        "risk": frame["risk_score"].to_numpy(),
        "timestamps": frame["decision_time_s"].to_numpy(),
        "subjects": frame["mac_rnti"].to_numpy(),
        "leases": frame["rnti_lease_id"].to_numpy(),
        "attack": frame["is_attack_epoch"].to_numpy(),
        "durations": frame["epoch_seconds"].to_numpy(),
    }


def _calibrate_and_evaluate(
    template: PolicyTemplate,
    budget: float,
    arrays: dict[str, np.ndarray],
    iterations: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    calibrated = calibrate_template(
        template,
        budget,
        **arrays,
        iterations=iterations,
    )
    if calibrated is None:
        return None
    spec, calibration = calibrated
    metrics = evaluate_candidate(spec, **arrays, split=TUNING_SPLIT)
    metrics.update(calibration)
    metrics["template_fixed"] = json.dumps(template.fixed, sort_keys=True)
    return metrics, spec.to_dict(), template.to_dict()


def select_matched_policies(metrics: pl.DataFrame) -> pl.DataFrame:
    """Select tuned comparators and the least-churning feasible proposal.

    Proposal feasibility is stated relative to the exactly budget-matched
    stateless policy.  In addition to the primary ALLOW-time exposure bound, a
    conservative ISOLATE-only bound prevents the intermediate RESTRICT action
    from hiding arbitrarily poor hard-containment performance.
    """

    required = {
        "candidate",
        "family",
        "friction_budget",
        "benign_friction",
        "malicious_allow",
        "malicious_not_isolated",
        "transitions_per_minute",
        "episode_coverage",
        "median_capped_delay_s",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"metrics missing required columns: {sorted(missing)}")
    if set(metrics["split"].unique().to_list()) != {TUNING_SPLIT}:
        raise ValueError("selection accepts controller-tuning rows only")

    selected: list[dict[str, Any]] = []
    for budget in sorted(metrics["friction_budget"].unique().to_list()):
        at_budget = metrics.filter(pl.col("friction_budget") == budget)
        stateless = at_budget.filter(pl.col("family") == "stateless")
        if stateless.height != 1:
            raise ValueError("each budget must have exactly one stateless reference")
        reference = stateless.row(0, named=True)
        selected.append({**reference, "selection_role": "reference"})

        for family in ("n_report", "ewma", "symmetric_hysteresis"):
            family_rows = at_budget.filter(pl.col("family") == family)
            if family_rows.is_empty():
                continue
            # Give every comparator the same coverage/delay protection before
            # minimizing exposure and then churn.
            feasible = family_rows.filter(
                (pl.col("episode_coverage")
                 >= reference["episode_coverage"] - EPISODE_COVERAGE_MARGIN)
                & (pl.col("median_capped_delay_s")
                   <= reference["median_capped_delay_s"] + DELAY_MARGIN_S)
            )
            if feasible.is_empty():
                feasible = family_rows
            chosen = feasible.sort(
                [
                    "malicious_allow",
                    "transitions_per_minute",
                    "malicious_not_isolated",
                    "candidate",
                ]
            ).row(0, named=True)
            selected.append({**chosen, "selection_role": "comparator"})

        proposals = at_budget.filter(pl.col("family") == "proposed")
        feasible = proposals.filter(
            (pl.col("malicious_allow")
             <= reference["malicious_allow"] + ALLOW_EXPOSURE_MARGIN)
            & (pl.col("malicious_not_isolated")
               <= reference["malicious_not_isolated"] + ISOLATION_EXPOSURE_MARGIN)
            & (pl.col("episode_coverage")
               >= reference["episode_coverage"] - EPISODE_COVERAGE_MARGIN)
            & (pl.col("median_capped_delay_s")
               <= reference["median_capped_delay_s"] + DELAY_MARGIN_S)
        )
        if feasible.is_empty():
            continue
        chosen = feasible.sort(
            [
                "transitions_per_minute",
                "malicious_allow",
                "malicious_not_isolated",
                "candidate",
            ]
        ).row(0, named=True)
        churn_reduction = 1.0 - float(chosen["transitions_per_minute"]) / float(
            reference["transitions_per_minute"]
        )
        selected.append(
            {
                **chosen,
                "selection_role": "proposed",
                "churn_reduction_vs_stateless": churn_reduction,
                "passes_minimum_churn_reduction": (
                    churn_reduction >= MINIMUM_CHURN_REDUCTION
                ),
            }
        )
    return pl.DataFrame(selected, strict=False).sort(
        ["friction_budget", "selection_role", "family"]
    )


def run_matched_search(
    *,
    score_path: Path,
    artifact_root: Path = ARTIFACT_ROOT_DEFAULT,
    budgets: Sequence[float] = FRICTION_BUDGETS,
    iterations: int = 18,
    n_jobs: int = 1,
) -> dict[str, Any]:
    arrays = _load_tuning_arrays(score_path)
    templates = generate_templates(arrays["risk"], arrays["attack"])
    tasks = [(template, float(budget)) for template in templates for budget in budgets]
    if n_jobs == 1:
        outputs = [
            _calibrate_and_evaluate(template, budget, arrays, iterations)
            for template, budget in tasks
        ]
    else:
        outputs = joblib.Parallel(n_jobs=n_jobs, verbose=10, batch_size=1)(
            joblib.delayed(_calibrate_and_evaluate)(
                template, budget, arrays, iterations
            )
            for template, budget in tasks
        )
    valid = [output for output in outputs if output is not None]
    metrics = pl.DataFrame([output[0] for output in valid], strict=False)
    specs = {output[1]["candidate"]: output[1] for output in valid}
    template_map = {item.template: item.to_dict() for item in templates}
    selected = select_matched_policies(metrics)

    results_dir = artifact_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "controller_matched_candidates_tune_v1.parquet"
    selected_path = results_dir / "controller_matched_selected_tune_v1.parquet"
    specs_path = results_dir / "controller_matched_specs_v1.json"
    templates_path = results_dir / "controller_matched_templates_v1.json"
    metrics.write_parquet(metrics_path, compression="zstd")
    selected.write_parquet(selected_path, compression="zstd")
    write_json_atomic(specs_path, specs)
    write_json_atomic(templates_path, template_map)
    return {
        "template_count": len(templates),
        "task_count": len(tasks),
        "feasible_candidates": len(valid),
        "family_counts": metrics.group_by("family").len().sort("family").to_dicts(),
        "selected_count": selected.height,
        "metrics_path": str(metrics_path.resolve()),
        "selected_path": str(selected_path.resolve()),
        "specs_path": str(specs_path.resolve()),
        "templates_path": str(templates_path.resolve()),
    }


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-path",
        type=Path,
        default=Path("artifacts/results/scores_logistic_seed1729_v1.parquet"),
    )
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT_DEFAULT)
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=list(FRICTION_BUDGETS),
        help="friction budgets to calibrate (default: full frozen frontier)",
    )
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    output = run_matched_search(
        score_path=args.score_path,
        artifact_root=args.artifact_root,
        budgets=args.budgets,
        iterations=args.iterations,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
