"""Generate deterministic publication tables and figures from locked artifacts.

This module is intentionally read-only with respect to experiment artifacts.  It
does not fit a model, tune a threshold, replay a controller, or resample data.
It renders the already locked point estimates and bootstrap intervals while
keeping three scopes distinct:

* the chronological held-out RNTI-level offline replay;
* the controller-tuning security--friction/stability frontier; and
* explicitly diagnostic robustness and sensitivity analyses.

The terminology is deliberately conservative.  A ``trace block`` is not called
a capture, a numeric RNTI is not treated as a durable identity, and an offline
action replay is not interpreted as causal attack prevention.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import polars as pl


PRIMARY_PROPOSAL = "proposed-template-047-B0.01"
PRIMARY_REFERENCE = "stateless-reference-0.01"
PRIMARY_BUDGET = 0.01

FAMILY_ORDER = {
    "proposed": 0,
    "stateless_reference": 1,
    "stateless": 1,
    "ewma": 2,
    "n_report": 3,
    "symmetric_hysteresis": 4,
}
FAMILY_LABELS = {
    "proposed": "Friction-budgeted",
    "stateless_reference": "Stateless",
    "stateless": "Stateless",
    "ewma": "EWMA",
    "n_report": "Two-report",
    "symmetric_hysteresis": "Symmetric hysteresis",
}
ATTACK_ORDER = {
    "portscan": 0,
    "ddos-ripper-C": 1,
    "dos-hulk-C": 2,
    "slowloris-C": 3,
}
ATTACK_LABELS = {
    "portscan": "Port scan",
    "ddos-ripper-C": "DDoS/Ripper",
    "dos-hulk-C": "DoS/Hulk",
    "slowloris-C": "Slowloris†",
}

# Okabe--Ito-derived colors plus redundant markers/hatches for accessibility.
FAMILY_COLORS = {
    "proposed": "#0072B2",
    "stateless_reference": "#4D4D4D",
    "stateless": "#4D4D4D",
    "ewma": "#E69F00",
    "n_report": "#009E73",
    "symmetric_hysteresis": "#CC79A7",
}
FAMILY_MARKERS = {
    "proposed": "o",
    "stateless_reference": "X",
    "stateless": "X",
    "ewma": "s",
    "n_report": "^",
    "symmetric_hysteresis": "D",
}
FAMILY_HATCHES = {
    "proposed": "//",
    "stateless_reference": "xx",
    "stateless": "xx",
    "ewma": "..",
    "n_report": "++",
    "symmetric_hysteresis": "\\\\",
}

INPUT_PATHS = {
    "aggregate": "artifacts/confirmatory/aggregate_metrics_v2.parquet",
    "episodes": "artifacts/confirmatory/attack_episode_summary_v2.parquet",
    "strata": "artifacts/confirmatory/stratified_action_metrics_v2.parquet",
    "inference_v3": "artifacts/confirmatory/inference_report_v3.json",
    "tuning_candidates": "artifacts/results/controller_matched_candidates_tune_v1.parquet",
    "tuning_selection": "artifacts/results/controller_strict_selection_tune_v1.parquet",
    "tuning_references": "artifacts/results/stateless_references_tune_v1.parquet",
    "hgb_bootstrap": "artifacts/hgb_sensitivity/confirmatory/paired_bootstrap_v2.json",
    "hgb_aggregate": "artifacts/hgb_sensitivity/confirmatory/aggregate_metrics_v2.parquet",
    "timebase_selection": "artifacts/timebase_1e6/results/controller_strict_selection_tune_v1.parquet",
    "timebase_reference": "artifacts/timebase_1e6/results/stateless_references_tune_v1.parquet",
    "lease_action": "artifacts/sensitivities/lease_timeout_action_metrics_v1.parquet",
    "lease_episodes": "artifacts/sensitivities/lease_timeout_episode_metrics_v1.parquet",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _float(value: Any) -> float:
    result = float(value)
    return result if math.isfinite(result) else float("nan")


def _metric(metrics: Iterable[Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    matches = [item for item in metrics if item["metric"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one {name!r} interval, found {len(matches)}")
    return matches[0]


def _only_row(frame: pl.DataFrame, **filters: Any) -> dict[str, Any]:
    selected = frame
    for column, value in filters.items():
        selected = selected.filter(pl.col(column) == value)
    if selected.height != 1:
        raise ValueError(f"expected one row for {filters}, found {selected.height}")
    return selected.row(0, named=True)


def heldout_policy_table(
    aggregate: pl.DataFrame,
    episodes: pl.DataFrame,
) -> pl.DataFrame:
    """Return the locked held-out policy comparison in publication units."""

    episode_rows = episodes.filter(
        (pl.col("attack_type") == "__any_attack__")
        & (pl.col("onset_stratum") == "__all__")
    ).select(
        "candidate",
        "episode_count",
        "episode_coverage",
        "mean_capped_delay_s",
        "median_capped_delay_s",
    )
    joined = aggregate.join(episode_rows, on="candidate", how="inner", validate="1:1")
    if joined.height != 5:
        raise ValueError(f"expected five locked held-out policies, found {joined.height}")
    records: list[dict[str, Any]] = []
    for row in joined.iter_rows(named=True):
        family = str(row["family"])
        records.append(
            {
                "policy_order": FAMILY_ORDER[family],
                "policy": FAMILY_LABELS[family],
                "role": (
                    "proposed"
                    if family == "proposed"
                    else "static reference"
                    if family == "stateless_reference"
                    else "comparator"
                ),
                "epochs": int(row["epochs"]),
                "benign_friction_pct": 100.0 * _float(row["benign_friction"]),
                "benign_isolation_pct": 100.0 * _float(row["benign_isolation"]),
                "malicious_allow_pct": 100.0 * _float(row["malicious_allow"]),
                "malicious_not_isolated_pct": 100.0
                * _float(row["malicious_not_isolated"]),
                "effective_transitions": int(row["effective_transitions"]),
                "transitions_per_1000_observed_epochs": 1000.0
                * int(row["effective_transitions"])
                / int(row["epochs"]),
                "attack_episode_count": int(row["episode_count"]),
                "episode_coverage_pct": 100.0 * _float(row["episode_coverage"]),
                "mean_capped_delay_s": _float(row["mean_capped_delay_s"]),
                "median_capped_delay_s": _float(row["median_capped_delay_s"]),
            }
        )
    return pl.DataFrame(records, strict=False).sort("policy_order")


def per_attack_table(
    episodes: pl.DataFrame,
    intervals: Mapping[str, Any],
    fallback_intervals: Mapping[str, Any] | None = None,
) -> pl.DataFrame:
    """Return proposal-versus-static per-attack points and trace-block CIs."""

    all_onsets = episodes.filter(pl.col("onset_stratum") == "__all__")
    records: list[dict[str, Any]] = []
    for attack_type in sorted(ATTACK_ORDER, key=ATTACK_ORDER.get):
        proposal = _only_row(
            all_onsets,
            candidate=PRIMARY_PROPOSAL,
            attack_type=attack_type,
        )
        reference = _only_row(
            all_onsets,
            candidate=PRIMARY_REFERENCE,
            attack_type=attack_type,
        )
        boot = intervals[attack_type]
        if "attack_exposure_difference" in boot:
            exposure = boot["attack_exposure_difference"]
            coverage = boot["episode_coverage_difference"]
            delay = boot["mean_capped_delay_difference_s"]
            exposure = {
                "estimate": exposure["estimate"],
                "lower": exposure["ci_lower"],
                "upper": exposure["ci_upper"],
            }
            coverage = {
                "estimate": coverage["estimate"],
                "lower": coverage["ci_lower"],
                "upper": coverage["ci_upper"],
            }
            delay = {
                "estimate": delay["estimate"],
                "lower": delay["ci_lower"],
                "upper": delay["ci_upper"],
            }
        else:
            exposure = boot["exposure_difference"]
            coverage = boot["coverage_difference"]
            delay = boot["mean_capped_delay_difference_s"]

        point_exposure = _float(proposal["malicious_allow"]) - _float(
            reference["malicious_allow"]
        )
        source_note = "formal inference report v3"
        if not math.isclose(_float(exposure["estimate"]), point_exposure, abs_tol=1e-12):
            if fallback_intervals is None:
                raise ValueError(
                    f"{attack_type} exposure interval estimate does not match locked point contrast"
                )
            exposure = fallback_intervals[attack_type]["exposure_difference"]
            if not math.isclose(_float(exposure["estimate"]), point_exposure, abs_tol=1e-12):
                raise ValueError(
                    f"{attack_type} fallback exposure interval also mismatches locked point contrast"
                )
            source_note = "v3 coverage/delay; validated v2 exposure fallback"
        is_slowloris = attack_type == "slowloris-C"
        records.append(
            {
                "attack_order": ATTACK_ORDER[attack_type],
                "attack": ATTACK_LABELS[attack_type].replace("†", ""),
                "attack_type": attack_type,
                "episodes": int(proposal["episode_count"]),
                "proposal_exposure_pct": 100.0 * _float(proposal["malicious_allow"]),
                "reference_exposure_pct": 100.0 * _float(reference["malicious_allow"]),
                "exposure_difference_pp": 100.0 * _float(exposure["estimate"]),
                "exposure_ci_lower_pp": 100.0 * _float(exposure["lower"]),
                "exposure_ci_upper_pp": 100.0 * _float(exposure["upper"]),
                "proposal_coverage_pct": 100.0 * _float(proposal["episode_coverage"]),
                "reference_coverage_pct": 100.0 * _float(reference["episode_coverage"]),
                "coverage_difference_pp": 100.0 * _float(coverage["estimate"]),
                "coverage_ci_lower_pp": 100.0 * _float(coverage["lower"]),
                "coverage_ci_upper_pp": 100.0 * _float(coverage["upper"]),
                "proposal_mean_capped_delay_s": _float(proposal["mean_capped_delay_s"]),
                "reference_mean_capped_delay_s": _float(reference["mean_capped_delay_s"]),
                "mean_delay_difference_s": _float(delay["estimate"]),
                "mean_delay_ci_lower_s": _float(delay["lower"]),
                "mean_delay_ci_upper_s": _float(delay["upper"]),
                "known_failure_mode": is_slowloris,
                "interval_source": source_note,
                "interpretation": (
                    "known failure: lower coverage and longer delay"
                    if is_slowloris
                    else "attack-specific descriptive contrast"
                ),
            }
        )
    return pl.DataFrame(records, strict=False).sort("attack_order")


def unseen_seen_table(
    strata: pl.DataFrame,
    novelty_inference: Mapping[str, Any],
) -> pl.DataFrame:
    """Return seen/unseen numeric-RNTI points and trace-block intervals."""

    records: list[dict[str, Any]] = []
    for order, stratum in enumerate(("rnti_seen", "rnti_unseen")):
        proposal = _only_row(strata, family="proposed", stratum=stratum)
        reference = _only_row(strata, family="stateless_reference", stratum=stratum)
        novelty_key = "seen" if stratum == "rnti_seen" else "unseen"
        novelty = novelty_inference[novelty_key]
        inference = novelty["inference"]
        metrics = inference["paired_additive_bootstrap"]["metrics"]
        exposure = _metric(metrics, "attack_exposure_difference")
        coverage = _metric(metrics, "episode_coverage_difference")
        churn = _metric(metrics, "transition_rate_ratio")
        friction = inference["proposed_friction_bootstrap"]["interval"]
        median = inference["paired_median_delay_bootstrap"]["interval"]
        records.append(
            {
                "stratum_order": order,
                "numeric_rnti_stratum": (
                    "Seen during model training" if stratum == "rnti_seen" else "Unseen during model training"
                ),
                "epochs": int(proposal["epochs"]),
                "numeric_rnti_count": int(novelty["numeric_rnti_count"]),
                "proposal_benign_friction_pct": 100.0 * _float(proposal["benign_friction"]),
                "reference_benign_friction_pct": 100.0 * _float(reference["benign_friction"]),
                "friction_difference_pp": 100.0
                * (_float(proposal["benign_friction"]) - _float(reference["benign_friction"])),
                "proposal_malicious_allow_pct": 100.0 * _float(proposal["malicious_allow"]),
                "reference_malicious_allow_pct": 100.0 * _float(reference["malicious_allow"]),
                "exposure_difference_pp": 100.0
                * (_float(proposal["malicious_allow"]) - _float(reference["malicious_allow"])),
                "exposure_ci_lower_pp": 100.0 * _float(exposure["ci_lower"]),
                "exposure_ci_upper_pp": 100.0 * _float(exposure["ci_upper"]),
                "proposal_not_isolated_pct": 100.0
                * _float(proposal["malicious_not_isolated"]),
                "reference_not_isolated_pct": 100.0
                * _float(reference["malicious_not_isolated"]),
                "proposal_transitions_per_1000_epochs": 1000.0
                * int(proposal["effective_transitions"])
                / int(proposal["epochs"]),
                "reference_transitions_per_1000_epochs": 1000.0
                * int(reference["effective_transitions"])
                / int(reference["epochs"]),
                "transition_rate_ratio": _float(churn["estimate"]),
                "transition_ratio_ci_lower": _float(churn["ci_lower"]),
                "transition_ratio_ci_upper": _float(churn["ci_upper"]),
                "coverage_difference_pp": 100.0 * _float(coverage["estimate"]),
                "coverage_ci_lower_pp": 100.0 * _float(coverage["ci_lower"]),
                "coverage_ci_upper_pp": 100.0 * _float(coverage["ci_upper"]),
                "proposal_friction_ci_lower_pct": 100.0 * _float(friction["ci_lower"]),
                "proposal_friction_ci_upper_pct": 100.0 * _float(friction["ci_upper"]),
                "proposal_friction_one_sided_95_upper_pct": 100.0
                * _float(friction["one_sided_upper"]),
                "proposal_friction_point_within_1pct": bool(
                    inference["descriptive_safeguards"]["friction_point_within_budget"]
                ),
                "proposal_friction_ucb_within_1pct": bool(
                    inference["descriptive_safeguards"][
                        "friction_one_sided_ucb_within_budget"
                    ]
                ),
                "median_capped_delay_difference_s": _float(median["estimate"]),
                "four_pairwise_gates_pass": all(
                    bool(gate["passed"]) for gate in inference["primary_gates"]
                ),
                "scope_note": "numeric RNTI only; no durable-identity interpretation",
            }
        )
    return pl.DataFrame(records, strict=False).sort("stratum_order")


def tuning_frontier_table(
    selection: pl.DataFrame,
    references: pl.DataFrame,
) -> pl.DataFrame:
    """Combine strict structural selections and stateless budget references."""

    records: list[dict[str, Any]] = []
    for row in selection.iter_rows(named=True):
        family = str(row["family"])
        records.append(
            {
                "budget_pct": 100.0 * _float(row["friction_budget"]),
                "family_order": FAMILY_ORDER[family],
                "policy_family": FAMILY_LABELS[family],
                "status": str(row["status"]).replace("_", " "),
                "candidate": row["candidate"],
                "realized_friction_pct": 100.0 * _float(row["benign_friction"]),
                "malicious_allow_pct": 100.0 * _float(row["malicious_allow"]),
                "malicious_not_isolated_pct": 100.0 * _float(row["malicious_not_isolated"]),
                "transitions_per_observed_rnti_minute": _float(row["transitions_per_minute"]),
                "episode_coverage_pct": 100.0 * _float(row["episode_coverage"]),
                "median_capped_delay_s": _float(row["median_capped_delay_s"]),
            }
        )
    for row in references.iter_rows(named=True):
        family = "stateless_reference"
        records.append(
            {
                "budget_pct": 100.0 * _float(row["friction_budget"]),
                "family_order": FAMILY_ORDER[family],
                "policy_family": FAMILY_LABELS[family],
                "status": "budget reference",
                "candidate": row["candidate"],
                "realized_friction_pct": 100.0 * _float(row["benign_friction"]),
                "malicious_allow_pct": 100.0 * _float(row["malicious_allow"]),
                "malicious_not_isolated_pct": 100.0 * _float(row["malicious_not_isolated"]),
                "transitions_per_observed_rnti_minute": _float(row["transitions_per_minute"]),
                "episode_coverage_pct": 100.0 * _float(row["episode_coverage"]),
                "median_capped_delay_s": _float(row["median_capped_delay_s"]),
            }
        )
    return pl.DataFrame(records, strict=False).sort(["budget_pct", "family_order"])


def lease_timeout_table(action: pl.DataFrame, episodes: pl.DataFrame) -> pl.DataFrame:
    """Return locked-controller lease-construction timeout sensitivity."""

    keep = ["proposed", "stateless_reference"]
    action_subset = action.filter(pl.col("family").is_in(keep)).select(
        "candidate",
        "family",
        "lease_timeout_s",
        "epochs",
        "benign_friction",
        "malicious_allow",
        "malicious_not_isolated",
        "transitions_per_1000_observed_epochs",
    )
    episode_subset = episodes.filter(pl.col("family").is_in(keep)).select(
        "candidate",
        "lease_timeout_s",
        "episode_count",
        "episode_coverage",
        "mean_capped_delay_s",
        "median_capped_delay_s",
    )
    joined = action_subset.join(
        episode_subset,
        on=["candidate", "lease_timeout_s"],
        how="inner",
        validate="1:1",
    )
    records: list[dict[str, Any]] = []
    for row in joined.iter_rows(named=True):
        family = str(row["family"])
        records.append(
            {
                "lease_timeout_s": _float(row["lease_timeout_s"]),
                "family_order": FAMILY_ORDER[family],
                "policy": FAMILY_LABELS[family],
                "epochs": int(row["epochs"]),
                "episodes": int(row["episode_count"]),
                "benign_friction_pct": 100.0 * _float(row["benign_friction"]),
                "malicious_allow_pct": 100.0 * _float(row["malicious_allow"]),
                "malicious_not_isolated_pct": 100.0 * _float(row["malicious_not_isolated"]),
                "transitions_per_1000_observed_epochs": _float(
                    row["transitions_per_1000_observed_epochs"]
                ),
                "episode_coverage_pct": 100.0 * _float(row["episode_coverage"]),
                "mean_capped_delay_s": _float(row["mean_capped_delay_s"]),
                "median_capped_delay_s": _float(row["median_capped_delay_s"]),
                "scope_note": "locked thresholds; lease construction changed without refit",
            }
        )
    return pl.DataFrame(records, strict=False).sort(["lease_timeout_s", "family_order"])


def robustness_table(
    inference: Mapping[str, Any],
    _hgb: Mapping[str, Any],
    timebase_selection: pl.DataFrame,
    timebase_reference: pl.DataFrame,
    lease: pl.DataFrame,
) -> pl.DataFrame:
    """Summarize confirmatory robustness and explicitly failed sensitivities."""

    records: list[dict[str, Any]] = []

    def bootstrap_record(
        analysis: str,
        grouping: str,
        report: Mapping[str, Any],
        median: Mapping[str, Any],
        friction: Mapping[str, Any] | None,
        status: str,
        limitation: str,
    ) -> dict[str, Any]:
        metrics = report["metrics"]
        exposure = _metric(metrics, "attack_exposure_difference")
        coverage = _metric(metrics, "episode_coverage_difference")
        churn = _metric(metrics, "transition_rate_ratio")
        delay = median["interval"]
        friction_interval = friction["interval"] if friction else None
        return {
            "analysis": analysis,
            "grouping_or_change": grouping,
            "trace_blocks": int(report["clusters"]),
            "exposure_difference_pp": 100.0 * _float(exposure["estimate"]),
            "exposure_95ci": f"[{100.0 * _float(exposure['ci_lower']):.2f}, {100.0 * _float(exposure['ci_upper']):.2f}] pp",
            "coverage_difference_pp": 100.0 * _float(coverage["estimate"]),
            "coverage_95ci": f"[{100.0 * _float(coverage['ci_lower']):.2f}, {100.0 * _float(coverage['ci_upper']):.2f}] pp",
            "transition_rate_ratio": _float(churn["estimate"]),
            "transition_ratio_95ci": f"[{_float(churn['ci_lower']):.3f}, {_float(churn['ci_upper']):.3f}]",
            "median_delay_difference_s": _float(delay["estimate"]),
            "proposed_friction_pct": (
                100.0 * _float(friction_interval["estimate"])
                if friction_interval
                else float("nan")
            ),
            "friction_one_sided_95_upper_pct": (
                100.0 * _float(friction_interval["one_sided_upper"])
                if friction_interval
                else float("nan")
            ),
            "evidence_status": status,
            "limitation": limitation,
        }

    primary = inference["primary"]
    primary_passes = sum(bool(gate["passed"]) for gate in primary["primary_gates"])
    records.append(
        bootstrap_record(
            "Primary logistic score",
            "15-min trace blocks",
            primary["paired_additive_bootstrap"],
            primary["paired_median_delay_bootstrap"],
            primary["proposed_friction_bootstrap"],
            f"{primary_passes}/4 predeclared gates pass; friction UCB crosses 1%",
            (
                "point friction is below 1%, but its one-sided 95% upper bound is above 1%; "
                "descriptive mean capped delay worsens"
            ),
        )
    )

    for novelty_key, label in (("seen", "Seen numeric RNTI"), ("unseen", "Unseen numeric RNTI")):
        novelty = inference["rnti_novelty_strata"][novelty_key]
        stratum_inference = novelty["inference"]
        passes = sum(bool(gate["passed"]) for gate in stratum_inference["primary_gates"])
        point_ok = bool(
            stratum_inference["descriptive_safeguards"]["friction_point_within_budget"]
        )
        ucb_ok = bool(
            stratum_inference["descriptive_safeguards"][
                "friction_one_sided_ucb_within_budget"
            ]
        )
        records.append(
            bootstrap_record(
                "Numeric-RNTI stratum",
                f"{label} ({novelty['numeric_rnti_count']} values)",
                stratum_inference["paired_additive_bootstrap"],
                stratum_inference["paired_median_delay_bootstrap"],
                stratum_inference["proposed_friction_bootstrap"],
                (
                    f"{passes}/4 pairwise gates pass; friction point "
                    f"{'within' if point_ok else 'above'} 1% and UCB "
                    f"{'within' if ucb_ok else 'above'} 1%"
                ),
                "descriptive numeric-RNTI occurrence stratum, not durable identity",
            )
        )
    one_hour = inference["one_hour_cluster_sensitivity"]
    one_hour_passes = sum(bool(gate["passed"]) for gate in one_hour["primary_gates"])
    records.append(
        bootstrap_record(
            "Coarser resampling",
            "1-h trace groups",
            one_hour["paired_additive_bootstrap"],
            one_hour["paired_median_delay_bootstrap"],
            one_hour["proposed_friction_bootstrap"],
            f"{one_hour_passes}/4 predeclared gates pass",
            "only 14 resampling groups; friction upper bound remains above 1%",
        )
    )
    hgb_sensitivity = inference["risk_model_sensitivity"]
    hgb_selected = hgb_sensitivity["selected_ewma_vs_stateless"]
    hgb_selected_passes = sum(
        bool(gate["passed"]) for gate in hgb_selected["primary_gates"]
    )
    records.append(
        bootstrap_record(
            "HGB score / constrained selection",
            "5-s EWMA selected; 15-min trace blocks",
            hgb_selected["paired_additive_bootstrap"],
            hgb_selected["paired_median_delay_bootstrap"],
            hgb_selected["proposed_friction_bootstrap"],
            f"{hgb_selected_passes}/4 controller gates pass; friction exceeds 1%",
            (
                "strict tune-only selection changes controller family under HGB; "
                "held-out friction is 1.186% with one-sided UCB 1.890%"
            ),
        )
    )
    hgb_diagnostic = hgb_sensitivity["diagnostic_asymmetric_vs_stateless"]
    records.append(
        bootstrap_record(
            "HGB asymmetric-family diagnostic",
            "15-min trace blocks",
            hgb_diagnostic["paired_additive_bootstrap"],
            hgb_diagnostic["paired_median_delay_bootstrap"],
            hgb_diagnostic["proposed_friction_bootstrap"],
            "strict coverage gate failed",
            "one-sided coverage lower bound is below the -5 pp noninferiority margin",
        )
    )

    episode_sensitivities = inference["episode_definition_sensitivities"]
    for key, label in (
        ("strict_zero_missing_gap", "Strict zero-gap episodes"),
        ("mixed_attack_epoch_exclusion", "Exclude mixed-attack epochs"),
    ):
        sensitivity = episode_sensitivities[key]
        summary = pl.DataFrame(sensitivity["summary"], strict=False)
        proposal = _only_row(summary, candidate=PRIMARY_PROPOSAL)
        reference = _only_row(summary, candidate=PRIMARY_REFERENCE)
        records.append(
            {
                "analysis": "Episode-definition sensitivity",
                "grouping_or_change": label,
                "trace_blocks": None,
                "exposure_difference_pp": 100.0
                * (_float(proposal["malicious_allow"]) - _float(reference["malicious_allow"])),
                "exposure_95ci": "point only",
                "coverage_difference_pp": 100.0
                * (_float(proposal["episode_coverage"]) - _float(reference["episode_coverage"])),
                "coverage_95ci": "point only",
                "transition_rate_ratio": float("nan"),
                "transition_ratio_95ci": "actions unchanged",
                "median_delay_difference_s": _float(proposal["median_capped_delay_s"])
                - _float(reference["median_capped_delay_s"]),
                "proposed_friction_pct": float("nan"),
                "friction_one_sided_95_upper_pct": float("nan"),
                "evidence_status": "descriptive; no action replay or refit",
                "limitation": (
                    f"episode count changes to {int(proposal['episode_count'])}; "
                    "coverage is definition-sensitive"
                ),
            }
        )

    reference = _only_row(timebase_reference, friction_budget=PRIMARY_BUDGET)
    proposal_rows = timebase_selection.filter(pl.col("family") == "proposed")
    has_proposal = proposal_rows.height > 0 and proposal_rows["candidate"].drop_nulls().len() > 0
    records.append(
        {
            "analysis": "Alternative timestamp scaling",
            "grouping_or_change": "raw timestamp / 1e6",
            "trace_blocks": None,
            "exposure_difference_pp": float("nan"),
            "exposure_95ci": "not run",
            "coverage_difference_pp": float("nan"),
            "coverage_95ci": "not run",
            "transition_rate_ratio": float("nan"),
            "transition_ratio_95ci": "not run",
            "median_delay_difference_s": float("nan"),
            "proposed_friction_pct": float("nan"),
            "friction_one_sided_95_upper_pct": float("nan"),
            "evidence_status": (
                "unexpected feasible proposal" if has_proposal else "no feasible proposed policy at 1%"
            ),
            "limitation": (
                "no held-out controller replay; 1% tuning stateless exposure "
                f"was {100.0 * _float(reference['malicious_allow']):.2f}%"
            ),
        }
    )

    proposal_lease = lease.filter(pl.col("policy") == "Friction-budgeted")
    reference_lease = lease.filter(pl.col("policy") == "Stateless")
    joined_lease = proposal_lease.join(
        reference_lease,
        on="lease_timeout_s",
        suffix="_reference",
        validate="1:1",
    ).with_columns(
        (
            pl.col("malicious_allow_pct") - pl.col("malicious_allow_pct_reference")
        ).alias("exposure_delta"),
        (
            pl.col("transitions_per_1000_observed_epochs")
            / pl.col("transitions_per_1000_observed_epochs_reference")
        ).alias("churn_ratio"),
    )
    records.append(
        {
            "analysis": "Lease-timeout sensitivity",
            "grouping_or_change": "5--300 s label-blind leases",
            "trace_blocks": None,
            "exposure_difference_pp": _float(joined_lease["exposure_delta"].mean()),
            "exposure_95ci": (
                f"range [{joined_lease['exposure_delta'].min():.2f}, "
                f"{joined_lease['exposure_delta'].max():.2f}] pp"
            ),
            "coverage_difference_pp": float("nan"),
            "coverage_95ci": "episode definition changes",
            "transition_rate_ratio": _float(joined_lease["churn_ratio"].mean()),
            "transition_ratio_95ci": (
                f"range [{joined_lease['churn_ratio'].min():.3f}, "
                f"{joined_lease['churn_ratio'].max():.3f}]"
            ),
            "median_delay_difference_s": _float(
                (
                    joined_lease["median_capped_delay_s"]
                    - joined_lease["median_capped_delay_s_reference"]
                ).max()
            ),
            "proposed_friction_pct": float("nan"),
            "friction_one_sided_95_upper_pct": float("nan"),
            "evidence_status": "descriptive direction stable",
            "limitation": "locked thresholds; episode counts vary with lease construction",
        }
    )
    return pl.DataFrame(records, strict=False)


def _latex_escape(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "--"
        return f"{value:.3f}"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _write_latex_table(
    frame: pl.DataFrame,
    path: Path,
    columns: Sequence[tuple[str, str]],
    caption: str,
    label: str,
) -> None:
    alignment = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        " & ".join(_latex_escape(header) for _, header in columns) + r" \\",
        r"\hline",
    ]
    for row in frame.iter_rows(named=True):
        lines.append(
            " & ".join(_latex_escape(row[column]) for column, _ in columns) + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_table(
    frame: pl.DataFrame,
    stem: str,
    output_dir: Path,
    latex_columns: Sequence[tuple[str, str]],
    caption: str,
    label: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    tex_path = output_dir / f"{stem}.tex"
    float_columns = [
        column
        for column, dtype in frame.schema.items()
        if dtype in (pl.Float32, pl.Float64)
    ]
    clean = frame.with_columns(
        [
            pl.when(pl.col(column).is_nan())
            .then(None)
            .otherwise(pl.col(column))
            .alias(column)
            for column in float_columns
        ]
    )
    clean.write_csv(csv_path, float_precision=6)
    _write_latex_table(clean, tex_path, latex_columns, caption, label)
    return [csv_path, tex_path]


def _configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def _save_figure(fig: Any, stem: str, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fixed_time = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "oran.reporting"},
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "oran.reporting",
            "Producer": "matplotlib",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    return [png, pdf]


def _bar_labels(axis: Any, bars: Any, decimals: int = 1) -> None:
    for bar in bars:
        height = float(bar.get_height())
        axis.annotate(
            f"{height:.{decimals}f}",
            (bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )


def plot_heldout_policy(frame: pl.DataFrame, output_dir: Path) -> list[Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    labels = frame["policy"].to_list()
    families = [
        "proposed", "stateless_reference", "ewma", "n_report", "symmetric_hysteresis"
    ]
    colors = [FAMILY_COLORS[item] for item in families]
    hatches = [FAMILY_HATCHES[item] for item in families]
    x = np.arange(frame.height)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.6), constrained_layout=True)
    panels = [
        ("benign_friction_pct", "Benign friction (%)", "(a) Friction"),
        ("malicious_allow_pct", "Malicious ALLOW time (%)", "(b) Exposure"),
        (
            "transitions_per_1000_observed_epochs",
            "Effective transitions / 1,000 epochs",
            "(c) Action churn",
        ),
    ]
    for axis, (column, ylabel, title) in zip(axes, panels, strict=True):
        bars = axis.bar(x, frame[column].to_numpy(), color=colors, edgecolor="#222222", linewidth=0.55)
        for bar, hatch in zip(bars, hatches, strict=True):
            bar.set_hatch(hatch)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels, rotation=28, ha="right")
        axis.set_ylim(bottom=0)
        _bar_labels(axis, bars)
    fig.suptitle("Locked policies on chronological held-out trace blocks", fontweight="bold", fontsize=10)
    fig.text(
        0.5,
        -0.04,
        "RNTI-level offline replay at the 1% tuning budget; bars are point estimates.",
        ha="center",
        fontsize=7,
    )
    paths = _save_figure(fig, "fig_heldout_policy_comparison", output_dir)
    plt.close(fig)
    return paths


def plot_per_attack(frame: pl.DataFrame, output_dir: Path) -> list[Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    y = np.arange(frame.height)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.6), sharey=True, constrained_layout=True)
    panels = [
        ("exposure_difference_pp", "exposure_ci_lower_pp", "exposure_ci_upper_pp", "Δ ALLOW time (pp)", "(a) Exposure"),
        ("coverage_difference_pp", "coverage_ci_lower_pp", "coverage_ci_upper_pp", "Δ coverage (pp)", "(b) Coverage"),
        ("mean_delay_difference_s", "mean_delay_ci_lower_s", "mean_delay_ci_upper_s", "Δ mean capped delay (s)", "(c) Delay"),
    ]
    for axis, (point, lower, upper, xlabel, title) in zip(axes, panels, strict=True):
        axis.axhspan(2.5, 3.5, color="#FDE0DD", alpha=0.55, zorder=0)
        axis.axvline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        for index, row in enumerate(frame.iter_rows(named=True)):
            value = _float(row[point])
            lo = _float(row[lower])
            hi = _float(row[upper])
            failure = bool(row["known_failure_mode"])
            axis.errorbar(
                value,
                index,
                xerr=np.array([[value - lo], [hi - value]]),
                fmt="D" if failure else "o",
                color="#D55E00" if failure else "#0072B2",
                markerfacecolor="white" if failure else "#0072B2",
                markeredgewidth=1.1,
                capsize=2.2,
                linewidth=1.1,
                markersize=4.5,
                zorder=3,
            )
        axis.set_xlabel(xlabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x")
        axis.grid(axis="y", visible=False)
        lo = min(0.0, float(frame[lower].min()))
        hi = max(0.0, float(frame[upper].max()))
        padding = max(0.12 * (hi - lo), 0.15)
        axis.set_xlim(lo - padding, hi + padding)
    axes[0].set_yticks(y, [ATTACK_LABELS[item] for item in frame["attack_type"].to_list()])
    axes[0].invert_yaxis()
    for label in axes[0].get_yticklabels():
        if "Slowloris" in label.get_text():
            label.set_color("#D55E00")
            label.set_fontweight("bold")
    fig.suptitle("Per-attack proposal minus stateless reference", fontweight="bold", fontsize=10)
    fig.text(
        0.5,
        -0.04,
        "95% trace-block bootstrap CIs. †Slowloris is an explicit failure mode: coverage and delay worsen.",
        ha="center",
        fontsize=7,
    )
    paths = _save_figure(fig, "fig_per_attack_contrasts", output_dir)
    plt.close(fig)
    return paths


def plot_unseen_seen(frame: pl.DataFrame, output_dir: Path) -> list[Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["Seen", "Unseen"]
    x = np.arange(2)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.5), constrained_layout=True)
    panels = [
        (
            "exposure_difference_pp",
            "exposure_ci_lower_pp",
            "exposure_ci_upper_pp",
            "Proposal − static ALLOW time (pp)",
            "(a) Exposure contrast",
        ),
        (
            "transition_rate_ratio",
            "transition_ratio_ci_lower",
            "transition_ratio_ci_upper",
            "Proposal / static churn",
            "(b) Stability contrast",
        ),
        (
            "proposal_benign_friction_pct",
            "proposal_friction_ci_lower_pct",
            "proposal_friction_ci_upper_pct",
            "Proposal benign friction (%)",
            "(c) Friction",
        ),
    ]
    for axis, (point, lower, upper, ylabel, title) in zip(axes, panels, strict=True):
        for index, row in enumerate(frame.iter_rows(named=True)):
            value = _float(row[point])
            lo = _float(row[lower])
            hi = _float(row[upper])
            unseen = index == 1
            axis.errorbar(
                index,
                value,
                yerr=np.array([[value - lo], [hi - value]]),
                fmt="D" if unseen else "o",
                color="#D55E00" if unseen else "#0072B2",
                markerfacecolor="white" if unseen else "#0072B2",
                markeredgewidth=1.1,
                capsize=3,
                linewidth=1.2,
                markersize=5,
                zorder=3,
            )
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
    axes[0].axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[1].axhline(0.75, color="#555555", linewidth=0.8, linestyle=":")
    axes[2].axhline(1.0, color="#555555", linewidth=0.8, linestyle=":")
    unseen = frame.row(1, named=True)
    axes[2].scatter(
        1,
        unseen["proposal_friction_one_sided_95_upper_pct"],
        marker="_",
        s=90,
        linewidth=1.5,
        color="#D55E00",
        zorder=4,
    )
    axes[2].annotate(
        f"UCB {unseen['proposal_friction_one_sided_95_upper_pct']:.2f}%",
        (1, unseen["proposal_friction_one_sided_95_upper_pct"]),
        xytext=(-3, 5),
        textcoords="offset points",
        ha="right",
        fontsize=6.2,
        color="#D55E00",
    )
    fig.suptitle("Descriptive numeric-RNTI strata", fontweight="bold", fontsize=10)
    fig.text(
        0.5,
        -0.04,
        "95% trace-block CIs; orange diamond: unseen. Dotted lines: 0.75 churn gate and 1% friction budget; RNTI is not identity.",
        ha="center",
        fontsize=7,
    )
    paths = _save_figure(fig, "fig_unseen_seen_rnti", output_dir)
    plt.close(fig)
    return paths


def plot_tuning_frontier(
    candidates: pl.DataFrame,
    frontier: pl.DataFrame,
    output_dir: Path,
) -> list[Path]:
    _configure_matplotlib()
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)
    family_keys = ["proposed", "stateless", "ewma", "n_report", "symmetric_hysteresis"]
    for family in family_keys:
        cloud = candidates.filter(pl.col("family") == family)
        if cloud.height:
            x = 100.0 * cloud["benign_friction"].to_numpy()
            axes[0].scatter(
                x,
                100.0 * cloud["malicious_allow"].to_numpy(),
                s=9,
                marker=FAMILY_MARKERS[family],
                color=FAMILY_COLORS[family],
                alpha=0.14,
                linewidths=0,
            )
            axes[1].scatter(
                x,
                cloud["transitions_per_minute"].to_numpy(),
                s=9,
                marker=FAMILY_MARKERS[family],
                color=FAMILY_COLORS[family],
                alpha=0.14,
                linewidths=0,
            )

        label = FAMILY_LABELS[family]
        selected = frontier.filter(pl.col("policy_family") == label).filter(
            pl.col("candidate").is_not_null() & pl.col("realized_friction_pct").is_finite()
        )
        for row in selected.iter_rows(named=True):
            diagnostic = row["status"] not in {"selected", "budget reference"}
            for axis, metric in zip(
                axes,
                ("malicious_allow_pct", "transitions_per_observed_rnti_minute"),
                strict=True,
            ):
                axis.scatter(
                    row["realized_friction_pct"],
                    row[metric],
                    s=30,
                    marker=FAMILY_MARKERS[family],
                    facecolor="white" if diagnostic else FAMILY_COLORS[family],
                    edgecolor=FAMILY_COLORS[family],
                    linewidth=1.0,
                    zorder=4,
                )

    for axis in axes:
        axis.set_xscale("log")
        axis.axvline(1.0, color="#000000", linewidth=0.8, linestyle=":")
        axis.set_xlabel("Realized benign friction (%)")
        axis.grid(axis="both")
    axes[0].set_ylabel("Malicious ALLOW time (%)")
    axes[0].set_title("(a) Security–friction", loc="left", fontweight="bold")
    axes[1].set_ylabel("Effective transitions / observed RNTI-minute")
    axes[1].set_title("(b) Stability–friction", loc="left", fontweight="bold")

    handles = [
        mlines.Line2D(
            [],
            [],
            color=FAMILY_COLORS[family],
            marker=FAMILY_MARKERS[family],
            linestyle="None",
            label=FAMILY_LABELS[family],
            markersize=5,
        )
        for family in family_keys
    ]
    axes[0].legend(handles=handles, frameon=False, ncol=2, loc="best")
    fig.suptitle("Matched-budget controller-tuning frontier", fontweight="bold", fontsize=10)
    fig.text(
        0.5,
        -0.04,
        "Controller-tuning trace blocks (not held-out test). Faint points: calibrated templates; open points: gate-failing diagnostics.",
        ha="center",
        fontsize=7,
    )
    paths = _save_figure(fig, "fig_tuning_security_stability_frontier", output_dir)
    plt.close(fig)
    return paths


def plot_lease_timeout(frame: pl.DataFrame, output_dir: Path) -> list[Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55), constrained_layout=True)
    panels = [
        ("malicious_allow_pct", "Malicious ALLOW time (%)", "(a) Exposure"),
        ("transitions_per_1000_observed_epochs", "Transitions / 1,000 epochs", "(b) Churn"),
        ("episode_coverage_pct", "Episode coverage (%)", "(c) Coverage"),
    ]
    for family, label in (("Friction-budgeted", "Friction-budgeted"), ("Stateless", "Stateless")):
        selected = frame.filter(pl.col("policy") == family).sort("lease_timeout_s")
        key = "proposed" if family == "Friction-budgeted" else "stateless_reference"
        for axis, (metric, ylabel, title) in zip(axes, panels, strict=True):
            axis.plot(
                selected["lease_timeout_s"].to_numpy(),
                selected[metric].to_numpy(),
                color=FAMILY_COLORS[key],
                marker=FAMILY_MARKERS[key],
                linewidth=1.25,
                markersize=4,
                label=label,
            )
            axis.set_xscale("log")
            axis.set_xlabel("Lease timeout (s)")
            axis.set_ylabel(ylabel)
            axis.set_title(title, loc="left", fontweight="bold")
            axis.grid(axis="both")
    axes[0].legend(frameon=False)
    fig.suptitle("Lease-construction sensitivity with locked thresholds", fontweight="bold", fontsize=10)
    fig.text(
        0.5,
        -0.04,
        "RNTI-level offline replay; changing the timeout also changes attack-episode counts.",
        ha="center",
        fontsize=7,
    )
    paths = _save_figure(fig, "fig_lease_timeout_sensitivity", output_dir)
    plt.close(fig)
    return paths


def _records(frame: pl.DataFrame, *, drop: Sequence[str] = ()) -> list[dict[str, Any]]:
    selected = frame.drop([column for column in drop if column in frame.columns])
    return selected.to_dicts()


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats so the digest is strict RFC-compatible JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def generate_outputs(project_root: Path, reports_root: Path) -> dict[str, Any]:
    """Render all reporting outputs and return the deterministic result digest."""

    project_root = project_root.resolve()
    reports_root = reports_root.resolve()
    paths = {key: project_root / relative for key, relative in INPUT_PATHS.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing locked reporting inputs: " + ", ".join(missing))

    aggregate = pl.read_parquet(paths["aggregate"])
    episodes = pl.read_parquet(paths["episodes"])
    strata = pl.read_parquet(paths["strata"])
    inference = _load_json(paths["inference_v3"])
    hgb = _load_json(paths["hgb_bootstrap"])

    heldout = heldout_policy_table(aggregate, episodes)
    attacks = per_attack_table(
        episodes,
        inference["per_attack"],
    )
    unseen = unseen_seen_table(strata, inference["rnti_novelty_strata"])
    tuning_candidates = pl.read_parquet(paths["tuning_candidates"])
    frontier = tuning_frontier_table(
        pl.read_parquet(paths["tuning_selection"]),
        pl.read_parquet(paths["tuning_references"]),
    )
    lease = lease_timeout_table(
        pl.read_parquet(paths["lease_action"]),
        pl.read_parquet(paths["lease_episodes"]),
    )
    robustness = robustness_table(
        inference,
        hgb,
        pl.read_parquet(paths["timebase_selection"]),
        pl.read_parquet(paths["timebase_reference"]),
        lease,
    )

    tables_dir = reports_root / "tables"
    figures_dir = reports_root / "figures"
    generated: list[Path] = []
    generated += _write_table(
        heldout.drop("policy_order"),
        "table_heldout_policy_comparison",
        tables_dir,
        [
            ("policy", "Policy"),
            ("benign_friction_pct", "Friction (%)"),
            ("malicious_allow_pct", "ALLOW (%)"),
            ("malicious_not_isolated_pct", "Not isolated (%)"),
            ("transitions_per_1000_observed_epochs", "Transitions/1k"),
            ("episode_coverage_pct", "Coverage (%)"),
            ("median_capped_delay_s", "Median delay (s)"),
        ],
        "Chronological held-out RNTI-level offline policy replay at the locked 1% tuning budget.",
        "tab:heldout-policy",
    )
    generated += _write_table(
        attacks.drop("attack_order"),
        "table_per_attack_contrasts",
        tables_dir,
        [
            ("attack", "Attack"),
            ("episodes", "Episodes"),
            ("exposure_difference_pp", "Delta ALLOW (pp)"),
            ("exposure_ci_lower_pp", "ALLOW CI low"),
            ("exposure_ci_upper_pp", "ALLOW CI high"),
            ("coverage_difference_pp", "Delta coverage (pp)"),
            ("mean_delay_difference_s", "Delta mean delay (s)"),
        ],
        "Proposal minus stateless-reference per-attack contrasts with 95% trace-block bootstrap intervals. Slowloris is the explicit failure mode.",
        "tab:per-attack",
    )
    generated += _write_table(
        unseen.drop("stratum_order"),
        "table_unseen_seen_rnti",
        tables_dir,
        [
            ("numeric_rnti_stratum", "Numeric RNTI stratum"),
            ("numeric_rnti_count", "RNTIs"),
            ("epochs", "Epochs"),
            ("exposure_difference_pp", "Delta ALLOW (pp)"),
            ("exposure_ci_lower_pp", "ALLOW CI low"),
            ("exposure_ci_upper_pp", "ALLOW CI high"),
            ("transition_rate_ratio", "Churn ratio"),
            ("proposal_benign_friction_pct", "Proposal friction (%)"),
            ("proposal_friction_one_sided_95_upper_pct", "Friction UCB (%)"),
        ],
        "Descriptive held-out numeric-RNTI strata; numeric RNTI is not a durable identity.",
        "tab:rnti-strata",
    )
    generated += _write_table(
        frontier.drop("family_order"),
        "table_tuning_frontier",
        tables_dir,
        [
            ("budget_pct", "Budget (%)"),
            ("policy_family", "Family"),
            ("status", "Status"),
            ("realized_friction_pct", "Friction (%)"),
            ("malicious_allow_pct", "ALLOW (%)"),
            ("transitions_per_observed_rnti_minute", "Transitions/min"),
        ],
        "Strict structural selections and diagnostics on controller-tuning trace blocks.",
        "tab:tuning-frontier",
    )
    generated += _write_table(
        robustness,
        "table_robustness_sensitivity",
        tables_dir,
        [
            ("analysis", "Analysis"),
            ("grouping_or_change", "Change"),
            ("exposure_difference_pp", "Delta ALLOW (pp)"),
            ("coverage_difference_pp", "Delta coverage (pp)"),
            ("transition_rate_ratio", "Churn ratio"),
            ("evidence_status", "Evidence status"),
        ],
        "Robustness and sensitivity results. Failed or unsupported analyses remain explicit.",
        "tab:robustness",
    )
    generated += _write_table(
        lease.drop("family_order"),
        "table_lease_timeout_sensitivity",
        tables_dir,
        [
            ("lease_timeout_s", "Timeout (s)"),
            ("policy", "Policy"),
            ("malicious_allow_pct", "ALLOW (%)"),
            ("transitions_per_1000_observed_epochs", "Transitions/1k"),
            ("episode_coverage_pct", "Coverage (%)"),
            ("median_capped_delay_s", "Median delay (s)"),
        ],
        "Locked-controller sensitivity to label-blind RNTI lease construction.",
        "tab:lease-timeout",
    )

    generated += plot_heldout_policy(heldout, figures_dir)
    generated += plot_per_attack(attacks, figures_dir)
    generated += plot_unseen_seen(unseen, figures_dir)
    generated += plot_tuning_frontier(tuning_candidates, frontier, figures_dir)
    generated += plot_lease_timeout(lease, figures_dir)

    proposal = _only_row(heldout, role="proposed")
    reference = _only_row(heldout, role="static reference")
    primary_inference = inference["primary"]
    paired_metrics = primary_inference["paired_additive_bootstrap"]["metrics"]
    exposure_interval = _metric(paired_metrics, "attack_exposure_difference")
    coverage_interval = _metric(paired_metrics, "episode_coverage_difference")
    churn_interval = _metric(paired_metrics, "transition_rate_ratio")
    median_interval = primary_inference["paired_median_delay_bootstrap"]["interval"]
    friction_interval = primary_inference["proposed_friction_bootstrap"]["interval"]

    digest: dict[str, Any] = {
        "schema_version": 1,
        "scope": {
            "evaluation": "chronological held-out trace blocks; RNTI-level offline replay",
            "actions": ["ALLOW", "RESTRICT", "ISOLATE"],
            "claim_boundary": (
                "action-state replay and stability/security tradeoffs only; no causal attack-prevention "
                "or durable-identity claim"
            ),
        },
        "primary_result": {
            "budget_pct": 100.0 * PRIMARY_BUDGET,
            "proposal_benign_friction_pct": proposal["benign_friction_pct"],
            "proposal_friction_95ci_pct": [
                100.0 * friction_interval["ci_lower"],
                100.0 * friction_interval["ci_upper"],
            ],
            "proposal_friction_one_sided_95_upper_pct": 100.0
            * friction_interval["one_sided_upper"],
            "proposal_minus_static_exposure_pp": 100.0 * exposure_interval["estimate"],
            "exposure_95ci_pp": [
                100.0 * exposure_interval["ci_lower"],
                100.0 * exposure_interval["ci_upper"],
            ],
            "proposal_minus_static_episode_coverage_pp": 100.0
            * coverage_interval["estimate"],
            "coverage_95ci_pp": [
                100.0 * coverage_interval["ci_lower"],
                100.0 * coverage_interval["ci_upper"],
            ],
            "proposal_divided_by_static_transition_rate": churn_interval["estimate"],
            "transition_ratio_95ci": [
                churn_interval["ci_lower"],
                churn_interval["ci_upper"],
            ],
            "transition_reduction_pct": 100.0 * (1.0 - churn_interval["estimate"]),
            "proposal_minus_static_median_capped_delay_s": median_interval["estimate"],
            "proposal_malicious_not_isolated_pct": proposal["malicious_not_isolated_pct"],
            "static_malicious_not_isolated_pct": reference["malicious_not_isolated_pct"],
            "predeclared_gate_results": primary_inference["primary_gates"],
            "all_four_predeclared_gates_pass": all(
                bool(gate["passed"]) for gate in primary_inference["primary_gates"]
            ),
            "descriptive_mean_capped_delay_difference_s": primary_inference[
                "descriptive_safeguards"
            ]["mean_capped_delay_difference_s"]["estimate"],
        },
        "per_attack": _records(attacks, drop=("attack_order",)),
        "unseen_seen_numeric_rnti": _records(unseen, drop=("stratum_order",)),
        "robustness_and_sensitivity": _records(robustness),
        "known_failures_and_limits": [
            "Slowloris coverage is lower and mean capped delay is longer under the proposal; pooled results must not conceal this.",
            "The proposal's held-out friction point estimate is below 1%, but the one-sided 95% trace-block bootstrap upper bound exceeds 1%.",
            "Strict HGB tuning selects EWMA and transfers all four controller gates, but held-out friction is 1.186% (one-sided upper bound 1.890%); the asymmetric HGB diagnostic separately misses the coverage gate.",
            "Under the unsupported raw/1e6 timestamp scaling, no proposed policy is feasible at the 1% tuning budget; no held-out replay was run.",
            "For unseen numeric RNTIs, proposed friction is 1.289% (one-sided 95% upper bound 1.727%), above the 1% budget; this stratum does not establish identity generalization.",
            "All results are offline RNTI-level action replays and do not establish causal attack prevention or deployment effects.",
        ],
        "figure_alt_text": {
            "fig_heldout_policy_comparison": "Three bar panels compare benign friction, malicious ALLOW time, and action churn for five locked policies on chronological held-out trace blocks.",
            "fig_per_attack_contrasts": "Three forest-plot panels show proposal-minus-static exposure, episode coverage, and delay contrasts with trace-block bootstrap intervals; Slowloris is marked as a failure.",
            "fig_unseen_seen_rnti": "Three interval panels show proposal-minus-static exposure, churn ratio, and proposal friction for numeric RNTIs seen and unseen during model training; unseen friction exceeds 1%, without an identity interpretation.",
            "fig_tuning_security_stability_frontier": "Two log-friction scatter panels show security and stability tradeoffs among calibrated policy templates on the controller-tuning split; gate-failing points are open.",
            "fig_lease_timeout_sensitivity": "Three line panels show exposure, churn, and episode coverage as the label-blind lease timeout varies from 5 to 300 seconds with thresholds locked.",
        },
        "input_artifact_sha256": {
            INPUT_PATHS[key]: _sha256(path) for key, path in sorted(paths.items())
        },
    }
    digest["generated_output_sha256"] = {
        str(path.relative_to(reports_root)): _sha256(path)
        for path in sorted(generated, key=lambda item: str(item))
    }
    digest = _json_safe(digest)
    digest_path = reports_root / "result_digest.json"
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.write_text(
        json.dumps(digest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return digest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render deterministic tables and figures from locked O-RAN artifacts."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    digest = generate_outputs(args.project_root, args.reports_root)
    print(
        json.dumps(
            {
                "result_digest": str(args.reports_root / "result_digest.json"),
                "primary_result": digest["primary_result"],
                "known_limit_count": len(digest["known_failures_and_limits"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
