# Publication readiness and research handoff

Study: **Friction-Budgeted Stabilization of RNTI-Level Containment in O-RAN**  
Handoff date: 2026-08-09  
Primary dataset: OpenIreland, *RAN Performance measurements for security threats*

## Decision

**Conditional GO for submission as a narrowly scoped, trace-driven IEEE
Networking Letters paper about RNTI-level policy stabilization.** The locked
primary controller passes all four predeclared held-out pairwise gates, and the
main contribution is defensible as the combination of:

1. matched-budget, security-constrained controller selection;
2. label-blind trace-block and RNTI-lease replay with one-epoch action lag;
3. an operational security--friction--stability frontier with chronological,
   attack-specific, and unseen-numeric-RNTI reporting; and
4. honest network-level actions for a currently attached RNTI: `ALLOW`,
   `RESTRICT`, or `ISOLATE`.

This is a **NO-GO** if the intended claim is durable identity, subscriber
authentication, adaptive MFA/access control, causal attack prevention, or
production O-RAN effectiveness. A temporary RNTI is an enforcement handle, not
a durable principal. If the paper must make genuine identity or adaptive-access
claims, an authentication dataset such as LANL must become primary; that study
is not implemented in this repository.

The GO remains conditional because the timestamp unit needs author
confirmation, Slowloris is a material failure mode, held-out friction is not
statistically certified below 1%, and no xApp/RIC deployment has been tested.

## 1. Provenance and critical dataset audit

| Item | Frozen value |
|---|---|
| Source file | `/nobackup/ashukuma/o_ran/dtst.csv` |
| File size | 794,453,804 bytes |
| SHA-256 | `4cc8498466eb7cb258412721ae94a2460f04a7da1235ac07e8e9cd20e15a76a7` |
| DOI / license | `10.17632/t2rzh9y4mp.1` / CC BY 4.0 |
| Raw shape | 3,175,140 rows, 47 columns |
| Labels | Four benign; port scan, DDoS/Ripper, DoS/Hulk, Slowloris |
| Primary time conversion | raw timestamp / `100000` |
| Derived representation | 525,271 causal 1-second epochs |
| Observable blocks / inferred leases | 273 / 5,892 |

The schema/ontology audit found no missing or unexpected source columns, no
unknown or null labels, and no mismatch between `mac_rnti` and the duplicate
`ue_ident` export. There is one reversal in source row order, so chronological
processing explicitly sorts by corrected event time. Constant fields,
identifier duplication, duplicate records, and label/RNTI/scenario coupling
were treated as leakage risks rather than predictors. An exact full-row scan
found 3,175,140 unique rows and zero duplicates. The final feature policy
allows only 25 dynamic radio/MAC KPIs and rejects RNTI/UE identifiers,
timestamps, mobility/scenario metadata, labels, `rf_error`, and sample-count or
post-event fields.

The primary raw/`100000` conversion maps the trace to 24--27 October 2022 and
produces an 89.6 ms median within-RNTI interval; 89.4% of same-label/mobility
intervals are 50--150 ms. That is compatible with the authors' related 100 ms
collection description, but it is still an empirical inference. Literal
raw/`1000000` maps to April 1975 and a median interval near 8.96 ms. Neither
encoding nor epoch has been author-confirmed.

Trace blocks are created without labels: a boundary occurs after a 300-second
global gap or at a maximum 900-second span. They are observable block
surrogates, not verified captures or campaigns. A lease begins at a block
boundary or after 30 seconds of inactivity for a numeric RNTI. Lease-timeout and
coarser-cluster sensitivities quantify the consequences of these choices.

### Critical go/no interpretation

The data pass the practical go/no test for the **narrow network-policy
question**. All four `id_ue` values and each of the five mobility categories
occur with all eight labels, so the label is not perfectly determined by those
scenario fields. Numeric-RNTI reuse is substantial: values spanning exactly
one label account for 395,851 of 3,175,140 rows (12.47%), while the remainder
span two to seven labels. Benign held-out segments cross tuned thresholds, all
evaluated controllers make nontrivial state transitions, the proposed policy
does not collapse to a constant action, and score discrimination transfers to
unseen numeric RNTI values (AUROC 0.9472).

The data do not pass a go/no test for identity generalization: no durable
subscriber identity exists, numeric RNTIs are reused, and the evaluation data
are effectively all `car` mobility. The study therefore makes no
mobility-generalization or identity claim.

## 2. Chronological evaluation protocol

Whole label-blind trace blocks are assigned chronologically.

| Split | Raw rows | Blocks | Causal epochs | Purpose |
|---|---:|---:|---:|---|
| Train | 1,643,891 | 130 | 246,193 | Fit logistic risk model |
| Calibration | 406,134 | 48 | 93,989 | Fit Platt calibrator |
| Controller tune | 557,265 | 39 | 83,528 | Match budgets and select policy |
| Test | 567,850 | 56 | 101,561 | Frozen controller-policy replay |
| **Total** | **3,175,140** | **273** | **525,271** | |

The test partition contains 932 inferred leases and 744 numeric RNTI values:
470 occurred before test and 274 did not. The full trace and preliminary
go/no-go behavior were inspected before protocol lock, and risk diagnostics are
available for every partition. Accordingly, `test` is a frozen chronological
**controller-policy outcome holdout**, not a never-inspected dataset.

Primary score model: regularized logistic regression on 25 KPIs, seed 1729,
then Platt calibration on the separate calibration split. Test AUROC is 0.9513;
seen and unseen numeric-RNTI AUROCs are 0.9541 and 0.9472. The HGB sensitivity
uses the same allowed features and split discipline.

The replay has one-second decisions and a one-epoch actuation lag. The first
effective state of each lease is `ALLOW`; controller state never crosses a
trace-block or lease boundary and never resets at a label transition. Labels
enter offline tuning/evaluation only, never controller state.

## 3. Locked friction-budgeted controller

At benign-friction budget `B = 0.01`, the exact tune-only selection chose
`proposed-template-047-B0.01`, an asymmetric sequential policy with immediate
entry and two-report recovery:

| Parameter | Value |
|---|---:|
| `restrict_enter` / `restrict_exit` | 0.9550303 / 0.9050303 |
| `isolate_enter` / `isolate_exit` | 0.9968761 / 0.9468761 |
| Entry / recovery reports | 1 / 2 |
| Minimum hold time / EWMA | 0 s / none |

The matched stateless reference threshold is 0.8753676. Both consume 0.998797%
benign friction on the tuning split. At that matched point, proposed malicious
`ALLOW` time is 23.360% versus 26.845%, and transitions are 2.480 versus 6.759
per observed RNTI-minute, a 63.32% tuning reduction. The novelty is not EWMA or
hysteresis; it is the budgeted constrained formulation, exact matched reference,
evaluation protocol, and measured frontier.

## 4. Primary held-out results

The following values are frozen on 101,561 test epochs, 56 trace blocks, and
752 primary attack episodes. Friction and exposure are time-weighted.

| Metric | Proposed | Stateless | Contrast |
|---|---:|---:|---:|
| Benign friction (`RESTRICT` or `ISOLATE`) | 0.821% | 0.643% | +0.179 pp |
| Benign `ISOLATE` time | 0.253% | 0.643% | -0.389 pp |
| Malicious `ALLOW` time | 22.330% | 26.812% | -4.482 pp |
| Malicious not-`ISOLATE` time | 29.571% | 26.812% | **+2.759 pp** |
| Transitions / 1,000 observed epochs | 36.313 | 85.230 | ratio 0.4261 |
| Transitions / observed RNTI-minute | 2.179 | 5.114 | -57.39% |
| Episode coverage (at least `RESTRICT`) | 82.979% | 85.771% | -2.793 pp |
| Median capped onset-to-`RESTRICT` delay | 1.000 s | 1.000 s | 0.000 s |
| Mean capped delay, descriptive | 5.703 s | 4.152 s | **+1.552 s** |

The decrease in malicious `ALLOW` time is not the same as stronger full
isolation. The proposed controller intentionally separates `RESTRICT` from
`ISOLATE`; consequently, malicious not-`ISOLATE` time is worse than the
stateless reference. Both metrics must remain in the paper.

### Predeclared formal gates

Inference uses 5,000 paired bootstrap resamples of complete observable trace
blocks, seed 1729. Gate decisions use one-sided 95% endpoints; the table also
shows two-sided 95% intervals where applicable.

| Gate | Point estimate (95% CI) | One-sided endpoint | Rule | Result |
|---|---:|---:|---:|---|
| Proposed minus stateless malicious-`ALLOW` exposure | -4.482 pp `[-5.095, -3.887]` | -3.986 pp | upper <= +2 pp | Pass |
| Proposed minus stateless episode coverage | -2.793 pp `[-3.964, -1.693]` | -3.778 pp | lower >= -5 pp | Pass |
| Proposed minus stateless median capped delay | 0 s `[0, 0]` | 0 s | upper <= +1 s | Pass |
| Proposed/stateless transition-rate ratio | 0.4261 `[0.4042, 0.4518]` | 0.4474 | upper <= 0.75 | Pass |

All four predeclared pairwise gates pass. This does **not** establish a
population-level 1% friction guarantee: proposed held-out friction is 0.8214%
with two-sided 95% CI `[0.5544%, 1.1208%]` and one-sided 95% upper bound
`1.0694%`. The point meets the budget; the conservative upper bound does not.
The +1.552-second mean-delay contrast, CI `[1.189, 1.944]`, is a mandatory
descriptive safeguard rather than a predeclared gate.

## 5. Per-attack containment and delay

These attack-specific contrasts are mandatory descriptive results, not a new
family of confirmatory tests.

| Attack (episodes) | Malicious `ALLOW`, proposed / static | Exposure difference, pp (95% CI) | Coverage, proposed / static | Mean delay, proposed / static | Interpretation |
|---|---:|---:|---:|---:|---|
| Port scan (248) | 2.673% / 10.648% | -7.975 `[-8.776, -7.277]` | 92.34% / 87.10% | 1.42 / 1.73 s | Clear benefit |
| DDoS/Ripper (164) | 2.097% / 10.354% | -8.257 `[-9.684, -7.049]` | 89.02% / 89.02% | 1.54 / 1.51 s | Exposure benefit; delay essentially unchanged |
| DoS/Hulk (146) | 1.802% / 5.441% | -3.639 `[-4.812, -2.667]` | 91.10% / 86.99% | 1.02 / 1.00 s | Exposure/coverage benefit |
| **Slowloris (272)** | **78.018% / 76.791%** | **+1.227 `[-0.493, 3.153]`** | **66.91% / 73.90%** | **13.35 / 9.14 s** | **Known failure: coverage -6.99 pp; delay +4.21 s** |

The pooled result must not conceal Slowloris. Its coverage-difference 95% CI is
`[-9.72, -4.38]` pp and its mean-delay difference is +4.210 seconds, 95% CI
`[3.329, 5.163]`.

## 6. Robustness, generalization, and failure analysis

### Seen versus unseen numeric RNTIs

The numeric-RNTI split is occurrence novelty, not identity novelty.

| Stratum | RNTIs / epochs | Exposure difference | Transition ratio | Proposed friction (one-sided 95% upper) | Conclusion |
|---|---:|---:|---:|---:|---|
| Seen before test | 470 / 62,185 | -4.693 pp, CI `[-5.612, -3.765]` | 0.4138, CI `[0.3878, 0.4435]` | 0.559% (0.795%) | Pairwise gates and budget bound pass |
| Unseen before test | 274 / 39,376 | -4.176 pp, CI `[-4.991, -3.344]` | 0.4460, CI `[0.4145, 0.4826]` | **1.289% (1.727%)** | Pairwise effects transfer; budget does not |

The unseen stratum supports transfer of the security/stability contrast to new
numeric RNTI values within this trace. It does not support subscriber identity
or external-network generalization.

### Risk-model sensitivity

The prespecified depth-3 HGB risk scorer has test AUROC 0.9748. The same locked
constrained selection procedure chooses a 5-second EWMA, not the asymmetric
family. On held-out blocks, selected EWMA versus its HGB stateless reference has
exposure difference -4.598 pp (95% CI `[-5.455, -3.784]`), coverage difference
-2.926 pp (`[-4.515, -1.498]`), median-delay difference 0 seconds, and
transition ratio 0.2452 (`[0.2210, 0.2682]`); all four pairwise controller gates
pass. Its held-out benign friction is 1.186%, however, with a one-sided 95%
upper bound of 1.890%, so the scorer-specific result misses the budget.

The asymmetric HGB candidate remains diagnostic because it was coverage
inferior by 0.59 percentage points in tuning. On test, its transition ratio is
0.4304 and exposure difference is -2.970 pp, but the one-sided coverage lower
endpoint is -5.572 pp, below the -5 pp margin. Thus the framework transfers,
while the asymmetric mechanism is not universally selected.

### Timestamp sensitivity

Under literal raw/`1000000`, the pipeline produces only 64,294 epochs and 29
observable blocks; logistic test AUROC is 0.9305. At the 1% tuning budget there
is no budget-matched asymmetric proposal. The EWMA point has only 14.5% churn
reduction, below the required 25%, and the stateless reference has 41.95%
malicious `ALLOW` exposure. No held-out controller replay or claim was made for
this unsupported timebase. Author confirmation of the time unit is therefore a
publication-critical provenance action.

### Lease and episode definitions

Changing the label-blind inactivity timeout from 5 to 300 seconds without
refitting scores or retuning thresholds preserves the main direction:
proposed-minus-static malicious `ALLOW` exposure ranges from -4.51 to -3.85 pp,
and transition-rate ratios range from 0.426 to 0.431. Proposed normalized churn
is 35.96--36.45 versus 84.24--85.34 transitions per 1,000 epochs. Episode count
changes from 1,533 at 5 seconds to 722 at 300 seconds, so coverage remains
definition-dependent.

The primary episode definition merges missing telemetry within the frozen
30-second lease but ends at an observed benign epoch. A documented protocol
amendment aligned this definition before statistical inference and changed no
actions or time-weighted outcomes. Strict zero-gap construction produces 6,474
episodes and coverage 43.93% versus 47.88% (difference -3.95 pp), with median
delay 1 second for both. Excluding 19 mixed-attack epochs leaves the primary
point results essentially unchanged.

### Cluster unit and mobility

One-hour time-only grouping leaves only 14 bootstrap clusters. All four
pairwise gates still pass, but proposed friction's one-sided upper bound rises
to 1.262%; this is supporting sensitivity, not stronger inference. The dataset
contains no trustworthy capture/campaign ID, and inferred groupings must not be
called independent captures. The test split is effectively all `car`, so no
pedestrian/static mobility generalization is available.

### Runtime feasibility

On one pinned Intel Xeon Gold 6246R logical CPU, 15 warm repetitions measured:

- logistic batch scoring: 3,526,528 reports/s, 0.284 microseconds/report;
- proposed online controller update: 304,611 reports/s, 3.283
  microseconds/report;
- exact proposed batch replay: 290,867 reports/s, 3.438
  microseconds/report; and
- serial scorer-plus-online-controller estimate: 3.566
  microseconds/report.

All benchmarked controllers exactly reproduce the frozen state vectors, and
rescoring matches at absolute tolerance `1e-12`. These are amortized in-memory
Python microbenchmarks. They exclude KPI collection, xApp transport,
serialization, RIC scheduling, policy installation, radio/core enforcement,
and network actuation; they are not near-RT RIC tail-latency measurements.

## 7. Claims that the evidence supports

Suitable claim:

> On the OpenIreland trace and a frozen logistic KPI scorer, tune-only
> friction-budgeted selection found an asymmetric RNTI-level controller that
> reduced chronological held-out action churn by 57.4% and malicious `ALLOW`
> time by 4.48 percentage points relative to a budget-matched stateless policy,
> while satisfying predeclared coverage and median-delay noninferiority gates.

Every use of that sentence should remain adjacent to the following boundaries:

- the friction point is below 1%, but its one-sided upper bound is not;
- malicious not-`ISOLATE` time and mean capped delay are worse;
- Slowloris is a failure mode;
- unseen-numeric-RNTI friction exceeds budget;
- a nonlinear scorer selects EWMA instead, and its held-out friction exceeds
  budget;
- trace blocks are surrogates, not verified independent captures; and
- results are offline action-state replay, not causal prevention or deployed
  service impact.

Do not use: “zero-trust identity was verified,” “the controller authenticates a
UE,” “attacks were prevented,” “the 1% budget is guaranteed,” “the result
generalizes across mobility,” or “near-RT RIC latency was demonstrated.”

## 8. End-to-end regeneration commands

Run from `/nobackup/ashukuma/xr/o_ran_publication`. Do not modify the locked
protocols after seeing test results.

### Environment and source verification

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
sha256sum /nobackup/ashukuma/o_ran/dtst.csv
```

Expected source SHA-256:
`4cc8498466eb7cb258412721ae94a2460f04a7da1235ac07e8e9cd20e15a76a7`.

### Primary pipeline

```bash
.venv/bin/python -m oran.capture_audit /nobackup/ashukuma/o_ran/dtst.csv \
  --report-output reports/capture_robustness_audit.json \
  --mapping-output artifacts/block_cluster_candidates.csv
.venv/bin/python -m oran.experiment prepare \
  --source /nobackup/ashukuma/o_ran/dtst.csv --artifact-root artifacts \
  --timestamp-scale 100000 --epoch-seconds 1
.venv/bin/python -m oran.experiment fit \
  --artifact-root artifacts --model-kind logistic --seed 1729
.venv/bin/python -m oran.matched_search \
  --score-path artifacts/results/scores_logistic_seed1729_v1.parquet \
  --artifact-root artifacts --iterations 18 --n-jobs 1
.venv/bin/python -m oran.strict_selection \
  --scores artifacts/results/scores_logistic_seed1729_v1.parquet \
  --candidates artifacts/results/controller_matched_candidates_tune_v1.parquet \
  --artifact-root artifacts --budgets 0.001 0.005 0.01 0.02 0.05
.venv/bin/python -m oran.confirmatory \
  --protocol configs/study_protocol_v1_locked.json \
  --scores artifacts/results/scores_logistic_seed1729_v1.parquet \
  --artifact-root artifacts --episode-merge-gap-s 0 --artifact-version v1
.venv/bin/python -m oran.confirmatory \
  --protocol configs/study_protocol_v1_locked.json \
  --scores artifacts/results/scores_logistic_seed1729_v1.parquet \
  --artifact-root artifacts --episode-merge-gap-s 30 --artifact-version v2
```

### HGB risk-score sensitivity

```bash
.venv/bin/python -m oran.experiment fit \
  --artifact-root artifacts --model-kind hgb --seed 1729
.venv/bin/python -m oran.matched_search \
  --score-path artifacts/results/scores_hgb_seed1729_v1.parquet \
  --artifact-root artifacts/hgb_sensitivity --iterations 18 --n-jobs 1
.venv/bin/python -m oran.strict_selection \
  --scores artifacts/results/scores_hgb_seed1729_v1.parquet \
  --candidates artifacts/hgb_sensitivity/results/controller_matched_candidates_tune_v1.parquet \
  --artifact-root artifacts/hgb_sensitivity \
  --budgets 0.001 0.005 0.01 0.02 0.05
.venv/bin/python -m oran.confirmatory \
  --protocol configs/study_protocol_hgb_sensitivity_v1_locked.json \
  --scores artifacts/results/scores_hgb_seed1729_v1.parquet \
  --artifact-root artifacts/hgb_sensitivity \
  --episode-merge-gap-s 30 --artifact-version v2
```

### Literal microsecond timebase sensitivity

The cut values are divided by ten with the timestamps so the chronological
partition semantics remain comparable.

```bash
.venv/bin/python -m oran.experiment prepare \
  --source /nobackup/ashukuma/o_ran/dtst.csv \
  --artifact-root artifacts/timebase_1e6 --timestamp-scale 1000000 \
  --epoch-seconds 1 --cuts 166674240 166678560 166682880
.venv/bin/python -m oran.experiment fit \
  --artifact-root artifacts/timebase_1e6 --model-kind logistic --seed 1729
.venv/bin/python -m oran.matched_search \
  --score-path artifacts/timebase_1e6/results/scores_logistic_seed1729_v1.parquet \
  --artifact-root artifacts/timebase_1e6 --iterations 18 --n-jobs 1
.venv/bin/python -m oran.strict_selection \
  --scores artifacts/timebase_1e6/results/scores_logistic_seed1729_v1.parquet \
  --candidates artifacts/timebase_1e6/results/controller_matched_candidates_tune_v1.parquet \
  --artifact-root artifacts/timebase_1e6 \
  --budgets 0.001 0.005 0.01 0.02 0.05
```

### Inference, lease sensitivity, runtime, reports, and integrity gates

```bash
.venv/bin/python -m oran.inference \
  --confirmatory-root artifacts/confirmatory \
  --cluster-mapping artifacts/block_cluster_candidates.csv \
  --hgb-root artifacts/hgb_sensitivity \
  --alternate-timebase-root artifacts/timebase_1e6 \
  --replicates 5000 --seed 1729
.venv/bin/python -m oran.sensitivity \
  --source /nobackup/ashukuma/o_ran/dtst.csv \
  --scores artifacts/results/scores_logistic_seed1729_v1.parquet \
  --protocol configs/study_protocol_v1_locked.json \
  --timeouts-s 5 10 30 60 300 --output-root artifacts/sensitivities
.venv/bin/python -m oran.benchmark --project-root . --repetitions 15 --warmups 2
.venv/bin/python -m oran.reporting --project-root . --reports-root reports
.venv/bin/python -m oran.repro_audit \
  --source /nobackup/ashukuma/o_ran/dtst.csv \
  --artifact-root artifacts \
  --protocol configs/study_protocol_v1_locked.json \
  --output reports/reproducibility_audit.json
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The fail-closed audit verifies source/protocol/score/candidate hashes, disjoint
chronological blocks, forbidden-feature exclusion, unique keys, complete
actions, one-epoch causal lag, exact metric recomputation, policy-independent
episode keys, single successful 1% proposal lock, and all four primary gates.

## 9. Artifact map

| Purpose | Authoritative artifact |
|---|---|
| Frozen primary protocol | `configs/study_protocol_v1_locked.json` |
| Episode amendment | `configs/protocol_amendment_v1a_episode_alignment.json` |
| HGB sensitivity lock | `configs/study_protocol_hgb_sensitivity_v1_locked.json` |
| Raw-data/epoch audit | `artifacts/audits/data_summary_v1.json` |
| Split and source manifest | `artifacts/manifests/split_manifest_v1.json` |
| Capture-surrogate audit | `reports/capture_robustness_audit.json` |
| Logistic model diagnostics | `artifacts/audits/risk_diagnostics_logistic_seed1729_v1.json` |
| Exact tuning frontier | `artifacts/results/controller_matched_candidates_tune_v1.parquet` |
| Strict selection table | `artifacts/results/controller_strict_selection_tune_v1.parquet` |
| Canonical locked policies | `artifacts/confirmatory/candidate_lock_v2.json` |
| Held-out action trace | `artifacts/confirmatory/action_trace_v2.parquet` |
| Aggregate held-out metrics | `artifacts/confirmatory/aggregate_metrics_v2.parquet` |
| Attack episodes | `artifacts/confirmatory/attack_episodes_v2.parquet` |
| Formal inference and sensitivities | `artifacts/confirmatory/inference_report_v3.json` |
| Lease-timeout sensitivity | `artifacts/sensitivities/` |
| Alternative-score sensitivity | `artifacts/hgb_sensitivity/` |
| Alternative-timebase sensitivity | `artifacts/timebase_1e6/` |
| Runtime manifest/raw trials | `artifacts/benchmarks/` |
| Human-readable runtime report | `reports/runtime_benchmark.md` |
| Publication tables/figures | `reports/tables/`, `reports/figures/` |
| Deterministic output digest | `reports/result_digest.json` |
| Fail-closed integrity audit | `reports/reproducibility_audit.json` |
| Letter and bibliography | `manuscript/main.tex`, `manuscript/references.bib` |

Important frozen hashes:

- primary protocol: `c862d5dbc42be0a27f92f39ec3a73829e21f1e710958c98664fde3be67c8f2cb`;
- logistic score frame: `438c8cdd333b307015b1a8972574179875ab384d6524962e0ffef066b4d75831`;
- canonical candidate lock content: `52eedcea2437afcfe8d7129fdf689c8ba86a18c06adb4ac0c4c95fab5b9e48c0`;
  and
- current inference report: `cb7ea80d0e9be24f6747bae9acae92a5e0140ec11d6ff8b1c7473a9cd511f98a`.

The candidate-lock file's file SHA differs from its canonical-content SHA by
design; the latter is the value recorded by the confirmatory run.

## 10. Required external actions before submission

1. **Obtain authoritative timestamp confirmation.** Ask the dataset authors for
   the timestamp divisor, epoch, and capture/run boundaries. Archive the reply.
   If `100000` is not confirmed, rerun the complete pipeline and reconsider the
   GO verdict; do not relabel the existing sensitivity as confirmation.
2. **Verify the CFP submission category in ScholarOne.** The CFP page/category
   wording may refer to a different or older special issue. Confirm the exact
   category with the guest editors or journal office and retain written
   confirmation before uploading.
3. **Keep deployment claims out unless a true O-RAN experiment is added.** A
   credible deployment extension needs an xApp/near-RT RIC path, E2/KPI
   collection, enforceable `RESTRICT`/`ISOLATE` mappings, end-to-end and tail
   latency, action success/failure telemetry, and benign QoS/service impact.
4. **Preserve the failure disclosures during page fitting.** Do not remove the
   Slowloris row, friction upper bound, unseen-RNTI budget failure, HGB-selected
   policy/budget drift, timebase uncertainty, or offline-action limitation to
   meet the page limit.
5. **Archive a final reproducibility bundle.** Rerun reporting, the fail-closed
   audit, and the complete test suite after the last artifact change; record
   hashes and software/CPU metadata; compile the final IEEE PDF; and verify the
   CFP's hard page limit including references.

## Final recommendation

Submit after the timestamp/category checks and final PDF/audit pass, using a
title and abstract centered on **friction-budgeted RNTI-level containment**.
Position zero trust as motivation and deployment context, not as an identity
result. Lead with the constrained operational frontier and chronological
capture-surrogate protocol; report the 57.4% churn reduction and -4.48 pp
malicious-`ALLOW` contrast; immediately disclose the friction uncertainty and
Slowloris/unseen-RNTI limitations. That is the strongest claim supported by the
available evidence.
