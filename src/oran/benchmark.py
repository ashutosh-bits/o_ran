"""Reproducible latency and memory benchmark for the frozen O-RAN pipeline.

The coordinator performs correctness checks, then launches every timed workload
in a fresh process.  Worker processes are pinned to one logical CPU when Linux
permits it and all known native thread pools are limited to one thread before
NumPy, Polars, or scikit-learn are imported.  File I/O, model deserialization,
protocol parsing, and input construction are outside every timed region.

Two controller paths are measured deliberately:

* ``online_update`` calls the production ``update`` method once per report and
  retains only the action vector.  This is the deployment-relevant path.
* ``batch_trace`` calls the exact ``run`` path used by the frozen offline replay;
  it also materializes the complete ``ControllerTrace`` for auditability.

The benchmark is descriptive, not a hard-real-time guarantee.  In particular,
reported per-report latency is total batch time divided by the report count; it
is not a tail-latency measurement of independently scheduled requests.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
}

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[2]
EPOCHS_DEFAULT = Path("artifacts/epochs/epochs_1s_v1.parquet")
SCORES_DEFAULT = Path("artifacts/results/scores_logistic_seed1729_v1.parquet")
MODEL_DEFAULT = Path("artifacts/models/logistic_seed1729_v1.joblib")
PROTOCOL_DEFAULT = Path("configs/study_protocol_v1_locked.json")
ACTION_TRACE_DEFAULT = Path("artifacts/confirmatory/action_trace_v2.parquet")
OUTPUT_ROOT_DEFAULT = Path("artifacts/benchmarks")
REPORT_DEFAULT = Path("reports/runtime_benchmark.md")
EVALUATION_SPLIT = "test"
EXPECTED_TEST_ROWS = 101_561


def _imports() -> tuple[Any, Any, Any, Any]:
    """Import numerical libraries lazily so worker thread limits apply first."""

    import joblib
    import numpy as np
    import polars as pl
    import sklearn

    return np, pl, sklearn, joblib


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    """Hash array shape, dtype, and contiguous bytes without string coercion."""

    np, _, _, _ = _imports()
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def summarize_timings(samples_s: Sequence[float], rows: int) -> dict[str, float | int]:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if not samples_s:
        raise ValueError("at least one timing sample is required")
    np, _, _, _ = _imports()
    samples = np.asarray(samples_s, dtype=float)
    if samples.ndim != 1 or not np.isfinite(samples).all() or (samples <= 0).any():
        raise ValueError("timing samples must be finite positive values")
    median_s = float(np.median(samples))
    result: dict[str, float | int] = {
        "repetitions": int(len(samples)),
        "median_s": median_s,
        "mean_s": float(np.mean(samples)),
        "stdev_s": float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
        "minimum_s": float(np.min(samples)),
        "p05_s": float(np.quantile(samples, 0.05)),
        "p25_s": float(np.quantile(samples, 0.25)),
        "p75_s": float(np.quantile(samples, 0.75)),
        "p95_s": float(np.quantile(samples, 0.95)),
        "maximum_s": float(np.max(samples)),
        "median_reports_per_s": float(rows / median_s),
        "median_us_per_report": float(1e6 * median_s / rows),
    }
    result["coefficient_of_variation"] = (
        float(result["stdev_s"]) / float(result["mean_s"])
    )
    return result


def _parse_proc_status_text(text: str) -> dict[str, int | None]:
    values: dict[str, int | None] = {"rss_kib": None, "hwm_kib": None}
    mapping = {"VmRSS": "rss_kib", "VmHWM": "hwm_kib"}
    for line in text.splitlines():
        key, separator, remainder = line.partition(":")
        if not separator or key not in mapping:
            continue
        fields = remainder.strip().split()
        if fields:
            values[mapping[key]] = int(fields[0])
    return values


def process_memory() -> dict[str, int | None]:
    try:
        return _parse_proc_status_text(Path("/proc/self/status").read_text())
    except (OSError, ValueError):
        return {"rss_kib": None, "hwm_kib": None}


def _memory_summary(
    after_import: dict[str, int | None],
    input_ready: dict[str, int | None],
    after_cold: dict[str, int | None],
    after_warm: dict[str, int | None],
) -> dict[str, Any]:
    prior_hwm = input_ready["hwm_kib"]
    final_hwm = after_warm["hwm_kib"]
    new_hwm = (
        None
        if prior_hwm is None or final_hwm is None
        else max(0, int(final_hwm) - int(prior_hwm))
    )
    return {
        "unit": "KiB",
        "after_import": after_import,
        "input_ready": input_ready,
        "after_first_invocation": after_cold,
        "after_repeated_invocations": after_warm,
        "new_high_water_increment_after_input_ready_kib": new_hwm,
        "interpretation": (
            "Linux whole-process resident memory; inputs and interpreter are included. "
            "The high-water increment is observational and allocator reuse can make it "
            "understate transient workload demand."
        ),
    }


def _pin_to_one_cpu() -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested": True,
        "supported": hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"),
        "allowed_before": None,
        "selected_cpu": None,
        "allowed_after": None,
        "error": None,
    }
    if not result["supported"]:
        return result
    try:
        before = sorted(os.sched_getaffinity(0))
        result["allowed_before"] = before
        if not before:
            raise RuntimeError("process has an empty CPU-affinity mask")
        selected = before[0]
        os.sched_setaffinity(0, {selected})
        result["selected_cpu"] = selected
        result["allowed_after"] = sorted(os.sched_getaffinity(0))
    except (OSError, RuntimeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _time_call(function: Callable[[], Any]) -> tuple[float, Any]:
    started = time.perf_counter_ns()
    output = function()
    elapsed_s = (time.perf_counter_ns() - started) / 1e9
    return elapsed_s, output


def _timed_repetitions(
    function: Callable[[], Any], *, repetitions: int, warmups: int
) -> tuple[list[float], Any]:
    if repetitions < 1 or warmups < 0:
        raise ValueError("repetitions must be positive and warmups non-negative")
    output: Any = None
    for _ in range(warmups):
        output = function()
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        samples: list[float] = []
        for _ in range(repetitions):
            elapsed_s, output = _time_call(function)
            samples.append(elapsed_s)
    finally:
        if was_enabled:
            gc.enable()
    return samples, output


def _test_feature_frame(epochs_path: Path, feature_columns: Sequence[str]) -> Any:
    _, pl, _, _ = _imports()
    return (
        pl.scan_parquet(epochs_path)
        .filter(pl.col("split") == EVALUATION_SPLIT)
        .select(list(feature_columns))
        .collect()
    )


def _test_controller_arrays(scores_path: Path) -> tuple[Any, Any, Any, Any]:
    np, pl, _, _ = _imports()
    frame = (
        pl.scan_parquet(scores_path)
        .filter(pl.col("split") == EVALUATION_SPLIT)
        .select(["risk_score", "decision_time_s", "rnti_lease_id", "mac_rnti"])
        .collect()
    )
    return (
        np.asarray(frame["risk_score"].to_numpy(), dtype=float),
        np.asarray(frame["decision_time_s"].to_numpy(), dtype=float),
        np.asarray(frame["rnti_lease_id"].to_numpy(), dtype=object),
        np.asarray(frame["mac_rnti"].to_numpy(), dtype=object),
    )


def online_update_states(
    controller: Any,
    scores: Any,
    timestamps: Any,
    lease_ids: Any,
    subject_ids: Any,
) -> Any:
    """Execute the production per-report API while retaining only action states."""

    np, _, _, _ = _imports()
    controller.clear()
    states = np.empty(len(scores), dtype=np.int8)
    for index, (score, timestamp, lease, subject) in enumerate(
        zip(scores, timestamps, lease_ids, subject_ids, strict=True)
    ):
        states[index] = int(
            controller.update(score, timestamp, lease, subject_id=subject).state
        )
    return states


def _load_spec(protocol_path: Path, candidate: str) -> Any:
    from .confirmatory import _locked_specs

    with protocol_path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    matches = [spec for spec in _locked_specs(protocol) if spec.candidate == candidate]
    if len(matches) != 1:
        raise ValueError(f"expected one locked specification for {candidate!r}")
    return matches[0]


def _worker_score(args: argparse.Namespace) -> dict[str, Any]:
    affinity = _pin_to_one_cpu()
    np, pl, sklearn, joblib = _imports()
    from threadpoolctl import threadpool_info, threadpool_limits

    from .model import load_model_bundle

    after_import = process_memory()
    with threadpool_limits(limits=1):
        bundle = load_model_bundle(args.model)
        frame = _test_feature_frame(args.epochs, bundle.feature_columns)
        if frame.height != EXPECTED_TEST_ROWS:
            raise ValueError(f"expected {EXPECTED_TEST_ROWS} test rows; got {frame.height}")
        input_ready = process_memory()
        cold_s, cold_output = _time_call(lambda: bundle.predict_risk(frame))
        after_cold = process_memory()
        if len(cold_output) != frame.height or not np.isfinite(cold_output).all():
            raise AssertionError("risk scorer returned invalid output")
        samples, final_output = _timed_repetitions(
            lambda: bundle.predict_risk(frame),
            repetitions=args.repetitions,
            warmups=args.warmups,
        )
        after_warm = process_memory()
        pools = threadpool_info()
    # Load the frozen target only after all timing and memory snapshots so the
    # validation vector cannot affect the measured working set.
    frozen_output = np.asarray(
        pl.scan_parquet(args.scores)
        .filter(pl.col("split") == EVALUATION_SPLIT)
        .select("risk_score")
        .collect()["risk_score"]
        .to_numpy(),
        dtype=float,
    )
    output_error = np.abs(np.asarray(final_output) - frozen_output)
    output_allclose = bool(
        np.allclose(final_output, frozen_output, rtol=0.0, atol=1e-12)
    )
    if not output_allclose:
        raise AssertionError("thread-limited worker scores differ from frozen scores")
    return {
        "workload": "risk_score_batch",
        "candidate": "calibrated_logistic",
        "rows": frame.height,
        "features": len(bundle.feature_columns),
        "feature_columns": list(bundle.feature_columns),
        "first_in_memory_s": cold_s,
        "first_in_memory_reports_per_s": frame.height / cold_s,
        "first_in_memory_us_per_report": 1e6 * cold_s / frame.height,
        "warm": summarize_timings(samples, frame.height),
        "raw_warm_seconds": samples,
        "output_sha256": array_sha256(final_output),
        "frozen_output_sha256": array_sha256(frozen_output),
        "output_allclose_atol_1e_12": output_allclose,
        "output_max_abs_error": float(output_error.max(initial=0.0)),
        "output_bytes": int(final_output.nbytes),
        "input_frame_estimated_bytes": int(frame.estimated_size()),
        "memory": _memory_summary(after_import, input_ready, after_cold, after_warm),
        "affinity": affinity,
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENVIRONMENT},
        "thread_pools": pools,
        "polars_thread_pool_size": int(pl.thread_pool_size()),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }


def _worker_controller(args: argparse.Namespace) -> dict[str, Any]:
    affinity = _pin_to_one_cpu()
    np, pl, sklearn, joblib = _imports()
    from threadpoolctl import threadpool_info, threadpool_limits

    from .policy_search import make_controller

    after_import = process_memory()
    with threadpool_limits(limits=1):
        scores, timestamps, leases, subjects = _test_controller_arrays(args.scores)
        if len(scores) != EXPECTED_TEST_ROWS:
            raise ValueError(f"expected {EXPECTED_TEST_ROWS} test rows; got {len(scores)}")
        spec = _load_spec(args.protocol, args.candidate)
        controller = make_controller(spec)
        if args.controller_mode == "online_update":
            invoke = lambda: online_update_states(
                controller, scores, timestamps, leases, subjects
            )
            output_bytes_multiplier = 1
        elif args.controller_mode == "batch_trace":
            invoke = lambda: controller.run(
                scores, timestamps, leases, subjects, clear=True
            )
            output_bytes_multiplier = 0
        else:
            raise ValueError(f"unknown controller mode: {args.controller_mode}")

        input_ready = process_memory()
        cold_s, cold_output = _time_call(invoke)
        after_cold = process_memory()
        cold_states = cold_output if args.controller_mode == "online_update" else cold_output.state
        if len(cold_states) != len(scores):
            raise AssertionError("controller returned the wrong state-vector length")
        samples, final_output = _timed_repetitions(
            invoke, repetitions=args.repetitions, warmups=args.warmups
        )
        after_warm = process_memory()
        final_states = (
            final_output
            if args.controller_mode == "online_update"
            else final_output.state
        )
        pools = threadpool_info()

    if output_bytes_multiplier:
        output_bytes = int(final_states.nbytes)
    else:
        output_bytes = int(
            sum(
                value.nbytes
                for value in (
                    final_output.timestamp_s,
                    final_output.subject_id,
                    final_output.lease_id,
                    final_output.raw_score,
                    final_output.evidence_score,
                    final_output.state,
                    final_output.previous_state,
                    final_output.transitioned,
                    final_output.lifecycle_start,
                )
            )
        )
    input_bytes = int(
        scores.nbytes + timestamps.nbytes + leases.nbytes + subjects.nbytes
    )
    return {
        "workload": f"controller_{args.controller_mode}",
        "candidate": spec.candidate,
        "family": spec.family,
        "controller_kind": spec.controller,
        "rows": len(scores),
        "first_in_memory_s": cold_s,
        "first_in_memory_reports_per_s": len(scores) / cold_s,
        "first_in_memory_us_per_report": 1e6 * cold_s / len(scores),
        "warm": summarize_timings(samples, len(scores)),
        "raw_warm_seconds": samples,
        "state_sha256": array_sha256(final_states),
        "state_counts": {
            str(int(state)): int(count)
            for state, count in zip(*np.unique(final_states, return_counts=True), strict=True)
        },
        "input_array_bytes": input_bytes,
        "output_array_bytes": output_bytes,
        "memory": _memory_summary(after_import, input_ready, after_cold, after_warm),
        "affinity": affinity,
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENVIRONMENT},
        "thread_pools": pools,
        "polars_thread_pool_size": int(pl.thread_pool_size()),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }


def _read_first_line(path: Path) -> str | None:
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or None


def _git_metadata(project_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def system_metadata(project_root: Path) -> dict[str, Any]:
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpus_visible": os.cpu_count(),
        "cpu_affinity_visible": allowed,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "cgroup_cpu_max": _read_first_line(Path("/sys/fs/cgroup/cpu.max")),
        "cgroup_memory_max": _read_first_line(Path("/sys/fs/cgroup/memory.max")),
        "clock": "time.perf_counter_ns (monotonic, highest available resolution)",
        "git": _git_metadata(project_root),
    }


def _subprocess_environment(project_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    source = str((project_root / "src").resolve())
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not existing else f"{source}{os.pathsep}{existing}"
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _run_worker(
    project_root: Path,
    common: list[str],
    *,
    workload: str,
    candidate: str | None = None,
    controller_mode: str | None = None,
) -> dict[str, Any]:
    command = [sys.executable, "-m", "oran.benchmark", "--worker", workload, *common]
    if candidate is not None:
        command.extend(["--candidate", candidate])
    if controller_mode is not None:
        command.extend(["--controller-mode", controller_mode])
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=_subprocess_environment(project_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "benchmark worker failed:\n"
            f"command: {' '.join(command)}\n"
            f"stdout: {completed.stdout}\n"
            f"stderr: {completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"worker emitted invalid JSON: {completed.stdout!r}") from exc


def _correctness_checks(
    *,
    epochs_path: Path,
    scores_path: Path,
    model_path: Path,
    protocol_path: Path,
    action_trace_path: Path,
) -> dict[str, Any]:
    np, pl, _, _ = _imports()
    from .confirmatory import _locked_specs
    from .model import load_model_bundle
    from .policy_search import make_controller

    bundle = load_model_bundle(model_path)
    features = _test_feature_frame(epochs_path, bundle.feature_columns)
    stored = (
        pl.scan_parquet(scores_path)
        .filter(pl.col("split") == EVALUATION_SPLIT)
        .select(
            [
                "trace_block_id",
                "rnti_lease_id",
                "mac_rnti",
                "decision_time_s",
                "epoch_seconds",
                "risk_score",
            ]
        )
        .collect()
    )
    epoch_keys = (
        pl.scan_parquet(epochs_path)
        .filter(pl.col("split") == EVALUATION_SPLIT)
        .select(
            [
                "trace_block_id",
                "rnti_lease_id",
                "mac_rnti",
                "decision_time_s",
                "epoch_seconds",
            ]
        )
        .collect()
    )
    if stored.height != EXPECTED_TEST_ROWS or features.height != EXPECTED_TEST_ROWS:
        raise AssertionError("frozen held-out row count changed")
    key_columns = epoch_keys.columns
    keys_match = all(
        np.array_equal(stored[column].to_numpy(), epoch_keys[column].to_numpy())
        for column in key_columns
    )
    if not keys_match:
        raise AssertionError("epoch feature and score tables are not row-aligned")
    observed_risk = bundle.predict_risk(features)
    expected_risk = np.asarray(stored["risk_score"].to_numpy(), dtype=float)
    absolute = np.abs(observed_risk - expected_risk)
    risk_allclose = bool(np.allclose(observed_risk, expected_risk, rtol=0.0, atol=1e-12))
    if not risk_allclose:
        raise AssertionError("fresh logistic scores differ from frozen scores")

    with protocol_path.open(encoding="utf-8") as handle:
        specs = _locked_specs(json.load(handle))
    scores, timestamps, leases, subjects = _test_controller_arrays(scores_path)
    controller_results: list[dict[str, Any]] = []
    for spec in specs:
        expected = (
            pl.scan_parquet(action_trace_path)
            .filter(pl.col("candidate") == spec.candidate)
            .select("decision_state")
            .collect()["decision_state"]
            .to_numpy()
        )
        observed = make_controller(spec).run(
            scores, timestamps, leases, subjects, clear=True
        ).state
        exact = bool(np.array_equal(observed, expected))
        if not exact:
            raise AssertionError(f"controller states differ for {spec.candidate}")
        controller_results.append(
            {
                "candidate": spec.candidate,
                "family": spec.family,
                "rows": int(len(observed)),
                "exact_state_match": exact,
                "state_sha256": array_sha256(observed),
            }
        )
    return {
        "held_out_rows": stored.height,
        "epoch_score_keys_exactly_aligned": keys_match,
        "risk_score_allclose_atol_1e_12": risk_allclose,
        "risk_score_max_abs_error": float(absolute.max(initial=0.0)),
        "fresh_risk_sha256": array_sha256(observed_risk),
        "frozen_risk_sha256": array_sha256(expected_risk),
        "controllers": controller_results,
    }


def _write_csvs(manifest: dict[str, Any], output_root: Path) -> dict[str, str]:
    raw_path = output_root / "benchmark_trials_v1.csv"
    summary_path = output_root / "benchmark_summary_v1.csv"
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for result in manifest["results"]:
        for index, seconds in enumerate(result["raw_warm_seconds"], start=1):
            raw_rows.append(
                {
                    "workload": result["workload"],
                    "candidate": result["candidate"],
                    "family": result.get("family", "risk_model"),
                    "trial": index,
                    "rows": result["rows"],
                    "elapsed_s": seconds,
                    "reports_per_s": result["rows"] / seconds,
                    "us_per_report": 1e6 * seconds / result["rows"],
                }
            )
        warm = result["warm"]
        summary_rows.append(
            {
                "workload": result["workload"],
                "candidate": result["candidate"],
                "family": result.get("family", "risk_model"),
                "rows": result["rows"],
                "repetitions": warm["repetitions"],
                "median_s": warm["median_s"],
                "p05_s": warm["p05_s"],
                "p95_s": warm["p95_s"],
                "median_reports_per_s": warm["median_reports_per_s"],
                "median_us_per_report": warm["median_us_per_report"],
                "coefficient_of_variation": warm["coefficient_of_variation"],
                "first_in_memory_s": result["first_in_memory_s"],
                "input_ready_rss_kib": result["memory"]["input_ready"]["rss_kib"],
                "process_hwm_kib": result["memory"]["after_repeated_invocations"]["hwm_kib"],
                "new_hwm_increment_kib": result["memory"][
                    "new_high_water_increment_after_input_ready_kib"
                ],
            }
        )
    for path, rows in ((raw_path, raw_rows), (summary_path, summary_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return {"trials_csv": str(raw_path), "summary_csv": str(summary_path)}


def _display_family(value: str) -> str:
    return value.replace("_", " ")


def render_report(manifest: dict[str, Any]) -> str:
    score = next(item for item in manifest["results"] if item["workload"] == "risk_score_batch")
    online = [
        item for item in manifest["results"] if item["workload"] == "controller_online_update"
    ]
    batch = [
        item for item in manifest["results"] if item["workload"] == "controller_batch_trace"
    ]
    metadata = manifest["system"]
    protocol = manifest["protocol"]

    lines = [
        "# Frozen pipeline runtime benchmark",
        "",
        "The frozen calibrated logistic scorer processed the 101,561 held-out epochs "
        f"at **{score['warm']['median_reports_per_s']:,.0f} reports/s** "
        f"(**{score['warm']['median_us_per_report']:.3f} µs/report**) in repeated "
        "single-thread batch scoring. The table below reports the deployment-oriented "
        "per-report controller API separately from the audit-oriented batch trace path.",
        "",
        "## Protocol",
        "",
        f"- Dataset partition: frozen chronological `test`, {protocol['held_out_rows']:,} one-second RNTI epochs.",
        f"- Timing: `perf_counter_ns`; one first in-memory invocation, {protocol['warmups']} unreported warmups, then {protocol['repetitions']} recorded repetitions.",
        "- Isolation: each workload ran in a fresh process, pinned to one logical CPU; "
        "OpenMP/BLAS/NumExpr/Polars thread limits were set to one before imports.",
        "- Excluded from timed regions: Parquet I/O, model deserialization, protocol parsing, "
        "input projection, correctness comparison, and report serialization.",
        "- `online_update` calls the production sequential `update` API and stores one byte "
        "per resulting state. `batch_trace` is the exact frozen replay implementation and "
        "materializes the full trace.",
        "- The reported µs/report is amortized batch wall time, not an independently "
        "scheduled request or hard-real-time tail-latency guarantee.",
        "",
        "## Results",
        "",
        "| Workload | Policy | First in-memory ms | Warm median ms / 101,561 | p05–p95 ms | Reports/s | µs/report | CV |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = [score, *online, *batch]
    for result in ordered:
        warm = result["warm"]
        family = result.get("family", "calibrated logistic")
        lines.append(
            f"| {result['workload']} | {_display_family(family)} | "
            f"{1000 * result['first_in_memory_s']:.2f} | "
            f"{1000 * warm['median_s']:.2f} | "
            f"{1000 * warm['p05_s']:.2f}–{1000 * warm['p95_s']:.2f} | "
            f"{warm['median_reports_per_s']:,.0f} | "
            f"{warm['median_us_per_report']:.3f} | "
            f"{100 * warm['coefficient_of_variation']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "A serial scorer-plus-controller point estimate can be obtained by summing the "
            "two medians because the frozen pipeline executes them sequentially. These sums "
            "are descriptive rather than independently timed end-to-end measurements:",
            "",
            "| Policy | Scorer + online controller ms | Reports/s | µs/report |",
            "|---|---:|---:|---:|",
        ]
    )
    for result in online:
        total = score["warm"]["median_s"] + result["warm"]["median_s"]
        lines.append(
            f"| {_display_family(result['family'])} | {1000 * total:.2f} | "
            f"{result['rows'] / total:,.0f} | {1e6 * total / result['rows']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Process memory",
            "",
            "Memory is Linux whole-process RSS/high-water RSS from `/proc/self/status`, "
            "measured in each fresh worker. It includes Python, loaded libraries, the model, "
            "and resident input arrays. The incremental HWM column is the *new* high-water "
            "mark observed after inputs were ready; allocator reuse can understate transient "
            "demand, so these values are descriptive rather than object-level allocations.",
            "",
            "| Workload | Policy | Input-ready RSS MiB | Process HWM MiB | New HWM MiB |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for result in ordered:
        memory = result["memory"]
        rss = memory["input_ready"]["rss_kib"]
        hwm = memory["after_repeated_invocations"]["hwm_kib"]
        increment = memory["new_high_water_increment_after_input_ready_kib"]
        fmt = lambda value: "n/a" if value is None else f"{value / 1024:.1f}"
        lines.append(
            f"| {result['workload']} | "
            f"{_display_family(result.get('family', 'calibrated logistic'))} | "
            f"{fmt(rss)} | {fmt(hwm)} | {fmt(increment)} |"
        )

    correct = manifest["correctness"]
    lines.extend(
        [
            "",
            "## Correctness gates",
            "",
            f"- Feature and score rows were exactly aligned on all five replay keys: **{correct['epoch_score_keys_exactly_aligned']}**.",
            f"- Recomputed logistic risk matched the frozen score vector at absolute tolerance 1e-12; maximum absolute error: `{correct['risk_score_max_abs_error']:.3g}`.",
            "- Every measured controller reproduced its frozen 101,561-element decision-state vector exactly.",
            "",
            "## Environment and interpretation",
            "",
            f"- CPU: {metadata['cpu_model']}; worker CPU: {manifest['results'][0]['affinity']['selected_cpu']}.",
            f"- Platform: {metadata['platform']}; visible logical CPUs: {metadata['logical_cpus_visible']}.",
            f"- Software: Python {score['software']['python']}, NumPy {score['software']['numpy']}, "
            f"Polars {score['software']['polars']}, scikit-learn {score['software']['scikit_learn']}.",
            f"- Load averages captured when the manifest was created: {metadata['load_average']}.",
            "",
            "These results establish feasibility of this Python reference implementation on "
            "one virtualized Xeon core. They do not establish near-real-time RIC latency: xApp "
            "transport, feature collection, serialization, scheduling, enforcement, and network "
            "actuation were not present in the offline dataset or timed here.",
            "",
            "Machine-readable raw repetitions, summary statistics, provenance hashes, affinity, "
            "thread-pool metadata, and memory snapshots are in `artifacts/benchmarks/`.",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve()
    epochs = args.epochs.resolve()
    scores = args.scores.resolve()
    model = args.model.resolve()
    protocol = args.protocol.resolve()
    action_trace = args.action_trace.resolve()
    output_root = args.output_root.resolve()
    report_path = args.report.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("[benchmark] validating frozen scores and controller decisions", flush=True)
    correctness = _correctness_checks(
        epochs_path=epochs,
        scores_path=scores,
        model_path=model,
        protocol_path=protocol,
        action_trace_path=action_trace,
    )
    with protocol.open(encoding="utf-8") as handle:
        from .confirmatory import _locked_specs

        specs = _locked_specs(json.load(handle))

    common = [
        "--epochs",
        str(epochs),
        "--scores",
        str(scores),
        "--model",
        str(model),
        "--protocol",
        str(protocol),
        "--repetitions",
        str(args.repetitions),
        "--warmups",
        str(args.warmups),
    ]
    print("[benchmark] calibrated logistic scorer", flush=True)
    results = [_run_worker(project_root, common, workload="score")]
    if not results[0]["output_allclose_atol_1e_12"]:
        raise AssertionError("timed worker changed the calibrated risk vector")
    for mode in ("online_update", "batch_trace"):
        for spec in specs:
            print(f"[benchmark] {mode}: {spec.family}", flush=True)
            results.append(
                _run_worker(
                    project_root,
                    common,
                    workload="controller",
                    candidate=spec.candidate,
                    controller_mode=mode,
                )
            )

    state_hashes = {
        item["candidate"]: item["state_sha256"]
        for item in correctness["controllers"]
    }
    for result in results:
        if result["workload"].startswith("controller_"):
            if result["state_sha256"] != state_hashes[result["candidate"]]:
                raise AssertionError(
                    f"timed worker changed states for {result['candidate']}"
                )

    manifest: dict[str, Any] = {
        "benchmark_version": 1,
        "system": system_metadata(project_root),
        "protocol": {
            "evaluation_split": EVALUATION_SPLIT,
            "held_out_rows": EXPECTED_TEST_ROWS,
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "fresh_process_per_workload": True,
            "cpu_affinity_target_count": 1,
            "thread_limit": 1,
            "timed_region_excludes_io_and_deserialization": True,
            "cold_definition": "first invocation with model and input already resident",
            "warm_definition": "recorded invocations after first call and unreported warmups",
        },
        "inputs": {
            "epochs": {"path": str(epochs), "sha256": sha256_file(epochs)},
            "scores": {"path": str(scores), "sha256": sha256_file(scores)},
            "model": {"path": str(model), "sha256": sha256_file(model)},
            "protocol": {"path": str(protocol), "sha256": sha256_file(protocol)},
            "action_trace": {
                "path": str(action_trace),
                "sha256": sha256_file(action_trace),
            },
        },
        "correctness": correctness,
        "results": results,
        "limitations": [
            "offline in-memory microbenchmark, not deployed near-real-time RIC timing",
            "amortized batch microseconds per report, not per-request tail latency",
            "excludes telemetry, transport, serialization, scheduling, and actuation",
            "shared virtualized host load and frequency can affect wall-clock results",
            "Linux RSS/HWM includes process and inputs; incremental memory is observational",
        ],
    }
    csv_paths = _write_csvs(manifest, output_root)
    manifest["outputs"] = csv_paths
    manifest_path = output_root / "benchmark_manifest_v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report_path.write_text(render_report(manifest))
    print(f"[benchmark] wrote {manifest_path}", flush=True)
    print(f"[benchmark] wrote {report_path}", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("score", "controller"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--epochs", type=Path, default=EPOCHS_DEFAULT)
    parser.add_argument("--scores", type=Path, default=SCORES_DEFAULT)
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
    parser.add_argument("--action-trace", type=Path, default=ACTION_TRACE_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    parser.add_argument("--candidate")
    parser.add_argument(
        "--controller-mode", choices=("online_update", "batch_trace")
    )
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.repetitions < 3:
        raise SystemExit("--repetitions must be at least 3")
    if args.warmups < 0:
        raise SystemExit("--warmups cannot be negative")
    if args.worker == "score":
        result = _worker_score(args)
        print(json.dumps(result, sort_keys=True))
        return
    if args.worker == "controller":
        if not args.candidate or not args.controller_mode:
            raise SystemExit("controller workers require --candidate and --controller-mode")
        result = _worker_controller(args)
        print(json.dumps(result, sort_keys=True))
        return
    run_benchmark(args)


if __name__ == "__main__":
    main()
