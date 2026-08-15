# Frozen pipeline runtime benchmark

The frozen calibrated logistic scorer processed the 101,561 held-out epochs at **3,526,528 reports/s** (**0.284 µs/report**) in repeated single-thread batch scoring. The table below reports the deployment-oriented per-report controller API separately from the audit-oriented batch trace path.

## Protocol

- Dataset partition: frozen chronological `test`, 101,561 one-second RNTI epochs.
- Timing: `perf_counter_ns`; one first in-memory invocation, 2 unreported warmups, then 15 recorded repetitions.
- Isolation: each workload ran in a fresh process, pinned to one logical CPU; OpenMP/BLAS/NumExpr/Polars thread limits were set to one before imports.
- Excluded from timed regions: Parquet I/O, model deserialization, protocol parsing, input projection, correctness comparison, and report serialization.
- `online_update` calls the production sequential `update` API and stores one byte per resulting state. `batch_trace` is the exact frozen replay implementation and materializes the full trace.
- The reported µs/report is amortized batch wall time, not an independently scheduled request or hard-real-time tail-latency guarantee.

## Results

| Workload | Policy | First in-memory ms | Warm median ms / 101,561 | p05–p95 ms | Reports/s | µs/report | CV |
|---|---|---:|---:|---:|---:|---:|---:|
| risk_score_batch | calibrated logistic | 43.35 | 28.80 | 25.90–31.94 | 3,526,528 | 0.284 | 8.1% |
| controller_online_update | stateless reference | 338.22 | 330.22 | 329.10–344.12 | 307,557 | 3.251 | 1.6% |
| controller_online_update | proposed | 341.47 | 333.41 | 332.63–340.12 | 304,611 | 3.283 | 1.2% |
| controller_online_update | ewma | 371.88 | 370.99 | 367.46–380.28 | 273,758 | 3.653 | 1.3% |
| controller_online_update | n report | 343.04 | 335.12 | 333.60–338.12 | 303,058 | 3.300 | 0.5% |
| controller_online_update | symmetric hysteresis | 348.14 | 333.22 | 331.08–341.53 | 304,791 | 3.281 | 1.1% |
| controller_batch_trace | stateless reference | 387.67 | 339.83 | 339.11–341.28 | 298,854 | 3.346 | 0.3% |
| controller_batch_trace | proposed | 395.52 | 349.17 | 345.17–353.54 | 290,867 | 3.438 | 0.9% |
| controller_batch_trace | ewma | 427.74 | 372.43 | 370.95–382.25 | 272,699 | 3.667 | 1.3% |
| controller_batch_trace | n report | 390.50 | 344.41 | 341.55–353.88 | 294,880 | 3.391 | 2.2% |
| controller_batch_trace | symmetric hysteresis | 408.46 | 343.39 | 341.10–346.68 | 295,758 | 3.381 | 0.5% |

A serial scorer-plus-controller point estimate can be obtained by summing the two medians because the frozen pipeline executes them sequentially. These sums are descriptive rather than independently timed end-to-end measurements:

| Policy | Scorer + online controller ms | Reports/s | µs/report |
|---|---:|---:|---:|
| stateless reference | 359.02 | 282,886 | 3.535 |
| proposed | 362.21 | 280,392 | 3.566 |
| ewma | 399.79 | 254,037 | 3.936 |
| n report | 363.92 | 279,075 | 3.583 |
| symmetric hysteresis | 362.01 | 280,544 | 3.565 |

## Process memory

Memory is Linux whole-process RSS/high-water RSS from `/proc/self/status`, measured in each fresh worker. It includes Python, loaded libraries, the model, and resident input arrays. The incremental HWM column is the *new* high-water mark observed after inputs were ready; allocator reuse can understate transient demand, so these values are descriptive rather than object-level allocations.

| Workload | Policy | Input-ready RSS MiB | Process HWM MiB | New HWM MiB |
|---|---|---:|---:|---:|
| risk_score_batch | calibrated logistic | 196.3 | 272.2 | 68.4 |
| controller_online_update | stateless reference | 191.4 | 191.4 | 0.0 |
| controller_online_update | proposed | 193.5 | 193.5 | 0.0 |
| controller_online_update | ewma | 190.3 | 190.3 | 0.0 |
| controller_online_update | n report | 190.3 | 190.3 | 0.0 |
| controller_online_update | symmetric hysteresis | 192.4 | 192.4 | 0.0 |
| controller_batch_trace | stateless reference | 193.7 | 222.2 | 28.5 |
| controller_batch_trace | proposed | 192.2 | 220.8 | 28.6 |
| controller_batch_trace | ewma | 188.1 | 218.7 | 30.5 |
| controller_batch_trace | n report | 189.9 | 218.3 | 28.3 |
| controller_batch_trace | symmetric hysteresis | 191.7 | 220.0 | 28.3 |

## Correctness gates

- Feature and score rows were exactly aligned on all five replay keys: **True**.
- Recomputed logistic risk matched the frozen score vector at absolute tolerance 1e-12; maximum absolute error: `2.78e-16`.
- Every measured controller reproduced its frozen 101,561-element decision-state vector exactly.

## Environment and interpretation

- CPU: Intel(R) Xeon(R) Gold 6246R CPU @ 3.40GHz; worker CPU: 0.
- Platform: Linux-4.18.0-553.124.1.el8_10.x86_64-x86_64-with-glibc2.28; visible logical CPUs: 6.
- Software: Python 3.11.13, NumPy 2.4.6, Polars 1.43.2, scikit-learn 1.9.0.
- Load averages captured when the manifest was created: [1.94, 1.43, 1.94].

These results establish feasibility of this Python reference implementation on one virtualized Xeon core. They do not establish near-real-time RIC latency: xApp transport, feature collection, serialization, scheduling, enforcement, and network actuation were not present in the offline dataset or timed here.

Machine-readable raw repetitions, summary statistics, provenance hashes, affinity, thread-pool metadata, and memory snapshots are in `artifacts/benchmarks/`.
