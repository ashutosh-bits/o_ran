# Manuscript handoff

`main.tex` is a compact IEEE Networking Letters draft for the locked study, and `references.bib` contains only references whose title, authors, venue/report, year, and DOI or official URL were checked against publisher, NIST, Mendeley, DBLP, or USENIX records. `cover_letter.md` is an optional editor-facing scope statement.

The abstract is within the requested 75--100-word range. The author block is intentionally `Anonymous Author(s)` and must be replaced before submission.

## Build

The locked draft was compiled on 2026-08-09 with Tectonic 0.16.9 and IEEEtran
v1.8b:

```sh
XDG_CACHE_HOME=/tmp/oran-tectonic-cache /tmp/tectonic-0.16.9/tectonic \
  main.tex --keep-logs --keep-intermediates --print
```

The resulting `main.pdf` is four letter-size pages including references. The
log identifies it as a 10-point document and contains no overfull boxes or
unresolved citations/references. It does contain four underfull-box warnings
and XeTeX/Tectonic font-shape substitutions; perform the final submission build
with the venue's current official IEEE LaTeX environment.

In a conventional TeX installation, from this directory:

```sh
latexmk -pdf main.tex
```

or, without `latexmk`:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Frozen evidence used

- Protocol: `../configs/study_protocol_v1_locked.json`
- Episode amendment: `../configs/protocol_amendment_v1a_episode_alignment.json`
- Formal inference: `../artifacts/confirmatory/inference_report_v3.json`
- Aggregate and per-attack metrics: `../artifacts/confirmatory/*_v2.*`
- Unseen-RNTI strata: `../artifacts/confirmatory/stratified_action_metrics_v2.parquet`
- HGB sensitivity: `../artifacts/hgb_sensitivity/`
- Timestamp sensitivity: `../artifacts/timebase_1e6/`
- Lease-timeout sensitivity: `../artifacts/sensitivities/`
- Runtime benchmark: `../artifacts/benchmarks/benchmark_manifest_v1.json`

## Submission checks

1. Replace the author placeholder and add acknowledgments/conflicts as appropriate.
2. Compile with the current IEEE Networking Letters template and confirm the five-page limit, including references.
3. Reconcile the final artifact/code repository URL and data DOI with the availability statement required by the venue.
4. Keep the terminology precise: label-blind `trace blocks`, not known captures; `unseen numeric RNTI`, not unseen user; and offline policy replay, not deployed prevention or authentication.
5. Do not strengthen the claims: the primary point friction is below 1%, but its one-sided 95% upper bound is 1.069%; Slowloris fails; the HGB-selected EWMA passes the four controller gates but has 1.186% held-out friction (one-sided upper bound 1.890%), while the asymmetric HGB diagnostic fails its coverage gate; raw/$10^6$ admits no feasible proposed policy at 1%.
