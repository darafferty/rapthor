# Performance Equivalence Runs

Compact milestone reports for the branch-vs-master performance gate belong in
this directory. Keep raw Measurement Sets, FITS products, h5parm files, full
logs, Dask HTML reports, and run working directories out of git.

Use `docs/source/development/performance_equivalence_contract.rst` as the
contract for deciding whether a run is valid and how to interpret the result.

## Baseline Gate

Run the first gate before making further performance-sensitive optimisation
changes. Start with the phase-only scenario because both branches can run it
without the known legacy slow-gain combination limitations.

Use short `--run-root` and `--repeatability-work-root` paths, preferably under
`/tmp`, when comparing against `master`. The legacy `master` helper scripts can
hit PyBDSF `AF_UNIX path too long` errors when run directories are deeply
nested.

Prepare-only smoke:

```bash
python3 scripts/dev/run_branch_equivalence.py \
  --scenario-id phase-only-core-performance-baseline \
  --base-parset docs/source/development/equivalence_runs/2026-07-05-phase-only-initial-solutions-master-ref/inputs/base/master_benchmark_phase_only.parset \
  --current-parset docs/source/development/equivalence_runs/2026-07-05-phase-only-initial-solutions-master-ref/inputs/current/current_benchmark_phase_only.parset \
  --run-root /tmp/rapthor-performance-gate-prepare-phase-only \
  --prepare-only \
  --repeatability-repetitions 3
```

Full advisory baseline:

```bash
python3 scripts/dev/run_branch_equivalence.py \
  --scenario-id phase-only-core-performance-baseline \
  --base-ref master \
  --base-parset docs/source/development/equivalence_runs/2026-07-05-phase-only-initial-solutions-master-ref/inputs/base/master_benchmark_phase_only.parset \
  --current-parset docs/source/development/equivalence_runs/2026-07-05-phase-only-initial-solutions-master-ref/inputs/current/current_benchmark_phase_only.parset \
  --run-root /tmp/rapthor-performance-gate-phase-only \
  --repeatability-work-root /tmp/rapthor-performance-gate-phase-only-work \
  --repeatability-repetitions 3 \
  --setup-base-env \
  --base-system-site-packages \
  --base-install-spec .
```

The runner writes elapsed wall-clock seconds for each branch repetition, parses
operation boundary timings from each run's `rapthor.log`, and reports
min/median/max plus current-vs-base median deltas. Archive the compact
JSON/Markdown report here after the full gate completes.

The first formal repeatability-aware phase-only gate pass is archived in
`2026-07-11-phase-only-core-repeatability-gate.md`.
