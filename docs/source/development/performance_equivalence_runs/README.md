# Performance Equivalence Runs

Compact milestone reports for the branch-vs-master performance gate belong in
this directory. Keep raw Measurement Sets, FITS products, h5parm files, full
logs, Dask HTML reports, and run working directories out of git.

Use `docs/source/development/performance_equivalence_contract.rst` as the
contract for deciding whether a run is valid and how to interpret the result.
The current combined science/performance verdict lives in the root
`EQUIVALENCE_REPORT.md`.

After every successful or decision-relevant performance gate rerun, archive the
compact Markdown/JSON performance report here and update the latest performance
section of `EQUIVALENCE_REPORT.md` in the same change. Track selected per-pair
compact reports only when they are needed to explain a tolerance or decision.
Keep raw products, full logs, Dask reports, and temporary run directories out
of git.

## Repeatability Gates

Run the repeatability-aware gate before making further performance-sensitive
optimisation changes, and rerun targeted scenarios after each optimisation
batch. Start with the phase-only scenario because both branches can run it
without the known legacy slow-gain combination limitations, then cover a
broader mixed-calibration path such as DD phase plus DI full-Jones.

Use short `--run-root` and `--repeatability-work-root` paths, preferably under
`/tmp`, when comparing against `master`. The legacy `master` helper scripts can
hit PyBDSF `AF_UNIX path too long` errors when run directories are deeply
nested.

Prepare-only smoke:

```bash
python3 scripts/dev/run_branch_equivalence.py \
  --scenario-id phase-only-core-performance-baseline \
  --base-parset docs/source/development/science_equivalence_runs/2026-07-05-phase-only-initial-solutions-master-ref/inputs/base/master_benchmark_phase_only.parset \
  --current-parset docs/source/development/science_equivalence_runs/2026-07-05-phase-only-initial-solutions-master-ref/inputs/current/current_benchmark_phase_only.parset \
  --run-root /tmp/rapthor-performance-gate-prepare-phase-only \
  --prepare-only \
  --repeatability-repetitions 3
```

Full advisory baseline:

```bash
python3 scripts/dev/run_branch_equivalence.py \
  --scenario-id phase-only-core-performance-baseline \
  --base-ref master \
  --base-parset docs/source/development/science_equivalence_runs/2026-07-05-phase-only-initial-solutions-master-ref/inputs/base/master_benchmark_phase_only.parset \
  --current-parset docs/source/development/science_equivalence_runs/2026-07-05-phase-only-initial-solutions-master-ref/inputs/current/current_benchmark_phase_only.parset \
  --run-root /tmp/rapthor-performance-gate-phase-only \
  --repeatability-work-root /tmp/rapthor-performance-gate-phase-only-work \
  --repeatability-repetitions 3 \
  --setup-base-env \
  --base-system-site-packages \
  --base-install-spec .
```

The runner writes elapsed wall-clock seconds for each branch repetition, parses
operation boundary timings from each run's `rapthor.log`, and reports
min/median/max plus current-vs-base median deltas. The same command also writes
the science-equivalence report and combined repeatability summary:

- `science-equivalence-report.json` and `science-equivalence-report.md`
- `performance-equivalence-report.json` and
  `performance-equivalence-report.md`
- `repeatability-summary.json` and `repeatability-summary.md`

Archive `performance-equivalence-report.*` here after the full gate completes.
Archive `science-equivalence-report.*` and any selected pair reports under
`docs/source/development/science_equivalence_runs/` when the run updates the
science evidence as well.

Current accepted gate evidence:

- `2026-07-11-phase-only-core-repeatability-gate.md`: phase-only core passes,
  with current median runtime `29.425%` faster than `master`
- `2026-07-12-dd-phase-plus-di-fulljones-repeatability-gate.md`: DD phase plus
  DI full-Jones passes, with current median runtime `37.821%` faster than
  `master`
