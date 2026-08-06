# Equivalence scenario resources

This directory contains the small, version-controlled inputs required to rerun
Rapthor's branch-vs-master science and performance checks. Historical reports
and generated products do not belong here.

The parsets use the dev-container workspace path `/app` so the legacy `master`
CWL runner and the current Prefect/Dask runner can consume the same generated
Measurement Set and sky models. Generate those ignored demo inputs first:

```bash
python scripts/dev/generate-prefect-demo-data.py --include-multi-sector --force
```

Run one scenario from inside the dev container:

```bash
python scripts/dev/run_branch_option_matrix.py \
  --matrix tests/resources/equivalence/option-matrix.json \
  --scenario phase-only-core \
  --run-root /tmp/rapthor-equivalence-phase-only \
  --setup-base-env \
  --base-system-site-packages
```

Omit `--scenario` to run the complete matrix. Scenarios with a `skip_reason`
record a known coverage limitation without attempting an invalid comparison.
The two core gate scenarios request three repetitions by default and therefore
produce both science- and performance-equivalence reports.

The separate saved-CWL gate still requires the ignored
`.pytest_cache/cwl-reference-artifacts/` reference bundle. That large bundle
must be retained in external artifact storage rather than committed here.
