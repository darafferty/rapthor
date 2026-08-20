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
  --run-root /tmp/eq-phase \
  --setup-base-env \
  --base-system-site-packages \
  --base-pip-constraint tests/resources/equivalence/master-constraints.txt
```

To investigate generated-initial-sky-model grouping, run the paired control and
imaging-BDA scenarios together:

```bash
python scripts/dev/run_branch_option_matrix.py \
  --matrix tests/resources/equivalence/option-matrix.json \
  --scenario initial-skymodel-regroup \
  --scenario initial-skymodel-bda-regroup \
  --run-root /tmp/eq-initial \
  --setup-base-env \
  --base-system-site-packages \
  --base-pip-constraint tests/resources/equivalence/master-constraints.txt
```

Both scenarios generate the initial image and sky model, regroup sources to a
single direction from a 1 Jy target flux, and then calibrate. The first keeps
the initial imaging data unaveraged; the second enables the production imaging
BDA settings. Each runs three repetitions per branch so the gate can distinguish
branch divergence from normal solver and imaging repeatability. Comparing the
pair localizes a difference to the generated-model path or its interaction with
BDA instead of conflating the two.

Active scenarios pin calibration and imaging BDA values explicitly and equally
on both branches. Phase-only, mixed-calibration, normalization, and prediction
rows remain no-BDA controls. The BDA averaging, BDA frequency-limit, and
generated-initial-model BDA rows use the production settings of averaging
enabled with `bda_timebase = bda_frequencybase = 20000.0`; the paired initial
model control disables imaging BDA while retaining the production calibration
BDA settings. Legacy CWL/Toil plus PyBDSF multiprocessing can exceed the
AF_UNIX socket path limit when nested under descriptive directory names. In
repeatability mode the matrix runner therefore uses a short unique directory
below the system temporary directory by default, while retaining full scenario
names in its reports. An explicit
`--repeatability-work-root` should likewise be short when site policy requires
scratch data elsewhere.

The master constraint file retains the NumPy 1 ABI used by the legacy master
container. This is required when its virtual environment reuses compiled
system astronomy packages such as python-casacore; installing NumPy 2 into that
environment makes those extensions fail before Rapthor starts.

Omit `--scenario` to run the complete matrix. Scenarios with a `skip_reason`
record a known coverage limitation without attempting an invalid comparison.
The two core gate scenarios request three repetitions by default and therefore
produce both science- and performance-equivalence reports.

The separate saved-CWL gate still requires the ignored
`.pytest_cache/cwl-reference-artifacts/` reference bundle. That large bundle
must be retained in external artifact storage rather than committed here.
