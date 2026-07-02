# Testing Playbook

Use the narrowest test that covers the changed contract, then broaden when the
change crosses module, payload, or scientific product boundaries.

If the prepared dev container is available through Podman, prefer running
formatting, linting, and tests inside it so compiled astronomy dependencies and
external tools match the project environment. Use the same commands below via
`podman exec -w /app <container> ...`.

## Quick Commands

```bash
python -m pytest tests/operations/test_image.py
python -m pytest -m "not integration" tests
tox -e lint
tox
```

For the current stabilization gate from `PLAN.md`, start with:

```bash
python -m pytest tests/lib/test_parset.py
python -m pytest tests/execution/test_config.py tests/execution/test_runtime_bootstrap.py tests/execution/test_runtime_bootstrap_process.py
python -m pytest tests/test_cli.py tests/execution/test_pipeline_flow.py tests/execution/test_payloads.py
tox -e lint
```

To mirror the current tox split manually in the prepared dev-container
environment:

```bash
python -m pytest tests/lib/test_field.py -m "not integration"
python -m pytest -m "not integration and prefect" tests
python -m pytest -m "not integration and not prefect" -n auto --dist worksteal --ignore=tests/lib/test_field.py tests
```

## What To Run By Change Type

| Change | Start With | Broaden To |
| --- | --- | --- |
| Domain model, `Field`, `Observation`, `Sector`, strategy parsing | `tests/lib/` focused file | non-integration tests |
| Operation adapter behavior | `tests/operations/` focused file | operation integration if external tools are involved |
| Payload validators or command builders | `tests/execution/` focused file | command reference or owner-package tests |
| Prefect/Dask scheduling or task-runner behavior | Prefect-marked `tests/execution/` tests | non-integration Prefect suite |
| Import boundaries or retired scripts | `tests/architecture/` | non-integration tests |
| CLI entry points | focused CLI/adapter tests near owner package | non-integration tests |
| Parset/default option | focused parsing/default tests plus docs/templates check | non-integration tests |
| Runtime bootstrap, preflight, or CLI startup | `tests/test_cli.py`, `tests/execution/test_config.py`, `tests/execution/test_runtime_bootstrap*.py` | user-facing `rapthor input.parset` smoke lane |
| Scientific product behavior | focused command/payload/finalizer tests | integration or equivalence check |
| Dask task boundaries or scalability | flow tests plus payload serialization/size guards | benchmark or rich-demo run |
| Benchmark harness/reporting | runner/report parsing tests | manual or scheduled benchmark job |
| Docs-only or `.agents/` change | `git diff --check` and link/path inspection | no test suite unless docs build is requested |

## Heavy Checks

Run integration tests only when the environment has the external tools and data
needed by the scenario:

```bash
RAPTHOR_TEST_RUN_ROOT=/tmp/rapthor-integration-runs python -m pytest -m integration -vv -ra --durations=0 tests/integration tests/operations/integration
```

Run equivalence after scientific logic changes, script-to-module migrations,
calibration strategy changes, or changes to FITS, h5parm, or sky-model products:

```bash
python scripts/dev/run_saved_cwl_equivalence.py --run-root /tmp/rapthor-equivalence --stop-on-failure
```

Use the demo when checking end-to-end runtime bootstrapping or orchestration:

```bash
python scripts/dev/run-rapthor-prefect-demo.py examples/generated/prefect_demo_rich/prefect_demo_rich.parset --run-dir /tmp/rapthor-prefect-demo --no-keep-server
```

For benchmark work, follow `PLAN.md`: define scenarios, run from clean working
directories, repeat at least three times on the same machine/container image,
report median plus min/max, and include command timings, Prefect task timings,
Dask scheduler gaps, memory/disk footprint, and output equivalence or checksum
status.

## Test Hygiene

- Put generated run roots under `/tmp` or another spacious filesystem.
- Do not assume internet access in unit tests.
- Tests marked `integration` are excluded from default non-integration pytest.
- Tests marked `internet` may require network access and should not become
  ordinary unit-test dependencies.
- If you cannot run the relevant test, say so and name the remaining risk.
