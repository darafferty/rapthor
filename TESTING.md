# Testing Rapthor

This is the shared guide for contributors and coding agents. Tool configuration
and markers are declared in [pyproject.toml](pyproject.toml).

## Environment

Prefer the prepared [dev container](.devcontainer/devcontainer.json), which has
compiled astronomy dependencies. A fresh host or tox environment may need to
build `python-casacore`, `everybeam` and other dependencies.

From the host, find the running container and run commands in `/app`:

```bash
podman ps
# Replace the value with the container name or ID from the output.
RAPTHOR_CONTAINER=rapthor-dev
podman exec -w /app "$RAPTHOR_CONTAINER" python -m pytest tests/lib/test_parset.py
```

For local development, use `python -m pip install --group=dev` and an editable
install (`python -m pip install -e .`), as in the container setup. Python support
and dependency versions are maintained in `pyproject.toml`.

## Choose checks for the change

Start with the focused test file. Broaden when the change crosses an operation,
worker, runtime or scientific-product boundary, or leaves an unresolved risk.
Do not run expensive suites for unrelated edits.

| Change | Initial checks | Broaden when needed |
| --- | --- | --- |
| Domain state or strategy parsing | Relevant `tests/lib/` file | Affected callers and non-integration suite |
| User option/default | `test_parset.py`, `test_parset_option_behavior.py`, `test_parset_option_coverage.py` under `tests/lib/` | Operation/execution behavior and docs/templates |
| Operation adapter or finalizer | Relevant `tests/operations/` file | Execution owner and restart cases |
| Payload, validator or command builder | Relevant `tests/execution/` file | Owner flow and command-reference tests |
| Ownership boundary | `tests/architecture/` and affected owner tests | Non-integration suite |
| Prefect/Dask scheduling | Prefect flow tests, using local Dask for scheduler behavior | Affected flows and real-run smoke test |
| CLI/bootstrap/preflight | `tests/test_cli.py`, execution config/bootstrap tests | CLI smoke test in the prepared environment |
| Scientific products | Focused command, payload and finalizer tests | External-tool integration and relevant science diagnostics |
| Documentation only | `git diff --check`, links and paths | Sphinx build for changed RST structure or directives |

For Python changes, format imports and changed files before final verification:

```bash
ruff check --fix --select I path/to/changed_file.py
ruff format path/to/changed_file.py
tox -e lint
```

The paths above are placeholders for the changed Python files. Docs-only edits
do not need Python formatting or runtime tests.

## Non-integration suite

Run one focused file during development:

```bash
python -m pytest tests/operations/test_image.py
```

For the complete non-integration suite, run the `serial` tests on their own and
then everything else in parallel, which mirrors the `tox -e py` commands:

```bash
export RAPTHOR_TEST_RUN_ROOT=/tmp/rapthor-test-unit
python -m pytest -m "not integration and serial" tests
python -m pytest -m "not integration and not serial" -n auto --dist worksteal tests
```

- Choose `-n <count>` instead of `-n auto` when resource limits require fewer
  workers, or use `PYTEST_XDIST_AUTO_NUM_WORKERS=2 tox -e py`. Flow tests share
  one Prefect server per worker, so more workers also use more memory.
- [tests/conftest.py](tests/conftest.py) isolates `PREFECT_HOME` by worker. Preserve
  this isolation. Concurrent pytest invocations can interfere through default
  run-root cleanup; use distinct `RAPTHOR_TEST_RUN_ROOT` values if needed.
- `tox` runs all configured environments, including integration tests and
  multiple Python versions. Select a specific available Python environment when
  that is sufficient; the command above uses the prepared environment.

## Coverage

`tox -e py` and `tox -e test_integration` both collect coverage. CI combines the
unit lane with the eight integration nodes in the `coverage_report` job, so the
reported figure includes code that only the end-to-end runs reach. Coverage
follows the `rapthor` subprocesses and Dask workers through the `patch` settings
under `[tool.coverage.run]` in [pyproject.toml](pyproject.toml), and
`[tool.coverage.paths]` maps the package tox installs back onto the checkout.
To reproduce the combined report locally, give each lane its own data file:

```bash
COVERAGE_FILE=.coverage.unit tox -e py
COVERAGE_FILE=.coverage.integration tox -e test_integration
coverage combine .coverage.unit .coverage.integration
coverage report
```

## Markers and test helpers

| Marker | Meaning |
| --- | --- |
| `integration` | External-tool or end-to-end tests, excluded from the commands above. |
| `internet` | Network-dependent tests; require network access or deselect explicitly. |
| `prefect` | Tests that start or share a Prefect test server. |
| `serial` | Tests that launch their own Prefect and Dask processes; run outside the parallel pool. |
| `slow` | Tests that can be explicitly deselected for a faster run. |

Request the `run_flow_for_test` fixture from the
[execution conftest](tests/execution/conftest.py) to run production flows with
optional fake shell operations. It defaults to the `sync` task runner and
depends on the session-scoped `prefect_server` fixture. Tests that invoke flows
or operations directly request `prefect_server` instead. Both are automatically
marked `prefect`; the server starts only once per pytest process.

Tests that start their own server in a subprocess need an explicit `prefect`
marker. Add `serial` to process-startup tests that need isolation from parallel
CPU load. Use local Dask for scheduling, resource and worker-boundary tests.
Builder, validator and finalizer tests normally call plain Python helpers.

[Option coverage](tests/lib/test_parset_option_coverage.py) checks options in
`defaults.parset` for direct test attention or an intentional allow-list reason.
It does not prove behavior or JSON-default consistency. Add behavior assertions,
remove obsolete allow-list entries, and synchronize option changes as described
in [AGENTS.md](AGENTS.md).

## External-tool integration and demo

Use an environment with DP3, WSClean, EveryBeam, IDG, PyBDSF and Casacore as required
by the scenario. Some fixtures download the small test Measurement Set on first
use; prepare its cache before an offline run. Ordinary unit tests should not
introduce network dependencies.

```bash
RAPTHOR_TEST_RUN_ROOT=/tmp/rapthor-integration-runs \
  python -m pytest -m integration -vv -ra --durations=0 \
  tests/integration tests/operations/integration tests/execution/integration
```

Alternatively use `tox -e test_integration`. CI splits this lane across eight
jobs using `CI_NODE_TOTAL` and `CI_NODE_INDEX`. pytest-split balances them using
[.test_durations](.test_durations), assigning the longest tests first and using
the average duration for new tests. After changing integration scenarios, add
`--store-durations` to the full pytest command above and commit the updated file.

Integration tests must live in one of the three directories above to run in CI.
Real-tool checks of one execution owner belong in `tests/execution/integration/`.
CI reuses the container's installed packages in both test lanes. Tox also passes
`EVERYBEAM_DATADIR` through so DP3 and WSClean can find the coefficient files
despite tox changing `VIRTUAL_ENV`.

Reuse [integration fixtures](tests/integration/conftest.py),
[helpers](tests/integration/utils.py), and
[the smoke-sized parset](tests/resources/integration_template.parset), which uses
local Dask. Assert relevant output records, command records, h5parm structure,
sky models, FITS metadata/statistics, diagnostics and `.done` markers. Give skips
or xfails a concrete environment or external-tool reason. Keep products under
`RAPTHOR_TEST_RUN_ROOT`; CI may retain them for artifact upload.

For a CLI/runtime smoke test, run from the repository root:

```bash
rapthor examples/prefect_demo.parset
```

This requires `tests/resources/test.ms` from the fixture download, or an adjusted
`input_ms` in the demo parset. Demo work directories are gitignored. See
[running](docs/source/running.rst) for server/dashboard setup. A small smoke run
checks execution; it does not establish scientific equivalence on larger data.

## Write tests a reviewer can understand

- Name the behavior and give parametrized cases readable IDs. Show setup, action
  and expected outcome without reconstructing the production algorithm.
- Assert observable contracts: commands, records, products, warnings or failures.
  Test internals only when they protect an explicit boundary, such as worker
  serialization or restart behavior.
- Keep unique setup nearby and shared fixtures shallow. Use a local helper for
  one file, package `conftest.py` for shared setup, and root fixtures only when
  widely needed. A little repetition can be clearer than a generic helper.
- Use small existing fixtures in `tests/resources/`, `tmp_path`, `caplog`,
  `pytest.raises(..., match=...)` and `pytest.approx` where appropriate.
- Control paths, environment, randomness and time. Fake downloads and external
  commands in unit tests; run real tools only for the behavior that needs them.
- Add focused regression coverage for behavioral fixes. Document scientific
  tolerances and non-obvious constraints beside the scenario.
- Diagnose slow tests with `--durations=30` or `--collect-only`; avoid repeated
  server startup, MS copies or end-to-end runs when one scenario can cover the
  relevant assertions. Share the Prefect server through `prefect_server` rather
  than starting one per test. Keep heavy optional imports out of broad
  collection paths.
