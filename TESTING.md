# Testing Rapthor

This document is the developer and agent-facing guide for testing Rapthor.
Keep it aligned with `.agents/testing_playbook.md`, `pyproject.toml`, and
`PLAN.md` whenever test commands, markers, fixtures, or expectations change.

## Testing Principles

- Start with the narrowest test that covers the changed contract, then broaden
  when the change crosses module, payload, runtime, or scientific-product
  boundaries.
- Prefer contract tests over implementation-detail tests. Good failures should
  explain what behavior changed.
- Use pytest features directly: fixtures, fixture factories, `tmp_path`,
  `monkeypatch`, `caplog`, `pytest.raises`, `pytest.approx`,
  `pytest.mark.parametrize`, and `pytest.param(..., id=...)`.
- Keep tests readable. Avoid very long setup blocks, one-row parametrizations,
  and copy-pasted scenario construction when a fixture or helper would make the
  intent clearer.
- Do not assume internet access in ordinary tests. Tests that require network
  access must be marked `internet`.
- Keep generated run products, large Measurement Sets, FITS products, Dask
  reports, coverage output, `.tox`, `.ruff_cache`, and temporary run roots out
  of source decisions.
- Prefer small fixtures from `tests/resources/` over adding new large data
  files.

## Clean Test Code

Test code is production code for confidence. Keep it clean, reviewed, and
designed for the next person who needs to understand Rapthor's behaviour.

- Apply the same care to tests that you apply to production code: clear names,
  small functions, simple control flow, and no incidental cleverness.
- Let the test structure mirror the contract being protected. A reader should
  see setup, action, and expected outcome without untangling unrelated details.
- Prefer a descriptive test name. Add a docstring when the name alone cannot
  explain the behavior, when scientific or runtime reasoning matters, or when a
  future developer needs context for why the case exists.
- Prefer domain language over generic test language: use names like
  `dd_slow_gain_strategy`, `facet_layout`, `prepared_ms`, and
  `normalization_h5parm`.
- Keep helpers narrow and honest. A helper named `make_image_payload` should
  build an image payload, not also patch global state, create files, and assert
  side effects.
- Avoid giant assertions that make failures opaque. Compare small named pieces
  or use helper assertions that report the scenario, option, product, or command
  that failed.
- Keep fixtures composable but shallow. If understanding a test requires
  following a long fixture chain, the setup is probably too hidden.
- Delete dead setup and stale comments when behaviour moves. Tests should not
  preserve historical scaffolding just because it once helped.
- Refactor tests opportunistically when changing adjacent behaviour. Do not add
  the next case to a monolithic file if a small scenario table or helper module
  would make future changes clearer.
- Do not duplicate setup or helper functions unnecessarily. If two tests build
  the same field, parset, payload, strategy, fake shell operation, or external
  command response, extract a shared fixture or helper with a domain-specific
  name.
- Share setup at the right level: local helper for one file, `conftest.py`
  fixture for a package, and only top-level `tests/conftest.py` when most of
  the suite benefits. Avoid making every helper global by default.
- Keep shared helpers extensible through small keyword overrides instead of
  long positional argument lists or hidden mutations.
- Do not deduplicate at the cost of readability. Repeated one-line setup can be
  clearer than a generic helper that hides the scenario.

## Environment

The prepared dev container is the preferred environment for formatting, linting,
unit tests, Prefect tests, integration tests, equivalence checks, and demo runs.
It includes compiled astronomy dependencies that are often not available in a
fresh local Python environment.

If the container is running through Podman, run commands like this:

```bash
podman exec -w /app <container-id-or-name> python3 -m pytest tests/lib/test_parset.py
```

Find a running container with:

```bash
podman ps
```

Local tox environments are useful for pure Python checks, but may try to build
packages such as `python-casacore` or `everybeam` if the host is not already
prepared.

## Formatting And Linting

After any Python code change, run:

```bash
ruff check --fix --select I
ruff format
```

Before handing over a larger change, also run:

```bash
tox -e lint
```

For docs-only changes, `git diff --check` is the minimum useful check.

## Quick Commands

Run one focused file while developing:

```bash
python -m pytest tests/operations/test_image.py
python -m pytest tests/lib/test_parset.py
python -m pytest tests/execution/test_config.py
```

Run all non-integration tests:

```bash
python -m pytest -m "not integration" tests
```

Run the default tox suite:

```bash
tox
```

Mirror the current tox split manually in the prepared dev-container
environment:

```bash
python -m pytest tests/lib/test_field.py -m "not integration"
python -m pytest -m "not integration and prefect" tests
python -m pytest -m "not integration and not prefect" -n auto --dist worksteal --ignore=tests/lib/test_field.py tests
```

The split matters:

- `tests/lib/test_field.py` does not support parallel execution.
- Tests using Prefect's test harness start a local Prefect test server and must
  run serially.
- The remaining non-integration tests can use xdist.

When reviewing suite speed, start with collection and duration signals:

```bash
python -m pytest -m "not integration" tests --collect-only -q
python -m pytest -m "not integration" tests --durations=30 --durations-min=0.25
```

Use duration output to decide whether a slow test belongs in the default lane,
needs a smaller fixture, should be marked `slow`, or should move to integration
or equivalence coverage.

## Test Layout

- `tests/lib/`: domain model, parset, strategy, records, observation, field,
  sector, and small utility tests.
- `tests/operations/`: operation adapter behavior. These tests should keep
  adapters thin and verify how domain state becomes execution inputs and
  finalized outputs.
- `tests/execution/`: execution-owned payloads, command builders, flow wiring,
  runtime bootstrap, shell adapters, artifacts, benchmarking, and helper
  modules.
- `tests/architecture/`: ownership-boundary tests that prevent retired or
  misplaced imports from returning.
- `tests/integration/` and `tests/operations/integration/`: external-tool and
  end-to-end behavior against small representative datasets.
- `tests/resources/`: small committed fixtures and parset templates.

## Pytest Markers

Markers are declared in `pyproject.toml`:

- `integration`: tests that require external tools or end-to-end runtime
  behavior and are excluded from ordinary non-integration runs.
- `internet`: tests that require network access.
- `prefect`: tests that start a Prefect test server and must run serially.
- `slow`: tests that are slow enough to deselect explicitly.

`tests/conftest.py` automatically marks collected tests whose source uses
`prefect_test_harness` as `prefect`. Add the marker explicitly only when the
automatic source check is not enough.

## What To Run By Change Type

| Change | Start With | Broaden To |
| --- | --- | --- |
| Domain model, `Field`, `Observation`, `Sector`, strategy parsing | Focused `tests/lib/` file | `python -m pytest -m "not integration" tests` |
| Parset/default option | `tests/lib/test_parset.py`, `tests/lib/test_parset_option_behavior.py`, `tests/lib/test_parset_option_coverage.py` | Docs/templates check plus non-integration tests |
| Operation adapter behavior | Focused `tests/operations/` file | Operation integration if external tools are involved |
| Payload validators or command builders | Focused `tests/execution/` file | Owner-package tests and command/reference tests |
| Prefect/Dask scheduling or task-runner behavior | Prefect-marked execution tests | Non-integration Prefect suite |
| Runtime bootstrap, preflight, or CLI startup | `tests/test_cli.py`, `tests/execution/test_config.py`, `tests/execution/test_runtime_bootstrap*.py` | User-facing `rapthor input.parset` smoke lane |
| Scientific product behavior | Focused command/payload/finalizer tests | Integration or equivalence check |
| Dask task boundaries or scalability | Flow tests plus payload serialization/size guards | Benchmark or rich-demo run |
| Benchmark harness/reporting | `tests/execution/test_benchmarking.py` and demo-data generator tests | Manual or scheduled benchmark job |
| Documentation only | `git diff --check` and link/path inspection | Docs build only when requested or risky |

## Parset Option Coverage

`tests/lib/test_parset_option_coverage.py` parses
`rapthor/settings/defaults.parset` and checks that every default option has
direct test attention or an explicit allow-list reason. When adding or renaming
a user-facing option:

1. Update `rapthor/settings/defaults.parset`.
2. Update `rapthor/settings/defaults.json` when relevant.
3. Update parset templates in `tests/resources/`.
4. Update user docs under `docs/source/`.
5. Add focused parser/default behavior coverage.
6. Add operation, execution, integration, or equivalence coverage if the option
   changes runtime or scientific behavior.

The allow-list in `test_parset_option_coverage.py` is for intentional
exceptions only. If an option now has a direct test mention, remove it from the
allow-list.

## Prefect Test Harness

Use Prefect's test harness for tests that need to execute production Prefect
flows without requiring a persistent external Prefect server:

```python
from prefect.testing.utilities import prefect_test_harness


def test_flow_behavior(payload):
    with prefect_test_harness(server_startup_timeout=None):
        result = flow_fn(payload)
    assert result["status"] == "completed"
```

Prefer `run_flow_for_test` from `tests/execution/conftest.py` when it fits:

```python
def test_flow_uses_payload(payload):
    result = run_flow_for_test(flow_fn, payload, shell_operation_cls=FakeShellOperation)
    assert result["output_path"].endswith(".fits")
```

Guidelines for Prefect tests:

- Default to `ExecutionConfig(task_runner="sync")` for behavior checks.
- Use local Dask only when testing scheduler, Dask resource, or task-runner
  behavior.
- Patch external command loading with fake shell operation classes when the
  test is about flow wiring rather than the external command itself.
- Keep payloads plain and serializable. Do not pass live `Field`, `Observation`,
  `Sector`, operation instances, file handles, or subprocess state across worker
  boundaries.
- Do not run Prefect-harness tests under xdist. The tox split runs them
  serially with `-m "not integration and prefect"`.
- Keep `PREFECT_HOME` isolated. Test setup already does this through
  `tests/conftest.py`; avoid overriding it unless the test is specifically
  about runtime bootstrap.

## Integration Tests With External Tools

Integration tests exercise the pipeline with external astronomy tools such as
DP3, WSClean, EveryBeam, IDG, PyBDSF, and Casacore. They are heavier than unit
tests and should be run only in an environment that has the required tools and
data access.

Run the integration suite with:

```bash
RAPTHOR_TEST_RUN_ROOT=/tmp/rapthor-integration-runs python -m pytest -m integration -vv -ra --durations=0 tests/integration tests/operations/integration
```

Or through tox:

```bash
tox -e test_integration
```

The shared integration template is intentionally smoke-sized so CI and local
dev-container runs exercise the same external-tool scenarios quickly. Use
benchmark and equivalence checks for larger, science-representative imaging
workloads.

Integration-test guidelines:

- Mark integration tests with `@pytest.mark.integration`.
- Add `@pytest.mark.internet` when a test depends on remote survey/catalog
  access.
- Use fixtures and helpers from `tests/integration/conftest.py` and
  `tests/integration/utils.py` rather than duplicating parset and command-log
  handling.
- Keep run roots under `/tmp` or `RAPTHOR_TEST_RUN_ROOT`; CI keeps integration
  products inside the project only so artifacts can be uploaded.
- Prefer `tests/resources/integration_template.parset` and small fixture sky
  models over bespoke large inputs.
- Assert backend-neutral products where possible: output records, command
  records, h5parm structure, sky-model summaries, FITS metadata/statistics, and
  expected `.done` markers.
- Skip or xfail environment-specific scenarios with a clear reason, for example
  Slurm-only behavior or known external-tool limitations.

CI may split integration tests with `pytest-split` using `CI_NODE_TOTAL` and
`CI_NODE_INDEX`.

## Equivalence, Demo, And Benchmark Checks

Use saved CWL equivalence after scientific logic changes, script-to-module
migrations, calibration strategy changes, or changes to FITS, h5parm, or
sky-model products:

```bash
python scripts/dev/run_saved_cwl_equivalence.py --run-root /tmp/rapthor-equivalence --stop-on-failure
```

Use the generated demo for end-to-end runtime bootstrapping or orchestration:

```bash
python scripts/dev/run-rapthor-prefect-demo.py examples/generated/prefect_demo_rich/prefect_demo_rich.parset --run-dir /tmp/rapthor-prefect-demo --no-keep-server
```

For benchmark work, follow `PLAN.md`. Benchmarks should report median plus
min/max across repetitions, command timings, Prefect task timings, Dask
scheduler gaps, memory/disk footprint, and output equivalence or checksum
status. Do not commit raw benchmark run directories or bulky artifacts.

## Tests As Living Documentation

Treat tests as executable examples of how Rapthor is expected to behave. A
developer should be able to open a test file, skim the scenario names and
parameter ids, and understand the supported behaviour without first reading the
production implementation.

Good tests tell a small story:

1. Given this parset, strategy, payload, field state, or external-tool result.
2. When this operation, command builder, flow, or helper runs.
3. Then these user-visible outputs, records, commands, warnings, or failures are
   expected.

Write tests for future readers:

- Put the behavioural idea in the test name, for example
  `test_image_only_cycle_preapplies_di_h5parm_without_skymodel`, not
  `test_h5parm_case_3`.
- Use `pytest.param(..., id="di-slow-gains-preapply")` so failures name the
  scenario in domain language.
- Prefer realistic names for domain fixtures: `target_field`,
  `image_only_strategy`, `calibration_h5parm`, `facet_layout`, and
  `prepared_measurement_set` are easier to follow than `obj`, `data`, or
  `mock1`.
- Keep setup close to the assertion when it is unique to one test. Move setup
  into a fixture only when several tests share the same concept.
- Avoid hiding important behaviour inside over-general helpers. A helper should
  remove noise, not make the reader hunt for the actual scenario.
- Group related scenarios in small tables when the contrast between rows is the
  point. If each row needs a paragraph of explanation, split it into separate
  named tests.
- Assert the public contract first: output records, command arguments, product
  paths, h5parm/skymodel/FITS summaries, warnings, exceptions, and task names
  when those names are part of the debugging contract.
- Keep low-level implementation assertions only when they protect an explicit
  boundary, such as serializable worker payloads or deterministic command
  builders.
- Use comments sparingly to explain why a scenario matters scientifically or
  operationally. Do not narrate code that is already clear.
- Use test docstrings for non-obvious reasoning that belongs with the scenario:
  scientific intent, migration history, external-tool constraints, restart
  semantics, or why a regression would be harmful.
- Prefer exact expected structures for small payloads and records. For large
  outputs, assert a named subset that captures the behaviour being protected.
- Make failure messages useful. If a comparison is nontrivial, include the
  option name, scenario id, product path, or command name in the assertion or
  helper output.

Tests should also be easy to extend:

- Add the next scenario by appending a readable `pytest.param` row when the
  setup and assertions are genuinely shared.
- Add a new fixture when a concept is reused, not just to shorten one test.
- Keep fixture dependencies shallow. A reader should be able to understand the
  setup path without following a long chain across files.
- Prefer fixture factories for domain objects that need small per-test
  variations.
- Consolidate repeated fakes and setup builders when they represent the same
  concept, for example fake shell operations, operation parsets, image/calibrate
  payload factories, and integration strategy writers.
- Keep tests deterministic by controlling paths, environment variables, random
  seeds, and time-sensitive values.
- Keep external state at the edge. Unit tests should fake shell operations,
  survey downloads, scheduler startup, and file discovery unless the test is
  explicitly an integration or runtime test.

## Keeping Tests Fast

Fast tests get run more often, so speed is part of correctness. Keep the
default non-integration lane quick enough that developers are willing to run it
before changing architecture or scientific contracts.

- Prefer pure unit tests for parsing, payload construction, command builders,
  resource calculations, and finalizer decisions.
- Use Prefect's test harness only when testing actual flow behaviour. Builder,
  validator, and finalizer tests should usually call plain Python helpers.
- Keep Prefect-harness tests serial and focused. If several tests start a
  Prefect test server just to reach the same helper logic, pull that logic into
  a direct unit test and leave one flow-level smoke test.
- Avoid subprocess calls in unit tests unless the subprocess boundary is the
  behaviour being tested. Patch shell runners or use fake shell operation
  classes for command orchestration.
- Copy or mutate Measurement Sets only in tests that truly need a writable MS.
  Reuse session fixtures, tiny synthetic files, or metadata-only payloads for
  command and flow wiring tests.
- Keep FITS, h5parm, and sky-model fixtures as small as the assertion allows.
  Cache expensive synthetic products in fixtures when they are read-only.
- Do not import heavy astronomy libraries in broad conftest files unless most
  tests need them during collection. Prefer local imports inside fixtures or
  tests for optional-heavy paths.
- Consolidate integration assertions around each expensive Rapthor run. If a
  single external-tool run can prove several product contracts, assert them
  together instead of launching another end-to-end scenario.
- Mark genuinely slow tests with `slow` and keep them out of the fast lane
  unless they protect a critical default contract.
- Track collection time as well as execution time. Slow collection usually
  means heavy imports, broad fixtures, or too much work at module import time.

Speed cleanup targets to keep under review:

- Large flow modules with many Prefect-harness tests, especially
  `tests/execution/test_calibrate_flow.py` and
  `tests/execution/test_image_flow.py`.
- Large operation modules whose scenario matrices can be split into smaller
  behaviour-focused files, especially `tests/operations/test_calibrate.py` and
  `tests/operations/test_image.py`.
- One-row parametrizations in `tests/lib/test_miscellaneous.py`; convert them
  to direct tests or fixtures unless they are being prepared for real scenario
  matrices.
- Repeated integration tests that launch Rapthor only to check behaviour that
  can be covered by a focused operation, execution, or command-builder test.

## Writing Maintainable Tests

- Name tests after behavior, not implementation.
- Use one assertion group per behavior. Multiple assertions are fine when they
  describe one contract.
- Use `pytest.param(..., id="readable-case")` for nontrivial parametrized
  matrices.
- Prefer fixture factories for repeated domain setup.
- Prefer `tmp_path` over hard-coded temporary paths.
- Prefer `caplog` over string parsing of console output.
- Use `pytest.raises(..., match=...)` for expected failures.
- Use `pytest.approx` for floating-point values and scientific tolerances.
- Keep external-command tests at the command-boundary level unless the external
  tool itself is intentionally part of an integration test.
- Add focused regression tests for bug fixes before broadening to full suites.
- Update this file when adding new marks, new test tiers, new helper fixtures,
  or new required verification commands.
