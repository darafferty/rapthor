# Repo Architecture Guide

Use this guide to decide where a change belongs before editing. Rapthor has a
deliberate split between domain state, operation adapters, execution-owned
mechanics, and user-facing settings.

`PLAN.md` is the current roadmap. It says the main architecture cleanup is far
enough along that new architecture work should favor stabilization, benchmarking,
Dask scalability guardrails, runtime UX, and contributor documentation over
compatibility shims or broad cosmetic reshuffling.

## Ownership Layers

| Layer | Main Paths | Owns | Keep Out |
| --- | --- | --- | --- |
| CLI | `rapthor/cli.py` | Parsing entry-point arguments and starting runs | Scientific workflow logic |
| Domain model | `rapthor/lib/` | `Field`, `Observation`, `Sector`, `Cluster`, `Operation`, parset and strategy state | Worker-only execution details |
| Operation adapters | `rapthor/operations/` | Thin adapters that translate domain state into operation inputs and finalize outputs | Command mechanics, subprocess details, heavy helper logic |
| Pipeline flow | `rapthor/execution/pipeline/flow.py` | Top-level Prefect flow, scheduling, operation ordering | Scientific option semantics better owned by domain/operation code |
| Execution owners | `rapthor/execution/<owner>/` | Payloads, command builders, validation, outputs, task wiring, migrated helper-script logic | `Field`, `Observation`, `Sector`, or operation instances in worker payloads |
| Settings | `rapthor/settings/` | Defaults exposed through parsets and JSON defaults | Behavior that should be tested in code |
| Docs/examples | `docs/source/`, `examples/` | User-facing behavior and runnable strategy/parset examples | Production pipeline code |
| Tests | `tests/lib`, `tests/operations`, `tests/execution`, `tests/architecture`, `tests/integration` | Contract, behavior, architecture, and end-to-end coverage | Large generated data |

`docs/source/development/architecture.rst` is the more detailed architecture
reference. It also describes a possible future `rapthor.application` or
`rapthor.use_cases` layer; that package is not present today, so do not invent
it for small changes.

## Change Placement

- New calibration command option: update the relevant
  `rapthor/execution/calibrate/` command or payload module first, then the
  operation adapter only if domain state needs to feed it.
- New imaging command option: start in `rapthor/execution/image/`, then update
  imaging operation hand-off code, defaults, docs, examples, and tests.
- New parset option: update defaults, parsing/domain state, payload builders,
  validators, docs, templates, examples, and focused tests together.
- New operation output: update the execution owner output record, operation
  finalizer, restart behavior, and tests that assert output discovery.
- New helper that runs external tools: place production code in the appropriate
  `rapthor/execution/<owner>/` package, not `scripts/dev` or `scripts/prod`.
- New scientific strategy behavior: start with `rapthor/lib/strategy.py` and
  operation consumption points, then update examples and tests.
- New preflight or runtime-UX behavior: start in `rapthor/execution/pipeline/`,
  `rapthor/execution/runtime_bootstrap.py`, `rapthor/execution/config.py`, and
  `docs/source/running.rst`.
- New benchmark or Dask scalability work: read `PLAN.md` first and add tests
  that make task boundaries, payload sizes, serialization, and run reports
  visible.

## Reading Order For A Change

1. Read the user-facing option or strategy docs if the behavior is exposed.
2. Read the domain object that stores the state.
3. Read the operation adapter that passes the state onward.
4. Read the execution owner payload, validator, command builder, and output
   finalizer.
5. Read the narrowest existing tests in the matching `tests/` subtree.

## Boundary Rules

- Keep operation adapters thin. If the work is about command construction,
  files, payload validation, or subprocess execution, it belongs under
  `rapthor/execution/<owner>/`.
- Keep Prefect/Dask task payloads plain and serializable. Do not send domain
  objects or operation instances to workers.
- Keep command builders deterministic. Tests should be able to compare emitted
  command tokens without relying on incidental ordering.
- Keep scripts under `scripts/dev` and `scripts/prod` as wrappers or developer
  utilities only.
- Keep generated local noise such as `__pycache__`, `.tox`, `.ruff_cache`,
  `runs`, `htmlcov`, build outputs, and temporary demo or integration roots out
  of source decisions.
