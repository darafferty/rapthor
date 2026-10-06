# Agent instructions

Rapthor is a radio astronomy pipeline for LOFAR direction-dependent calibration,
with ongoing SKA-Low support. Production workflows use Prefect/Dask to run DP3,
WSClean and other astronomy tools.

## Working rules

- Check the worktree before editing; preserve unrelated changes. Do not commit
  unless asked.
- Keep fixes scoped. Follow the surrounding code, use comments for non-obvious
  reasoning, and preserve module loggers such as `rapthor:calibrate`.
- Keep generated data, large Measurement Sets, downloaded archives and run/build
  products out of version control. Do not modify original input Measurement Sets.
- Format changed Python code and imports, then run checks appropriate to the
  change as described in [TESTING.md](TESTING.md). Report checks that could not run.
- Update user documentation when behavior, options or runtime requirements change.
  Keep architecture diagrams aligned with ownership and execution changes.

## Code ownership and contracts

- `rapthor/lib/` owns domain state and parset/strategy interpretation.
- `rapthor/operations/` contains thin adapters: gather domain state, call execution
  code, then finalize results back into the field.
- `rapthor/execution/<owner>/` owns operation payloads, validation, command
  builders, output discovery, helper logic and Prefect flow wiring.
- `rapthor/cli.py` owns the CLI; dependency, entry-point and tool configuration
  belongs in [pyproject.toml](pyproject.toml).

Preserve these execution contracts:

- Worker payloads contain plain serializable values and file paths. Do not pass
  live `Field`, `Observation`, `Sector`, operation instances, tables, file handles
  or subprocess state across worker boundaries.
- Keep command builders deterministic. Give tasks useful domain names and test
  scheduling or serialization boundaries when changing them.
- New products need output records, finalizer state and restart handling. Skipped
  commands must still return every record field required downstream; discovery
  must not pick up products from the wrong cycle.
- Distinguish worker-local scratch from shared storage. Clean up only after all
  consumers finish, and preserve recovery products on failed runs.

## Scientific and option changes

- `calibration_strategy` controls solve types and their order. Preserve supported
  sequences and the initial-model-dependent early phase-only cycles; legacy solve
  toggles are compatibility inputs, not the interface for new work.
- Keep DI, DD, full-Jones, normalization and screens distinct, and retain explicit
  apparent-sky/true-sky and generate/apply states.
- Distinguish solutions used as optimizer seeds from applied corrections. Their
  cycle and direction compatibility rules differ; consult the science reference.
- For a user option, update both applicable defaults (`defaults.parset` and
  `defaults.json`), parsing/domain state, operation inputs, execution payloads,
  validators and commands, docs/examples, test templates and behavior tests.

## Read references when relevant

Use the section needed for the task; routine edits do not require reading every
reference.

| Task | Reference |
| --- | --- |
| Tests, formatting or test environment | [TESTING.md](TESTING.md) |
| Ownership, flow/task boundaries or runtime architecture | [Architecture](docs/source/development/architecture.rst) |
| Scientific semantics: solves, corrections, beams, models or averaging | [Science reference](.agents/scientific_glossary.md) |
| User options or strategy behavior | [Parset](docs/source/parset.rst), [strategy](docs/source/strategy.rst) |
| CLI, task runners, containers or cluster execution | [Running](docs/source/running.rst), [installation](docs/source/installation.rst) |
| Product names, restart behavior or migration compatibility | [Products](docs/source/products.rst), [upgrading](docs/source/upgrading.rst) |
| Planned architecture, scalability or development workflow work | [Post-merge plan](PLAN.md) |
