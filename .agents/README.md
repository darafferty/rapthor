# Agent Playbook

This directory holds short, high-signal guides for AI coding agents working in
Rapthor. `AGENTS.md` remains the top-level instruction file; use these files
when you need fast context before changing a specific part of the system.

`AGENTS.md` intentionally repeats only critical guardrails and routing. Keep
detailed workflows here so agents get both a short contract and focused help
without reading the same long guidance twice.

## Reading Route

1. Start with `AGENTS.md` at the repository root.
2. Read `.agents/scientific_glossary.md` before changing calibration,
   prediction, imaging, sky-model, strategy, or scientific documentation logic.
3. Read `PLAN.md` before changing architecture boundaries, benchmarking,
   runtime bootstrap, preflight behavior, task granularity, or scalability code.
4. Pick the smallest guide below that matches the task.
5. Confirm the behavior in code and tests before editing.

## Guides

- `scientific_glossary.md`: radio interferometry, self-calibration vocabulary,
  Rapthor concepts, and science-facing configuration recommendations.
- `repo_architecture.md`: where behavior belongs across `rapthor/lib`,
  `rapthor/operations`, `rapthor/execution`, settings, docs, examples, and tests.
- `pipeline_contracts.md`: contracts that keep Prefect/Dask payloads,
  operation outputs, h5parm products, command builders, and restart behavior
  stable.
- `parset_strategy_guide.md`: how parset options and strategy values flow
  through defaults, docs, payloads, command builders, examples, and tests.
- `external_tools.md`: what Rapthor uses DP3, WSClean, EveryBeam, IDG/IDGCal,
  PyBDSF, Casacore, containers, Dask, and Prefect for.
- `testing_playbook.md`: focused test commands and confidence checks by change
  type.

## Working Style

- Prefer the repo's existing owner-package boundaries over new abstractions.
- Keep edits scoped to the requested behavior and the contracts it touches.
- Use focused tests first, then broaden only when the changed behavior crosses
  module, payload, or scientific product boundaries.
- Treat generated run products, Measurement Sets, FITS files, and integration
  outputs as evidence, not source files to casually edit.
- Keep this directory concise. If a guide starts becoming full user
  documentation, move that material to `docs/source/` and leave a pointer here.
