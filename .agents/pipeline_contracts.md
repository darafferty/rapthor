# Pipeline Contracts

These contracts keep Rapthor reproducible across normal runs, restarts,
Prefect/Dask workers, external tools, and scientific comparisons.

The current roadmap in `PLAN.md` explicitly calls for Dask scalability
guardrails. When changing flow task boundaries or worker inputs, add tests that
make payload serialization, payload size, submitted task units, and useful task
names visible.

## Serializable Payloads

- Payloads sent to Prefect/Dask tasks must be plain serializable data.
- Do not send `Field`, `Observation`, `Sector`, `Operation`, open file handles,
  table objects, or live subprocess state to workers.
- Add payload-size or serialization guard tests when changing image,
  calibration, predict, mosaic, or concatenate task boundaries.
- Prefer shared execution payload validators for strings, basenames, lists, file
  records, and optional paths.
- Keep payload field names domain-specific enough to avoid ambiguity:
  `calibrator_patch_names`, `dd_h5parm_filename`, `modeldatacolumn`,
  `cycle_number`.

## Operation Finalizers

- Operation adapters gather domain state, call execution-owned code, and
  finalize outputs back into the `Field`.
- A new output product usually needs an output record, finalizer state, restart
  handling, and focused tests.
- Keep solution cycle numbers visible. Do not silently reuse solutions from an
  older cycle unless the strategy and operation contract explicitly say so.
- Preserve the distinction between DI, DD, full-Jones, normalization, and screen
  products.

## h5parm And Solution Products

- DD scalar/diagonal solutions, DI full-Jones solutions, normalization products,
  and future screen products are different scientific objects.
- h5parm direction names must match sky-model patch names or supplied
  `facet_layout` directions intentionally.
- Track whether a product is current-cycle, previous-cycle, DI, DD, full-Jones,
  or normalization. Names such as `dd_h5parm_cycle_number` and
  `fulljones_h5parm_cycle_number` are not decoration; they are safety rails.
- A change to solution discovery, filtering, combination, or naming needs
  restart and finalizer tests.

## Command Builders

- Command builders should be deterministic and easy to compare in tests.
- Use option dataclasses when they clarify a stable group of arguments, but do
  not hide scientific intent behind a generic abstraction.
- Mirror external tool argument names when that reduces translation mistakes:
  `msin`, `msin.datacolumn`, `solve1.mode`, `applycal.parmdb`.
- Add or update command reference tests when command tokens change.

## Restart And Idempotence

- Pipeline outputs often use done markers, output records, and discovered files
  to support restarts.
- A restarted run must not pick up stale h5parm, sky-model, FITS, or catalog
  products from the wrong cycle.
- If a command can be skipped because outputs already exist, verify the output
  record still contains every downstream field needed by later operations.
- Debugging docs and preflight output should name `.done` markers,
  `.outputs.json`, command records, Prefect run names, Dask dashboard views, and
  external-command stderr consistently.

## Task Boundary Visibility

- Flow tests should assert the intended task units when a boundary matters for
  scheduling, restart, or dashboard clarity.
- Task names should carry useful domain identifiers such as operation,
  calibration mode, cycle, sector, chunk, observation, image type, or epoch.
- Keep DP3, WSClean, IDG, and PyBDSF as coarse external commands unless a
  benchmark shows a clear benefit from finer integration.

## Scientific Product State

- Keep apparent-sky and true-sky products explicit in names and payloads.
- Keep image products, model products, residuals, dirty images, catalogs, masks,
  and diagnostics distinguishable.
- Do not collapse `apply_amplitudes`, `apply_fulljones`,
  `apply_normalizations`, `generate_screens`, and `apply_screens` into one
  generic apply flag.
- When changing products that feed later cycles, check photometry, astrometry,
  RMS, dynamic range, source count, unflagged fraction, restoring beam, and
  diagnostic artifacts.
