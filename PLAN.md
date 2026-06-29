# Rapthor Architecture Refactor Plan

Status snapshot: 2026-06-29.

## Goal

Make the Prefect/Dask Rapthor pipeline easy to understand, extend, test, debug,
and scale while preserving the user-facing CLI workflow:

```bash
rapthor input.parset
```

This Prefect/Dask implementation has not been released. We should therefore
prefer the clean production architecture over preserving unreleased Python APIs,
compatibility shims, migration aliases, or test-only surfaces.

## Current State

The core cleanup phase is largely complete.

Completed foundations:

- Operation execution is organized by owner package:
  - `rapthor.execution.image`
  - `rapthor.execution.calibrate`
  - `rapthor.execution.concatenate`
  - `rapthor.execution.predict`
  - `rapthor.execution.mosaic`
  - `rapthor.execution.pipeline`
- Image and calibration operation adapters are package-based:
  - `rapthor.operations.image.base`
  - `rapthor.operations.image.initial`
  - `rapthor.operations.image.normalize`
  - `rapthor.operations.image.plan`
  - `rapthor.operations.image.diagnostics`
  - `rapthor.operations.calibrate.base`
  - `rapthor.operations.calibrate.plan`
- Broad execution facades, migration shims, normalized command wrappers,
  test-only production APIs, and unused runtime abstractions have been removed.
- Command builders are deterministic and covered by focused command tests.
- Image and calibration command builders now use small dataclass option objects
  instead of very long argument lists where the grouping is stable.
- Payload contracts and validation live with the operation-specific execution
  packages.
- Scheduler-independent execution work is split from Prefect task wiring for
  the complex image and calibration flows.
- Current-cycle calibration solution handling is covered at both operation-input
  and DP3-command levels.
- Architecture boundary tests exist and should be extended as new boundaries
  settle.

Known follow-ups from completed work:

- Run a Sphinx docs build once the docs environment has `sphinx` installed.
- Avoid running multiple pytest processes in parallel unless each run has a
  separate `RAPTHOR_TEST_RUN_ROOT`.
- Prefect can emit late logging shutdown warnings after passing flow tests.
  Track separately if this becomes noisy in CI.

## Remaining Roadmap

Work through the remaining tasks in this order.

### 1. Script-To-Module Migration

Goal: make script logic importable and testable so future Dask workers can pass
small data objects in memory where that is sensible.

Status: complete for the current `rapthor/scripts/*.py` helper set. The
remaining work is wrapper retirement, not more script migration.

Tasks:

- Audit `rapthor/scripts/*.py` and record each helper's owner, inputs, outputs,
  external dependencies, data size, and current tests.
- Choose the first low-risk helper. Prefer one of:
  - `make_region_file.py`
  - `check_image_beam.py`
  - `blank_image.py`
  - a small mosaic helper
- Avoid starting with heavy helpers such as:
  - `calculate_image_diagnostics.py`
  - `normalize_flux_scale.py`
  - `subtract_sector_models.py`
  - `process_gains.py`
- Extract an importable function with a narrow typed signature.
- Keep the CLI wrapper stable.
- Add function-vs-CLI parity tests.
- Only switch production execution from shell/script to direct function calls
  when the payload is small, serializable, and safe for Dask workers.
- Once production execution uses the importable module path and parity is
  proven by tests, remove the original script wrapper instead of keeping an
  unused legacy entry point.

Progress:

- Started with `check_image_beam.py`: extracted the FITS header update logic to
  `rapthor.execution.image.beam.ensure_image_beam`, kept the script as the CLI
  wrapper, and added direct function plus CLI parity tests.
- Continued with `blank_image.py`: extracted template-mask creation to
  `rapthor.execution.image.masking.blank_image`, kept the script as the CLI
  wrapper, added direct function plus CLI parity tests, and removed the unused
  `region_file` option from the blank-mask command path.
- Continued with `make_region_file.py`: extracted DS9 facet-region creation to
  `rapthor.execution.regions.make_ds9_region_from_skymodel`, kept the script as
  the CLI wrapper, and replaced the placeholder script test with direct function
  plus CLI parity coverage.
- Continued with the mosaic helper group: extracted template creation,
  regridding, and mosaic averaging to `rapthor.execution.mosaic.images`, kept
  the three scripts as CLI wrappers, and added direct function plus CLI parity
  coverage for each script.
- Continued with the image-cube helper group: extracted FITS cube creation and
  PyBDSF catalog creation to `rapthor.execution.image.cubes`, kept both scripts
  as CLI wrappers, and added direct function plus CLI parity coverage.
- Continued with `concat_ms.py`: extracted Measurement Set concatenation and
  command construction to `rapthor.execution.concatenate.measurement_sets`,
  kept the script as the CLI wrapper, and kept direct plus CLI parity coverage.
- Continued with `adjust_h5parm_sources.py`: extracted h5parm source-table
  alignment to `rapthor.execution.calibrate.h5parm_sources`, kept the script as
  the CLI wrapper, and added direct function plus CLI parity coverage.
- Continued with `collect_screen_h5parms.py`: extracted screen-solution h5parm
  collection to `rapthor.execution.calibrate.screen_h5parms`, kept the script
  as the CLI wrapper, and added real h5py-backed direct plus CLI parity tests.
- Continued with `filter_skymodel.py` and `restore_skymodel.py`: extracted
  image skymodel filtering to `rapthor.execution.image.skymodel_filter` and
  skymodel restoration to `rapthor.execution.image.restoration`, kept both
  scripts as CLI wrappers, and added direct function plus CLI parity coverage.
- Continued with `combine_h5parms.py`: extracted h5parm combination helpers to
  `rapthor.execution.calibrate.h5parm_combination`, kept the script as the CLI
  wrapper, and moved unit coverage to the execution helper with CLI argument
  parity coverage.
- Continued with `process_gains.py`: extracted gain flagging, smoothing, and
  normalization to `rapthor.execution.calibrate.gain_processing`, kept the
  script as the CLI wrapper, and moved unit coverage to the execution helper
  with CLI argument parity coverage.
- Continued with `add_sector_models.py`: extracted sector model summing to
  `rapthor.execution.predict.sector_model_addition`, kept the script as the CLI
  wrapper, and added direct function plus CLI argument parity coverage.
- Continued with `subtract_sector_models.py`: extracted sector model
  subtraction and covariance-weight helpers to
  `rapthor.execution.predict.sector_model_subtraction`, kept the script as the
  CLI wrapper, and added direct function plus CLI argument parity coverage.
- Continued with `normalize_flux_scale.py`: extracted flux-scale normalization
  helpers to `rapthor.execution.image.flux_normalization`, kept the script as
  the CLI wrapper, and moved unit coverage to the execution helper with CLI
  parity coverage.
- Continued with `calculate_image_diagnostics.py`: extracted image diagnostic
  calculation helpers to `rapthor.execution.image.diagnostic_calculation`, kept
  the script as the CLI wrapper, left argument parsing in the wrapper, and moved
  unit coverage to the execution helper with CLI argument parity coverage.

Script audit and migration order:

- Done: `check_image_beam.py` -> image WSClean output; FITS header in-place;
  small payload; `astropy`; direct and CLI parity tests now exist.
- Done: `blank_image.py` -> image preparation; FITS template/input plus sector
  vertices to mask FITS; small-to-medium file payload; `lsmtool` and `astropy`;
  direct and CLI parity tests now exist.
- Done: `make_region_file.py` -> image and calibration prediction preparation;
  sky model plus facet bounds to DS9 region; small payload; `lsmtool`; direct
  and CLI parity tests now exist.
- Done: `make_mosaic_template.py`, `regrid_image.py`, and `make_mosaic.py` ->
  mosaic execution; FITS images plus vertices to template/regridded/mosaic FITS
  products; medium file payloads; direct and CLI parity tests now exist.
- Done: `make_image_cube.py` and `make_catalog_from_image_cube.py` -> image
  normalization; channel FITS files and cube metadata to cube/catalog products;
  medium file payloads; direct and CLI parity tests now exist.
- Done: `concat_ms.py` -> concatenate and image preparation; Measurement Set
  paths to copied or concatenated Measurement Set output; medium-to-large file
  payloads; direct and CLI parity tests now exist.
- Done: `adjust_h5parm_sources.py` -> calibration collection/prediction;
  skymodel patch positions plus h5parm source tables; medium file payloads;
  direct and CLI parity tests now exist.
- Done: `collect_screen_h5parms.py` -> calibration screen collection; screen
  h5parm files concatenated over time; medium file payloads; direct and CLI
  parity tests now exist.
- Done: `filter_skymodel.py` and `restore_skymodel.py` -> image filtered-model
  products and restored skymodel FITS products; skymodels, FITS files, and
  WSClean restore path; medium file payloads; direct and CLI parity tests now
  exist.
- Done: `combine_h5parms.py` -> calibration h5parm combination; solution
  h5parm files to merged h5parm products; medium file payloads; helper tests
  now import the execution module and CLI argument parity coverage exists.
- Done: `process_gains.py` -> calibration gain processing; h5parm amplitude
  and phase solution tables; medium file payloads; helper tests now import the
  execution module and CLI argument parity coverage exists.
- Done: `add_sector_models.py` -> predict model-data preparation; input
  Measurement Set plus sector model Measurement Sets to summed model-data
  output; large file payloads; direct function and CLI argument parity coverage
  now exist.
- Done: `subtract_sector_models.py` -> predict model-data subtraction; input
  Measurement Set plus sector model Measurement Sets to per-sector residual
  outputs and optional calibration weights; large file payloads; direct
  function and CLI argument parity coverage now exist.
- Done: `normalize_flux_scale.py` -> image flux-scale normalization; PyBDSF
  source catalog, Measurement Set metadata, reference catalogs/skymodels, and
  output normalization h5parm; medium FITS/catalog payloads; helper tests now
  import the execution module and CLI parity coverage exists.
- Done: `calculate_image_diagnostics.py` -> image diagnostics; FITS image/RMS
  products, PyBDSF source catalog, representative Measurement Set metadata,
  optional comparison skymodels/surveys, JSON diagnostics, and plot products;
  helper tests now import the execution module and CLI argument parity coverage
  exists.
- Script helper logic has been migrated to importable modules. Keep the thin
  script wrappers until production execution has switched to direct module calls
  where the payloads are safe for Dask workers, then remove wrappers with parity
  coverage in place.

Done when:

- The script audit is complete, each helper has an importable function, tests
  cover the importable module path, and wrapper deletion is tracked separately.

### 2. Switch Prefect Flows From Script Commands To Python Work Units

Goal: use the migrated helper modules directly from production Prefect tasks
instead of invoking Rapthor script wrappers as shell commands.

Status: complete for migrated helpers, except the intentionally retained
`filter_skymodel.py` subprocess fallback for daemonic Dask workers.

Keep shell execution for true external tools:

- DP3
- WSClean and WSClean-MP
- IDGCal Python DP3 steps
- `fpack`
- `plotrapthor`
- external h5parm collectors or other third-party executables not yet migrated

Use direct Python calls for migrated Rapthor helpers:

- Image preparation:
  - `blank_image.py` -> `rapthor.execution.image.masking.blank_image`
  - `make_region_file.py` -> `rapthor.execution.regions.make_ds9_region_from_skymodel`
  - `check_image_beam.py` -> `rapthor.execution.image.beam.ensure_image_beam`
- Image products:
  - `make_image_cube.py` -> `rapthor.execution.image.cubes.make_image_cube`
  - `make_catalog_from_image_cube.py` ->
    `rapthor.execution.image.cubes.make_catalog_from_image_cube`
  - `filter_skymodel.py` -> `rapthor.execution.image.skymodel_filter.filter_image_skymodel`
  - `restore_skymodel.py` -> `rapthor.execution.image.restoration.restore_skymodel`
  - `normalize_flux_scale.py` ->
    `rapthor.execution.image.flux_normalization.normalize_flux_scale`
  - `calculate_image_diagnostics.py` ->
    `rapthor.execution.image.diagnostic_calculation.calculate_image_diagnostics`
- Mosaic:
  - `make_mosaic_template.py` -> `rapthor.execution.mosaic.images.make_mosaic_template`
  - `regrid_image.py` -> `rapthor.execution.mosaic.images.regrid_image`
  - `make_mosaic.py` -> `rapthor.execution.mosaic.images.make_mosaic`
- Calibration:
  - `collect_screen_h5parms.py` ->
    `rapthor.execution.calibrate.screen_h5parms.collect_screen_h5parms`
  - `combine_h5parms.py` -> `rapthor.execution.calibrate.h5parm_combination.combine_h5parms`
  - `process_gains.py` ->
    `rapthor.execution.calibrate.gain_processing.process_gain_solutions`
  - `adjust_h5parm_sources.py` ->
    `rapthor.execution.calibrate.h5parm_sources.adjust_h5parm_source_coordinates`
  - calibration `make_region_file.py` calls -> shared region helper
- Predict post-processing:
  - `add_sector_models.py` -> `rapthor.execution.predict.sector_model_addition.add_sector_models`
  - `subtract_sector_models.py` ->
    `rapthor.execution.predict.sector_model_subtraction.subtract_sector_models`
- Concatenation:
  - Replace `concat_ms.py` shell calls with the migrated validation and command
    selection helpers, but keep DP3/TAQL/copy execution on `run_external_command`
    unless the underlying Measurement Set operation is pure Python.

Recommended order:

1. Add a tiny working-directory/logging helper only if the first slice shows
   repeated boilerplate. Direct Python helpers must run with the same effective
   working directory as the script wrapper did, and all relative output
   filenames must still resolve inside the pipeline working directory.
2. Convert the lowest-risk image helpers first: beam check, blank mask, and
   region file creation.
3. Convert mosaic helpers next because the task boundary is already one image
   type and the functions are pure Python file transformations. Keep mosaic
   compression as an `fpack` shell command.
4. Convert image product helpers: cube creation, PyBDSF catalog creation,
   skymodel filtering, filtered-model restoration, diagnostics, and flux
   normalization.
5. Convert calibration helper scripts in collection/prediction while keeping
   DP3, WSClean draw-model, `plotrapthor`, and any external h5parm collector as
   shell commands.
6. Convert predict add/subtract post-processing after the image/calibration
   pattern is stable, because these helpers touch large Measurement Sets and
   need careful Dask-worker filesystem assumptions.
7. Convert concatenate last, preserving existing external-command logging for
   DP3/TAQL/copy and using the migrated module only for validation, ordering,
   and command selection.
8. Remove now-unused `build_*_command` functions for migrated Rapthor scripts
   from production command modules once their flow tests no longer need them.
   Keep thin CLI wrappers until a later cleanup pass proves they are unused
   outside tests and production execution.

Progress:

- Done: low-risk image helpers now run as direct Python work units instead of
  shelling out to Rapthor script wrappers:
  - `blank_image.py` -> `rapthor.execution.image.masking.blank_image`
  - image `make_region_file.py` ->
    `rapthor.execution.regions.make_ds9_region_from_skymodel`
  - `check_image_beam.py` -> `rapthor.execution.image.beam.ensure_image_beam`
- Flow tests now spy on those direct helper calls, and the image shell fake only
  needs to model true external commands plus script wrappers that have not yet
  been converted.
- Done: mosaic helpers now run as direct Python work units while `fpack` remains
  shell-based:
  - `make_mosaic_template.py` -> `rapthor.execution.mosaic.images.make_mosaic_template`
  - `regrid_image.py` -> `rapthor.execution.mosaic.images.regrid_image`
  - `make_mosaic.py` -> `rapthor.execution.mosaic.images.make_mosaic`
- Mosaic flow tests now spy on direct helper calls, and the mosaic shell fake
  only accepts `fpack`.
- Done: image product helpers now run as direct Python work units from the
  image flow:
  - `make_image_cube.py` -> `rapthor.execution.image.cubes.make_image_cube`
  - `make_catalog_from_image_cube.py` ->
    `rapthor.execution.image.cubes.make_catalog_from_image_cube`
  - `filter_skymodel.py` -> `rapthor.execution.image.skymodel_filter.filter_image_skymodel`
  - `restore_skymodel.py` -> `rapthor.execution.image.restoration.restore_skymodel`
  - `normalize_flux_scale.py` ->
    `rapthor.execution.image.flux_normalization.normalize_flux_scale`
  - `calculate_image_diagnostics.py` ->
    `rapthor.execution.image.diagnostic_calculation.calculate_image_diagnostics`
- Image flow tests now spy on those direct helper calls, and the image shell
  fake only models DP3, TAQL, WSClean/WSClean-MP, bright-source WSClean
  restore, and `fpack`.
- Done: calibration collection and prediction helpers now run as direct Python
  work units while DP3, WSClean draw-model, `H5parm_collector.py`, and
  `plotrapthor` remain shell-based:
  - `collect_screen_h5parms.py` ->
    `rapthor.execution.calibrate.screen_h5parms.collect_screen_h5parms`
  - `combine_h5parms.py` ->
    `rapthor.execution.calibrate.h5parm_combination.combine_h5parms`
  - `process_gains.py` ->
    `rapthor.execution.calibrate.gain_processing.process_gain_solutions`
  - `adjust_h5parm_sources.py` ->
    `rapthor.execution.calibrate.h5parm_sources.adjust_h5parm_source_coordinates`
  - calibration `make_region_file.py` ->
    `rapthor.execution.regions.make_ds9_region_from_skymodel`
- Calibration flow tests now patch those direct helper calls, and the
  calibration shell fake only models true external commands plus the external
  h5parm collector and plotting executable.
- Done: predict add/subtract post-processing now runs as direct Python work
  units while DP3 prediction remains shell-based:
  - `add_sector_models.py` ->
    `rapthor.execution.predict.sector_model_addition.add_sector_models`
  - `subtract_sector_models.py` ->
    `rapthor.execution.predict.sector_model_subtraction.subtract_sector_models`
- Predict post-processing helpers now accept an explicit output directory so
  Prefect/Dask tasks do not need to change the process working directory.
- The now-unused predict `build_add_sector_models_command` and
  `build_subtract_sector_models_command` wrapper builders were removed from
  production code instead of leaving test-only compatibility surfaces.
- Predict flow tests now patch those direct helper calls, and the predict shell
  fake only models DP3.
- Done: concatenate validation and command selection now use the migrated
  Measurement Set helper directly:
  - production flow no longer shells out to `concat_ms.py`
  - frequency concatenation still runs DP3 as an external command
  - time concatenation still runs TAQL as an external command
  - single-input epochs still run copy as an external command
  - the unused concatenate wrapper-command module was removed
- Note: image source filtering remains a special Dask-worker case. The helper
  is importable and still runs directly in normal Python processes, but
  daemonic Dask worker processes run `filter_skymodel.py` as a subprocess
  because lsmtool/PyBDSF creates Python multiprocessing children internally.
- Done: removed migrated Rapthor script command builders from production command
  modules where the owning flow has already switched to direct Python helpers:
  - mosaic keeps only the `fpack` compression command builder
  - calibration keeps only DP3, WSClean draw-model, `H5parm_collector.py`, and
    `plotrapthor` command builders
  - image keeps DP3, WSClean/WSClean-MP, `fpack`, and the daemonic-worker
    `filter_skymodel.py` fallback command builders
- Done: image preparation now uses the shared Measurement Set concatenation
  helper directly. Imaging time-concatenation runs the selected external `taql`
  command instead of shelling out through `concat_ms.py`.
- Done: command reference fixtures were pruned so they no longer contain stale
  entries for migrated Rapthor script wrapper builders.
- Next: keep the `filter_skymodel.py` fallback until PyBDSF/lsmtool
  multiprocessing can be isolated without breaking Dask worker execution.

Testing tasks:

- Update flow tests so fake shell operations only see true external commands.
- Patch or spy on direct helper functions in flow tests to prove the Prefect
  tasks call Python helpers with the expected arguments.
- Add assertions that migrated Rapthor script names no longer appear in the
  production shell-command list.
- Keep script-level direct-function and CLI parity tests until the wrapper
  scripts are intentionally removed.
- Update restart/failure integration tests that currently inject failing wrapper
  scripts on `PATH`; direct function calls need failure fixtures that exercise
  the imported helper path instead.
- Run focused execution tests after each owner package:
  - `tests/execution/test_image_flow.py`
  - `tests/execution/test_mosaic_flow.py`
  - `tests/execution/test_calibrate_flow.py`
  - `tests/execution/test_predict_flow.py`
  - `tests/execution/test_concatenate_flow.py`

Done when:

- Production Prefect flows no longer invoke migrated Rapthor helper scripts as
  shell commands, except documented compatibility fallbacks.
- Shell command logs contain only real external tools.
- Direct helper calls preserve existing output records, skip/restart behavior,
  and artifact publication.
- Dask worker payloads remain plain serializable file/path metadata rather than
  in-memory FITS tables, Measurement Sets, h5parm handles, or domain objects.

### 3. Retire Legacy Script Wrappers

Goal: remove unused CLI wrapper files and test-only compatibility surfaces now
that production execution uses importable modules.

Keep for now:

- `filter_skymodel.py`, because image execution still uses it as a subprocess
  fallback inside daemonic Dask workers.
- `bin/plotrapthor`, because calibration still treats `plotrapthor` as an
  external plotting executable.
- `mpi_runner.sh` until a targeted packaging/runtime audit proves it is unused.

Tasks:

- Add a deletion-readiness architecture test that fails when production code
  imports `rapthor.scripts` or invokes migrated Rapthor helper script names.
  Allow only documented exceptions such as `filter_skymodel.py`.
- Update `bin/concat_linc_files` to import
  `rapthor.execution.concatenate.measurement_sets.concat_ms` directly, then
  `concat_ms.py` can be retired with the other wrappers.
- Remove retired wrapper paths from `pyproject.toml` `script-files` and from
  `rapthor` package data.
- Remove wrapper files from `rapthor/scripts/` in small batches by owner:
  - concatenate: `concat_ms.py`
  - image preparation: `blank_image.py`, `check_image_beam.py`,
    `make_region_file.py`
  - mosaic: `make_mosaic_template.py`, `regrid_image.py`, `make_mosaic.py`
  - image products: `make_image_cube.py`, `make_catalog_from_image_cube.py`,
    `restore_skymodel.py`, `normalize_flux_scale.py`,
    `calculate_image_diagnostics.py`
  - calibration: `adjust_h5parm_sources.py`, `collect_screen_h5parms.py`,
    `combine_h5parms.py`, `process_gains.py`
  - predict: `add_sector_models.py`, `subtract_sector_models.py`
- Remove or rewrite `tests/scripts/test_*` files that only exercise deleted
  wrappers. Keep direct helper tests under `tests/execution` or move any
  missing direct-helper assertions there before deleting wrapper parity tests.
- Update `docs/source/development/architecture.rst` so `rapthor.scripts` is no
  longer described as a production pipeline layer. Document the remaining
  `filter_skymodel.py` Dask-worker exception.
- Revisit restart/failure integration tests that use PATH-injected wrapper
  scripts. Convert those failures to patch the importable helper path or keep
  them explicitly tied to the remaining `filter_skymodel.py` fallback.

Recommended order:

1. Add the deletion-readiness architecture test and fix `bin/concat_linc_files`.
2. Remove low-risk wrappers first: image preparation and mosaic wrappers.
3. Remove concatenate, calibration, predict, and image-product wrappers once
   their direct-helper tests are confirmed to cover the same behaviour.
4. Leave `filter_skymodel.py` until the PyBDSF/lsmtool multiprocessing issue is
   isolated or the Dask worker strategy changes.

Done when:

- `rapthor/scripts/` contains only documented live entry points.
- Package metadata no longer installs retired wrappers.
- Production code and command fixtures contain no retired script names.
- Tests exercise importable helper modules rather than deleted CLI wrappers.

### 4. Dask Scalability Contracts

Goal: prove the pipeline can scale across multiple workers/nodes without
passing domain objects, huge nested state, or local-only paths by accident.

Tasks:

- Add payload-size and serialization guard tests for:
  - image sectors
  - calibration chunks
  - predict model tasks
  - mosaic image types
  - concatenate epochs
- Add tests that assert each flow submits the intended task units.
- Extend resource-request coverage beyond image WSClean MPI paths.
- Check that all worker payloads are plain serializable data, not `Field`,
  `Observation`, `Sector`, or operation instances.
- Document which steps are currently distributed by Dask and which still run as
  coarse external commands.

Done when:

- A developer can see, from tests and docs, what the Dask task boundaries are
  and what data each boundary receives.

### 5. Runtime Preflight And Dry-Run

Goal: make failures obvious before expensive external tools start.

Tasks:

- Expand dry-run output to show:
  - planned operation order
  - task groups
  - resource hints
  - expected outputs
  - external command/script adapters
  - unsupported multi-node features
- Improve preflight messages for:
  - missing external tools
  - unsupported container configuration
  - Slurm/external-Dask mismatch
  - missing Dask scheduler
  - MPI WSClean assumptions
- Keep dry-run and preflight code independent of Prefect task objects where
  possible.

Done when:

- A user can run a preflight/dry-run path and understand likely runtime failures
  without reading flow code.

### 6. Operation And Runner Simplification

Goal: keep code quantity controlled as new module boundaries settle.

Tasks:

- Review these modules for remaining large or mixed-responsibility functions:
  - `rapthor.operations.image.base`
  - `rapthor.operations.calibrate.base`
  - `rapthor.execution.image.sector`
  - `rapthor.execution.calibrate.collection`
  - `rapthor.execution.shell`
- Extract only when the new helper has a clear scientific/workflow name or
  removes meaningful duplication.
- Do not create empty pass-through modules.
- Keep finalizer side effects easy to audit.

Done when:

- The remaining large methods are either split into named work units or kept
  intentionally with a clear reason.

### 7. Contributor Documentation

Goal: make the pipeline pleasant for developers and scientists to improve.

Tasks:

- Add short docs/checklists for:
  - adding a parset option
  - modifying an operation
  - adding an external command helper
  - converting a script to an importable module
  - adding a new flow task boundary
- Include fast test lanes for each change type.
- Point contributors to the owner module for payloads, commands, outputs,
  operation adapters, and Prefect flow wiring.

Done when:

- A new contributor can identify the right module and test file for a typical
  change without reading the whole pipeline.

### 8. Target Environment Validation

Goal: validate the architecture in the environments it is meant to run in.

Tasks:

- Define opt-in validation runs for:
  - external Dask
  - Slurm-launched Dask
  - MPI WSClean
  - `prefect_command_profile = perf`
- Keep these separate from the default non-integration suite.
- Document required system tools, scheduler assumptions, and expected runtime
  artifacts.

Done when:

- CI or developer docs clearly separate fast local tests from target-environment
  validation.

## Testing Strategy

Use the narrowest test lane that matches the change, then run a broader lane
before handing off larger slices.

Fast lanes:

- Pure planning helpers: `tests/operations`
- Payload contracts: `tests/execution/test_payloads.py` and the relevant
  operation flow test
- Command builders: `tests/execution/test_commands.py`,
  `tests/execution/test_reference_fixtures.py`, or operation-specific command
  tests
- Prefect wiring: relevant `tests/execution/test_*_flow.py`
- Architecture boundaries: `tests/architecture`
- Script conversions: matching `tests/scripts` file plus CLI/function parity
  tests

Broader lanes:

- Operation adapters:
  - `tests/operations/test_image.py`
  - `tests/operations/test_calibrate.py`
  - corresponding predict/mosaic/concatenate tests
- Execution flows:
  - `tests/execution/test_image_flow.py`
  - `tests/execution/test_calibrate_flow.py`
  - smaller flow tests for predict, mosaic, and concatenate
- End-to-end behavior:
  - selected `tests/integration` files only when required tools are available

Use the dev container for tests and formatting.

## Architecture Rules

- Preserve the CLI/parset user workflow.
- Do not preserve unreleased Python APIs, aliases, or compatibility shims unless
  they protect the CLI or a documented scientific contract.
- Keep `rapthor.lib` independent of Prefect, Dask, Slurm, shell execution, and
  artifact publication.
- Keep Prefect flows thin: task wiring, scheduling, runtime integration, and
  aggregation belong there.
- Keep operation adapters focused on lifecycle, reading live `Field` state,
  payload construction, flow invocation, and finalizer side effects.
- Keep command builders pure and deterministic.
- Keep Dask worker payloads plain and serializable.
- Use operation-owned packages for non-trivial execution code:
  - `commands.py`
  - `payloads.py`
  - `outputs.py` when useful
  - focused work-unit modules
  - `flow.py`
- Use operation-owned packages for non-trivial operation code:
  - `base.py`
  - `plan.py`
  - variant modules such as `initial.py` or `normalize.py`
  - diagnostics/finalizer helpers only when they have real ownership
- Prefer stdlib dataclasses for stable internal command argument groups.
- Keep Pydantic in mind as a future option for parset, payload, and
  script-module boundary validation, but do not introduce it until a boundary
  clearly benefits from runtime validation and error reporting.

## Definition Of Done For Future Refactor Slices

Each slice should leave the codebase better than it found it:

- Behavior is preserved unless the change is explicitly scoped as a feature
  change.
- Owner modules are clear.
- No new broad facades or compatibility shims are added.
- Tests cover the moved behavior at the lowest useful level.
- Docs or contributor notes are updated when ownership or workflow changes.
- Code quantity is justified by readability, testability, or scalability.
