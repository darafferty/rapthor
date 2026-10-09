# Rapthor Pre-Merge Review

Reviewed 2026-10-09 · migration branch `gec-468-ai-migrate-to-prefect` at
`2271eb89` against master at `ff1f29d9`

Merge as planned, but resolve or rule out three defects that can silently
change scientific products (E01, R01, R03) before production runs that use
those paths, and fix R04–R06 before production-scale or multi-node use. The
review found 15 new defects: 10 on the migration branch only and 5 that also
exist on master. All 21 items already in [PLAN.md](PLAN.md) still apply. Every
finding below is recorded as a task in PLAN.md.

## Findings at a glance

Sorted by severity. "Both" means the defect also exists on master and is not a
migration regression.

| ID | Finding | Branches | Severity | Area |
| --- | --- | --- | --- | --- |
| E01 | Combined phases applied twice when DD solutions are pre-applied for imaging (`dde_method = single`) | Both | High (that mode) | Imaging |
| R01 | Image-only cycles after calibration: predict drops the carried-forward solutions | Migration only | High | Predict, imaging |
| R03 | DI and normalization corrections may be missing from DD solves | Migration only | High if confirmed | Calibration |
| R04 | Every plot is re-published to Prefect after every operation | Migration only | High at scale | Runtime |
| R05 | Remote Dask workers do not report to the run's Prefect API; MPI under Slurm untested | Migration only | High for clusters | Runtime |
| E02 | Phase-only DD cycles discard the medium-phase solutions | Both | Medium–high (decision) | Calibration |
| R02 | An omitted `calibration_strategy` carries over from the previous cycle | Migration only | Medium | Strategy |
| R06 | A Slurm configuration without a scheduler starts 12 local workers | Migration only | Medium | Runtime |
| R07 | DP3/WSClean child processes survive cancellation | Migration only | Medium | Runtime |
| E03 | Built-in selfcal final step is the same object as the last selfcal step | Both | Low–medium | Strategy |
| R08 | `rapthor -r` does not reset operations staged on global scratch | Migration only | Low–medium | Reset |
| R09 | New model-mosaic default changes the product and fails on empty sky models | Migration only | Low–medium | Mosaic |
| E04 | `rapthor -r` cannot reset repeated final cycles | Both | Low | Reset |
| E05 | WSClean-dependent tests are not marked as integration tests | Both | Low | Tests |
| R10 | Astrometry offsets JSON is no longer kept | Migration only | Low | Products |

## Before production use

1. **Before any production run on the merged branch:** decide E02, and fix or
   rule out E01 (if `dde_method = single` is used), R01 (image-only cycles after
   calibration) and R03 (flux-scale normalization, WSClean prediction).
2. **Before production-size runs:** fix R04.
3. **Before multi-node use:** fix R05, R06 and R07.
4. **Then** the scientific ports P01, P03, P04 and P06, the build and CI items
   P09–P12, and the remaining items in PLAN.md.

## Defects on both branches

These exist on master today, so they need fixing on merged master whether or
not the migration lands.

**E01 — Combined phases applied twice with `dde_method = single`.** When DD
solutions are pre-applied during imaging preparation, the applycal step list
gains a `mediumphase` step for DD `medium_phase` solves, for example
`[fastphase,mediumphase,slowgain]`. Neither the branch's DP3 command nor master's
`prepare_imaging_data.cwl` defines `applycal.mediumphase.*`, so DP3 falls back to
`applycal.parmdb` and `applycal.correction=phase000`. The combined `phase000`
table is then applied twice. Master commit `0ebc0690` introduced this, and the
migration ported it: `build_image_applycal_steps()` in
`rapthor/operations/image/plan.py`. The branch's unit tests assert the defective
list. **Fix:** apply each soltab once (drop `mediumphase` when a combined product
is selected, or give it its own h5parm and soltab), then compare a
`dde_method = single` run with facet-based application.

**E02 — Phase-only DD cycles discard the medium-phase solutions.** In a DD
`["fast_phase", "medium_phase"]` cycle, which the built-in strategy uses for
its early cycles with supplied or downloaded sky models, DP3 solves both, but the
finalizer copies only `fast_phases.h5parm` to `field-solutions.h5`. The branch
flow already builds the combined product and DI cycles use theirs, but the DD
finalizer ignores it. The docs say `field-solutions.h5` holds all solves
combined. This dates from master commit `b7e58780`. **Fix:** decide whether
these cycles should apply fast plus medium phases or skip the medium solve, then
implement it consistently.

**E03 — Built-in selfcal final step aliases the last selfcal step.**
`set_selfcal_strategy()` appends the last selfcal dict itself as the final step,
then sets `channel_width_hz = 4e6` on it. The last selfcal cycle therefore also
images with 4 MHz channels, and later in-place changes leak between the two.
When selfcal reaches its last cycle with equal data fractions and no QUV imaging,
no final pass runs. **Fix:** append a deep copy.

**E04 — Repeated final cycles cannot be reset.** `modifystate.run()` only lists
operations up to `len(strategy_steps)`, so cycles repeated with
`ntimes_to_repeat_final_cycle > 0` beyond that are not offered.

**E05 — Unmarked tool-dependent tests.** `test_integration_restore_skymodel`
and `test_integration_restore_skymodel_compressed` run WSClean but lack the
`integration` marker, so the non-integration commands in TESTING.md fail on a
machine without WSClean. These were the only two failures in this review's
unit-test run.

## Defects on the migration branch only

**R01 — Image-only cycles after calibration.** Master's `Field.update()` reuses
the existing patch layout in an image-only cycle that follows calibration
(`reuse_solution_layout`, from `0ebc0690`); the branch always regroups. Imaging
compensates by building facets from the previous cycle's sky model, but
`Predict._get_applycal_h5parm_filename()` rejects a DD h5parm from an earlier
cycle, or one missing any patch, and then predicts with no solutions, logging
only a warning. Outlier, bright-source and other-sector models are then
subtracted uncorrupted while imaging applies the carried solutions. Predict runs
whenever there are outlier sectors, several sectors, bright-source peeling or
reweighting. With two consecutive image-only cycles, imaging also takes the
previous cycle's sky model although the h5parm is older. **Fix:** one
carry-forward rule for predict and imaging, with a clear error on a direction
mismatch.

**R03 — DI and normalization pre-application in DD solves.** With
`use_wsclean_predict`, the branch omits the leading `applycal` DP3 step (a unit
test asserts it), so DI solutions and normalization are not applied before DD
solves; master keeps the step. Master also passes the normalization to the
slow-gain solve as `solve3.applycal.steps` and
`solve3.applycal.normalization.parmdb`; the branch computes
`calibration_applycal_steps` but never emits them. **Fix:** establish from DP3's
behaviour where these corrections must be applied for each prediction mode, and
check that slow-gain amplitudes do not re-absorb the normalization.

**R04 — Unbounded Prefect artifact publication.** After every operation,
`_run_operation()` re-reads every file under `plots/` (all cycles), embeds each
as a base64 data URL and creates a new artifact version. Calls and database size
grow with operations × plots, which reaches thousands of PNGs in runs with tens
of directions. Publication errors are not caught, so a Prefect API failure aborts
a run whose operation succeeded, and plot artifacts cannot be disabled.
**Fix:** publish only new products, link rather than embed large files, add an
off switch and make publication best-effort.

**R05 — Multi-node runtime unverified.** The Slurm job script in `running.rst`
starts `dask worker` without `PREFECT_API_URL` or `PREFECT_HOME`, while the
default temporary Prefect API exists only in the launching process. A local
probe (Prefect 3.7.7, prefect-dask 0.3.7) emulating that setup showed the worker
starting its own temporary Prefect server, event-emission errors, and its task
run missing from the launcher's API. MPI imaging runs `mpirun` from inside a
Dask worker that already occupies an `srun` step. The Slurm integration test only
checks scheduler connectivity. **Fix:** a real multi-node run, then require or
propagate the Prefect API for external Dask and document MPI constraints.

**R06 — Slurm configuration silently runs locally.** With
`batch_system = slurm` and no scheduler, Rapthor starts a local Dask cluster with
`max_nodes` workers (12 by default), each using all cores; the Slurm preflight
check passes because it sees the rewritten configuration. The docs also say
`local_dask_workers = 0` means one worker, but the code uses `max_nodes`.

**R07 — Orphaned child processes.** The shell runner kills only the `bash` or
GNU `time` wrapper on interruption, so DP3, WSClean or `mpirun` keep running and
writing after Ctrl-C, cancellation or worker shutdown. Full command output is
also kept in memory. **Fix:** run commands in their own process group and
terminate the group.

**R02 — Omitted `calibration_strategy`.** Master resets it per cycle and
resolves it from the legacy flags (fast and medium phase by default). The branch
carries the previous cycle's value forward, and a first-cycle omission gets the
full four-solve default, adding slow gains compared with master. The docs say an
omission always means the full default, and the warning says "default value of
None".

**R08 — Reset with global scratch.** After an interrupted operation,
`pipelines/<operation>` is a symlink to scratch. `modifystate.run()` uses
`shutil.rmtree(..., ignore_errors=True)`, which silently fails on a symlink, so
the next run resumes from the staged workspace.

**R09 — Model-mosaic default.** `model_mosaic_method = wsclean` renders
multi-sector model mosaics from the filtered sector sky models, so the mosaic
`model-pb` image contains only components kept by source filtering, and the
operation fails when every sector sky model is empty. The upgrade guide does not
mention it.

**R10 — Astrometry offsets product.** Master copies each sector's
`astrometry_offsets.json` to `plots/image_N`; the branch deletes it with the
operation's temporary outputs.

## Plan items re-verified

Every item already in PLAN.md is still open on `2271eb89`. D04 is new.

| Item | Evidence on the migration branch |
| --- | --- |
| P01 | No `wsclean_predict_beam_interval`; DP3 calibration still sets `beammode=array_factor`; prediction has no facet beam |
| P02 | WSClean prediction runs per facet and frequency group with `-no-reorder`; no reordered-data reuse |
| P03 | `Field` raises when observations have different station lists |
| P04 | Supplied survey catalogs get correction 1.0 and error 0.0; `normalization_reference_frequencies` missing from `defaults.json` |
| P05 | No catalog-prefetch helper or workflow |
| P06 | `bda_frequencybase = 20000.0` in both sections (master 5000.0); `defaults.parset` comment says 0 |
| P07 | File log handler still wraps emission in ANSI colouring; console handler added beside existing handlers |
| P08 | `ShellCommandError` carries command and return code only, no log path or excerpt |
| P09 | `Docker/Dockerfile` and `ci/ubuntu_24_04-base` still rewrite `/etc/apt/sources.list` |
| P10 | Standalone `Dockerfile` installs `numpy<2`; CI image uses NumPy 2 |
| P11 | `requires-python >=3.9`, yet `zip(..., strict=True)` in image diagnostics needs 3.10 |
| P12 | Single `.gitlab-ci.yml`; tox uses `-n auto`; no Astron/SKA configurations |
| P13 | No `Docker/extract_version_hashes.sh` |
| P14 | No end-to-end assertion on `facets_rms` (master's `test_sector_diagnostics.py` absent) |
| P15 | No parallel-gridding integration matrix (master's `test_wsclean_parallel_gridding.py` absent) |
| P16 | `ms_for_normalisation` fixture does not reset `FLAG` or `WEIGHT` |
| P17 | `preparation.rst` still says DI calibration must be in `DATA`; `Sector` docstring lacks the literal block |
| D01 | Strategy validation accepts only 6 DD and 5 DI solve sequences |
| D02 | One region helper, `make_ds9_region_from_skymodel()`, serves all formats |
| D03 | `fetch_commit_hashes.sh` resolves SAGECal `HEAD` instead of master's pin |
| D04 (new) | `defaults_skalow.parset` deleted on the branch; master still ships it, though it names a missing strategy file |
| L01 | Shared-facet test's enabled case is `xfail(run=False)` |

## Master defects already fixed by the migration

Keep these fixes when porting master code:

- Master's multi-sector `Image.finalize()` reads `sector_diagnostics[0]` for every
  sector, so all sectors report sector 1's diagnostics. The branch reads each
  sector's own file.
- After outlier or bright-source peeling, master's subtraction still reads
  `data_colname` from the peeled copy, which fails when it is not `DATA`. The
  branch reads the peeled output column (migration commit `4091e26d`).

## How the review was done

- **Revisions:** migration HEAD `2271eb89` against master `ff1f29d9` (merge base
  `2e21be62`, 64 master-only commits). Source review of the orchestration,
  operation adapters, execution owners, DP3/WSClean command builders against
  master's CWL steps, runtime bootstrap, defaults, parset documentation and
  container/CI files.
- **Tests run:** the non-integration, non-Prefect unit suite (excluding
  `test_field.py`) in a fresh Python 3.12 environment without DP3 or WSClean:
  1376 passed, 2 failed (E05).
- **Probe:** a local Dask scheduler and worker started without Prefect settings,
  for R05.
- **Not done:** the field, Prefect-server and integration suites; any DP3,
  WSClean or real-data run. E01, R03 and part of R05 rely on DP3 and Prefect
  behaviour inferred from source, so reproduce each before fixing it.
