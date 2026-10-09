Merge request !473 · GEC-468 · reviewed at 2271eb89

# Rapthor MR 473 Review

Review of [Migrate Rapthor from CWL/Toil to Prefect/Dask](https://git.astron.nl/RD/rapthor/-/merge_requests/473), focused on bugs and inconsistencies. Code style and coding standards were not reviewed.

**Scope.** Source review of the new `rapthor/execution/` layer, the operation adapters and the `rapthor/lib/` changes, compared with current `origin/master`. Items already listed in `PLAN.md` (P01–P17, D01–D03, L01) are left out.

**Not done.** No test suite or pipeline run. The only thing executed was a check of `fpack`'s behaviour when its output file already exists.

For pasting into the MR thread.

## High priority 5

Calibration · imaging

### DD slow-gain amplitudes are dropped when DD calibration runs before DI

- `field.apply_amplitudes` is set by `scan_h5parms()` from `field.h5parm_filename` alone.
- DI calibration then sets `h5parm_filename` to `di-solutions.h5`.
- With `{"dd": [fast, medium, slow, medium], "di": ["fast_phase"]}`, a supported and documented order, `apply_amplitudes` is False after DI finishes.
- Imaging then applies only `phase000` per facet, and DD predict leaves out the `slowgain` step.
- Master kept a separate `dd_apply_amplitudes` for this. The branch removed it and passes `apply_amplitudes` as the DD value.

[operations/calibrate/base.py:945](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/operations/calibrate/base.py#L945) [operations/image/base.py:301](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/operations/image/base.py#L301) [operations/image/base.py:613](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/operations/image/base.py#L613) [operations/predict.py:201](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/operations/predict.py#L201)

Strategy

### The default calibration strategy changed for user strategy files

- For a cycle with neither `calibration_strategy` nor the legacy flags, master runs fast + medium. The branch runs fast + medium + slow_gains + medium.
- Existing strategy files that relied on `do_slowgain_solve` defaulting to False now silently get amplitude solves.
- This contradicts the upgrade guide ("the format of the strategy file is unchanged") and isn't listed under the behaviour differences.

[lib/strategy.py:11](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/lib/strategy.py#L11) [lib/field.py:2082](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/lib/field.py#L2082)

Strategy

### `calibration_strategy` carries over between cycles

- Master resets the attribute when a step leaves it out. The branch dropped that reset, so a calibration cycle without the key reuses the previous cycle's sequence.
- `strategy.rst` says the default sequence is used in that case.
- Carrying it over may be deliberate for image-only cycles. For calibration cycles the code and the docs disagree.
- The preflight warning also prints "Using the default value of None".

[lib/field.py:1981 (Field.update)](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/lib/field.py#L1981)

Runtime · Slurm

### The Slurm guard never fires, and the local worker count follows `max_nodes`

- `bootstrapped_runtime` rewrites `local_dask` to `external_dask` (pointing at a local cluster) before `preflight_execution` runs, so the `slurm_requires_external_dask` check always passes.
- With `batch_system = slurm` and no scheduler configured, Rapthor starts a local cluster with `max_nodes` workers (12 by default for Slurm). Each worker runs DP3 or WSClean with all `max_threads`, on one node.
- `local_dask_worker_count` is `max(1, max_nodes)`, but the parset docs say "0 = one worker".
- `slurm_cluster_spec` never reads `os.environ` in production, so `SLURM_NNODES` is ignored.

[execution/runtime_bootstrap.py:217](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/execution/runtime_bootstrap.py#L217) [execution/config.py:146](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/execution/config.py#L146) [execution/slurm.py:37](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/execution/slurm.py#L37)

Restart

### Re-running after a failure can get stuck

Master ran each step in a fresh Toil directory. The branch re-runs inside the existing operation directory, and three steps don't handle leftovers:

- **fpack** exits with an error if the `.fz` file already exists (checked locally: return code 255). A re-run of an image or mosaic flow that failed after compression then fails every time.
- **Concatenate** calls `select_concatenation_command` with `overwrite=False`, so a partial output MS raises `FileExistsError`.
- **Imaging concat** skips concatenation whenever the output directory exists, so a partial copy left by an interruption is silently reused.

[execution/image/commands.py:173](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/execution/image/commands.py#L173) [execution/mosaic/commands.py:22](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/execution/mosaic/commands.py#L22) [execution/concatenate/measurement_sets.py:146](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/execution/concatenate/measurement_sets.py#L146) [execution/image/preparation.py:70](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/execution/image/preparation.py#L70)

## Medium priority 7

These need a decision or a check against DP3 behaviour.

Calibration

### `field-solutions.h5` holds only fast phases for DD `[fast_phase, medium_phase]`

- `_dd_active_solution` copies the collected fast h5parm.
- The flow meanwhile builds and source-adjusts a combined fast + medium product and returns it as `combined_solutions`, which the operation ignores.
- Older master code did the same, but `operations.rst` says the file holds all solves combined. This affects the default phase-only selfcal cycles.

[operations/calibrate/base.py:1013](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/operations/calibrate/base.py#L1013)

Calibration

### Master's per-solve `applycal` is gone

- Master passed `solve3.applycal.steps` and `solve3.applycal.normalization.parmdb` to the slow-gain solve.
- The branch computes `calibration_applycal_steps`, but nothing reads it.
- `SOLVE_SLOT_ARGUMENTS` has entries for those keys, but `_solve_slots_for_chunk` never fills them in.

[operations/calibrate/base.py:690](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/operations/calibrate/base.py#L690) [execution/calibrate/commands.py](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/execution/calibrate/commands.py)

Predict

### Normalization in predict now uses a separate `applycal` step

- Normalization moved from `predict.applycal` to a top-level `applycal` with `usemodeldata=True`, after `predict.operation=replace`.
- Please confirm that DP3 applies this to the predicted main buffer. The only tests check the command string.

[execution/predict/commands.py:54](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/execution/predict/commands.py#L54) (commit 5a6b8620)

Parset

### WSClean predict now switches off solve BDA

Master doesn't do this, and it isn't documented.

[lib/parset.py:336](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/lib/parset.py#L336)

Runtime

### `prefect_retries` has no effect

It is parsed, validated and documented, but no task is given `retries`.

[execution/config.py:84](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/execution/config.py#L84)

Monitoring

### Plot artifacts are re-published on every operation

- After every operation, every file under `plots/` is base64-encoded and sent to Prefect again, so the cost grows with every cycle.
- There is no option to turn it off.
- An API error isn't caught, so it can fail the run after the operation itself succeeded.

[execution/pipeline/flow.py:440](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/execution/pipeline/flow.py#L440) [execution/artifacts.py:286](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/execution/artifacts.py#L286)

Concurrency

### Concurrent plot tasks can pick up each other's files

Each plot task finds its plots with a before/after glob of `*.png`. When plot tasks for different solves run at the same time on several workers, their output records can include each other's plots.

[execution/calibrate/collection.py:250](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/execution/calibrate/collection.py#L250)

## Low priority 6

### Predict output globs match other time chunks

The patterns `{obs}*.sector_*` and `*_field` run in the shared directory, so with several time chunks of one MS each task's output records also include the other chunks' MSs.

[execution/predict/flow.py:243](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/execution/predict/flow.py#L243)

### `np.NaN` no longer exists in NumPy 2

Used in `sector_model_subtraction.py`. It only matters when `use_compression` is set, which no current caller does.

[execution/predict/sector_model_subtraction.py:223](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/execution/predict/sector_model_subtraction.py#L223)

### Python log messages from Dask workers aren't kept

Messages logged by Python helpers that run inside Dask workers never reach `logs/rapthor.log`. With the temporary Prefect server, the Prefect copy is deleted when the run ends.

### `logging.basicConfig` runs at import

Called at module level in the restoration module, so importing it changes the root logger's configuration.

[execution/image/restoration.py:15](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/execution/image/restoration.py#L15)

### Fallback resource metrics mix concurrent commands

When GNU `time` is missing, per-command metrics come from `RUSAGE_CHILDREN`, which includes other commands running on the same worker.

[execution/shell.py](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/execution/shell.py)

### Reset leaves logs behind

`rapthor -r` doesn't clear `logs/<op>/`, `commands.jsonl` or `tasks.jsonl`, so later runs append to stale records.

[modifystate.py](https://git.astron.nl/RD/rapthor/-/blob/2271eb89e7c06534df27f87609f8c4c1f5a09e3f/rapthor/modifystate.py)