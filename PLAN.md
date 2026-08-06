# Next Feature: DP3 Calibration Memory Checks

Add advisory memory checks for DP3 calibration so users can identify likely
out-of-memory configurations before processing starts and again when each
cycle's actual calibration facet count is known.

The calculation should match the current-DP3 model in
`../ska-sdp-ical/notebooks/dp3_calibrate_memory.py`, but the implementation and
tests in this repository must remain generic to Rapthor. Do not copy
SKA-specific scenarios or add SKA-specific test data here.

## Intended Behaviour

- Run a pre-flight check after the processing strategy has been loaded and
  validated. At this point, use each calibration cycle's `max_directions` as a
  conservative direction count because the actual facets are not known.
- Run a second check immediately after `Field.update()` for every calibration
  cycle. At this point, use the actual number of calibration patches in
  `field.num_patches` and the resolved observation solve parameters.
- Evaluate each enabled DP3 solve for each observation and report the largest
  per-task estimate. Do not sum memory across observations because they are
  separate workflow tasks.
- Compare the estimate with `cluster_specific.mem_per_node_gb` when it is
  positive. Otherwise, use the memory available on the current machine,
  following Rapthor's existing `get_available_memory()` convention.
- Log a concise `INFO` result when the estimate fits and a `WARNING` containing
  "likely out of memory" when it exceeds the limit. Include the cycle, solve,
  observation, direction count, estimated peak, limit, and headroom or overage.
- Keep the check advisory. It must not stop processing or alter calibration
  settings.
- Skip checks cleanly for cycles without calibration or without a DP3 solve.

## Memory Model

Implement the calculation as pure domain logic using decimal gigabytes:

```text
time_steps = ceil(solution_interval_seconds / sampling_interval_seconds)
samples = baselines * channels * time_steps * (directions + 1)

visibility_copies_gb = samples * 4 * 8 / 1e9
weights_gb = samples * 4 * 4 / 1e9
weighted_data_gb = samples * 4 * 8 / 1e9
peak_memory_gb = visibility_copies_gb + weights_gb + weighted_data_gb
```

This is the current-DP3 peak of 80 bytes per sample. The legacy solve buffer is
not part of the peak. The extra direction represents the original data buffer.
Derive the baseline count from the observation station count, including
autocorrelations: `nstations * (nstations + 1) // 2`.

Use the observation's raw channel count as a conservative bound. Calibration BDA
is baseline-dependent, so there is no single reduced channel count that can be
substituted without potentially understating memory.

## Implementation Tasks

1. Add the reusable estimator in `rapthor/lib/cluster.py`.
   - Add a result type or clearly documented dictionary containing `time_steps`,
     each memory component, and `peak_memory_gb`.
   - Validate that channels, baselines, sampling interval, solution interval,
     and directions are positive.
   - Keep the helper independent of `Field`, `Observation`, logging, and the CWL
     runner so it can be tested directly.

2. Make calibration solve resolution reusable.
   - Move or expose the solve-selection logic currently owned by
     `Calibrate._requested_calibration_solves()` and its solve metadata maps.
   - Preserve explicit `calibration_strategy` mode and solve order.
   - Preserve legacy expansion from `do_slowgain_solve` and
     `do_fulljones_solve` when no explicit strategy is supplied.
   - Make `Calibrate` and the memory checker use the same source of truth so
     their enabled solves and interval keys cannot drift apart.

3. Add a memory-assessment helper in `rapthor/process.py` or a small dedicated
   library module if keeping it pure makes testing materially simpler.
   - Accept a field, cycle number, direction count, and whether observation
     parameters have already been resolved.
   - Map `fast_phase`, `medium_phase`, `slow_gains`, and `full_jones` to the
     corresponding timestep settings or resolved observation parameter keys.
   - Evaluate all enabled DI and DD solves for all observations.
   - Treat DI solves as one direction. Use the supplied maximum or actual patch
     count for DD solves.
   - Return the controlling estimate and its inputs separately from logging.

4. Add the pre-flight check to `process.run()`.
   - Run it after `set_strategy()` and `validate_strategy()` and before initial
     sky-model generation or calibration work.
   - Check every self-calibration and final calibration step, including repeated
     final-cycle configurations without producing duplicate identical messages.
   - Use strategy timestep values and each step's `max_directions` for DD solves.
   - Handle image-only and DI-only strategies without requiring
     `max_directions`.

5. Add the resolved check to `process.run_steps()`.
   - Run it after `field.update()` has established `field.num_patches` and before
     prediction or calibration operations start.
   - Resolve observation calibration parameters once before estimating memory;
     reuse them when `Calibrate.set_input_parameters()` runs rather than doing
     duplicate stateful work.
   - Use the actual rounded DP3 solution intervals from each observation.
   - Ensure image-only cycles that reuse an existing solution layout do not
     trigger a calibration-memory warning when no solve will run.

6. Add structured logging.
   - Put the summary and capacity comparison at `INFO` or `WARNING` level.
   - Put component values and calculation inputs at `DEBUG` level.
   - State whether the limit is configured per-node memory or memory observed on
     the current machine.
   - Make pre-flight messages explicitly identify `max_directions` as an upper
     bound and cycle messages identify the facet count as resolved.

7. Document the behaviour.
   - Add a short user-facing section to the appropriate documentation describing
     when checks run, what is estimated, why the result is advisory, and how
     `mem_per_node_gb=0` is interpreted.
   - Note the decimal-GB convention, the 80-byte current-DP3 model, and the
     conservative treatment of BDA.

## Test Tasks

1. Extend `tests/lib/test_cluster.py` with direct estimator tests.
   - Verify hand-calculated small synthetic cases and ceiling of fractional time
     steps.
   - Verify component totals and the 80-byte peak.
   - Verify invalid inputs raise clear `ValueError`s.
   - Do not import the ska-sdp-ical notebook or reproduce its named SKA
     configurations as test cases.

2. Extend `tests/test_process.py` with focused orchestration tests.
   - Use the existing Rapthor `field`, `parset`, and test Measurement Set fixtures
     where observation metadata is relevant.
   - Prefer small synthetic field/observation doubles when testing selection,
     comparison, and logging in isolation.
   - Verify pre-flight uses `max_directions` and resolved checks use
     `field.num_patches`.
   - Verify DI uses one direction and DD uses the maximum or resolved facets.
   - Verify the largest observation/solve estimate controls the result without
     summing independent tasks.
   - Verify configured-memory and mocked `get_available_memory()` paths.
   - Verify fitting, likely-OOM, no-calibration, and repeated-final-cycle cases.

3. Extend `tests/operations/test_calibrate.py` only as needed to prove that
   calibration and memory assessment share solve resolution and resolved
   observation parameters.

4. Keep all repository tests Rapthor-specific or generic.
   - Use `tests/resources/`, the standard Rapthor test MS, existing sky models,
     or minimal synthetic metadata.
   - Do not add AA2, AA*, SKA-Low, ska-sdp-ical notebook, or other SKA-specific
     fixtures and assertions. Those scenarios belong in ska-sdp-ical.
   - Do not require internet access or external data downloads for the new unit
     tests.

## Verification

Run focused checks first:

```bash
python3 -m pytest tests/lib/test_cluster.py tests/test_process.py
python3 -m pytest tests/operations/test_calibrate.py
tox -e lint
```

Then run the non-integration suite:

```bash
python3 -m pytest -m "not integration" tests
```

Run integration tests only if implementation changes calibration workflow
inputs or execution order; the memory calculation and logging alone should not
require new integration coverage.

## Completion Criteria

- Both checks run at the intended lifecycle points and produce actionable logs.
- The resolved cycle check uses actual facets and actual rounded DP3 intervals.
- Calibration and memory checks share solve-selection logic.
- No existing workflow behaviour is changed when an estimate exceeds memory.
- Focused and non-integration tests pass using only Rapthor or synthetic data.
- Documentation describes assumptions and limitations.
