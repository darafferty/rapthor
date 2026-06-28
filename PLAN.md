# Rapthor Architecture Refactor Plan

Status snapshot: 2026-06-28.

## Goal

Make the post-migration Python/Prefect/Dask codebase easier to understand,
extend, test, and operate without changing Rapthor's scientific behaviour or
public parset/strategy contract.

The target architecture should let a developer or scientist answer these
questions quickly:

- Where do I add or change a parset option?
- Where do I translate domain objects into an execution payload?
- Where do I build the external tool command?
- Where does Prefect/Dask scheduling happen?
- Where are operation outputs recorded and finalized?
- Which tests prove that a change preserved command lines, output records, and
  restart behaviour?
- How do I dry-run, debug, profile, and scale a run without reading every flow
  implementation first?

## Progress Tracker

Last updated: 2026-06-28.

Completed:

- Initial architecture guardrails:
  - added `docs/source/development/architecture.rst`
  - linked the development architecture page from `docs/source/index.rst`
  - added `tests/architecture/test_import_boundaries.py`
  - documented `rapthor.execution` and the previous flow package as broad
    migration-era compatibility facades
  - added `tests/architecture` to the explicit Ruff format/check target lists
- Output record consolidation:
  - moved finalizer-compatible record creation, validation, and path extraction
    into `rapthor.lib.records`
  - migrated internal imports away from `rapthor.execution.outputs` and removed
    that compatibility shim
  - moved output-record contract tests to `tests/lib/test_records.py`
  - pruned `rapthor.execution` facade exports for output-record helpers
  - replaced duplicated flow-local file/directory record path helpers in
    concatenate, mosaic, predict, image, and calibration flows
  - added focused tests for required and optional record path extraction
- Typed payload contract proving ground:
  - added `ConcatenatePayload` and `ConcatenateEpochPayload` in
    `rapthor.execution.payloads`
  - validated incoming concatenate payloads into the typed shape before command
    execution or Prefect task submission
  - added focused tests for missing and malformed epoch input filenames so
    structurally invalid but serializable payloads fail before reaching workers
- Small-flow typed payload extension:
  - added mosaic and predict payload contracts in `rapthor.execution.payloads`
  - validated mosaic image-type payloads and predict model/post-processing task
    payloads before command execution or Prefect task submission
  - added focused tests for malformed mosaic image lists and predict directions
    so invalid worker payloads fail before shell commands run
- Image typed payload extension:
  - added image, image-sector, prepare-task, and image-cube payload contracts in
    `rapthor.execution.payloads`
  - validated image payloads into typed sector/task shapes before sync or
    Prefect task execution
  - added a focused test for malformed image prepare-task payloads so invalid
    worker payloads fail before shell commands run
- Calibration typed payload extension:
  - added calibration, chunk, solve-slot, output, and image-predict payload
    contracts in `rapthor.execution.payloads`
  - validated calibration payload chunks and image-predict setup before sync or
    Prefect task execution
  - added a focused test for malformed calibration chunk payloads so invalid
    worker payloads fail before shell commands run
- Shared command utility proving ground:
  - added shared boolean, comma-join, and bracketed-list token helpers in
    `rapthor.execution.commands`
  - migrated mosaic and predict command builders away from duplicated
    flow-local token helpers
  - added shared helpers for optional prefixed tokens, separate CLI
    option/value tokens, boolean flags, expanded list-valued options, and DP3
    `key=value` assignments
  - migrated image and calibration command builders away from duplicated
    flow-local option-appending helpers
  - added focused tests for the shared token and option helpers while
    preserving existing command-token behaviour
- Image command-builder extraction:
  - added `rapthor.execution.image.commands` as the first operation-specific
    command module
  - moved image DP3, WSClean, MPI WSClean, helper-script, a-term config, and
    normalized fixture command builders out of `rapthor.execution.image.flow`
  - pruned broad facade re-exports for image command helpers after package
    consolidation
  - moved focused image command-builder tests to import directly from
    `rapthor.execution.image.commands`
- Image payload mapping extraction:
  - added `rapthor.execution.image.payloads` for image payload construction and
    incoming payload validation
  - moved image payload mapping and validation out of
    `rapthor.execution.image.flow`
  - moved image-specific `TypedDict` payload contracts out of the shared
    `rapthor.execution.payloads` module and into
    `rapthor.execution.image.payloads`
  - updated the image operation adapter and focused tests to import the payload
    builder from the new owner module
  - pruned broad facade re-exports for `image_payload_from_inputs` after package
    consolidation
- Image sector/output split and package consolidation:
  - added `rapthor.execution.image.outputs` for image-specific file and
    directory discovery contracts
  - moved non-Prefect image sector execution into
    `rapthor.execution.image.sector`
  - slimmed `rapthor.execution.image.flow` to Prefect task wiring, payload
    validation, sync execution entry points, and final result aggregation
  - consolidated image command, payload, output, and sector modules under the
    `rapthor.execution.image` package instead of keeping temporary flat modules
  - stopped re-exporting image command helpers, `image_payload_from_inputs`, and
    `run_image_sector` from the broad execution facades; internal code imports
    from owner modules
- Calibration command-builder extraction:
  - added `rapthor.execution.calibrate.commands` as the first calibration
    package module
  - moved DDECal, IDGCal, model drawing, region-file, h5parm collection,
    h5parm combination, gain processing, source adjustment, plotting, and
    normalized fixture command builders out of `rapthor.execution.calibrate.flow`
  - updated focused calibration command-builder tests to import directly from
    `rapthor.execution.calibrate.commands`
  - pruned broad facade re-exports for calibration command helpers; internal
    code imports command builders from the owner module
- Calibration payload mapping extraction:
  - added `rapthor.execution.calibrate.payloads` for calibration payload
    contracts, construction, and incoming payload validation
  - moved calibration-specific `TypedDict` payload contracts out of the shared
    `rapthor.execution.payloads` module and into
    `rapthor.execution.calibrate.payloads`
  - moved `calibrate_payload_from_inputs` and calibration payload validators out
    of `rapthor.execution.calibrate.flow`
  - updated the Calibrate operation adapter and focused tests to import the
    payload builder directly from the calibration package
  - pruned broad facade re-exports for `calibrate_payload_from_inputs` now that
    this branch has no released public API to preserve
- Calibration runner split:
  - added `rapthor.execution.calibrate.runner` for scheduler-independent
    calibration task bodies, image-predict preparation, plotting, collection,
    combination, source adjustment, and gain processing
  - kept simple required-file checks private to the runner until calibration has
    output-discovery contracts large enough to justify a dedicated module
  - slimmed `rapthor.execution.calibrate.flow` to Prefect task wrappers,
    payload validation, sync execution entry points, and orchestration
  - updated focused calibration tests to exercise plotting and collection
    helpers through the runner owner module
  - pruned broad facade re-exports for `run_calibrate_chunk`
- Calibrate solve-plan helper extraction:
  - added `rapthor.operations.calibrate_plan` for pure strategy-to-solve-slot
    planning
  - moved `CalibrationSolve`, solve output naming, requested-solve defaults, and
    solve-plan expansion out of `rapthor.operations.calibrate`
  - kept thin `Calibrate` wrapper methods so finalizer and existing tests can
    still ask the operation for its current solve plan
  - extended operation tests to exercise the pure solve-plan helper directly
- Calibrate DP3-step helper extraction:
  - moved calibration DP3 step-chain decisions for BDA, pre-application, and
    image-based predict into `rapthor.operations.calibrate_plan`
  - kept `Calibrate._build_dp3_steps()` as a thin wrapper over the pure helper
  - added direct helper coverage for BDA/slow-gain and image-based-predict step
    ordering
- Calibrate pre-apply helper extraction:
  - moved DD calibration pre-apply step selection into
    `rapthor.operations.calibrate_plan`
  - kept `Calibrate._build_applycal()` responsible for current-cycle file
    resolution and FileRecord conversion
  - added direct helper coverage for scalar, slow-gain, full-Jones, and
    normalization pre-apply ordering
- Calibrate solve-slot input helper extraction:
  - moved per-slot data-use, solutions-per-direction, smoothness scaling,
    antenna constraint, and optional reference-value mapping into
    `rapthor.operations.calibrate_plan`
  - kept `Calibrate` responsible for reading values from `Field`, resolving
    core-station constraints, and mutating operation inputs
  - reused the same helper for default DD slot inputs and explicit strategy
    solve-slot remapping
  - added direct helper coverage for scalar phase and slow-gain slot inputs
- Image applycal planning helper extraction:
  - added `rapthor.operations.image_plan` for pure prepare-data applycal step
    planning
  - moved calibration-strategy-to-applycal-step selection and scalar h5parm
    preference logic out of `Image._build_applycal_steps()`
  - kept `Image` responsible for current-cycle h5parm resolution, FileRecord
    conversion, and operation input mutation
  - added direct helper coverage for DD scalar preference and facet h5parm
    selection without pre-application
- Image prepare-data step helper extraction:
  - moved prepare-data DP3 step ordering for applycal, averaging, BDA, and
    screen compatibility into `rapthor.operations.image_plan`
  - kept `Image` responsible for deciding whether applycal steps exist,
    checking channel regularity, and formatting the final payload string
  - added direct helper coverage for averaging+BDA, pre-application, irregular
    channels, and screen mode
- Image WSClean control helper extraction:
  - moved polarization-link/join selection and clean-iteration disabling into
    `rapthor.operations.image_plan`
  - kept `Image` responsible for reading sector iteration counts and field
    flags before adding values to the payload
  - moved `is_only_pol_I` into the Image planning helper module so Stokes-I
    decisions have one owner
  - added direct helper coverage for Stokes-I, linked polarization, joined
    polarization, and disabled cleaning
- Image facet solution-control helper extraction:
  - moved facet `soltabs`, scalar-visibility, and diagonal-visibility selection
    into `rapthor.operations.image_plan`
  - kept `Image` responsible for gathering facet geometry, parallel gridding
    settings, and field solution flags before adding values to the payload
  - added direct helper coverage for phase-only scalar, diagonal amplitudes,
    scalarized amplitudes, and full-Stokes facet imaging
- Image screen-interval helper extraction:
  - moved the screen-mode imaging interval calculation into
    `rapthor.operations.image_plan`
  - kept `Image` responsible for reading the first observation and adding the
    computed interval to the flow payload
  - added direct helper coverage for exact, rounded-up, and minimum-one-sample
    interval cases, plus an adapter assertion that `Image` uses the helper rule
- Image MPI resource-control helper extraction:
  - moved per-sector MPI node and CPU payload selection into
    `rapthor.operations.image_plan`
  - kept `Image` responsible for reading sector count, batch system, and parset
    resource values before adding MPI controls to the flow payload
  - added direct helper coverage for static Slurm, launcher-reserved Slurm, and
    minimum-one-node allocation cases, plus an adapter assertion for MPI payload
    wiring
- Calibrate station-selection helper extraction:
  - moved core-station, superterp-station, and DP3 core-baseline selection rules
    into `rapthor.operations.calibrate_plan`
  - kept `Calibrate` wrapper methods as thin adapters over the pure helpers so
    existing operation tests and callers still exercise the same behaviour
  - added direct helper coverage beside the operation-wrapper assertions
- Operation adapter thinning checkpoint:
  - moved the decision-heavy Image and Calibrate planning clusters into
    `rapthor.operations.image_plan` and `rapthor.operations.calibrate_plan`
  - left operation classes responsible for lifecycle setup, reading live `Field`
    state, converting file records, invoking Prefect flows, and finalizer side
    effects
  - treated predict, mosaic, and concatenate as already thin enough for this
    stage; extract more only when a future change reveals duplication or a
    specific testability problem
- Pipeline dry-run plan foundation:
  - added `rapthor.execution.pipeline.plan` as a Prefect-free owner for
    pipeline-level feature detection and serializable operation-plan summaries
  - moved pipeline feature detection and supported-feature declarations out of
    `rapthor.execution.pipeline.flow`
  - added a dry-run/debug helper that reports expected operation order, cycle
    number, calibration mode, final/selfcal context, and selfcal-check markers
    without mutating a `Field` or starting Prefect/Dask work
  - added architecture coverage so the pipeline-plan helper stays free of
    Prefect, Dask, flow, and operation imports
- Pipeline facade cleanup:
  - removed `rapthor.process` as an internal/public Python compatibility facade;
    the user-facing CLI remains `rapthor input.parset`
  - changed `bin/rapthor` to call `rapthor.execution.pipeline.flow.pipeline_flow`
    directly
  - moved scheduler-independent lifecycle helpers for final-pass decisions,
    observation chunking, and diagnostics report writing into
    `rapthor.execution.pipeline.lifecycle`
  - removed legacy process-facade tests that patched operation globals on
    `rapthor.process`; pipeline-step orchestration is now tested through
    `rapthor.execution.pipeline.flow`
- Execution package facade cleanup:
  - removed broad package-level re-exports from `rapthor.execution` and
    the previous flow package
  - kept execution package initializers lightweight so importing one execution
    submodule does not import every Prefect flow, command builder, and runtime
    helper
  - added architecture coverage to prevent the broad facades from being rebuilt
    accidentally
- Operation flow-execution bridge cleanup:
  - moved the Prefect-flow execution helper out of `rapthor.lib.Operation` and
    into `rapthor.operations.flow_execution`
  - updated operation adapters to call the execution bridge directly with their
    parset-derived runtime settings
  - removed the last documented domain-to-execution architecture exception and
    simplified the import-boundary test so `rapthor.lib` has no allowlist
- Normalized command wrapper cleanup:
  - removed production `normalized_*_command` wrappers that only delegated to
    `normalize_command(build_*_command(...))`
  - kept command normalization as a test concern by calling `normalize_command`
    directly in golden-fixture tests
  - reduced command modules and flow modules to the command builders used by
    execution code
- Dead-code cleanup:
  - removed unused `merge_list_flatten()` from `rapthor.operations.image`
  - verified that no production, test, or documentation references remained
- Low-risk duplicate consolidation:
  - moved Prefect run-context detection into `rapthor.execution.prefect_context`
  - moved shared payload basename, string-list, and integer-list validators into
    `rapthor.execution.payloads`
  - moved shared solution-cycle parsing onto the `Field` model
  - added focused tests for payload validation and solution-cycle parsing
- Runtime abstraction cleanup:
  - removed the unused `RuntimeSpec`, `build_runtime_spec()`, and
    `build_command_environment()` layer and deleted `rapthor.execution.runtime`
  - kept direct `ResourceRequest` validation where production shell execution
    currently needs it
  - merged `run_flow_with_task_runner` into `rapthor.execution.task_runner` so
    Prefect task-runner construction and flow application have one owner
- Shell/output validation cleanup:
  - added `rapthor.execution.outputs` for shared required file/directory and
    glob-to-record helpers
  - kept operation-specific shell wrappers where they preserve explicit
    `Concatenate`, `Predict`, `Mosaic`, `Calibration`, and image output messages
- Image WSClean command construction cleanup:
  - extracted the shared WSClean command prefix and common option list used by
    no-DDE, facet, and screen command builders
  - kept mode-specific WSClean tokens and flags in the mode-specific builders so
    command differences remain easy to review
- Operation package and pipeline module consolidation:
  - moved operation Prefect adapters into operation-owned `flow.py` modules:
    `rapthor.execution.image.flow`, `rapthor.execution.calibrate.flow`,
    `rapthor.execution.concatenate.flow`, `rapthor.execution.predict.flow`, and
    `rapthor.execution.mosaic.flow`
  - added `rapthor.execution.concatenate`, `rapthor.execution.predict`, and
    `rapthor.execution.mosaic` packages with `commands.py`, `payloads.py`, and
    `flow.py` modules instead of keeping small-flow commands and payloads in a
    shared flow bucket
  - moved the top-level orchestration modules under
    `rapthor.execution.pipeline` and renamed the internal API to
    `pipeline_flow`, `run_pipeline`, `run_pipeline_steps`,
    `collect_pipeline_features`, and `build_pipeline_step_plan`
  - removed the previous flow package from active imports because
    this unreleased branch does not need compatibility shims
  - moved `run_flow_with_task_runner` into `rapthor.execution.task_runner` so
    Prefect task-runner selection and flow application have one owner
- Verified in the dev container:
  - `python3 -m pytest tests/architecture tests/execution/test_task_runner.py tests/execution/test_payloads.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_concatenate_flow.py tests/execution/test_mosaic_flow.py tests/execution/test_predict_flow.py tests/execution/test_reference_fixtures.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_image_flow.py tests/execution/test_calibrate_flow.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_pipeline_flow.py tests/lib/test_observation.py -q --tb=short`
  - `python3 -m pytest tests/operations/test_concatenate.py tests/operations/test_mosaic.py tests/operations/test_predict.py tests/operations/test_image.py tests/operations/test_calibrate.py -q --tb=short`
  - `python3 -m py_compile bin/rapthor`
  - import smoke checks for operation package owners, `rapthor.execution.task_runner.run_flow_with_task_runner`, and removed old `rapthor.execution.flows`/`flow_runtime` paths
  - `python3 -m pytest tests/architecture -q --tb=short`
  - `python3 -m pytest tests/lib/test_records.py tests/execution/test_payloads.py tests/execution/test_commands.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_concatenate_flow.py tests/execution/test_mosaic_flow.py tests/execution/test_predict_flow.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_image_flow.py tests/execution/test_calibrate_flow.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_reference_fixtures.py tests/lib/test_operation.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_concatenate_flow.py tests/execution/test_payloads.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_mosaic_flow.py tests/execution/test_predict_flow.py tests/execution/test_payloads.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_image_flow.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_payloads.py tests/architecture -q --tb=short`
  - `python3 -m pytest tests/execution/test_calibrate_flow.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_commands.py tests/execution/test_mosaic_flow.py tests/execution/test_predict_flow.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_commands.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_image_flow.py tests/execution/test_calibrate_flow.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_image_flow.py -q --tb=short`
  - `python3 -m pytest tests/operations/test_image.py -q --tb=short`
  - `python3 -m pytest tests/architecture -q --tb=short`
  - `python3 -m pytest tests/execution/test_image_outputs.py tests/execution/test_image_flow.py -q --tb=short`
  - `python3 -m pytest tests/operations/test_image.py -q --tb=short`
  - `python3 -m pytest tests/architecture -q --tb=short`
  - `python3 -m pytest tests/execution/test_payloads.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_image_flow.py -q --tb=short`
  - `python3 -m pytest tests/architecture -q --tb=short`
  - `python3 -m pytest tests/execution/test_calibrate_flow.py -q --tb=short`
  - `python3 -m pytest tests/operations/test_calibrate.py -q --tb=short`
  - `python3 -m pytest tests/architecture -q --tb=short`
  - `python3 -m pytest tests/operations/test_concatenate.py tests/operations/test_mosaic.py tests/operations/test_predict.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_pipeline_flow.py -q --tb=short`
  - `python3 -m pytest tests/lib/test_observation.py -q --tb=short`
  - `python3 -m pytest tests/integration/test_cli.py -q --tb=short`
  - `python3 -m pytest tests/lib/test_records.py tests/architecture tests/execution/test_reference_fixtures.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_concatenate_flow.py tests/execution/test_mosaic_flow.py tests/execution/test_predict_flow.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_image_flow.py tests/execution/test_calibrate_flow.py -q --tb=short`
  - `python3 -m pytest tests/operations/test_image.py tests/operations/integration/test_image_to_mosaic.py -q --tb=short`
  - `python3 -m pytest tests/architecture tests/execution/test_prefect_logging.py -q --tb=short`
  - `python3 -m pytest tests/architecture tests/lib/test_operation.py tests/operations/test_flow_execution.py -q --tb=short`
  - `python3 -m pytest tests/operations/test_concatenate.py tests/operations/test_mosaic.py tests/operations/test_predict.py -q --tb=short`
  - `python3 -m pytest tests/operations/test_image.py tests/operations/test_calibrate.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_concatenate_flow.py tests/execution/test_mosaic_flow.py tests/execution/test_predict_flow.py tests/execution/test_reference_fixtures.py -q --tb=short`
  - `python3 -m pytest tests/execution/test_image_flow.py tests/execution/test_calibrate_flow.py -q --tb=short`
  - `python3 -c "import rapthor.execution as execution; import rapthor.execution.image.commands as image_commands; import rapthor.execution.image.payloads as image_payloads; import rapthor.execution.image.sector as image_sector; import rapthor.execution.image.flow as image_flow; assert image_commands.build_wsclean_no_dde_command; assert image_payloads.image_payload_from_inputs; assert image_flow.validate_image_payload is image_payloads.validate_image_payload; assert image_sector.run_image_sector; assert not hasattr(execution, 'run_image_sector'); assert not hasattr(execution, 'image_payload_from_inputs')"`
  - `python3 -c "from rapthor.execution.image.payloads import ImagePayload, ImageSectorPayload, image_payload_from_inputs; import rapthor.execution.payloads as shared_payloads; assert ImagePayload; assert ImageSectorPayload; assert image_payload_from_inputs; assert not hasattr(shared_payloads, 'ImagePayload'); assert not hasattr(shared_payloads, 'ImageSectorPayload')"`
  - `python3 -c "import rapthor.execution as execution; import rapthor.execution.calibrate.commands as commands; assert commands.build_ddecal_solve_command; assert not hasattr(execution, 'build_ddecal_solve_command')"`
  - `python3 -c "from rapthor.execution.calibrate.payloads import CalibratePayload, calibrate_payload_from_inputs, validate_calibrate_payload; import rapthor.execution as execution; import rapthor.execution.payloads as shared_payloads; assert CalibratePayload; assert calibrate_payload_from_inputs; assert validate_calibrate_payload; assert not hasattr(shared_payloads, 'CalibratePayload'); assert not hasattr(execution, 'calibrate_payload_from_inputs')"`
  - `python3 -c "import rapthor.execution as execution; import rapthor.execution.calibrate.runner as runner; import rapthor.execution.calibrate.flow as flow; assert runner.run_calibrate_chunk; assert runner.collect_plot_and_combine; assert flow.calibrate_chunk_task; assert not hasattr(execution, 'run_calibrate_chunk')"`
  - `python3 -c "from rapthor.operations.calibrate_plan import build_calibration_solve_plan, requested_calibration_solves; solves, defaulted = requested_calibration_solves('dd', None, True); plan = build_calibration_solve_plan('dd', solves, defaulted_strategy=defaulted); assert [solve.step for solve in plan] == ['solve1', 'solve2', 'solve3', 'solve4']; assert plan[-1].output_prefix == 'medium2_phase'"`
  - `python3 -c "from rapthor.operations.calibrate_plan import build_calibration_dp3_steps; assert build_calibration_dp3_steps(0, 0, all_channels_regular=True, use_image_based_predict=True, do_slowgain_solve=False, solve_steps=['solve1'], preapply_solutions=True) == ['predict', 'applybeam', 'applycal', 'solve1']"`
  - `python3 -c "from rapthor.operations.calibrate_plan import build_calibration_preapply_steps; assert build_calibration_preapply_steps('dd', has_di_h5parm=True, has_fulljones_h5parm=True, apply_amplitudes=True, apply_normalizations=True) == ['fastphase', 'slowgain', 'fulljones', 'normalization']"`
  - `python3 -c "from rapthor.operations.calibrate_plan import build_calibration_solve_slot_inputs; assert build_calibration_solve_slot_inputs(1, 'slow', ntimechunks=2, datause='dual', solutions_per_direction=[[1], [1]], smoothness_dd_factors=[[3.0], [4.0]], smoothnessconstraint=12.0, include_smoothnessreffrequency=True)['solve1_smoothnessconstraint'] == 4.0"`
  - `python3 -c "from rapthor.operations.image_plan import build_image_applycal_steps; steps, selected = build_image_applycal_steps({'di': ['fast_phase'], 'dd': ['fast_phase', 'slow_gains']}, dd_h5parm='dd.h5', di_h5parm='di.h5', has_fulljones_h5parm=False, use_facets=False, apply_amplitudes=True, apply_normalizations=False, apply_none=False); assert steps == ['fastphase', 'slowgain']; assert selected == 'dd.h5'"`
  - `python3 -c "from rapthor.operations.image_plan import build_image_prepare_data_steps; assert build_image_prepare_data_steps(preapply_solutions=True, average_visibilities=True, image_bda_timebase=10.0, all_channels_regular=True, apply_screens=True) == ['applybeam', 'shift', 'applycal', 'avg']"`
  - `python3 -c "from rapthor.operations.image_plan import build_image_wsclean_control_inputs; assert build_image_wsclean_control_inputs('IQUV', 'link', [100, 200], disable_clean=False) == ('I', False, [100, 200])"`
  - `python3 -c "from rapthor.operations.image_plan import build_image_facet_solution_controls; assert build_image_facet_solution_controls('I', apply_amplitudes=True, apply_diagonal_solutions=True) == {'soltabs': 'amplitude000,phase000', 'diagonal_visibilities': True, 'scalar_visibilities': False}"`
  - `python3 -c "from rapthor.operations.image_plan import build_image_screen_interval; assert build_image_screen_interval(slow_timestep_sec=10.0, timepersample=2.0, numsamples=20) == [0, 15]"`
  - `python3 -c "from rapthor.operations.image_plan import build_image_mpi_resource_controls; assert build_image_mpi_resource_controls(nsectors=2, max_nodes=8, cpus_per_task=12, batch_system='slurm') == {'mpi_nnodes': [3, 3], 'mpi_cpus_per_task': [12, 12]}"`
  - `python3 -c "from rapthor.operations.calibrate_plan import build_calibration_core_baseline_selection; assert build_calibration_core_baseline_selection('HBA', ['CS003HBA0', 'RS106HBA0', 'DE601HBA']) == '[CR]*&&;!DE601HBA'"`
  - `python3 -c "from rapthor.execution.pipeline.plan import build_pipeline_step_plan; assert [item['operation'] for item in build_pipeline_step_plan([{'do_calibrate': False, 'do_predict': False, 'do_image': True, 'do_check': True}])] == ['image', 'mosaic', 'check_selfcal']"`
  - `python3 -c "from rapthor.execution.pipeline.lifecycle import do_final_pass, chunk_observations, make_report; assert do_final_pass and chunk_observations and make_report"`
  - `python3 -m py_compile bin/rapthor`
  - `python3 -c "import importlib.util; assert importlib.util.find_spec('rapthor.process') is None"`
  - `python3 -c "import importlib.util; assert importlib.util.find_spec('rapthor.execution.outputs') is None"`
  - `python3 -c "import rapthor.execution as execution; assert not hasattr(execution, 'run_pipeline'); assert not hasattr(execution, 'build_task_runner'); from rapthor.execution import prefect_logging; assert prefect_logging.publish_python_logs_to_prefect"`
  - `python3 -c "from rapthor.lib.operation import Operation; assert not hasattr(Operation, 'run_prefect_flow'); import rapthor.operations.flow_execution as flow_execution; assert flow_execution.run_prefect_flow"`
  - targeted Ruff format, lint, and import-sort checks for the new architecture
    tests, touched execution facade modules, output record helpers, and touched
    flow modules
  - targeted Ruff format, lint, and import-sort checks for the concatenate
    typed-payload slice
  - targeted Ruff format, lint, and import-sort checks for the mosaic/predict
    typed-payload slice
  - targeted Ruff format, lint, and import-sort checks for the image
    typed-payload slice
  - targeted Ruff format, lint, and import-sort checks for the calibration
    typed-payload slice
  - targeted Ruff format, lint, and import-sort checks for the shared command
    token and option helper slices
  - targeted Ruff format, lint, and import-sort checks for the image command
    extraction slice
  - targeted Ruff format, lint, and import-sort checks for the image payload
    extraction slice
  - targeted Ruff format, lint, and import-sort checks for the image package
    consolidation and sector/output split slice
  - targeted Ruff format, lint, and import-sort checks for moving image payload
    contracts into `rapthor.execution.image.payloads`
  - targeted Ruff format, lint, and import-sort checks for the calibration
    command extraction slice
  - targeted Ruff format, lint, and import-sort checks for the calibration
    payload extraction slice
  - targeted Ruff format, lint, and import-sort checks for the calibration
    runner split
  - targeted Ruff format, lint, and import-sort checks for the Calibrate
    solve-plan helper extraction
  - targeted Ruff format, lint, and import-sort checks for the Calibrate
    DP3-step helper extraction
  - targeted Ruff format, lint, and import-sort checks for the Calibrate
    pre-apply helper extraction
  - targeted Ruff format, lint, and import-sort checks for the Calibrate
    solve-slot input helper extraction
  - targeted Ruff format, lint, and import-sort checks for the Image applycal
    planning helper extraction
  - targeted Ruff format, lint, and import-sort checks for the Image
    prepare-data step helper extraction
  - targeted Ruff format, lint, and import-sort checks for the Image WSClean
    control helper extraction
  - targeted Ruff format, lint, and import-sort checks for the Image facet
    solution-control helper extraction
  - targeted Ruff format, lint, and import-sort checks for the Image
    screen-interval helper extraction
  - targeted Ruff format, lint, and import-sort checks for the Image MPI
    resource-control helper extraction
  - targeted Ruff format, lint, and import-sort checks for the Calibrate
    station-selection helper extraction
  - targeted Ruff format, lint, and import-sort checks for the process dry-run
    plan foundation
  - targeted Ruff format, lint, and import-sort checks for the process facade
    cleanup
  - targeted Ruff format, lint, import-sort, records, architecture, and
    affected-flow checks for output-shim removal
  - targeted Ruff format, lint, import-sort, architecture, and import smoke
    checks for execution package facade cleanup
  - targeted Ruff format, lint, import-sort, architecture, operation lifecycle,
    and import smoke checks for operation flow-execution bridge cleanup
  - targeted Ruff format, lint, import-sort, and focused command/flow fixture
    checks for normalized command wrapper cleanup

Known follow-up from the completed slice:

- Run a Sphinx docs build once the docs environment has `sphinx` installed.
- Avoid running multiple pytest processes in parallel locally without setting
  separate `RAPTHOR_TEST_RUN_ROOT` values; concurrent test startup can race while
  cleaning `.pytest_cache/rapthor-runs`.
- The image/calibration flow tests passed, but Prefect emitted a late logging
  shutdown warning after the passing summary. Track separately if it becomes
  noisy in CI.

Next slice:

- Begin Dask scalability and script-to-module migration preparation by auditing
  script helpers and choosing one small, low-risk script to expose as an
  importable function while keeping its CLI wrapper stable.

Immediate next tasks from the 2026-06-28 plan/code review:

Execution and operation cleanup queue, in recommended order:

1. Completed 2026-06-28: remove confirmed dead code.
   - Deleted `merge_list_flatten()` from `rapthor.operations.image`; no
     production, test, or documentation references remained.
2. Completed 2026-06-28: consolidate low-risk exact duplicates.
   - Moved the duplicate Prefect run-context check used by artifacts and Prefect
     logging into `rapthor.execution.prefect_context`.
   - Moved duplicated payload validators such as basename, string-list, and
     integer-list validation into `rapthor.execution.payloads`, keeping error
     messages stable.
   - Moved repeated solution-cycle parsing onto the `Field` model so Image,
     Predict, and Calibrate do not drift.
3. Completed 2026-06-28: resolve the unused runtime abstraction.
   - Removed `rapthor.execution.runtime.RuntimeSpec`, `build_command_environment()`,
     and `build_runtime_spec()` because they were tested but not used by production
     execution code, then deleted the empty `rapthor.execution.runtime` module.
   - Kept direct `ResourceRequest` validation and `thread_environment()` usage in
     the image-sector WSClean MPI path, where the production need is explicit.
4. Completed 2026-06-28: consolidate repeated shell/output checks only where it
   removes drift.
   - Added `rapthor.execution.outputs` for shared required file/directory checks,
     glob discovery, compressed-file records, and temporary-directory cleanup.
   - Moved calibration and image sector output checks to the shared helpers.
   - Updated concatenate, predict, and mosaic shell wrappers to use shared
     required-output helpers while keeping operation-specific failure messages
     visible at the call sites.
5. Completed 2026-06-28: simplify repeated WSClean command construction.
   - Extracted the shared WSClean command prefix and common option list used by
     no-DDE, facet, and screen command builders.
   - Preserved the existing golden command fixtures and kept mode-specific
     options visible in each builder so scientists can still review the command
     differences.
6. Remove test-only direct flow runners and test the production flow seams.
   - Remove `run_image_flow()`, `run_calibrate_flow()`, `run_mosaic_flow()`,
     `run_predict_flow()`, and `run_concatenate_flow()` from production modules
     once their tests are moved to production-used entry points or lower-level
     production runners.
   - Refactor the production Prefect entry points so they are easy to test
     without keeping parallel synchronous flow implementations. Prefer small
     production-owned orchestration helpers, injectable shell/runtime
     collaborators, and focused runner tests over exported functions whose only
     caller is the test suite.
   - Keep fast tests by asserting command and output behaviour at the smallest
     production-used seam: command builders, payload builders, operation runners
     such as `run_image_sector()`, calibration runner functions, and the actual
     `*_flow()` Prefect entry points with mocked shell loading where needed.
   - Do this next, before splitting large runner/payload functions or introducing
     command argument objects, so later cleanup does not preserve or expand the
     test-only API surface.
7. Split large payload and runner functions by real work unit.
   - Candidate functions are `image_payload_from_inputs()`,
     `calibrate_payload_from_inputs()`, `run_image_sector()`, and
     `_collect_plot_and_combine_dd_phase()`.
   - Split along units scientists recognise, such as prepare task, image cube,
     normalization, screen solve, scalar solve collection, and DD phase
     collection.
8. Introduce command argument objects for long command builders.
   - Replace very long command-builder signatures, such as the WSClean builders,
     with small frozen dataclasses grouped by real concepts: common WSClean
     options, no-DDE options, facet options, screen options, MPI launch options,
     and prepare-data options.
   - Keep command builders deterministic and side-effect free: argument objects
     should only describe the command, while builders should still return plain
     command tokens that golden fixture tests can compare.
   - Prefer stdlib dataclasses for this internal execution layer first. Evaluate
     Pydantic later for parset, payload, and script-module boundary validation,
     not as a dependency inside hot command-construction paths.
   - Do this after splitting `run_image_sector()` and related payload builders,
     because those splits will reveal the stable data groups. Do it before the
     script-to-module migration so new Python module APIs do not inherit the
     current long-argument style.
9. Revisit large operation adapters after the low-risk cleanup.
   - `Image.set_input_parameters()`, `Image.finalize()`, and
     `Calibrate.set_input_parameters()` remain the largest operation-side
     methods.
   - Extract only decision-heavy or duplicated pieces; keep finalizer side
     effects easy to audit.

Broader follow-on tasks after the cleanup queue:

1. Script-to-module audit and first conversion candidate.
   - Inventory each `rapthor/scripts/*.py` helper by command owner, input/output
     size, existing tests, external dependency risk, and whether it can sensibly
     run in-process on a Dask worker.
   - Start with a small, already tested helper such as `make_region_file.py`,
     `check_image_beam.py`, `blank_image.py`, or one mosaic helper rather than
     `normalize_flux_scale.py`, `calculate_image_diagnostics.py`,
     `subtract_sector_models.py`, or `process_gains.py`.
   - Extract one importable function, keep the command-line wrapper stable, and
     add CLI-vs-function parity tests.
2. Dask scalability contract checks.
   - Add focused tests that prove each flow submits the intended task units
     without passing domain objects or large nested state to workers.
   - Add payload-size/serialization guard tests for image sectors, calibration
     chunks, predict model tasks, mosaic image types, and concatenate epochs.
   - Extend resource-request coverage beyond image-sector WSClean paths so
     calibration, predict, mosaic, and concatenate command tasks have explicit
     thread/resource expectations where appropriate.
3. Runtime preflight and dry-run improvements.
   - Expand the dry-run/plan view so it reports task groups, resource hints,
     expected outputs, command/script adapters, and unsupported multi-node
     features before external tools run.
   - Improve preflight messages for missing tools, unsupported container mode,
     Slurm/external-Dask mismatch, missing Dask scheduler, and MPI WSClean
     assumptions.
4. Large runner/module simplification pass.
   - Review `rapthor.execution.calibrate.runner`, `rapthor.execution.image.sector`,
     `rapthor.execution.shell`, `rapthor.operations.calibrate`, and
     `rapthor.operations.image` before adding more abstractions.
   - Split only where there is a clearer scientific work unit, repeated command
     execution pattern, or testability/debugging benefit.
5. Contributor documentation slice.
   - Add short docs/checklists for adding a parset option, modifying an
     operation, adding an external command helper, and converting a script to an
     importable module.
   - Include the fast test lane for each change type and the expected owner
     module for payload, command, output, flow, operation, and pipeline changes.
6. Target-environment validation plan.
   - Define opt-in validation runs for external Dask, Slurm-launched Dask,
     MPI WSClean, and `prefect_command_profile = perf`.
   - Keep these separate from the default non-integration suite and document the
     required environment assumptions.

Remaining major stages:

- Manage code quantity and remove duplicated/dead compatibility code, including
  temporary shims, as slices land.
- Convert scripts to importable modules only when touched, while preserving CLI
  compatibility.
- Validate Dask scalability, external-Dask/Slurm, MPI WSClean, and
  target-environment assumptions.

## Refactor Principles

- Keep every refactor slice behaviour-preserving unless it is explicitly scoped
  as a feature change.
- Preserve the existing public parset, strategy, CLI, output file, restart, and
  finalizer contracts.
- Prefer pure functions and typed data contracts at module boundaries.
- Keep Rapthor domain objects out of Dask worker payloads; pass plain,
  serializable values only.
- Keep external-tool command builders deterministic and independently testable.
- Keep Prefect flows thin: orchestration, task wiring, artifacts, and runtime
  concerns belong there; domain extraction and command construction should not.
- Use `rapthor.execution.image` as the reference package shape for other
  operation flows, scaling the number of modules to the complexity of the
  operation.
- Optimize for fast contributor feedback: most behaviour should be testable
  without Prefect, Dask, Slurm, internet access, or external radio astronomy
  tools.
- Prefer observable and debuggable interfaces: dry-run output, clear preflight
  errors, stable logs, reproducible work directories, and actionable artifacts.
- Move broad code only after tests pin the current behaviour.
- Update documentation and examples whenever the contribution path changes.

## Clean Architecture Boundaries

The refactor should follow the dependency rule: inner layers must not import
outer layers. Scientific domain code should stay independent of Prefect, Dask,
Slurm, shell execution, artifact publication, and external command runners.

Use these layers as the architectural guide:

- Domain: field, observation, sector, cluster, strategy, parset-derived scientific
  state, and finalizer-visible operation state.
- Application/use cases: operation planning, parset-to-payload mapping,
  restart/reuse decisions, pipeline feature detection, and workflow decisions
  that are independent of a specific scheduler.
- Interface adapters: command builders, output-record conversion, script
  wrappers, filesystem record handling, and adapters between domain/use-case
  concepts and serializable execution payloads.
- Frameworks/drivers: Prefect flows and tasks, Dask task runners, Slurm
  integration, shell execution, artifact publishing, dashboards, logging
  integrations, and runtime resource checks.

Dependency direction should be one-way:

- `rapthor.lib` must not import `rapthor.execution`, Prefect, Dask, Slurm, or
  shell-command infrastructure.
- Application/use-case helpers may depend on domain objects and plain typed
  payload contracts, but not on Prefect task objects or Dask runtime state.
- Command builders should return deterministic token lists and should not run
  commands, publish artifacts, or inspect Prefect context.
- Prefect flows should depend inward on payload contracts, command builders, and
  pure helpers; pure helpers should not depend outward on Prefect flows.
- Runtime integrations should be replaceable adapters around stable ports such
  as command execution, artifact publishing, work-directory layout, task
  scheduling, and resource validation.

When a dependency needs to cross outward, define a small protocol or adapter
interface first. For example, use injectable collaborators for command
execution, artifact publication, runtime scheduling, and lifecycle hooks so
tests can exercise the use case without starting Prefect, Dask, Slurm, or real
external tools.

## Operation Package Pattern

The `rapthor.execution.image` package is the reference architecture for
remaining operation refactors. For each non-trivial operation, prefer an
operation-owned package shaped around clear responsibilities:

- `rapthor.execution.<operation>.commands` for deterministic command token
  builders and normalized command fixtures.
- `rapthor.execution.<operation>.payloads` for operation-specific typed payload
  contracts, payload construction, and payload validation.
- `rapthor.execution.<operation>.outputs` for operation-specific output
  discovery and finalizer-compatible records.
- `rapthor.execution.<operation>.<unit>` for scheduler-independent task bodies,
  named after the work unit scientists recognise, such as `sector`, `chunk`,
  `epoch`, `model`, or `image_type`.
- `rapthor.execution.<operation>.flow` as the thin Prefect adapter responsible
  for task wiring, scheduling, runtime integration, and result aggregation.
- `rapthor.execution.pipeline.flow`, `plan`, and `lifecycle` for top-level
  pipeline orchestration, dry-run planning, feature detection, and lifecycle
  collaborators.

Use the full package shape for complex flows such as image and calibration. For
smaller flows such as concatenate, mosaic, and predict, adopt the same ownership
rules without creating empty pass-through modules; add a module only when it
owns real behaviour, reduces duplication, or gives tests a clearer target.

Tests should mirror the package boundaries: command tests for command builders,
payload tests for contracts and validation, output tests for file discovery,
runner tests with shell execution mocked, and flow tests for Prefect wiring.
Prune broad `rapthor.execution` facade exports and any migration-era flow
facades as soon as internal imports point at the owning operation package.

Add architecture fitness tests as the boundaries settle. These can start as
simple import-boundary tests using `ast` or `modulefinder` before introducing
another dependency. The checks should fail if domain modules import execution
frameworks, if pure payload/command modules import Prefect, or if tests begin to
depend on broad facade exports that hide ownership.

## Target Module Shape

The exact package names can evolve while implementing the plan, but the intended
responsibilities are:

- `rapthor.lib`: domain model, parset/strategy interpretation, field/sector/
  observation state, operation lifecycle primitives, and finalizer-compatible
  record utilities.
- `rapthor.operations`: operation adapters that connect domain objects to flow
  payload builders, run the selected flow, update field state, and handle
  operation finalization.
- `rapthor.application` or `rapthor.use_cases`: a possible new home for
  scheduler-independent operation planning, typed payload contracts, parset/
  field-to-payload mapping, restart decisions, and workflow decisions that should
  not depend on Prefect/Dask.
- `rapthor.execution.payloads`: the shared serialization guard for worker-safe
  payload values. Operation-specific payload contracts should live in the owning
  operation package.
- `rapthor.execution.commands`: shared command token utilities plus small
  operation-specific command modules where useful.
- `rapthor.execution.<operation>.flow`: operation-specific Prefect
  orchestration only: task boundaries, scheduling, retries/failure handling,
  artifact publication, and task-runner integration.
- `rapthor.execution.pipeline`: top-level pipeline orchestration, lifecycle
  hooks, feature detection, dry-run planning, and preflight integration.
- `rapthor.execution.task_runner`, `outputs`, `resources`, `slurm`, `workdirs`,
  `artifacts`, and `shell`: reusable runtime infrastructure.
- `rapthor.scripts`: standalone helper scripts used by external command
  builders, kept testable with small fixtures.

## Step-By-Step Plan

### 0. Establish The Refactor Safety Net

Outcome: contributors can refactor with confidence because the current behaviour
is pinned before code moves.

- Record the current architecture boundaries in this plan and in developer docs.
- Keep the existing command/output reference fixtures as regression anchors.
- Add missing focused tests before moving behaviour out of large modules.
- For each refactor slice, run the narrowest relevant tests first, then a broader
  non-integration lane before merging.
- Avoid large formatting-only rewrites during behaviour-preserving moves; they
  make scientific parity review harder.

Suggested first checks:

```bash
python3 -m pytest tests/execution/test_commands.py tests/lib/test_records.py -q --tb=short
python3 -m pytest tests/execution/test_payloads.py tests/operations -q --tb=short
```

### 1. Define Stable Internal Boundaries

Outcome: the codebase has a small number of documented internal contracts instead
of accidental imports from large implementation modules.

- Audit `rapthor.execution.__init__` and operation package initializers.
- Decide which imports are stable internal API and which are test convenience
  exports.
- Move tests toward direct imports from the module that owns the behaviour.
- Keep temporary compatibility exports only where existing users are likely to
  rely on them.
- For every compatibility shim or facade, record the owner, reason, expected
  removal condition, import migration path, and tests that prove it can be
  removed.
- Add architecture fitness checks for forbidden imports and intended dependency
  direction.
- Add a small ownership map that links package areas to their test directories
  and common change workflows.
- Add a short `docs/source/development/architecture.rst` page describing the
  domain, operation, payload, command, flow, and runtime layers.
- Add an import-boundary note for new contributors: operation adapters build
  payloads, payload builders create serializable contracts, command builders
  create deterministic token lists, flows orchestrate execution.

Completion criteria:

- New contributors can identify the owning module for payload, command, output,
  flow, and finalizer changes.
- `__init__` exports are either intentionally documented or scheduled for
  deprecation.
- Temporary compatibility shims have explicit removal criteria and are tracked
  in this plan or the development architecture docs.
- CI has at least a lightweight import-boundary check so clean architecture does
  not rely on review memory alone.

### 2. Consolidate Output Record Handling

Outcome: Rapthor has one finalizer-compatible record API for files,
directories, optional values, nested lists, path extraction, validation, copying,
and cleanup.

- Compare any execution-side output-record helpers with `rapthor.lib.records`.
- Pick the long-term home for record creation and validation. Prefer the domain
  layer if finalizers and operations both need the API.
- Move shared helpers into that home:
  - `file_record` and `directory_record`
  - required and optional path extraction
  - basename validation where command builders depend on it
  - nested record validation for lists and optional records
  - copy, move, and cleanup helpers used by finalizers
- Remove compatibility shims once flows/tests/downstream-supported imports use
  the final record API.
- Replace local `_file_record_path`, `_directory_record_path`, and
  `_optional_file_record_path` helpers in flow modules with the shared API.
- Add or update import-boundary/search tests so new code does not start using a
  shim that is scheduled for removal.
- Add tests for malformed records, missing paths, optional records, nested lists,
  and path extraction error messages.

Completion criteria:

- Flow modules no longer contain duplicated record path extraction helpers.
- Operation finalizers and Prefect flows validate the same record contract.
- Existing output reference tests still pass.
- The old execution-side output-record shim has been removed, or any replacement
  shim remains documented with an explicit removal trigger.

### 3. Introduce Typed Payload Contracts

Outcome: payload builders, Prefect tasks, and tests stop depending on large
untyped dictionaries whose shape is hard to discover.

- Start with smaller flows such as concatenate, mosaic, and predict as proving
  grounds, then apply the same pattern to the highest-risk flows: image and
  calibrate.
- Add small `TypedDict` or dataclass contracts in `rapthor.execution.payloads`
  or operation-specific payload modules; move them toward a scheduler-independent
  application/use-case package if the dependency boundary becomes clearer that
  way.
- Keep a future Pydantic adoption path open for boundary validation. Do not adopt
  it as part of this plan yet, but keep payload/config/output contracts easy to
  express as Pydantic models later by using explicit fields, simple types, and
  clear conversion points.
- Keep payload values plain and serializable: strings, numbers, booleans, lists,
  dictionaries, and `None`.
- Provide explicit conversion helpers such as `from_operation_inputs(...)`,
  `to_payload()`, or `validate_payload(...)`.
- If Pydantic is evaluated later, start with a narrow spike such as runtime
  config, output records, or one small payload; require better error messages,
  plain-dict export before Dask submission, and no dependency-direction leaks
  before broader adoption.
- Keep the existing `assert_serializable_payload()` check at task boundaries.
- Add tests that exercise the payload builders without starting Prefect.
- Keep payload keys stable until downstream code has migrated.

Recommended extraction order:

1. Shared record/path fields.
2. Concatenate, mosaic, and predict payloads as smaller proving grounds.
3. Image sector payloads and image flow payloads.
4. Calibrate solver/chunk/screen/collection payloads.
5. Pipeline-flow lifecycle payloads if they become more complex.

Completion criteria:

- A developer can inspect one payload type to see required and optional fields.
- Payload tests fail at construction time for missing or invalid high-value
  fields.
- Dask workers continue to receive only serializable data.

### 4. Consolidate Command Builder Utilities

Outcome: external-tool command construction is deterministic, readable, and easy
to test without running DP3, WSClean, EveryBeam, IDG, or helper scripts.

- Expand `rapthor.execution.commands` with shared utilities already repeated in
  flow modules:
  - boolean tokens
  - optional flag/value appending
  - list token expansion
  - path-list joins
  - normalized command wrappers
  - shell-safe display strings
- Keep operation-specific command builders close to their domain until the best
  module split is obvious.
- Extract command builder groups from large flows into modules such as:
  - `rapthor.execution.commands.image`
  - `rapthor.execution.commands.calibrate`
  - `rapthor.execution.commands.predict`
  - `rapthor.execution.commands.mosaic`
  - `rapthor.execution.commands.concatenate`
- Preserve existing function names through compatibility imports while tests are
  migrated.
- Add focused tests for each command builder group using command token fixtures.

Completion criteria:

- Flow modules call command builders but do not assemble long command token lists
  inline.
- All command builders are importable and testable without Prefect.
- Command reference fixtures remain stable unless an intentional scientific or
  runtime change is documented.

### 5. Split The Image Flow By Responsibility

Outcome: `rapthor.execution.image.flow` becomes a readable orchestration layer
instead of one large module containing command construction, payload mapping,
sector execution, output discovery, and artifact logic.

- Extract image command builders first, using the command plan above.
- Extract image payload mapping into a pure payload module.
- Extract sector work into focused helpers:
  - prepare imaging data
  - concatenate time chunks
  - mask and region preparation
  - WSClean command selection
  - image cube and catalog creation
  - source filtering and diagnostics
  - output discovery and validation
- Extract artifact publication wrappers only if they are reusable or obscure the
  flow.
- Keep the Prefect flow responsible for task wiring, scheduling, result
  aggregation, and publishing flow-level artifacts.
- Split tests to mirror the new boundaries:
  - command builders
  - payload mapping
  - sector execution with shell mocked
  - output discovery contracts
  - flow-level orchestration

Completion criteria:

- The image flow module can be read top to bottom as orchestration.
- Most image logic can be tested without Prefect or external radio astronomy
  tools.
- Image operation tests still prove finalizer-visible field state and restart
  behaviour.

### 6. Split The Calibrate Flow By Responsibility

Outcome: `rapthor.execution.calibrate.flow` becomes a readable orchestration
layer with solver, screen, collection, plotting, and combine logic separated.

- Extract calibration command builders.
- Extract solver payload mapping and validation.
- Split pure helpers for:
  - DDECal solve command setup
  - IDGCal phase and phase/gain solve setup
  - draw-model and region setup
  - solve chunk execution
  - screen collection
  - H5Parm collection and combination
  - plot solution selection and artifact publication
  - source adjustment and gain processing
- Make solve-mode branching explicit and testable.
- Keep Prefect tasks small and named after the unit of work scientists recognise.
- Split tests by command, payload, chunk execution, collect/combine, and
  flow-level orchestration.

Completion criteria:

- Adding a new solve mode or solver command does not require editing unrelated
  plotting, collection, or artifact code.
- Calibration branch coverage improves without invoking external tools.
- Existing calibration output records and field finalization remain unchanged.

### 7. Thin Operation Adapters

Outcome: operation classes express lifecycle and finalizer effects, while pure
helpers do parset/field/input-to-payload mapping.

- For `Image` and `Calibrate`, identify methods that only read parset/field state
  and build flow inputs.
- Move those mappings into pure helper modules or functions with focused tests.
- Keep operation classes responsible for:
  - lifecycle setup
  - restart/done/output file handling through the base `Operation`
  - calling the selected Prefect flow
  - updating field attributes expected by later operations
  - copying/cleaning outputs
- Preserve monkeypatch-friendly operation constructors used by process tests.
- Repeat the same pattern for predict, mosaic, and concatenate only where it
  simplifies real behaviour.

Completion criteria:

- A scientist can review finalizer side effects without reading command builder
  code.
- A developer can test parset-to-payload mapping without constructing a full
  runtime environment.
- Operation adapters are smaller and follow the same shape across operations.

### 8. Clarify Pipeline Orchestration

Outcome: top-level orchestration remains easy to reason about as more runtimes,
feature flags, and scientific modes are added.

- Keep `rapthor input.parset` as the user-facing entry point.
- Keep `rapthor.execution.pipeline.flow` responsible for Prefect/Dask pipeline
  scheduling.
- Continue using injectable lifecycle hooks and operation factories for tests.
- Move feature detection helpers into a small, documented module if they grow.
- Keep preflight validation close to runtime capability checks.
- Document how pipeline features map to strategy steps and parset options.
- Add tests for new feature flags before wiring them into execution.

Completion criteria:

- Pipeline tests can cover operation ordering, skip conditions, error propagation,
  restart/reset behaviour, and compatibility helpers without real commands.
- Runtime preflight failures produce actionable messages.

### 9. Improve Test Structure And Coverage

Outcome: tests form a clear confidence ladder that makes the pipeline safe and
pleasant to improve.

Use these test layers deliberately:

- Domain unit tests: pure `rapthor.lib` behaviour, parset/strategy rules, field
  state transitions, sector/observation logic, and finalizer-visible state.
- Contract tests: typed payloads, output records, path extraction, serialization
  safety, resource requests, work-directory layout, and preflight validation.
- Command-builder golden tests: deterministic token lists for DP3, WSClean,
  EveryBeam, IDG, helper scripts, MPI launchers, and wrapper commands.
- Script-module tests: importable Python function behaviour plus CLI wrapper
  compatibility for each converted script.
- Flow orchestration tests: Prefect entry points with shell/external tools mocked
  so task wiring, retries, artifacts, output validation, and failure handling are
  covered without real radio astronomy tools.
- Operation adapter tests: lifecycle setup, restart/reuse, finalizer side
  effects, copy/clean behaviour, and field hand-off to later operations.
- Pipeline tests: operation ordering, skip conditions, strategy feature
  detection, preflight failures, reset/restart behaviour, and public helper
  compatibility.
- Dask scheduling tests: sync/local-Dask/external-Dask configuration,
  serializable payloads, task submission shape, resource hints, task-runner
  fallback behaviour, and safeguards against large object transfer.
- Integration smoke tests: representative external-tool scenarios for DP3,
  WSClean, EveryBeam, PyBDSF, diagnostics, mosaic hand-off, CLI execution, and
  restart.
- Target-environment tests: Slurm, external Dask, MPI WSClean, shared filesystem
  assumptions, and container/runtime deployment checks.
- Performance and observability checks: command timing, worker memory, Dask
  performance reports, artifact quality, log completeness, and regression
  signals for task granularity.
- Documentation and example smoke tests: example parsets, strategies, and common
  commands remain runnable or at least parseable.

Keep the default non-integration suite free of external-tool, internet, Slurm,
and multi-node requirements. Mark tests explicitly so contributors understand
which lane they are running: unit/contract tests should be fast, Prefect tests
should remain serial, integration tests should be environment-aware, and
target-environment/performance tests should be opt-in.

Test data and fixtures:

- Split oversized tests along the same boundaries as the code.
- Add builders for common payloads, records, sectors, fields, parsets, shell
  results, resource requests, and Dask task-runner configurations.
- Prefer small FITS, H5Parm, sky-model, region, and Measurement Set fixtures
  already in `tests/resources/`.
- Keep large Measurement Sets and downloaded archives out of version control.
- Maintain command and output reference fixtures as reviewed golden contracts.
- Add a documented fixture-update workflow so intentional command/output changes
  are easy to review and accidental churn is obvious.

Quality gates:

- Target the next practical milestone at 85% non-integration coverage.
- Do not add a hard coverage gate until CI is stable above the chosen threshold
  for several runs.
- Use coverage to expose hard-to-test areas, then extract pure helpers instead
  of testing through full flows only.
- Add branch-focused tests for decision-heavy code even when line coverage looks
  acceptable.
- Add architecture fitness tests for dependency direction and forbidden imports.
- Add CLI-vs-function parity tests as scripts become modules.
- Add Dask payload-size and serialization checks for new in-process tasks.
- Keep `pytest-socket` style network restrictions for unit tests; mark and
  isolate anything that needs internet access.

High-value coverage areas:

- malformed command-log and artifact contexts
- `perf` success, failure, and no-sample paths
- FITS preview edge cases
- Dask/task-runner fallback behaviour
- worker-resource validation and multi-node scheduling assumptions
- payload-size and large-object serialization safeguards
- subprocess-vs-in-process script parity
- dry-run/preflight output for common user mistakes
- CLI restart/reset and pipeline-flow compatibility helpers
- `Field` regrouping, target selection, normalization scaling, and empty model
  branches
- pure script helpers such as `subtract_sector_models.py`,
  `collect_screen_h5parms.py`, `check_image_beam.py`, `blank_image.py`,
  `combine_h5parms.py`, `process_gains.py`, and `make_region_file.py`

Completion criteria:

- The most important branches in payload mapping, command construction, output
  validation, orchestration, finalization, and Dask scheduling are covered by
  focused tests.
- Integration tests remain smoke/regression coverage for real external tools
  rather than the only way to validate business logic.
- Contributors can choose a fast, documented test lane for the layer they
  changed and know when broader validation is required.

### 10. Improve Contribution Documentation

Outcome: developers and scientists have a clear path for common changes.

- Add a development architecture guide under `docs/source/development/`.
- Add "How to add a new parset option" documentation:
  - defaults
  - docs/examples
  - domain object
  - payload mapping
  - command builder
  - flow/task usage
  - tests
- Add "How to add or modify an operation" documentation:
  - operation adapter
  - payload contract
  - command builder
  - flow task
  - output records
  - finalizer effects
  - focused and integration tests
- Add "How to add a new external command helper" documentation:
  - script location
  - command builder
  - unit fixtures
  - artifact/logging expectations
- Keep examples in sync with supported runtime options.

Completion criteria:

- A new contributor can make a small operation or parset change by following the
  guide without reverse-engineering image or calibration internals first.

### 11. Improve User And Developer Experience

Outcome: Rapthor is not only clean internally, but also pleasant to run, debug,
profile, and extend.

- Add or improve a dry-run/plan view that shows selected strategy steps,
  operation order, expected task groups, key command lines, resource hints,
  output locations, and unsupported feature warnings without running external
  tools.
- Make preflight failures actionable: include the parset key, strategy feature,
  missing executable, missing Python package, runtime setting, or filesystem
  assumption that caused the failure.
- Keep logs structured enough for humans and CI artifacts: operation name,
  sector/chunk identifiers, command labels, timing, retries, worker/resource
  hints, and output paths should be easy to find.
- Keep work directories predictable and documented so failed runs can be
  inspected and restarted without guessing where state lives.
- Provide small example parsets and strategies for common workflows: DI-only,
  DD-only, DI-then-DD, imaging-only, normalization, local Dask, external Dask,
  and Slurm/external-Dask when validated.
- Keep command-line entry points stable and avoid requiring contributors to know
  Prefect internals for normal development.
- Add contributor templates or checklists for common changes:
  - new parset option
  - new operation
  - new external command
  - script-to-module conversion
  - new integration scenario
- Keep error messages and docs scientist-friendly: explain what changed, why it
  matters scientifically or operationally, and what to try next.

Completion criteria:

- A new developer can make a small tested change using documented commands and
  fixtures within one focused workflow.
- A scientist can run a dry-run/preflight, understand what Rapthor will do, and
  diagnose common configuration or environment issues without reading source
  code.
- Logs, artifacts, and work directories make failed runs inspectable and
  restartable.

### 12. Manage Code Quantity And Complexity

Outcome: the refactor reduces cognitive load rather than creating more files,
facades, and boilerplate than the pipeline needs.

- Treat code volume as a maintainability signal, not a target by itself. A split
  is successful when responsibilities are easier to find, tests are faster to
  write, and old duplication disappears.
- Prefer deleting migration-era compatibility code, duplicated helpers, dead
  branches, unused exports, stale fixtures, and unused parset plumbing before
  adding new abstractions.
- Treat compatibility shims as technical debt with expiry conditions. They should
  be tiny, tested, documented, and removed in the next safe slice after imports
  are migrated.
- Track a lightweight before/after snapshot for large refactor slices:
  - largest modules by line count
  - broadest public export lists
  - duplicated helper functions
  - slowest focused tests
  - most commonly patched files
- Use soft complexity budgets to trigger review, not mechanical rewrites:
  - files approaching 700-900 lines should have a split plan
  - functions that need long comments, many flags, or deeply nested branches
    should be candidates for extraction
  - modules with many unrelated tests should be split by responsibility
  - public facade exports should stay intentionally small
- Avoid abstraction for its own sake. Add a protocol, class, or new package only
  when it removes real duplication, protects a clean boundary, enables testing,
  or makes runtime substitution clearer.
- Do not keep production-module functions solely for tests. If a helper is useful
  only because tests need a seam, either test through a production-used boundary
  or refactor the production code to expose the smaller real work unit that the
  pipeline itself uses.
- Keep data structures boring and explicit. Prefer `TypedDict`, small
  dataclasses, or plain functions over deep inheritance unless the existing
  operation lifecycle needs inheritance.
- Keep future validation libraries such as Pydantic at the boundaries, if they
  are adopted later. They should improve config/payload/output error messages,
  not become a deep dependency of domain objects or hot Dask task loops.
- Keep debug paths close to the code they explain:
  - each operation should have predictable input, output, log, and artifact
    locations
  - each task group should have clear labels for sector, chunk, mode, and solver
  - each failure should include enough context to reproduce the command or
    Python function call
  - each run should leave a concise manifest of parset, strategy, runtime,
    feature flags, task runner, and output records
- Add small debug helpers rather than ad hoc print/log blocks. For example,
  prefer reusable command summaries, payload summaries, and output-record
  summaries that tests can assert.
- Review total code after each major phase. If the refactor has mostly moved
  code around without reducing duplication, public surface area, or debugging
  friction, pause and simplify before continuing.
- Schedule a shim cleanup pass after each completed refactor stage. Search for
  compatibility modules, broad facade exports, deprecated names, and transitional
  allowlist entries, then remove any that no longer protect a real user-facing
  contract.

Completion criteria:

- Large files shrink into modules with clear single responsibilities, and the
  new module count does not create a maze of pass-through wrappers.
- Net new code is justified by deleted duplication, clearer tests, cleaner
  dependency direction, or better runtime/debug behaviour.
- Temporary compatibility code has a named removal path, and completed stages
  delete shims that are no longer needed.
- Contributors can locate the owner of a behaviour without searching across
  many similarly named helper modules.
- Debug output is structured and reusable enough that tests can protect it.

### 13. Dask Scalability And Script-To-Module Migration

Outcome: Rapthor is prepared for multi-node Dask execution and future in-process
Python tasks without mixing architectural cleanup with a broad script rewrite.

- Do not convert all scripts to in-process Python tasks as the first refactor.
  Prepare for that change now by defining stable Python APIs and keeping CLI
  compatibility.
- When touching a script, move the core behaviour into an importable function and
  keep the script entry point as a thin argument-parsing wrapper.
- Keep command builders as compatibility adapters until the corresponding Python
  function has focused tests and has been validated inside the relevant Prefect
  flow.
- Prioritize conversion candidates where the data is small, the behaviour is
  pure, subprocess overhead is noticeable, and Dask can schedule the work
  cleanly.
- Convert high-volume data paths only after profiling task granularity, memory
  use, serialization cost, and network transfer between workers.
- Treat in-memory data passing as an optimization, not a default. For large
  Measurement Sets, FITS images, H5Parm files, sky models, image cubes, and
  other heavy products, prefer passing compact metadata and keeping bulk data in
  shared storage, object storage, memory-mapped formats, or chunked formats such
  as Zarr/HDF5 where appropriate.
- Design Dask tasks around data locality: avoid moving large arrays or tables
  between workers when a worker can read the required chunk from shared storage.
- Add lightweight task contracts for future in-process execution:
  - serializable task inputs
  - explicit output records
  - no hidden global runtime state
  - deterministic work-directory usage
  - injectable filesystem, command-execution, and artifact collaborators where
    useful
- Keep external-tool calls such as DP3, WSClean, EveryBeam, IDG, and PyBDSF as
  command-driven adapters unless there is a tested Python API that is stable,
  performant, and deployable across worker nodes.
- Use Dask dashboard, worker memory metrics, and performance reports to decide
  whether a script should become an in-process task, remain a subprocess task, or
  be split into smaller chunk-aware tasks.

Recommended conversion order:

1. Pure helpers that operate on small text or metadata files.
2. FITS, region, sky-model, and H5Parm helpers with small fixture coverage.
3. Helpers whose output feeds another Python step and can avoid unnecessary disk
   round-trips.
4. Chunk-aware image/catalog/cube helpers where Dask can distribute independent
   work safely.
5. Heavy external-tool replacements only after representative multi-node
   profiling proves the benefit.

Completion criteria:

- Each converted script has an importable Python function, a thin CLI wrapper,
  focused unit tests, and unchanged command-line behaviour.
- Prefect flows can choose between subprocess execution and in-process Python
  execution behind a stable adapter.
- Multi-node Dask runs avoid avoidable large-object serialization and keep task
  inputs small enough to schedule reliably.

### 14. Runtime And Scalability Validation

Outcome: the cleaner architecture still supports local development, external
Dask, Slurm, and future SKA-Low scaling work.

- Keep runtime concerns isolated in `execution.config`, `runtime`,
  `task_runner`, `resources`, `slurm`, and `workdirs`.
- Use Dask dashboard and performance reports to review task granularity after
  code boundaries are clearer.
- Validate external-Dask and Slurm in representative allocations.
- Validate MPI WSClean in the intended deployment stack.
- Validate `prefect_command_profile = perf` in development and CI container
  runtimes where host kernel permissions allow sampling.
- Use rich demo runs to review command summary charts, logs, artifacts, and
  flamegraphs.

Completion criteria:

- The architecture supports runtime changes without pushing Slurm/Dask details
  into operation adapters or command builders.
- Profiling artifacts remain useful for DP3/WSClean bottleneck analysis.

### 15. Final Polish And Maintenance

Outcome: the refactor lands as a sequence of small, reviewable improvements.

- Run Ruff/formatting after each cluster of code moves.
- Remove compatibility shims only after imports and docs are migrated.
- Keep deprecations explicit and documented.
- Update `README.md`, release notes, and examples when user-facing behaviour or
  contribution paths change.
- Keep generated demo data, large reference artifacts, and run outputs ignored by
  VCS.

## Suggested Implementation Order

1. Document internal boundaries and audit public exports.
2. Consolidate output records.
3. Add typed payload contracts for concatenate, mosaic, and predict.
4. Completed shared command utilities for image/calibration option handling.
5. Completed image payload mapping extraction.
6. Move image sector execution and output discovery into focused helpers.
7. Completed calibration command builders and payload mapping extraction.
8. Completed calibration runner split.
9. Thin `Image` and `Calibrate` operation adapters.
10. Split tests to match the new modules.
11. Add dry-run/preflight and developer-experience improvements where they
    support the refactor.
12. Remove duplicated/dead code and check complexity before adding new layers.
13. Add script-to-module wrappers for touched scripts without broad conversion.
14. Add contributor documentation for common change paths.
15. Profile Dask task granularity and data movement before converting heavy
    scripts to in-process tasks.
16. Validate broader non-integration tests, then representative integration/demo
    runs.

## Useful Commands

Serial Prefect-server lane:

```bash
python3 -m pytest -m "not integration and prefect" -k "not test_field.py" tests -q --tb=short
```

Parallel non-Prefect lane:

```bash
python3 -m pytest -m "not integration and not prefect" -n auto -k "not test_field.py" tests -q --tb=short
```

Focused pipeline-flow checks:

```bash
python3 -m pytest tests/execution/test_pipeline_flow.py tests/lib/test_observation.py -q --tb=short
```

Focused output/payload/command checks:

```bash
python3 -m pytest tests/lib/test_records.py tests/execution/test_payloads.py tests/execution/test_commands.py -q --tb=short
```

Focused operation adapter checks:

```bash
python3 -m pytest tests/operations -q --tb=short
```

Architecture fitness checks, once added:

```bash
python3 -m pytest tests/architecture -q --tb=short
```

Script/module parity checks, as scripts are converted:

```bash
python3 -m pytest tests/scripts -q --tb=short
```

Code-size and large-module snapshot:

```bash
rg --files rapthor tests | xargs wc -l | sort -n
```

Non-integration coverage check:

```bash
python3 -m pytest -m "not integration" --cov=rapthor --cov-report=term-missing tests
```

Rich local demo:

```bash
scripts/dev/generate-prefect-demo-data.py --force
scripts/dev/run-rapthor-prefect-demo.py \
  --task-runner local_dask \
  --dask-dashboard-address :8787 \
  --dask-performance-report \
  examples/generated/prefect_demo_rich/prefect_demo_rich.parset
```

## Merge Criteria For Refactor Slices

- Public parset, strategy, CLI, restart, output-record, and finalizer behaviour is
  unchanged unless an intentional behaviour change is documented.
- Command token fixtures and output reference fixtures remain stable or are
  updated with a clear reason.
- New modules have focused tests at the same abstraction level.
- The chosen test lane is named in the merge notes: domain, contract, command,
  script/module, flow, operation, process, Dask scheduling, integration,
  target-environment, performance, or docs/example smoke.
- Architecture fitness checks pass for any slice that moves modules or changes
  imports.
- Refactor slices explain code-volume impact: what was deleted, what was added,
  whether public surface area grew, and why any extra abstraction is worth it.
- Script conversions include CLI compatibility tests and Python function tests.
- Dask-facing changes include payload serialization checks, resource/scheduling
  checks, and a decision about subprocess versus in-process execution.
- Debuggability changes include reusable summaries, structured log context, or
  run-manifest/output-record coverage where appropriate.
- User-facing changes include preflight/error-message coverage and docs/example
  updates.
- Prefect-specific tests are isolated from high xdist fan-out.
- Non-integration tests pass for the touched area, plus the relevant broader
  lane before merge.
- Docs are updated when a change affects contribution flow, user-facing options,
  runtime behaviour, or operation semantics.

## Deferred Follow-Up

- Slurm/external-Dask validation remains deferred until it can be run inside a
  representative Slurm allocation.
- MPI WSClean validation remains deferred until it can be run with the intended
  MPI/WSClean deployment stack.
- Dask task-granularity/resource optimization should happen after the module
  boundaries are cleaner, using Dask dashboards and performance reports.
- Revisit `hybrid_screens` only if it becomes a supported target workflow.
- Revisit `shared_facet_rw` after WSClean shared-facet read/write behaviour is
  reliable in the intended environment.
