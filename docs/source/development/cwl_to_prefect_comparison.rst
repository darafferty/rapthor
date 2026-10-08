.. _cwl_prefect_comparison:

Mapping the migration from CWL to Prefect
=========================================

This page is a high level overview of what changed in the Prefect/Dask branch
(``gec-468-ai-migrate-to-prefect``) compared to ``master``. It explains what
moved, what was rewritten, what stayed the same, what was added
or removed, and how large each change is. It does not repeat the reasons for
the change (see :doc:`adr_replace_cwl_toil_with_prefect_dask`) or describe
the new design (see :ref:`architecture`). The :ref:`last section
<cwl_architecture>` gives the C4 diagrams of the CWL version, drawn in the
same way as those of the Prefect version, so that the two can be compared
directly.

The numbers on this page were taken from three revisions:

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Revision
     - Date
     - Role on this page
   * - ``2e21be62``, the common ancestor
     - 2026-05-11
     - The last commit shared by ``master`` and the branch. The merge request
       diff, and every line count below, compare the branch with this revision
       (``git diff -M 2e21be62 b5884ade``, 380 files).
   * - ``ff1f29d9``, ``master``
     - 2026-10-08
     - ``master`` today, 64 commits after the ancestor. The file maps use the
       file names of this revision; see :ref:`master_moved` for what those
       commits changed.
   * - ``b5884ade``, the branch
     - 2026-10-06
     - The branch, 632 commits after the ancestor.


The two execution models
------------------------

The command, the parset, the strategy, the operations and the order in which
they run are the same in both versions. What changed is how an operation is
turned into work, and how that work is run.

.. rst-class:: wrap-code

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * -
     - CWL version (``master``)
     - Prefect version (branch)
   * - Start
     - The ``bin/rapthor`` script calls ``rapthor.process.run``.
     - The ``rapthor`` entry point (``rapthor/cli.py``) starts or connects to
       Prefect and Dask, then runs the ``pipeline`` flow.
   * - Run loop
     - ``rapthor/process.py``: the cycles, the selfcal convergence check and
       the final pass.
     - ``rapthor/execution/pipeline/flow.py`` holds the same loop;
       ``lifecycle.py`` has the final pass, the time chunking and the report.
   * - Description of the work of one operation
     - A Jinja2 template of a CWL workflow (``rapthor/pipeline/parsets``) is
       rendered to ``pipelines/<operation>/pipeline_parset.cwl``, and the
       inputs are written to ``pipeline_inputs.json``.
     - A payload of plain values (``rapthor/execution/<operation>/payloads.py``)
       is built from the ``Field`` (``builders.py``) and checked
       (``validation.py``). The inputs are still written to
       ``pipeline_inputs.json``.
   * - Steps
     - One CWL ``CommandLineTool`` file per command
       (``rapthor/pipeline/steps``), run by the CWL runner.
     - One Prefect task per step (``rapthor/execution/<operation>/flow.py``).
       The command line is built by a function in ``commands.py``.
   * - Python steps
     - Stand-alone scripts (``rapthor/scripts``), installed as commands and
       called from the CWL steps.
     - Functions in ``rapthor/execution/<operation>``, called from the task.
       PyBDSF and LoSoTo steps still run as separate ``python -m`` commands.
   * - Execution engine
     - A separate ``toil-cwl-runner`` (or ``cwltool`` or ``streamflow``)
       process for each operation. Toil's batch system runs the steps as local
       processes or as Slurm jobs.
     - Prefect flows inside the ``rapthor`` process. The tasks run on Dask
       workers, started by Rapthor on one machine or by a user in a Slurm
       allocation (``rapthor/execution/task_runner.py``).
   * - Containers
     - Optional: the CWL runner starts each step in a Docker, uDocker or
       Singularity image (``use_container``).
     - The whole run is started inside the image. ``use_container`` is
       rejected.
   * - Restart state
     - Toil's job store under ``pipelines/<operation>/jobstore``, plus a
       ``.done`` file and ``.outputs.json`` per operation.
     - ``.done``, ``.outputs.json`` and ``pipeline_outputs.json`` per operation
       (see :ref:`products`); no job store.
   * - Output records
     - CWL ``File`` and ``Directory`` objects, with checksums and sizes, read
       from the runner's output JSON (``rapthor/lib/cwl.py``).
     - Records with ``class`` and ``path`` only (``rapthor/lib/records.py``).
   * - Logs and monitoring
     - ``logs/<operation>/pipeline.log`` and the Toil job logs;
       ``plotrapthor`` for timing plots.
     - The Prefect and Dask dashboards, ``logs/<operation>/<task>.log``,
       ``logs/commands.jsonl`` and ``logs/tasks.jsonl`` (see
       :ref:`monitoring_rapthor`).

Intentional differences in the processing and the products are listed in
:ref:`migration_behavior_differences`.


What stayed the same
--------------------

* The ``rapthor <parset>`` command and its ``-q``, ``-v`` and ``-r`` options.
* The parset sections and almost all options (the differences are in
  :ref:`options_dependencies_tooling`), and the strategy files.
* The operations, their names (``calibrate_1``, ``image_1``, ...) and the
  order in which they run. The run loop in ``rapthor/process.py`` was moved
  with few changes.
* The run state: ``Field``, ``Observation``, ``Sector``, ``Parset`` and the
  strategy code in ``rapthor/lib``. The table below counts the functions and
  methods of the modules that exist on both ``master`` and the branch. A
  function is *modified* when its body differs by at least one line
  (docstrings are ignored).
* The products in the working directory and the ``.done`` marker used to
  resume a run.
* The external tools and libraries: DP3, WSClean, EveryBeam, IDG, PyBDSF,
  LSMTool and LoSoTo. The command lines are now built by Python functions and
  the expected command for each step is recorded in
  ``tests/execution/fixtures/command_reference.json``.
* The sky models in ``rapthor/skymodels`` (byte-identical) and the ``debug``
  helpers.

.. rst-class:: wrap-code

.. list-table:: Functions of the shared modules, branch compared with ``master`` today
   :header-rows: 1
   :widths: 40 12 12 12 12 12

   * - Module
     - On master
     - Unchanged
     - Modified
     - Added
     - Removed
   * - ``rapthor/lib/field.py``
     - 32
     - 19
     - 9
     - 2
     - 4
   * - ``rapthor/lib/observation.py``
     - 15
     - 12
     - 3
     - 0
     - 0
   * - ``rapthor/lib/sector.py``
     - 13
     - 11
     - 2
     - 0
     - 0
   * - ``rapthor/lib/strategy.py``
     - 7
     - 4
     - 3
     - 3
     - 0
   * - ``rapthor/lib/parset.py``
     - 9
     - 5
     - 4
     - 0
     - 0
   * - ``rapthor/lib/miscellaneous.py``
     - 18
     - 16
     - 0
     - 0
     - 2
   * - ``rapthor/lib/fitsimage.py``
     - 22
     - 17
     - 5
     - 1
     - 0
   * - ``rapthor/lib/calibration_memory.py``
     - 11
     - 8
     - 2
     - 5
     - 1
   * - ``rapthor/lib/cluster.py``
     - 3
     - 2
     - 1
     - 0
     - 0
   * - ``rapthor/lib/context.py``
     - 6
     - 6
     - 0
     - 0
     - 0
   * - ``rapthor/lib/operation.py``
     - 13
     - 5
     - 6
     - 4
     - 2
   * - ``rapthor/operations/predict.py``
     - 12
     - 7
     - 5
     - 6
     - 0
   * - ``rapthor/operations/mosaic.py``
     - 4
     - 1
     - 3
     - 2
     - 0
   * - ``rapthor/operations/concatenate.py``
     - 4
     - 2
     - 2
     - 1
     - 0
   * - ``rapthor/_logging.py``, ``rapthor/modifystate.py``
     - 5
     - 1
     - 4
     - 0
     - 0
   * - **Total**
     - **174**
     - **116**
     - **49**
     - **24**
     - **9**

Two thirds of the functions of these modules are identical to ``master``
today. Compared with the common ancestor the figures are 34 unchanged and
112 modified, because the branch has carried over most of the changes made
on ``master`` since May (see :ref:`master_moved`).


Where the code went
-------------------

The diagram shows where the lines of ``master`` ended up. The widths are the
numbers of lines on ``master`` today; the destinations are rewritten, so the
branch's modules are not the same size as their sources.

.. mermaid::
   :caption: Destination of the code on ``master``, weighted by its number of lines

   ---
   config:
     sankey:
       showValues: true
       width: 900
       height: 520
       nodeAlignment: justify
   ---
   sankey-beta

   master: CWL workflows,branch: flows and payloads,5694
   master: CWL steps,branch: command builders and tasks,6103
   master: top-level CWL,removed,571
   master: scripts,branch: step modules,6771
   master: scripts,removed,347
   master: CWL runner and Toil,branch: runtime,742
   master: CWL runner and Toil,branch: run state (lib),215
   master: run state (lib),branch: run state (lib),7020
   master: operations,branch: operations,3137
   master: run loop and bin,branch: cli and pipeline flow,888
   master: run loop and bin,removed,561
   master: testing helpers,branch: tests/conftest,339
   master: settings,branch: settings,607
   master: settings,removed,309

Packages and modules
~~~~~~~~~~~~~~~~~~~~

.. rst-class:: wrap-code

.. list-table::
   :header-rows: 1
   :widths: 30 8 34 28

   * - On ``master``
     - Lines
     - On the branch
     - What happened
   * - ``rapthor/process.py``
     - 554
     - ``rapthor/execution/pipeline/flow.py``, ``lifecycle.py``, ``plan.py``
     - Moved. The run loop is the same; the final pass, the chunking and the
       report are in ``lifecycle.py``.
   * - ``bin/rapthor``, ``bin/concat_linc_files``
     - 103
     - ``rapthor/cli.py``, ``rapthor/execution/concatenate/linc_cli.py``
     - Moved; installed as entry points from ``pyproject.toml``.
   * - ``bin/plotrapthor``, ``rapthor/scripts/plot_rapthor_timing.py``
     - 398
     - —
     - Removed. The Prefect dashboard and ``logs/tasks.jsonl`` replace the
       timing plots.
   * - ``rapthor/lib``: ``field``, ``observation``, ``sector``, ``parset``,
       ``strategy``, ``miscellaneous``, ``fitsimage``, ``cluster``,
       ``context``, ``calibration``, ``calibration_memory``
     - 6,684
     - The same files
     - Kept and edited in place (see the function table above).
       ``calibration.py`` was reduced to the solve-type metadata.
   * - ``rapthor/lib/operation.py``
     - 336
     - ``rapthor/lib/operation.py`` (250), ``rapthor/operations/flow_execution.py`` (24)
     - Rewritten. Template rendering, runner calls and log parsing are gone;
       subclasses implement ``execute_workflow``.
   * - ``rapthor/lib/cwl.py``
     - 215
     - ``rapthor/lib/records.py`` (293)
     - Replaced: ``CWLFile`` and ``CWLDir`` became ``FileRecord`` and
       ``DirectoryRecord``; ``NpEncoder`` and the copy and clean helpers
       moved with them.
   * - ``rapthor/lib/cwlrunner.py``, ``rapthor/lib/toil_batch_systems``
     - 742
     - ``rapthor/execution/config.py``, ``runtime_bootstrap.py``,
       ``task_runner.py``, ``slurm.py``, ``shell.py``, ``capabilities.py``
     - Replaced; no code in common.
   * - ``rapthor/operations/calibrate.py``
     - 1,195
     - ``rapthor/operations/calibrate/base.py`` (1,096), ``plan.py`` (364)
     - Split. The ``CalibrationSolve`` planning moved to ``plan.py``.
   * - ``rapthor/operations/image.py``
     - 1,404
     - ``rapthor/operations/image/base.py`` (860), ``initial.py``,
       ``normalize.py``, ``diagnostics.py``, ``plan.py``
     - Split, one class per file; the diagnostic reporting and the planning
       helpers were separated.
   * - ``rapthor/operations/concatenate.py``, ``mosaic.py``, ``predict.py``
     - 537
     - The same files
     - Kept; they now build a payload and run a flow instead of writing a
       CWL workflow.
   * - ``rapthor/pipeline/parsets/*.cwl`` (10 workflow templates)
     - 5,694
     - ``rapthor/execution/<operation>/flow.py``, with ``builders.py``,
       ``payloads.py`` and ``validation.py``
     - Rewritten as Python. The ``calibrate``, ``predict`` and ``image``
       workflows each had DI and DD (and sector) variants; the flows take the
       mode from the payload.
   * - ``rapthor/pipeline/steps/*.cwl`` (45 step definitions)
     - 6,103
     - ``rapthor/execution/<operation>/commands.py`` and the task functions
     - Rewritten as command builders (see :ref:`cwl_steps_map`).
   * - ``rapthor/pipeline/execution`` (top-level CWL workflow that runs
       Rapthor itself as one step, with its examples)
     - 571
     - —
     - Removed, no replacement.
   * - ``rapthor/scripts/*.py`` (24 scripts)
     - 7,107
     - ``rapthor/execution/<operation>/*.py``
     - Moved into importable modules (see :ref:`scripts_map`).
   * - ``rapthor/scripts/mpi_runner.sh``
     - 11
     - ``build_mpi_wsclean_launch_command`` in
       ``rapthor/execution/image/commands.py``
     - Replaced.
   * - ``rapthor/testing.py``
     - 339
     - ``tests/conftest.py``
     - Moved into the test suite.
   * - ``sbin/compare_workflow_results.py``
     - 454
     - —
     - Removed; it compared CWL outputs.
   * - ``rapthor/settings/defaults_skalow.parset``
     - 309
     - —
     - Removed. Nothing on ``master`` refers to it.
   * - —
     -
     - ``rapthor/execution/*.py`` (20 modules, 4,061 lines)
     - New: the Prefect and Dask runtime, command running and logging,
       scratch and workspace handling, artifacts, resources and preflight
       checks.
   * - —
     -
     - ``rapthor/lib/parset_paths.py``
     - New: resolves path values in the parset before the run starts.

.. _scripts_map:

Scripts
~~~~~~~

Four scripts were similar enough to be recorded as renames by ``git``
(``process_gains`` 74 %, ``combine_h5parms`` 74 %,
``calculate_image_diagnostics`` 82 %, ``normalize_flux_scale`` 88 %). The
others were rewritten around the same functions and share fewer lines with
their origin. Scripts marked † were added to ``master`` after the common
ancestor.

.. rst-class:: wrap-code

.. list-table::
   :header-rows: 1
   :widths: 30 6 36 28

   * - ``rapthor/scripts/``
     - Lines
     - ``rapthor/execution/``
     - Called from
   * - ``add_sector_models.py``
     - 228
     - ``predict/sector_model_addition.py``
     - ``postprocess_N`` (Predict)
   * - ``adjust_h5parm_sources.py``
     - 134
     - ``calibrate/h5parm_sources.py``
     - ``calibrate/collection.py``, ``calibrate/prediction.py``
   * - ``blank_image.py``
     - 95
     - ``image/masking.py``
     - ``image/preparation.py``
   * - ``calculate_image_diagnostics.py``
     - 994
     - ``image/diagnostic_calculation.py``
     - ``calculate_image_diagnostics`` (Image)
   * - ``check_image_beam.py``
     - 55
     - ``image/beam.py``
     - ``finish_wsclean_images`` (Image)
   * - ``collect_screen_h5parms.py``
     - 130
     - ``calibrate/screen_h5parms.py``
     - ``collect_screen_h5parms`` (Calibrate)
   * - ``combine_h5parms.py``
     - 794
     - ``calibrate/h5parm_combination.py``
     - ``combine_h5parms`` (Calibrate)
   * - ``concat_ms.py``
     - 257
     - ``concatenate/measurement_sets.py``
     - ``concatenate_epoch_N`` (Concatenate), ``concatenate_visibilities``
       (Image)
   * - ``correct_astrometry.py`` †
     - 256
     - ``image/astrometry.py``
     - ``finalize`` (Image)
   * - ``fetch_skymodel.py`` †
     - 56
     - —
     - Removed; only the top-level CWL workflow used it.
   * - ``filter_skymodel.py``
     - 298
     - ``image/skymodel_filter.py``, ``skymodel_filter_cli.py``
     - ``filter_skymodel`` (Image), as a ``python -m`` command
   * - ``make_catalog_from_image_cube.py``
     - 144
     - ``image/cubes.py``, ``cube_catalog_cli.py``
     - ``make_catalog_from_image_cube`` (Image), as a ``python -m`` command
   * - ``make_image_cube.py``
     - 79
     - ``image/cubes.py``
     - ``make_image_cube`` (Image)
   * - ``make_mosaic.py``, ``make_mosaic_template.py``, ``regrid_image.py``
     - 279
     - ``mosaic/images.py``
     - ``make_mosaic_template``, ``mosaic_<image type>`` (Mosaic)
   * - ``make_region_file.py``
     - 87
     - ``regions.py``
     - ``make_predict_region`` (Calibrate), ``image/preparation.py``
   * - ``normalize_flux_scale.py``
     - 908
     - ``image/flux_normalization.py``
     - ``normalize_flux_scale`` (Image)
   * - ``plot_rapthor_timing.py``
     - 291
     - —
     - Removed with ``plotrapthor``.
   * - ``process_gains.py``
     - 562
     - ``calibrate/gain_processing.py``
     - ``process_<solve>`` (Calibrate)
   * - ``restore_skymodel.py``
     - 218
     - ``image/restoration.py``
     - ``restore_skymodel`` (Image)
   * - ``subtract_sector_models.py``
     - 802
     - ``predict/sector_model_subtraction.py``
     - ``postprocess_N`` (Predict)
   * - ``wsclean_predict.py`` †
     - 440
     - ``calibrate/prediction.py``
     - ``read_predict_facets``, ``wsclean_predict_chunk_N`` (Calibrate)

.. _cwl_steps_map:

CWL steps
~~~~~~~~~

Each CWL step on ``master`` ran one command. On the branch the same command
is built by a function (all named ``build_*_command``) and run by a task. The
task names are those shown in the Prefect dashboard (see
:ref:`architecture_tasks`). Steps marked † were added to ``master`` after the
common ancestor.

.. rst-class:: wrap-code

.. list-table::
   :header-rows: 1
   :widths: 30 12 36 22

   * - ``rapthor/pipeline/steps/``
     - Ran
     - On the branch (``rapthor/execution/``)
     - Task
   * - ``concat_ms_files``
     - ``concat_ms.py``
     - ``concatenate/measurement_sets.py``
     - ``concatenate_epoch_N``, ``concatenate_visibilities``
   * - ``ddecal_solve``
     - DP3
     - ``calibrate/commands.py`` (``build_calibration_solve_command``),
       ``calibrate/solves.py``
     - ``solve_chunk_N``
   * - ``idgcal_solve_phase``, ``idgcal_solve_phase_and_gain``
     - DP3
     - ``build_idgcal_solve_phase_command``,
       ``build_idgcal_solve_phase_and_gain_command``
     - ``screen_chunk_N``
   * - ``collect_h5parms``
     - LoSoTo ``H5parm_collector.py``
     - ``build_collect_h5parms_command``, ``calibrate/collection.py``
     - ``collect_<solve>``
   * - ``collect_screen_h5parms``
     - script
     - ``calibrate/screen_h5parms.py``
     - ``collect_screen_h5parms``
   * - ``process_gains``
     - script
     - ``calibrate/gain_processing.py``
     - ``process_<solve>``
   * - ``plot_solutions``
     - ``plotrapthor``
     - ``build_plot_solutions_command``, ``calibrate/plotting.py``,
       ``plotting_cli.py``
     - ``plot_<solve>``
   * - ``combine_h5parms``
     - script
     - ``calibrate/h5parm_combination.py``
     - ``combine_h5parms``
   * - ``adjust_h5parm_sources``
     - script
     - ``calibrate/h5parm_sources.py``
     - ``adjust_normalization_h5parm``
   * - ``make_region_file``
     - script
     - ``regions.py``
     - ``make_predict_region``
   * - ``wsclean_draw_model``
     - WSClean
     - ``build_draw_model_command``
     - ``draw_model``
   * - ``wsclean_predict``, ``wsclean_predict_readpatches`` †
     - ``wsclean_predict.py``
     - ``build_wsclean_predict_command``, ``calibrate/prediction.py``
     - ``read_predict_facets``, ``wsclean_predict_chunk_N``
   * - ``predict_model_data``
     - DP3
     - ``predict/commands.py`` (``build_predict_model_data_command``)
     - ``dp3_predict_chunk_N``
   * - ``subtract_sector_models``, ``add_sector_models``
     - scripts
     - ``predict/sector_model_subtraction.py``,
       ``predict/sector_model_addition.py``
     - ``postprocess_N``
   * - ``prepare_imaging_data``
     - DP3
     - ``image/commands.py`` (``build_prepare_imaging_data_command``),
       ``image/preparation.py``
     - ``prepare_chunk_N``
   * - ``blank_image``
     - script
     - ``image/masking.py``
     - ``prepare_chunk_N``
   * - ``wsclean_image_no_dde``, ``wsclean_image_facets``,
       ``wsclean_image_screens``
     - WSClean
     - ``build_wsclean_no_dde_command``, ``build_wsclean_facets_command``,
       ``build_wsclean_screens_command``, ``image/wsclean.py``
     - ``wsclean_image``
   * - ``wsclean_mpi_image_*`` (3 steps) and ``mpi_runner.sh``
     - ``wsclean-mp``
     - ``build_wsclean_mpi_*_command``, ``build_mpi_wsclean_launch_command``
     - ``wsclean_image``
   * - ``wsclean_restore``
     - WSClean
     - ``build_wsclean_restore_command``
     - ``finish_wsclean_images``
   * - ``check_image_beam``
     - script
     - ``image/beam.py``
     - ``finish_wsclean_images``
   * - ``make_residual_data`` †
     - DP3
     - ``build_make_residual_visibilities_command``,
       ``image/residual_visibilities.py``
     - ``make_residual_visibilities``
   * - ``filter_skymodel``
     - script
     - ``build_filter_skymodel_command``, ``image/skymodel_filter.py``
     - ``filter_skymodel``
   * - ``calculate_image_diagnostics``
     - script
     - ``image/diagnostic_calculation.py``, ``image/diagnostics.py``
     - ``calculate_image_diagnostics``
   * - ``correct_astrometry`` †
     - script
     - ``image/astrometry.py``
     - ``finalize``
   * - ``make_image_cube``, ``make_catalog_from_image_cube``
     - scripts
     - ``image/cubes.py``, ``build_make_catalog_from_image_cube_command``
     - ``make_image_cube``, ``make_catalog_from_image_cube``
   * - ``normalize_flux_scale``
     - script
     - ``image/flux_normalization.py``
     - ``normalize_flux_scale``
   * - ``make_skymodel_image``
     - ``restore_skymodel.py``
     - ``image/restoration.py``
     - ``restore_skymodel``
   * - ``compress_sector_images``
     - ``fpack``
     - ``build_compress_sector_images_command``
     - ``compress_images``
   * - ``make_mosaic_template``, ``regrid_image``, ``make_mosaic``
     - scripts
     - ``mosaic/images.py``
     - ``make_mosaic_template``, ``mosaic_<image type>``
   * - ``compress_mosaic_image``
     - ``fpack``
     - ``mosaic/commands.py`` (``build_compress_mosaic_command``)
     - ``mosaic_<image type>``
   * - —
     - WSClean
     - ``build_draw_model_mosaic_command``, ``mosaic/model_rendering.py``
     - New: mosaics of the model images (:term:`model_mosaic_method`).
   * - ``merge_array_directories``, ``merge_array_files``, ``pick_file``,
       ``select_file``, ``writable_dir``
     - CWL expressions, ``touch``
     - —
     - Not needed: Python lists and paths take their place.

.. _master_moved:

Master has moved since the branch was made
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the common ancestor, ``master`` has gained 64 commits that change 176
files (+15,605 / −4,808 lines). Nearly all of those files are also changed by
the branch, so expect conflicts when ``master`` is merged in. Most of the
changes have already been carried over; the port audit in ``PLAN.md`` tracks
the rest. The files below were *added* to ``master`` and are unknown to the
branch, so a plain merge keeps them. Each row names the branch's counterpart.

.. rst-class:: wrap-code

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Added on ``master``
     - On the branch
   * - ``rapthor/scripts/wsclean_predict.py``, ``steps/wsclean_predict.cwl``,
       ``steps/wsclean_predict_readpatches.cwl``
     - ``rapthor/execution/calibrate/prediction.py`` and
       ``build_wsclean_predict_command``
   * - ``rapthor/scripts/correct_astrometry.py``,
       ``steps/correct_astrometry.cwl``, ``tests/scripts/test_correct_astrometry.py``
     - ``rapthor/execution/image/astrometry.py``,
       ``tests/execution/test_image_astrometry.py``
   * - ``steps/make_residual_data.cwl``
     - ``build_make_residual_visibilities_command``
   * - ``rapthor/scripts/fetch_skymodel.py``, ``pipeline/execution/fetch_model.cwl``
     - Nothing (the top-level CWL workflow was removed)
   * - ``rapthor/testing.py``, ``tests/test_testing.py``
     - ``tests/conftest.py``
   * - ``rapthor/lib/calibration.py`` (``resolve_calibration_strategy``,
       ``resolve_calibration_solves``), ``tests/lib/test_calibration.py``
     - The branch keeps ``calibration.py`` for the solve-type metadata only;
       strategy resolution is in ``rapthor/lib/strategy.py``, tested in
       ``tests/lib/test_strategy.py``
   * - ``tests/integration/test_image_only_applycal.py``,
       ``test_legacy_calibration_strategy.py``, ``test_sector_diagnostics.py``,
       ``test_wsclean_parallel_gridding.py``, ``.test_durations``
     - ``tests/integration/test_focused_scenarios.py`` covers image-only runs
       that apply supplied or earlier solutions. The legacy strategy flags,
       sector diagnostics and parallel gridding scenarios have no integration
       test on the branch
   * - ``tests/resources``: ``failed_workflow_sample.log``,
       ``integration_field_solutions.h5``, ``integration_normalization_*.txt``,
       ``manual_testing.parset``, ``manual_testing_strategy.py``,
       ``test_image_from_reg.fits``, ``test_image_regions_rendered.fits``
     - Not used by the branch's tests
   * - ``.gitlab-ci.common.yml``, ``.gitlab-ci.astron.yml``,
       ``.gitlab-ci.ska.yml``, ``ci/ubuntu_26_04-base``, ``ci/ubuntu_26_04-rapthor``,
       ``Docker/extract_version_hashes.sh``, ``tests/scripts/test_extract_version_hashes.py``
     - The branch has a single ``.gitlab-ci.yml`` and the Ubuntu 24.04 images
   * - ``CALIBRATION_STRATEGY.md``
     - ``.agents/scientific_glossary.md`` and :ref:`calibration_strategy_details`

The option ``wsclean_predict_beam_interval`` was also added to ``master`` and
is not on the branch.


Size of the change
------------------

All counts are from ``git diff -M`` between the common ancestor and the
branch. A file is *renamed* when ``git`` finds at least half of its lines in
a file with another name. Lines are counted in files of the following kinds:

.. rst-class:: wrap-code

.. list-table::
   :header-rows: 1
   :widths: 24 15 10 10 10 31

   * - Category
     - Files added / removed / modified / renamed
     - Added
     - Removed
     - Net
     - Notes
   * - Python: tests
     - 61 / 27 / 30 / 2
     - 29,961
     - 6,087
     - +23,874
     - ``tests/execution`` is new; ``tests/cwl`` and ``tests/scripts`` are
       gone
   * - Python: package
     - 90 / 24 / 15 / 4
     - 22,722
     - 7,975
     - +14,747
     - ``rapthor/execution`` is new; ``rapthor/scripts`` and the CWL runner
       code are gone
   * - CWL workflows and steps
     - 0 / 60 / 0 / 0
     - 0
     - 11,178
     - −11,178
     - All of ``rapthor/pipeline``
   * - Test resources
     - 5 / 3 / 4 / 0
     - 2,149
     - 117
     - +2,032
     - Reference commands, outputs and feature matrix in
       ``tests/execution/fixtures``
   * - User documentation
     - 3 / 3 / 13 / 0
     - 1,904
     - 311
     - +1,593
     - New ``architecture``, ADR and ``upgrading`` pages; the flowchart image
       was replaced by a Mermaid diagram
   * - Developer notes
     - 4 / 0 / 2 / 0
     - 641
     - 7
     - +634
     - ``AGENTS.md``, ``PLAN.md``, ``TESTING.md``, ``.agents/``
   * - Scripts in ``bin``, ``sbin`` and ``mpi_runner.sh``
     - 0 / 5 / 0 / 0
     - 0
     - 675
     - −675
     - Entry points moved into the package
   * - Examples
     - 3 / 0 / 5 / 0
     - 368
     - 108
     - +260
     - ``prefect_demo.parset`` and its strategy
   * - Parset defaults
     - 0 / 1 / 2 / 0
     - 145
     - 372
     - −227
     - ``defaults_skalow.parset`` removed; new ``[cluster]`` options
   * - Build, CI and containers
     - 1 / 3 / 6 / 1
     - 327
     - 477
     - −150
     - ``pyproject.toml``, ``.gitlab-ci.yml``, ``ci/``, ``Docker/``,
       ``.devcontainer/``
   * - Debug helpers
     - 0 / 0 / 3 / 0
     - 9
     - 7
     - +2
     -
   * - **Total**
     - **167 / 126 / 80 / 7**
     - **58,226**
     - **27,314**
     - **+30,912**
     -

.. mermaid::
   :caption: Thousands of lines added (blue) and removed (grey) per category. Both bars start at zero; the grey bar is drawn in front.

   ---
   config:
     xyChart:
       width: 760
       height: 420
     themeVariables:
       xyChart:
         plotColorPalette: "#1168bd, #999999"
   ---
   xychart-beta horizontal
       x-axis ["Tests", "Package", "CWL", "Test resources", "User docs", "Dev notes", "bin and sbin", "Examples", "Defaults", "Build and CI"]
       y-axis "Thousands of lines" 0 --> 30
       bar [30.0, 22.7, 0, 2.1, 1.9, 0.6, 0, 0.4, 0.1, 0.3]
       bar [6.1, 8.0, 11.2, 0.1, 0.3, 0, 0.7, 0.1, 0.4, 0.5]

Python lines by kind
~~~~~~~~~~~~~~~~~~~~

Each line of a Python file was classed as code, docstring, comment or blank,
using the Python parser, in both the old and the new version of the file.

.. list-table::
   :header-rows: 1
   :widths: 22 13 13 13 13 13 13

   * - Kind
     - Package: added
     - Package: removed
     - Package: net
     - Tests: added
     - Tests: removed
     - Tests: net
   * - Code
     - 18,492
     - 5,075
     - +13,417
     - 25,248
     - 3,975
     - +21,273
   * - Docstrings
     - 1,480
     - 1,517
     - −37
     - 265
     - 734
     - −469
   * - Comments
     - 313
     - 681
     - −368
     - 68
     - 518
     - −450
   * - Blank
     - 2,437
     - 696
     - +1,741
     - 4,380
     - 859
     - +3,521

The package has more than twice the code it had, with the same number of
docstring lines: the ratio of docstring to code lines fell from 0.41 at the
ancestor to 0.18 on the branch (0.13 inside ``rapthor/execution``).

Size at each revision
~~~~~~~~~~~~~~~~~~~~~

Code lines only (docstrings, comments and blank lines excluded), by area.
CWL, parset and RST files are counted whole.

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Area
     - Ancestor
     - ``master``
     - Branch
   * - ``rapthor/lib``
     - 3,812
     - 4,728
     - 4,520
   * - ``rapthor/operations``
     - 1,627
     - 2,338
     - 2,769
   * - ``rapthor/execution``
     - —
     - —
     - 14,460
   * - ``rapthor/scripts``
     - 3,028
     - 4,193
     - —
   * - ``rapthor`` top level (``process``, ``cli``, ``modifystate``, ``_logging``, ``testing``)
     - 484
     - 650
     - 228
   * - ``bin``, ``sbin`` (Python)
     - 322
     - 322
     - —
   * - **Package Python code**
     - **9,273**
     - **12,231**
     - **21,977**
   * - Package docstring lines
     - 3,812
     - 4,249
     - 3,900
   * - Package comment lines
     - 1,478
     - 1,592
     - 1,149
   * - CWL files (``rapthor/pipeline``)
     - 11,177 (60 files)
     - 12,368 (65 files)
     - —
   * - Parset defaults (``rapthor/settings``)
     - 884
     - 916
     - 657
   * - **Test Python code**
     - **7,279**
     - **11,030**
     - **28,398**
   * - Documentation (RST and ``conf.py``)
     - 2,122
     - 2,296
     - 3,713

Tests
~~~~~

The number of test functions, by directory. The branch has twice the tests
of ``master``; most of the new ones are in ``tests/execution``, which tests
the flows, payloads, command builders and runtime without running DP3 or
WSClean.

.. rst-class:: wrap-code

.. list-table::
   :header-rows: 1
   :widths: 22 12 12 12 42

   * - Directory
     - Ancestor
     - ``master``
     - Branch
     - What it covers on the branch
   * - ``tests/lib``
     - 177
     - 199
     - 205
     - The run state, parset and strategy; new files for parset option
       behaviour and coverage, parset paths, output records and solution
       cycles. ``test_cwl.py`` and ``test_cwlrunner.py`` are gone.
   * - ``tests/operations``
     - 63
     - 96
     - 125
     - The operation classes: payload construction and finalizers.
   * - ``tests/scripts``
     - 110
     - 128
     - —
     - Moved to ``tests/execution`` (two files recorded as renames).
   * - ``tests/cwl``
     - 35
     - 35
     - —
     - Removed with CWL: a mock that generated outputs from the CWL
       definitions, and tests that ran small workflows with ``cwltool``.
   * - ``tests/execution``
     - —
     - —
     - 647
     - One ``test_<operation>_flow.py`` per flow, the pipeline flow, command
       builders, payload serialization across the Dask boundary, the runtime
       (bootstrap, task runner, shell, scratch, Slurm, artifacts), and the
       step modules that were scripts. ``fixtures/command_reference.json``
       records the expected command line of every step.
   * - ``tests/architecture``
     - —
     - —
     - 4
     - Import rules: ``rapthor/lib`` must not import Prefect, Dask or
       ``rapthor.execution``; pure execution modules must not import the
       frameworks; retired script wrappers must not be used.
   * - ``tests/integration``
     - 12
     - 29
     - 35
     - Runs on real data with DP3 and WSClean; new focused scenarios and a
       Slurm execution test.
   * - ``tests`` (top level)
     - 7
     - 17
     - 14
     - ``test_cli.py`` replaces ``test_process.py``; ``test_modifystate.py``
       and ``test_integration_defaults.py``.
   * - **Total**
     - **404**
     - **504**
     - **1,030**
     -

Tests that start a Prefect test server carry the ``prefect`` marker and run
serially; the rest run in parallel (see ``TESTING.md``).


.. _options_dependencies_tooling:

Options, dependencies and tooling
---------------------------------

Parset options
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * -
     - Options
   * - Removed
     - ``cwl_runner``; ``dir_local`` (deprecated, use ``local_scratch_dir``)
   * - Changed
     - ``use_container`` and ``container_type`` are still parsed but a run
       with ``use_container = True`` stops with an error. ``batch_system``
       keeps the values ``single_machine``, ``slurm`` and ``slurm_static``,
       but with ``slurm`` Rapthor no longer submits jobs: it connects to Dask
       workers started in your allocation. ``max_nodes``, ``cpus_per_task``
       and ``mem_per_node_gb`` keep their meaning for chunking and resource
       limits.
   * - Added to ``[cluster]``
     - ``local_dask_workers``, ``dask_scheduler``, ``dask_dashboard_address``,
       ``prefect_task_runner``, ``prefect_api_url``, ``prefect_api_mode``,
       ``prefect_run_tags``, ``prefect_retries``, ``prefect_log_commands``,
       ``prefect_stream_output``, ``prefect_command_profile``,
       ``prefect_publish_fits_previews``,
       ``prefect_publish_postage_stamp_previews``,
       ``prefect_postage_stamp_preview_count``,
       ``prefect_postage_stamp_preview_size_px``,
       ``prefect_fits_preview_clip_percentile``, ``filter_skymodel_ncores``
   * - Added to ``[imaging]``
     - ``model_mosaic_method``
   * - On ``master`` only
     - ``wsclean_predict_beam_interval`` (added after the ancestor)

See :ref:`rapthor_parset` for the descriptions and :ref:`upgrading` for the
changes to the calibration strategy options.

Dependencies and packaging
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * -
     - ``pyproject.toml``
   * - Removed
     - ``toil[cwl]``, ``cwltool``, ``streamflow``, ``jinja2``, ``pyyaml``;
       ``requirements.txt``
   * - Added
     - ``prefect[dask,shell]`` 3.4.11 to 3.x, ``prefect-dask``,
       ``prefect-shell``, ``dask[distributed]``, ``bokeh``, ``fastapi``
       (pinned below 0.137), ``h5py``, ``psutil``; ``lsmtool`` from PyPI
       (1.9.0 or later) instead of the git repository; a ``docs`` dependency
       group with Sphinx and ``sphinxcontrib-mermaid``
   * - Entry points
     - ``[project.scripts]`` defines ``rapthor`` and ``concat_linc_files``.
       The ``script-files`` list of 22 scripts and the explicit ``packages``
       list are replaced by ``packages.find``. ``plotrapthor`` is gone.
   * - Tests
     - A ``prefect`` marker; the ``tox`` test environment runs the marked
       tests serially before the rest.
   * - CI and containers
     - ``ci/ubuntu_22_04-*`` became ``ci/ubuntu_24_04-*``, and the dev
       container builds from ``ci/ubuntu_24_04-base`` instead of its own
       ``Dockerfile``. ``Docker/Dockerfile`` was updated for the new
       dependencies (47 lines changed).


.. _cwl_architecture:

Architecture of the CWL version
-------------------------------

The diagrams below describe ``master`` in the same way as :ref:`architecture`
describes the branch, so that each level can be compared with its
counterpart. The system context (the astronomer, LINC, DP3, WSClean and the
survey catalogues) is the same in both versions and is not repeated.

Containers
~~~~~~~~~~

.. mermaid::
   :caption: Container diagram for a run of the CWL version (C4 level 2)

   %%{init: {"flowchart": {"nodeSpacing": 35, "rankSpacing": 45, "wrappingWidth": 340}}}%%
   flowchart TB
       astronomer["<b>Astronomer</b><br/>[Person]"]

       subgraph rapthor["Rapthor [Software system]"]
           cli["<b>rapthor command</b><br/>[Container: Python]<br/><br/>Reads the parset and<br/>strategy. Writes a CWL<br/>workflow and its inputs<br/>for each operation"]
           runner["<b>CWL runner</b><br/>[Container: Toil, cwltool<br/>or StreamFlow]<br/><br/>One process per operation.<br/>Runs the steps, locally<br/>or as Slurm jobs"]
           jobstore[("<b>Job store</b><br/>[Container: disk]<br/><br/>Toil's state of the<br/>running workflow")]
           steps["<b>CWL steps</b><br/>[Container: processes]<br/><br/>DP3, WSClean and Rapthor<br/>scripts, optionally in a<br/>container image"]
           workdir[("<b>Working directory</b><br/>[Container: disk]<br/><br/>Products, logs and<br/>restart records")]
           scratch[("<b>Scratch directories</b><br/>[Container: disk]<br/><br/>Temporary and<br/>intermediate files")]
       end

       tools["<b>DP3 and WSClean</b><br/>[Software systems]<br/><br/>Run as commands"]
       surveys["<b>Survey catalogues</b><br/>[Software systems]"]
       ms[("<b>Input<br/>Measurement Sets</b><br/>[Disk]")]

       astronomer -- "Runs" --> cli
       cli -- "Downloads initial<br/>sky model from" --> surveys
       cli -- "Starts, one<br/>per operation" --> runner
       runner -- "Returns<br/>outputs to" --> cli
       runner -- "Records<br/>state in" --> jobstore
       runner -- "Starts" --> steps
       steps -- "Run" --> tools
       steps -- "Download comparison<br/>catalogues from" --> surveys
       tools -- "Read and<br/>write" --> workdir
       tools -- "Write to" --> scratch
       tools -- "Read" --> ms

       classDef person fill:#08427b,stroke:#052e56,color:#ffffff
       classDef system fill:#1168bd,stroke:#0b4884,color:#ffffff
       classDef container fill:#438dd5,stroke:#2e6295,color:#ffffff
       classDef component fill:#85bbf0,stroke:#5d82a8,color:#000000
       classDef external fill:#999999,stroke:#6b6b6b,color:#ffffff
       classDef boundary fill:#ffffff,stroke:#444444,stroke-dasharray:5 5,color:#444444
       class astronomer person
       class cli,runner,jobstore,steps,workdir,scratch container
       class tools,surveys,ms external
       class rapthor boundary

Compared with the Prefect version: the CWL runner and its job store are
replaced by Prefect flows inside the ``rapthor`` command, with the Prefect
server for state and the Dask scheduler and workers for running the tasks.
The steps were separate processes started by the runner for every command;
the tasks now run inside long-lived worker processes, which start DP3 and
WSClean.

Components
~~~~~~~~~~

.. mermaid::
   :caption: Component diagram for the CWL version (C4 level 3)

   %%{init: {"flowchart": {"nodeSpacing": 35, "rankSpacing": 45, "wrappingWidth": 340}}}%%
   flowchart TB
       subgraph pkg["Rapthor Python package and workflow files"]
           cli["<b>Command line</b><br/>[Component: bin/rapthor]<br/><br/>Reads the options and<br/>starts or resets a run"]
           process["<b>Run loop</b><br/>[Component: rapthor.process]<br/><br/>Runs the operations of<br/>each cycle and checks<br/>selfcal convergence"]
           operations["<b>Operations</b><br/>[Component: rapthor.operations]<br/><br/>Fill the workflow template<br/>and inputs, and copy<br/>products into place"]
           lib["<b>Run state</b><br/>[Component: rapthor.lib]<br/><br/>Parset, strategy, Field,<br/>Observation and Sector"]
           opbase["<b>Operation base</b><br/>[Component: rapthor.lib.operation,<br/>cwl, cwlrunner,<br/>toil_batch_systems]<br/><br/>Renders the template, starts<br/>the runner, reads outputs"]
           templates["<b>Workflow templates</b><br/>[Component:<br/>rapthor/pipeline/parsets]<br/><br/>Jinja2 CWL workflows,<br/>one per operation"]
           stepdefs["<b>Step definitions</b><br/>[Component:<br/>rapthor/pipeline/steps]<br/><br/>CWL CommandLineTools,<br/>one per command"]
           scripts["<b>Scripts</b><br/>[Component: rapthor/scripts]<br/><br/>Python steps installed<br/>as commands"]
       end

       runner["<b>CWL runner</b><br/>[Container]"]
       tools["<b>DP3 and WSClean</b><br/>[Software systems]"]
       workdir[("<b>Working directory</b><br/>[Container: disk]")]

       cli -- "Runs" --> process
       process -- "Runs in order" --> operations
       process -- "Builds and updates" --> lib
       operations -- "Read and update" --> lib
       operations -- "Extend" --> opbase
       operations -- "Copy products to" --> workdir
       opbase -- "Renders" --> templates
       opbase -- "Starts" --> runner
       templates -- "Reference" --> stepdefs
       runner -- "Runs" --> stepdefs
       stepdefs -- "Call" --> scripts
       stepdefs -- "Call" --> tools

       classDef person fill:#08427b,stroke:#052e56,color:#ffffff
       classDef system fill:#1168bd,stroke:#0b4884,color:#ffffff
       classDef container fill:#438dd5,stroke:#2e6295,color:#ffffff
       classDef component fill:#85bbf0,stroke:#5d82a8,color:#000000
       classDef external fill:#999999,stroke:#6b6b6b,color:#ffffff
       classDef boundary fill:#ffffff,stroke:#444444,stroke-dasharray:5 5,color:#444444
       class cli,process,operations,lib,opbase,templates,stepdefs,scripts component
       class runner,workdir container
       class tools external
       class pkg boundary

Compared with the Prefect version: the run loop became the pipeline flow;
the workflow templates, step definitions and scripts became the operation
flows (``rapthor.execution.<operation>``); and the operation base with the
runner wrappers became the runtime (``rapthor.execution``). The command line,
the operations and the run state kept their places.

Deployment on a single machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With :term:`batch_system` = ``single_machine``, Toil runs the steps as local
processes. The recommended way to run was to start Rapthor inside the Rapthor
Docker or Singularity image, with ``use_container`` off.

.. mermaid::
   :caption: Deployment diagram for a run of the CWL version on a single machine

   %%{init: {"flowchart": {"nodeSpacing": 35, "rankSpacing": 45, "wrappingWidth": 340}}}%%
   flowchart TB
       subgraph machine["One machine"]
           direction TB
           cli["<b>rapthor command</b><br/>[Container: Python]"]
           runner["<b>toil-cwl-runner</b><br/>[Container: Toil]<br/><br/>Started by Rapthor for<br/>each operation"]
           steps["<b>CWL steps</b><br/>[Container: processes]<br/><br/>Started by Toil for<br/>each command"]
           tools["<b>DP3 and WSClean</b><br/>[Software systems]<br/><br/>Use the cores and threads<br/>allowed by the parset"]
           disk[("<b>Disk</b><br/>[dir_working, job stores,<br/>input data and scratch]")]
       end

       cli -- "Starts" --> runner
       runner -- "Starts" --> steps
       steps -- "Run" --> tools
       tools -- "Read and write" --> disk

       classDef person fill:#08427b,stroke:#052e56,color:#ffffff
       classDef system fill:#1168bd,stroke:#0b4884,color:#ffffff
       classDef container fill:#438dd5,stroke:#2e6295,color:#ffffff
       classDef component fill:#85bbf0,stroke:#5d82a8,color:#000000
       classDef external fill:#999999,stroke:#6b6b6b,color:#ffffff
       classDef boundary fill:#ffffff,stroke:#444444,stroke-dasharray:5 5,color:#444444
       class cli,runner,steps,disk container
       class tools external
       class machine boundary

Deployment on a cluster
~~~~~~~~~~~~~~~~~~~~~~~

With :term:`batch_system` = ``slurm``, the ``rapthor`` command and the Toil
runner ran on the login node, outside any container, and Toil submitted one
Slurm job per CWL step (with the cores and memory from ``cpus_per_task`` and
``mem_per_node_gb``). With ``use_container`` on, each job ran its command in
the Singularity or uDocker image. ``slurm_static`` used a Rapthor-specific
Toil batch system that ran the steps with ``srun`` inside an existing
reservation. WSClean with MPI was started through ``mpi_runner.sh``.

.. mermaid::
   :caption: Deployment diagram for a run of the CWL version on a Slurm cluster

   %%{init: {"flowchart": {"nodeSpacing": 35, "rankSpacing": 45, "wrappingWidth": 340}}}%%
   flowchart TB
       subgraph head["Login node"]
           cli["<b>rapthor command</b><br/>[Container: Python]"]
           runner["<b>toil-cwl-runner</b><br/>[Container: Toil]<br/><br/>Submits one Slurm job<br/>per step"]
       end
       subgraph nodes["Compute nodes"]
           job["<b>Slurm job</b><br/>[Container: process]<br/><br/>One CWL step, in the<br/>Rapthor image when<br/>use_container is set"]
           tools["<b>DP3 and WSClean</b><br/>[Software systems]<br/><br/>WSClean can span<br/>nodes with MPI"]
           local[("<b>Local scratch</b><br/>[dir_local or<br/>local_scratch_dir]")]
       end
       subgraph storage["Shared file system"]
           shared[("<b>Shared disk</b><br/>[dir_working, job stores,<br/>input data, global_scratch_dir]")]
       end

       cli -- "Starts" --> runner
       runner -- "Submits jobs<br/>with sbatch" --> job
       job -- "Runs" --> tools
       tools -- "Write temporary<br/>files to" --> local
       tools -- "Read and<br/>write" --> shared
       runner -- "Records<br/>state in" --> shared

       classDef person fill:#08427b,stroke:#052e56,color:#ffffff
       classDef system fill:#1168bd,stroke:#0b4884,color:#ffffff
       classDef container fill:#438dd5,stroke:#2e6295,color:#ffffff
       classDef component fill:#85bbf0,stroke:#5d82a8,color:#000000
       classDef external fill:#999999,stroke:#6b6b6b,color:#ffffff
       classDef boundary fill:#ffffff,stroke:#444444,stroke-dasharray:5 5,color:#444444
       class cli,runner,job,local,shared container
       class tools external
       class head,nodes,storage boundary

Compared with the Prefect version: Rapthor no longer talks to Slurm. You
start a Dask scheduler and one worker per node inside your allocation, and
Rapthor connects to the scheduler; the steps run in those workers instead of
in separate Slurm jobs. The containers are the other way round: on
``master`` the login-node process ran outside the image and each job inside
it; on the branch the whole run, workers included, is started inside the
image.


Suggested reading order
-----------------------

#. :ref:`upgrading` for the intentional differences in behaviour and
   products, then the tables on this page.
#. ``rapthor/lib``: mostly edits to known code. Start with ``operation.py``,
   ``records.py`` and ``parset_paths.py``, which are new or rewritten.
#. ``rapthor/operations``: compare ``calibrate/base.py`` and
   ``image/base.py`` with ``master``'s ``calibrate.py`` and ``image.py``.
   The ``set_input_parameters`` and ``finalize`` methods keep their shape;
   ``execute_workflow`` replaces the CWL runner call.
#. ``rapthor/execution/<operation>``: read ``flow.py`` from the top, then
   check each function in ``commands.py`` against the CWL step of the same
   name using the :ref:`step table <cwl_steps_map>` and
   ``tests/execution/fixtures/command_reference.json``.
#. ``rapthor/execution`` runtime modules: ``config.py``,
   ``runtime_bootstrap.py``, ``task_runner.py``, ``shell.py``,
   ``workspace.py`` and ``scratch.py``, ``artifacts.py``.
#. Tests: ``tests/execution/test_<operation>_flow.py`` mirrors each flow;
   ``tests/architecture`` enforces the import rules of
   :ref:`architecture`.
