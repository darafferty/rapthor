.. _architecture:

Architecture
============

Rapthor is a Python program. It reads the parset and the processing strategy,
decides which operations to run in each cycle, and runs DP3 and WSClean to do
the calibration and imaging. Two Python libraries do the bookkeeping:

* `Prefect <https://docs.prefect.io/>`_ keeps track of what has run, what
  failed and what was logged, and provides a dashboard for following a run.
* `Dask <https://docs.dask.org/>`_ runs the work, on one machine or spread
  over the nodes of a cluster.

The diagrams on this page follow the `C4 model <https://c4model.com/>`_. They
start with Rapthor as a single box and zoom in one level at a time: the system
context, the containers (the separate programs and stores that make up a
running Rapthor), and the components (the parts of the Python package). The
last two diagrams show where everything runs on a single machine and on a
cluster.

In every diagram each box gives a name, the kind of thing it is in square
brackets, and what it does. Blue boxes belong to Rapthor and grey boxes are
software or data outside it. A cylinder is data on disk and an arrow reads in
the direction it points. A dashed outline groups the things inside it; in the
deployment diagrams each dashed outline is a machine or a group of machines.

See :ref:`structure` for the order in which the operations are run and
:ref:`operations` for what each operation produces. The decision to run the
processing with Prefect and Dask instead of CWL and Toil, and the reasons for
it, are recorded in :doc:`adr_replace_cwl_toil_with_prefect_dask`. The
corresponding diagrams of the CWL version, and a map from its code to the
code described here, are in :doc:`cwl_to_prefect_comparison`.


System context
--------------

The system context shows Rapthor, the person who uses it, and the other
software it depends on.

.. mermaid::
   :caption: System context diagram for Rapthor (C4 level 1)

   %%{init: {"flowchart": {"nodeSpacing": 35, "rankSpacing": 45, "wrappingWidth": 340}}}%%
   flowchart TB
       astronomer["<b>Astronomer</b><br/>[Person]<br/><br/>Writes the parset and<br/>strategy, starts the run<br/>and checks the results"]
       linc["<b>Initial calibration pipeline</b><br/>[Software system]<br/><br/>LINC for LOFAR. Makes the<br/>calibrated Measurement Sets<br/>that Rapthor starts from"]
       rapthor["<b>Rapthor</b><br/>[Software system]<br/><br/>Self-calibrates and images<br/>LOFAR HBA and SKA-Low data,<br/>correcting for direction-<br/>dependent effects"]
       dp3["<b>DP3</b><br/>[Software system]<br/><br/>Calibration solves,<br/>prediction, subtraction<br/>and applying solutions"]
       wsclean["<b>WSClean</b><br/>[Software system]<br/><br/>Imaging and deconvolution,<br/>with IDG for gridding and<br/>EveryBeam for the beam"]
       surveys["<b>Survey catalogues</b><br/>[Software systems]<br/><br/>TGSS, LoTSS, GSM, NVSS,<br/>VLSSr, WENSS, Pan-STARRS,<br/>queried over the internet"]

       astronomer -- "Runs and monitors" --> rapthor
       linc -- "Supplies input<br/>Measurement Sets" --> rapthor
       rapthor -- "Runs" --> dp3
       rapthor -- "Runs" --> wsclean
       rapthor -- "Downloads sky<br/>models from" --> surveys

       classDef person fill:#08427b,stroke:#052e56,color:#ffffff
       classDef system fill:#1168bd,stroke:#0b4884,color:#ffffff
       classDef container fill:#438dd5,stroke:#2e6295,color:#ffffff
       classDef component fill:#85bbf0,stroke:#5d82a8,color:#000000
       classDef external fill:#999999,stroke:#6b6b6b,color:#ffffff
       classDef boundary fill:#ffffff,stroke:#444444,stroke-dasharray:5 5,color:#444444
       class astronomer person
       class rapthor system
       class linc,dp3,wsclean,surveys external

Rapthor does not modify the input Measurement Sets. It downloads from the
survey catalogues only when an initial sky model, a flux-scale comparison or an
astrometry comparison is needed and no file has been supplied in the parset;
downloads can be switched off with :term:`allow_internet_access`.

Rapthor also uses PyBDSF (source finding), LSMTool (sky models) and LoSoTo
(calibration solution plots). These are Python libraries that are installed
with Rapthor and run inside it, so they are not shown as separate systems.


Containers
----------

In the C4 model a *container* is something that runs or stores data on its
own: a program, a service or a directory. It has nothing to do with Docker or
Singularity containers. The diagram below shows the containers that make up
one Rapthor run.

.. mermaid::
   :caption: Container diagram for a Rapthor run (C4 level 2)

   %%{init: {"flowchart": {"nodeSpacing": 35, "rankSpacing": 45, "wrappingWidth": 340}}}%%
   flowchart TB
       astronomer["<b>Astronomer</b><br/>[Person]"]

       subgraph rapthor["Rapthor [Software system]"]
           cli["<b>rapthor command</b><br/>[Container: Python]<br/><br/>Reads the parset and<br/>strategy and runs the<br/>operations of each cycle"]
           prefect["<b>Prefect server</b><br/>[Container: Prefect]<br/><br/>Records state, logs<br/>and artifacts.<br/>Has a dashboard"]
           scheduler["<b>Dask scheduler</b><br/>[Container: Dask]<br/><br/>Hands tasks to<br/>the workers.<br/>Has a dashboard"]
           workers["<b>Dask workers</b><br/>[Container: Python]<br/><br/>Run the tasks of each<br/>operation, one at a time"]
           workdir[("<b>Working directory</b><br/>[Container: disk]<br/><br/>Products, logs and<br/>restart records")]
           scratch[("<b>Scratch directories</b><br/>[Container: disk]<br/><br/>Temporary and<br/>intermediate files")]
       end

       tools["<b>DP3 and WSClean</b><br/>[Software systems]<br/><br/>Run as commands"]
       surveys["<b>Survey catalogues</b><br/>[Software systems]"]
       ms[("<b>Input<br/>Measurement Sets</b><br/>[Disk]")]

       astronomer -- "Runs" --> cli
       astronomer -- "Follows<br/>the run in" --> prefect
       cli -- "Reports to" --> prefect
       cli -- "Downloads<br/>from" --> surveys
       cli -- "Submits<br/>tasks to" --> scheduler
       scheduler -- "Assigns<br/>tasks to" --> workers
       workers -- "Start" --> tools
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
       class cli,scheduler,workers,prefect,workdir,scratch container
       class tools,surveys,ms external
       class rapthor boundary

The ``rapthor`` command
    The process you start. It holds the state of the run (the observations,
    the sky model, the imaging sectors, the calibration patches and the current
    cycle) and decides what happens next. It does little heavy processing
    itself. When an operation finishes, it copies the products from
    ``dir_working/pipelines`` to their final place in the working directory.

Prefect server
    Stores what happened in the run and serves the Prefect dashboard. The
    ``rapthor`` command and the workers both report to it. If you do nothing,
    Prefect starts a temporary server that exists only while Rapthor is
    running. To keep the history and use the dashboard, start a server
    yourself and tell Rapthor where it is (see
    :ref:`persistent_prefect_dashboard`).

Dask scheduler and workers
    The scheduler hands out tasks and the workers run them. By default Rapthor
    starts a scheduler and workers on the machine it runs on and stops them
    when it exits. On a cluster you start the scheduler and one worker per
    node yourself and give Rapthor the scheduler address (see
    :ref:`running_on_cluster`). Each worker runs one task at a time; the
    task starts DP3 or WSClean, which then use as many threads as the parset
    allows.

Working directory
    The directory set by :term:`dir_working`. All the workers and the
    ``rapthor`` command must be able to read and write it, so on a cluster it
    has to be on a shared disk. Its contents are described in :ref:`products`.

Scratch directories
    Optional. :term:`local_scratch_dir` is a fast disk on each node for
    temporary files; :term:`global_scratch_dir` is a shared disk for
    intermediate files that pass between tasks.

Survey catalogues
    The ``rapthor`` command downloads the initial sky model, and the workers
    download the catalogues used for the flux-scale and astrometry
    comparisons. On a cluster the nodes therefore need internet access, unless
    the sky models are supplied in the parset.


Components
----------

The component diagram zooms in on the ``rapthor`` command and the workers,
which run the same Python package. Each box is a part of the package. To keep
the diagram readable, two relationships are left out: the command line reads
the parset through the run state, and it uses the runtime to start or connect
to Prefect and Dask before the pipeline flow begins.

.. mermaid::
   :caption: Component diagram for the Rapthor Python package (C4 level 3)

   %%{init: {"flowchart": {"nodeSpacing": 35, "rankSpacing": 45, "wrappingWidth": 340}}}%%
   flowchart TB
       subgraph pkg["Rapthor Python package"]
           cli["<b>Command line</b><br/>[Component: rapthor.cli]<br/><br/>Reads the options and<br/>starts or resets a run"]
           pipeline["<b>Pipeline flow</b><br/>[Component:<br/>rapthor.execution.pipeline]<br/><br/>Runs the operations of<br/>each cycle and checks<br/>selfcal convergence"]
           operations["<b>Operations</b><br/>[Component: rapthor.operations]<br/><br/>Calibrate, Predict, Image,<br/>Mosaic and Concatenate.<br/>Prepare inputs and copy<br/>products into place"]
           lib["<b>Run state</b><br/>[Component: rapthor.lib]<br/><br/>Parset, strategy, Field,<br/>Observation and Sector"]
           flows["<b>Operation flows</b><br/>[Component:<br/>rapthor.execution.calibrate,<br/>predict, image, ...]<br/><br/>Split the work into tasks<br/>and build the commands"]
           runtime["<b>Runtime</b><br/>[Component: rapthor.execution]<br/><br/>Connects to Prefect and<br/>Dask, runs and logs<br/>commands"]
       end

       prefect["<b>Prefect server</b><br/>[Container]"]
       dask["<b>Dask scheduler<br/>and workers</b><br/>[Containers]"]
       tools["<b>DP3 and WSClean</b><br/>[Software systems]"]
       workdir[("<b>Working directory</b><br/>[Container: disk]")]

       cli -- "Runs" --> pipeline
       pipeline -- "Runs in order" --> operations
       pipeline -- "Builds and updates" --> lib
       operations -- "Read and update" --> lib
       operations -- "Hand inputs to" --> flows
       operations -- "Copy products to" --> workdir
       flows -- "Run tasks with" --> runtime
       runtime -- "Reports to" --> prefect
       runtime -- "Submits tasks to" --> dask
       runtime -- "Starts" --> tools

       classDef person fill:#08427b,stroke:#052e56,color:#ffffff
       classDef system fill:#1168bd,stroke:#0b4884,color:#ffffff
       classDef container fill:#438dd5,stroke:#2e6295,color:#ffffff
       classDef component fill:#85bbf0,stroke:#5d82a8,color:#000000
       classDef external fill:#999999,stroke:#6b6b6b,color:#ffffff
       classDef boundary fill:#ffffff,stroke:#444444,stroke-dasharray:5 5,color:#444444
       class cli,pipeline,operations,lib,flows,runtime component
       class prefect,dask,workdir container
       class tools external
       class pkg boundary

Three points explain most of the layout:

* **The run state stays in the ``rapthor`` command.** The ``Field`` object and
  its observations and sectors never leave that process. An operation hands
  its flow a plain description of the work (file names, numbers and option
  values), and that is all a worker ever sees.
* **An operation is the unit of restart.** Each operation writes its inputs,
  its outputs and a ``.done`` marker under ``dir_working/pipelines``. When a
  run is resumed, finished operations are skipped and their recorded outputs
  are loaded instead (see :ref:`resuming_rapthor`).
* **DP3 and WSClean are run as ordinary commands.** Every command line is
  built by a small function in the operation's package, so the exact command
  that was run can be read back from ``dir_working/logs/commands.jsonl``.


.. _architecture_tasks:

Operations, flows and tasks
---------------------------

Each operation runs one Prefect *flow*, and each flow is split into *tasks*.
The flow and task names below are the ones shown in the Prefect dashboard. A
task that runs an external command also writes its output to
``dir_working/logs/<operation>/<task>.log``.

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Operation
     - Flow run name
     - Main tasks
   * - Concatenate
     - ``concatenate_1``
     - ``concatenate_epoch_N`` joins the frequency bands of one epoch.
   * - Calibrate
     - ``calibrate_dd_X``, ``calibrate_di_X``
     - ``solve_chunk_N`` runs DP3 on one time chunk. For each solve in the
       calibration strategy, ``collect_<solve>``, ``process_<solve>`` and
       ``plot_<solve>`` gather, smooth and plot the solutions.
       ``combine_h5parms`` and ``finalize_solutions`` make the final solution
       table. When the model is predicted from images (see
       :term:`use_image_based_predict` and :term:`use_wsclean_predict`),
       ``make_predict_region`` and then ``draw_model`` or
       ``wsclean_predict_chunk_N`` run first.
   * - Predict
     - ``predict_dd_X``, ``predict_di_X``
     - ``dp3_predict_chunk_N`` predicts the model visibilities and
       ``postprocess_N`` subtracts them from, or adds them to, the data.
   * - Image
     - ``image_X``
     - ``prepare_chunk_N`` and ``concatenate_visibilities`` prepare the
       visibilities. ``wsclean_image`` and ``finish_wsclean_images`` make the
       images. ``filter_skymodel`` and ``calculate_image_diagnostics`` make
       the sky model and the diagnostics. ``compress_images`` and
       ``finalize`` finish the sector. Depending on the parset and strategy,
       ``make_residual_visibilities``, ``make_image_cube``,
       ``normalize_flux_scale`` and ``restore_skymodel`` are added.
   * - Mosaic
     - ``mosaic_X``
     - ``make_mosaic_template`` and one ``mosaic_<image type>`` task per image
       type. Nothing is run when there is only one imaging sector.

``X`` is the cycle number and ``N`` counts the time chunks or epochs. The
image flow is also used for the initial image, where the flow run is named
``initial_image``, and for the flux-scale normalization image. The
normalization run is also named ``image_X`` in the dashboard, but its files
are kept under ``normalize_X`` in the working directory. When there is more
than one imaging sector, the image task names start with the sector name, for
example ``sector_2_wsclean_image``.

Tasks carry a tag that names the tool they run (``dp3``, ``wsclean``,
``pybdsf``, ``casacore``, ``fpack`` or ``python``), plus any tags set with
:term:`prefect_run_tags`, so runs and tasks can be filtered in the dashboard.


Deployment
----------

On a single machine
~~~~~~~~~~~~~~~~~~~

This is the default (:term:`batch_system` = ``single_machine``). Everything
runs on one machine, usually inside the Rapthor Docker or Singularity image
(see :ref:`using_containers`). Rapthor starts the Dask scheduler and workers
itself and stops them at the end of the run. The number of workers is set by
:term:`local_dask_workers`.

.. mermaid::
   :caption: Deployment diagram for a run on a single machine

   %%{init: {"flowchart": {"nodeSpacing": 35, "rankSpacing": 45, "wrappingWidth": 340}}}%%
   flowchart TB
       subgraph machine["One machine"]
           direction TB
           cli["<b>rapthor command</b><br/>[Container: Python]"]
           prefect["<b>Prefect server</b><br/>[Container: Prefect]<br/><br/>Temporary, unless you<br/>run your own"]
           scheduler["<b>Dask scheduler</b><br/>[Container: Dask]<br/><br/>Started and stopped<br/>by Rapthor"]
           workers["<b>Dask workers</b><br/>[Container: Python]<br/><br/>local_dask_workers<br/>processes"]
           tools["<b>DP3 and WSClean</b><br/>[Software systems]<br/><br/>Use the cores and threads<br/>allowed by the parset"]
           disk[("<b>Disk</b><br/>[dir_working, input data<br/>and scratch]")]
       end

       cli -- "Reports to" --> prefect
       cli -- "Submits tasks to" --> scheduler
       scheduler -- "Assigns tasks to" --> workers
       workers -- "Start" --> tools
       tools -- "Read and write" --> disk

       classDef person fill:#08427b,stroke:#052e56,color:#ffffff
       classDef system fill:#1168bd,stroke:#0b4884,color:#ffffff
       classDef container fill:#438dd5,stroke:#2e6295,color:#ffffff
       classDef component fill:#85bbf0,stroke:#5d82a8,color:#000000
       classDef external fill:#999999,stroke:#6b6b6b,color:#ffffff
       classDef boundary fill:#ffffff,stroke:#444444,stroke-dasharray:5 5,color:#444444
       class cli,prefect,scheduler,workers,disk container
       class tools external
       class machine boundary

By default there is one worker, so tasks run one after another and each DP3 or
WSClean command has the whole machine. Set :term:`local_dask_workers` to run
several tasks at the same time, for example to image several sectors at once,
and lower :term:`max_threads` to match so that the machine is not
oversubscribed.

On a cluster
~~~~~~~~~~~~

With :term:`batch_system` = ``slurm``, Rapthor uses a Dask scheduler and
workers that you start inside your Slurm allocation, with one worker on each
node (see :ref:`external_dask_runtime`). Rapthor does not submit Slurm jobs
itself.

.. mermaid::
   :caption: Deployment diagram for a run on several nodes of a cluster

   %%{init: {"flowchart": {"nodeSpacing": 35, "rankSpacing": 45, "wrappingWidth": 340}}}%%
   flowchart TB
       subgraph alloc["Slurm allocation"]
           subgraph head["Launch node"]
               cli["<b>rapthor command</b><br/>[Container: Python]"]
               scheduler["<b>Dask scheduler</b><br/>[Container: Dask]<br/><br/>Started by you"]
           end
           subgraph nodes["Each node"]
               worker["<b>Dask worker</b><br/>[Container: Python]<br/><br/>One per node,<br/>started by you"]
               tools["<b>DP3 and WSClean</b><br/>[Software systems]<br/><br/>WSClean can span<br/>nodes with MPI"]
               local[("<b>Local scratch</b><br/>[local_scratch_dir]")]
           end
       end
       subgraph service["Any reachable host"]
           prefect["<b>Prefect server</b><br/>[Container: Prefect]<br/><br/>Optional"]
       end
       subgraph storage["Shared file system"]
           shared[("<b>Shared disk</b><br/>[dir_working, input data,<br/>global_scratch_dir]")]
       end

       cli -- "Submits<br/>tasks to" --> scheduler
       cli -- "Reports to" --> prefect
       scheduler -- "Assigns<br/>tasks to" --> worker
       worker -- "Starts" --> tools
       tools -- "Write temporary<br/>files to" --> local
       tools -- "Read and<br/>write" --> shared

       classDef person fill:#08427b,stroke:#052e56,color:#ffffff
       classDef system fill:#1168bd,stroke:#0b4884,color:#ffffff
       classDef container fill:#438dd5,stroke:#2e6295,color:#ffffff
       classDef component fill:#85bbf0,stroke:#5d82a8,color:#000000
       classDef external fill:#999999,stroke:#6b6b6b,color:#ffffff
       classDef boundary fill:#ffffff,stroke:#444444,stroke-dasharray:5 5,color:#444444
       class cli,scheduler,worker,local,shared,prefect container
       class tools external
       class alloc,head,nodes,storage,service boundary

Each observation is split in time into chunks, about one per node when all
of the data are used, so that calibration and prediction can run on all the
nodes at once. When :term:`use_mpi` is set, WSClean is started with ``mpirun``
and spreads the imaging of one sector over several nodes.

If no Prefect server is given, each run uses its own temporary one, so
several independent Rapthor runs can share a cluster without interfering with
each other.


For contributors
----------------

The table below shows where a change belongs. The aim is that the run state,
the scientific decisions and the mechanics of running commands stay separate,
so that each can be read and tested on its own.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Location
     - What belongs there
   * - ``rapthor/lib``
     - The parset, the strategy and the ``Field``, ``Observation`` and
       ``Sector`` classes. No Prefect or Dask code.
   * - ``rapthor/operations``
     - One class per operation. It turns the run state into the inputs of the
       flow, and afterwards copies the products into place and updates the
       ``Field``. Keep these classes thin.
   * - ``rapthor/execution/pipeline``
     - The top-level flow: the order of the operations and the selfcal loop.
   * - ``rapthor/execution/<operation>``
     - Everything needed to run one operation: the description of its inputs,
       the checks on them, the functions that build the DP3 and WSClean
       commands, the tasks, the flow, and the Python steps that used to be
       separate scripts (for example sky model filtering and solution
       plotting).
   * - ``rapthor/execution``
     - Shared code for starting Prefect and Dask, running and logging
       commands, scratch directories, artifacts and resource checks.
   * - ``rapthor/settings``
     - The default values of all parset options.

Rules that keep this working:

* The inputs handed to a flow or a task must be plain data: strings, numbers,
  lists and dictionaries. Do not pass ``Field``, ``Observation``, ``Sector``
  or operation objects, or open files, to a worker.
* Functions that build a command line must depend only on their arguments, so
  that a test can check the exact command without running it.
* When adding a parset option, update the defaults, :ref:`rapthor_parset`,
  the operation class, the flow inputs and their checks, the command builder
  and the tests in the same change.
* Most Python steps are called directly from a task. A step is run as a
  separate ``python -m`` command only when it needs its own process, as PyBDSF
  and LoSoTo do.

Environment of external commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each command inherits the environment of the worker that starts it. A task
can change variables for its own command only, using the helper functions in
``rapthor.execution.environments``; the worker's environment is never
changed. The following policies apply:

* WSClean imaging is given thread limits that match the threads requested for
  its task, for both local and MPI runs.
* DP3 calibration and prediction commands, WSClean imaging and sky-model
  filtering remove the ``MALLOC_TRIM_THRESHOLD_`` variable that Dask sets on
  its workers. This restores glibc's adaptive allocation thresholds in those
  processes. Look at peak memory as well as run time when comparing
  performance, since allocator reuse can retain more memory.
* Sky-model filtering always starts a fresh Python process, including
  single-core runs. It sets ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``,
  ``MKL_NUM_THREADS`` and ``BLIS_NUM_THREADS`` to ``1`` before libraries are
  loaded. PyBDSF's ``ncores`` setting controls process parallelism; native
  thread pools should not multiply it. PyBDSF commands keep their temporary
  files under ``/tmp`` so multiprocessing socket paths stay short.

Changes of this kind are recorded with the command in
``dir_working/logs/commands.jsonl``; a variable that was removed is shown
with the value ``null``.
