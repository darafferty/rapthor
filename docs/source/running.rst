.. _running:

Running Rapthor
===============

.. _starting_rapthor:

Starting a Rapthor run
----------------------

.. note::
    For runs on a single machine, the recommended method of running Rapthor is
    to run everything within a container (see :ref:`using_containers` for
    details).

Rapthor can be run from the command line as follows:

.. code-block:: console

    $ rapthor rapthor.parset

where ``rapthor.parset`` is the parset described in :ref:`rapthor_parset`. A
number of options are available and are described below:

.. code-block:: console

    Usage: rapthor parset

    Options:
      --version             show program's version number and exit
      -h, --help            show this help message and exit
      -q                    enable quiet mode
      -r RESET, --reset=RESET
                            reset one or more operations so that
                            they will be rerun
      -v                    enable verbose mode

Rapthor begins a run by checking the input measurement sets. Next, Rapthor
will determine the DD calibrators from the input sky model and begin self
calibration and imaging. Rapthor uses Prefect/Dask to handle operation
execution, task orchestration, logging, artifacts, and restart state. Each
Rapthor operation is done in a separate flow. See :ref:`structure` for an overview of the various
operations that Rapthor performs and their relation to one another, and see
:ref:`operations` for details of each operation and their primary data products.


Low-friction local runtime
--------------------------

For the simplest local run, use the installed command directly:

.. code-block:: console

    $ rapthor input.parset

Interactive testers do not need Slurm or Spack. Use the dev container, an
existing site environment, or an editable install of this branch as long as the
required radio astronomy tools and Python dependencies are available.

With the default runtime settings, Rapthor lets Prefect use its temporary local
API/server when no ``PREFECT_API_URL`` or ``prefect_api_url`` is configured and
starts one local Dask scheduler for the run. The top-level pipeline and
operation flows attach to that scheduler, so the Dask dashboard can show one
continuous task stream rather than several short-lived local clusters. This is
the lowest-friction mode, but the temporary Prefect server and local Dask
scheduler stop when Rapthor exits. Rapthor uses an isolated temporary Prefect
home in this mode so runs are not written into a persistent local Prefect
dashboard by accident. Export ``PREFECT_API_URL`` or set ``prefect_api_url`` in
the parset when you want a persistent dashboard. Rapthor also sets
``PREFECT_SERVER_ANALYTICS_ENABLED=false`` for the run.

For production systems that need to launch several Rapthor jobs before a
Postgres-backed Prefect service is available, prefer this no-server mode for
each independent job. Give every job a unique ``global.dir_working`` and leave
``PREFECT_API_URL`` unset, or set ``prefect_api_mode = ephemeral``. This avoids
multiple jobs sharing a local SQLite-backed Prefect server. Use a managed
Prefect API with Postgres when persistent dashboard history is required.

To keep a persistent Prefect dashboard, start a server in one terminal and
explicitly export its API URL before running Rapthor:

.. code-block:: console

    $ scripts/dev/start-prefect-server.sh

.. code-block:: console

    $ export PREFECT_API_URL=http://127.0.0.1:4200/api
    $ rapthor input.parset

The setup script starts only Prefect; it does not invoke Rapthor. It uses an
isolated temporary Prefect home, waits for the API health check, and cleans up
when stopped with ``Ctrl+C``. Execute it directly rather than sourcing it. When
a Prefect API URL is configured, Rapthor checks it before launch and logs the
matching dashboard URL.

Choose another port with ``scripts/dev/start-prefect-server.sh --port 14200``
and use the matching ``http://127.0.0.1:14200/api`` URL for Rapthor.

To make related runs easier to find in the Prefect dashboard, add optional
run tags in the parset:

.. code-block:: ini

    [cluster]
    prefect_run_tags = demo, multi-sector

Rapthor attaches these tags to the pipeline, operation, and task runs launched
with the shared Prefect runner.

Running several jobs with dashboards
-------------------------------------

Several interactive jobs can use one Prefect server. Export the same
``PREFECT_API_URL`` in each job terminal, give every parset a unique
``global.dir_working``, and use distinct ``prefect_run_tags``. The dashboard
shows each job as a separate pipeline flow run with its own flow-run page,
subflows, tasks, logs, and artifacts. Those flow-run pages can be opened in
separate browser tabs for side-by-side monitoring.

To give every job a completely separate dashboard and Prefect database, start
one isolated server per job on a unique port:

.. code-block:: console

    $ scripts/dev/start-prefect-server.sh --port 14200
    $ scripts/dev/start-prefect-server.sh --port 14201

Run those commands in separate server terminals. In two additional terminals,
connect each Rapthor process to the corresponding server:

.. code-block:: console

    $ export PREFECT_API_URL=http://127.0.0.1:14200/api
    $ rapthor job-a.parset

.. code-block:: console

    $ export PREFECT_API_URL=http://127.0.0.1:14201/api
    $ rapthor job-b.parset

The dashboards are then available at ``http://127.0.0.1:14200`` and
``http://127.0.0.1:14201``. Local Dask runs must also use different
``dask_dashboard_address`` ports, for example ``:18787`` and ``:18788``. Leave
``PREFECT_HOME`` unset so each setup script creates unique temporary state, or
set a different ``PREFECT_HOME`` directory for each server.

This multi-dashboard setup is intended for manual testing. Do not use one
local SQLite-backed Prefect server as a shared production service for many
parallel jobs. Until a Postgres-backed service is available, use Rapthor's
isolated no-server mode for concurrent production jobs.

Developers testing the Prefect/Dask branch with real parsets should also read
the manual-testing guide at :ref:`manual_testing_prefect_dask`. It gives a
short checklist for adapting ``master`` parsets, choosing runtime settings,
inspecting outputs, and recording evidence for the branch switch decision.

To use an existing Dask cluster, either set ``dask_scheduler`` in the parset or
export ``DASK_SCHEDULER``:

.. code-block:: console

    $ dask scheduler
    $ dask worker tcp://127.0.0.1:8786

.. code-block:: console

    $ export DASK_SCHEDULER=tcp://127.0.0.1:8786
    $ rapthor input.parset

If ``PREFECT_API_URL`` is set but you want Prefect's one-off local API/server
for this run, set ``prefect_api_mode = ephemeral`` in the ``[cluster]`` section
of the parset. Use ``prefect_api_mode = external`` when a run must fail unless
the configured Prefect API is reachable.


Optional Prefect dashboard demo helper
--------------------------------------

The explicit server setup above followed by ``rapthor input.parset`` is the
recommended workflow for testers using their own data. A local helper remains
available for repeatable bundled demos and benchmark automation:

.. code-block:: console

    $ scripts/dev/run-rapthor-prefect-demo.py examples/prefect_demo.parset

The helper starts an isolated local Prefect server when one is not already available,
prints the dashboard URL, materializes relative parset paths to absolute paths,
runs the parset through the Prefect process flow, and keeps the server open
when the run finishes so the completed run remains visible. Press ``Ctrl+C``
after inspecting it; the helper then stops the server and removes its isolated
state. Its SQLite database uses system temporary storage, normally node-local
on a cluster, rather than the user's normal Prefect home. The server also
configures a relative UI API path, so the dashboard works through a single SSH
port forward. By default, the helper also overrides
``global.dir_working`` to a
fresh ``rapthor-work`` directory inside the demo run directory, so repeated runs
do not reuse pipeline state. Pass ``--no-unique-working-dir`` to use the working
directory from the parset, or ``--working-dir /path/to/work`` to choose one
explicitly. Use ``--no-keep-server`` for a one-shot run that stops its temporary
server and removes its temporary state before exiting. The demo parset uses the
small local test Measurement Set
and ``examples/prefect_demo_strategy.py``. To attach to an existing server
instead:

.. code-block:: console

    $ PREFECT_API_URL=http://127.0.0.1:4200/api \
      scripts/dev/run-rapthor-prefect-demo.py --no-start-server /path/to/rapthor.parset

Use ``--task-runner local_dask``, ``--task-runner sync``, or
``--task-runner external_dask`` to override the parset for a demo run. The demo
parset uses ``local_dask`` by default. For demo runs, the helper starts a
persistent local Dask cluster, rewrites the runtime parset to use that scheduler,
and serves the Dask dashboard on port ``8787``:

.. code-block:: console

    $ scripts/dev/run-rapthor-prefect-demo.py \
      --task-runner local_dask \
      --local-dask-workers 2 \
      --dask-dashboard-address :8787 \
      examples/prefect_demo.parset

The helper prints the Dask dashboard URL, usually
``http://127.0.0.1:8787/status``. The Dask dashboard is available only while the
local Dask cluster is running, so keep it open during the Rapthor run. The
Workers tab should show the configured local workers, and Task Stream should
show operation-level Prefect tasks as they are submitted. Use
``local_dask_workers`` in the parset, or ``--local-dask-workers`` with the demo
helper, to change the number of local Dask workers without changing
``max_nodes``. If the demo runs inside a dev container, forward port ``8787``
from the container to the host before opening the URL in your browser. Pass
``--no-start-dask`` to use
Prefect's temporary ``local_dask`` clusters directly; those can be harder to
monitor because they are created lazily by each flow. With
``--task-runner external_dask``, use the dashboard for the external scheduler.
To save the Dask scheduler, task stream, and profiling information as a
standalone HTML file, pass ``--dask-performance-report``. The report is written
to the demo run directory by default:

.. code-block:: console

    $ scripts/dev/run-rapthor-prefect-demo.py \
      --dask-performance-report \
      examples/prefect_demo.parset

Use ``--dask-performance-report-path /path/to/report.html`` to choose the
output path.

The demo parset streams external command output to Prefect task logs by default without
the repeated Prefect Shell ``PID ... stream output`` prefixes. Pass
``--no-stream-output`` or set ``prefect_stream_output = False`` to suppress
external command output in the dashboard. Rapthor's Python logging is also
forwarded to the active Prefect flow or task run. Plot files are also published as
Prefect artifacts as plotting tasks and operation finalizers create them, so
calibration PNGs and image diagnostic PDFs can be inspected from the dashboard
during the run. Image diagnostic JSON files are rendered as Markdown artifacts
with formatted JSON content. FITS image product previews are disabled by
default because they add runtime and disk usage for large datasets. Set
``prefect_publish_fits_previews = True`` for demo or debugging runs to render
those FITS products to PNG previews and publish them as image artifacts; the
demo parsets enable this, while the benchmark parset leaves it disabled. FITS
products, files under ``dir_working/plots``, and numeric image diagnostics are
still produced either way. Set ``prefect_publish_postage_stamp_previews = True``
to write source-centred PNG crops around the brightest catalog sources under
``dir_working/.rapthor-artifacts/postage-stamps`` and publish them from Prefect
when a run context is active; the demo parsets enable these smaller previews,
while the benchmark parset leaves them disabled. Each postage stamp is rendered
from the ``image-pb`` product, includes the catalog source coordinates in the
FITS WCS units, and uses the same full-image configurable percentile clipping
as whole-field FITS previews. At the end of the process flow, Rapthor publishes
an index artifact for everything found under ``dir_working/plots``.
When running in the VS Code development container, the plot index rewrites
``/app`` paths to the host workspace path using ``RAPTHOR_HOST_WORKSPACE`` so
the local file links can be opened from the browser.
Rapthor also records external command timings in
``dir_working/logs/commands.jsonl`` and publishes a command timing summary as a
Prefect Markdown artifact. With ``prefect_command_profile = auto`` (the
default), streamed external commands are also profiled with GNU
``/usr/bin/time -v`` when available, or with Python's process resource counters
as a portable fallback. The command log and Prefect artifact then include CPU
percentage, user/system time, peak resident memory, filesystem input/output
counts, page faults, and context switches for tools such as DP3 and WSClean.
Rapthor also publishes a compact PNG summary chart showing the slowest commands
and their CPU, memory, and I/O profile. The checked-in demo parset and generated
demo/benchmark parsets set ``prefect_command_profile = time`` because it works
in the standard rootless development container and avoids requiring elevated
profiling permissions. Set ``prefect_command_profile = off`` to disable
resource profiling.

For deeper native CPU profiling, ``prefect_command_profile = perf`` attempts to
run Linux ``perf record``. Treat this as an advanced, opt-in mode for a
disposable or rootful profiling container rather than the normal development
container. It requires host support, suitable ``perf_event`` permissions, and
container permissions that allow ``perf_event_open``. When successful, Rapthor
writes ``perf.data``, ``perf.script``, collapsed ``perf.folded`` stacks, and
``perf.flamegraph.svg`` files under ``dir_working/logs/profiles/``. Generated
flamegraph SVGs are also published as Prefect image artifacts and linked from
the command timing Markdown artifact. If ``perf`` is blocked, Rapthor falls back
to lower-level resource metrics.

The checked-in ``examples/prefect_demo.parset`` uses a very small test
Measurement Set so it starts quickly. For a more representative local demo with
five bright point-source groups, 48 time slots, multiple frequency bins, and two
calibration chunks, generate the ignored demo inputs first:

.. code-block:: console

    $ scripts/dev/generate-prefect-demo-data.py --force

This writes the following files under
``examples/generated/prefect_demo_rich/``:

- ``prefect_demo_rich.ms`` and matching apparent/true sky models
- ``prefect_demo_benchmark_strategy.py``
- ``prefect_demo_rich.parset`` for local dashboard demos
- ``prefect_demo_benchmark.parset`` for the benchmark harness and CI

The generator uses DP3 prediction to populate the visibilities, then adds
synthetic time/frequency antenna phases and thermal noise so calibration
solution plots have visible structure.

Both generated parsets use the same benchmark strategy: DI phase, DD
phase/faceting, the legacy DD default solve order
``["fast_phase", "medium_phase", "slow_gains", "medium_phase"]``, full-Jones
calibration, imaging, mosaicking, and source filtering. The local demo parset
keeps workstation-friendly resource defaults, while the benchmark parset leaves
thread counts to be derived from benchmark runtime overrides.

To run the generated local demo in the Prefect and Dask dashboards, use the
local demo parset:

.. code-block:: console

    $ scripts/dev/run-rapthor-prefect-demo.py \
      examples/generated/prefect_demo_rich/prefect_demo_rich.parset

On a smaller workstation, keep the same generated data and strategy but override
the runtime resources:

.. code-block:: console

    $ scripts/dev/run-rapthor-prefect-demo.py \
      --task-runner local_dask \
      --local-dask-workers 1 \
      --cpus-per-task 4 \
      --max-threads 4 \
      examples/generated/prefect_demo_rich/prefect_demo_rich.parset

To exercise the multiple-sector imaging and mosaicking path, generate the
additional quadrant-balanced benchmark dataset:

.. code-block:: console

    $ scripts/dev/generate-prefect-demo-data.py --force --include-multi-sector

This also writes ``prefect_demo_multisector.ms``, matching apparent/true sky
models, and ``prefect_demo_multisector_benchmark.parset``. The multi-sector sky
model places bright source groups well inside the four quadrants of a
``2 x 2`` sector grid, so the run exercises sectorized imaging and mosaic
assembly without relying on sources that sit close to sector boundaries. The
multi-sector parset uses ``dde_method = single`` so each sector applies the
nearest DD solution during imaging; the single-sector benchmark remains the
full-DD facet-imaging coverage.

For a modest local dashboard test, run:

.. code-block:: console

    $ MULTISECTOR_RUN_DIR="runs/prefect-demo-multisector-$(date +%Y%m%d-%H%M%S)"
    $ scripts/dev/run-rapthor-prefect-demo.py \
      examples/generated/prefect_demo_rich/prefect_demo_multisector_benchmark.parset \
      --run-dir "$MULTISECTOR_RUN_DIR" \
      --local-dask-workers 2 \
      --cpus-per-task 4 \
      --max-threads 4 \
      --filter-skymodel-ncores 4 \
      --dask-performance-report

For a CI-like 60-core resource shape, use:

.. code-block:: console

    $ MULTISECTOR_RUN_DIR="runs/prefect-demo-multisector-ci-shape-$(date +%Y%m%d-%H%M%S)"
    $ scripts/dev/run-rapthor-prefect-demo.py \
      examples/generated/prefect_demo_rich/prefect_demo_multisector_benchmark.parset \
      --run-dir "$MULTISECTOR_RUN_DIR" \
      --local-dask-workers 2 \
      --cpus-per-task 30 \
      --max-threads 30 \
      --filter-skymodel-ncores 15 \
      --dask-performance-report

If a Prefect server is already running, pass its API URL explicitly:

.. code-block:: console

    $ MULTISECTOR_RUN_DIR="runs/prefect-demo-multisector-$(date +%Y%m%d-%H%M%S)"
    $ scripts/dev/run-rapthor-prefect-demo.py \
      examples/generated/prefect_demo_rich/prefect_demo_multisector_benchmark.parset \
      --api-url http://127.0.0.1:4200/api \
      --no-start-server \
      --run-dir "$MULTISECTOR_RUN_DIR" \
      --local-dask-workers 2 \
      --cpus-per-task 4 \
      --max-threads 4 \
      --filter-skymodel-ncores 4 \
      --dask-performance-report

After the run starts, the log should confirm that Rapthor built four imaging
sectors and scheduled the mosaic operation:

.. code-block:: console

    $ rg "Using 4 imaging sectors|Operation mosaic" \
      "$MULTISECTOR_RUN_DIR/rapthor-work/logs/rapthor.log"

Pass ``--strategy /path/to/strategy.py`` to
``generate-prefect-demo-data.py`` when the local demo parset should reference a
different strategy. The benchmark parset always references the generated
benchmark strategy so CI and benchmark runs stay comparable.

To run the same scenario through the benchmark harness instead of the dashboard
helper, use ``ci-benchmark``. This command keeps the run small enough for a
local smoke check:

.. code-block:: console

    $ scripts/dev/run_benchmark_baseline.py \
      --scenario ci-benchmark \
      --prepare-inputs \
      --repetitions 1 \
      --local-dask-workers 1 \
      --cpus-per-task 4 \
      --max-threads 4


Running with Slurm and external Dask
------------------------------------

The Prefect/Dask execution path supports Slurm by starting one Dask scheduler
and one Dask worker per allocated node inside a single Slurm allocation. The
launch scripts export ``DASK_SCHEDULER`` before running ``rapthor`` so the
parset does not need to contain a fixed scheduler address.

For multi-node imaging with WSClean MPI, the same allocation must expose the
Dask scheduler to all workers and the WSClean MPI launch must match the
allocated node count. If a Prefect dashboard is used, ``PREFECT_API_URL`` must
also be reachable from the Dask worker nodes. This path should be validated in
a representative staging allocation before production use.

The parset should select the Slurm/external-Dask mode:

.. code-block:: ini

    [cluster_specific]
    batch_system = slurm
    prefect_task_runner = external_dask
    max_nodes = 4
    cpus_per_task = 32
    mem_per_node_gb = 256

For the MPI WSClean imaging path, also set:

.. code-block:: ini

    [imaging]
    use_mpi = True

For a live demonstration, the Slurm launcher should print or write SSH tunnel
commands for both dashboards. The intended local browser endpoints are:

.. code-block:: text

    Prefect dashboard: http://127.0.0.1:14200
    Dask dashboard:    http://127.0.0.1:18787/status

The tunnel usually forwards from the local machine through the login/head node
to the Prefect host and the first compute node running the Dask scheduler. The
Prefect API URL used by Rapthor must still be reachable from Dask workers
inside the allocation; the SSH tunnel is only for the local browser view.

Use the production template when the allocation should start a temporary
Prefect server:

.. code-block:: console

    $ RAPTHOR_PARSET=/path/to/rapthor.parset sbatch scripts/prod/run-rapthor-slurm.sbatch

For production-like Slurm tests, prefer a loadable Spack environment once the
Prefect/Dask branch recipe is available. Interactive non-Slurm testers may use
other working environments. The Spack recipe should live alongside the legacy
``py-rapthor`` recipe in ``../ska-sdp-spack/packages/`` and should load Prefect,
Dask distributed, DP3, WSClean including ``wsclean-mp``, and the Python
astronomy dependencies required by Rapthor.

Use the development template when a persistent Prefect API is already running:

.. code-block:: console

    $ PREFECT_API_URL=http://prefect.example:4200/api RAPTHOR_PARSET=/path/to/rapthor.parset sbatch scripts/dev/run-rapthor-slurm-dev.sbatch

Both templates can source site-specific environment setup by setting
``RAPTHOR_ENV_SCRIPT``. Dask workers are single-threaded by default so each
worker runs one Prefect task engine at a time. External tool parallelism remains
controlled by Rapthor settings such as ``cpus_per_task`` and ``max_threads``.
Only set ``RAPTHOR_DASK_WORKER_THREADS`` for a specific cluster after checking
that multi-threaded Prefect task execution is stable there.

The Slurm integration check is skipped by default. To validate a staging
allocation, run the integration suite from inside the Slurm job with
``RAPTHOR_RUN_SLURM_INTEGRATION=1``.


.. _using_containers:

Using a (u)Docker/Singularity image
-----------------------------------

Rapthor can use containers in two ways: by running Rapthor completely within a
container (for use on a single machine) or by installing Rapthor locally and running
only the operations within the container (for use with multiple nodes of a
compute cluster).


Running everything in a container (single-machine mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For runs on a single machine (i.e., when :term:`batch_system` = ``single_machine``),
the recommended method of running Rapthor is to run everything within a container. To
use this method, first obtain the container image as follows:

For Docker:

.. code-block:: console

    $ docker pull astronrd/rapthor

For uDocker:

.. code-block:: console

    $ udocker pull astronrd/rapthor

For Singularity:

.. code-block:: console

    $ singularity pull docker://astronrd/rapthor


Then start the run, making sure that all necessary volumes are accessible from
inside the container, e.g.,:

.. code-block:: console

    $ docker run --rm <docker_options> -v <mount_points>:<mount_points> -w $PWD astronrd/rapthor rapthor rapthor.parset

.. code-block:: console

    $ udocker run --rm <docker_options> -v <mount_points>:<mount_points> -w $PWD astronrd/rapthor rapthor rapthor.parset

.. code-block:: console

    $ singularity exec --bind <mount_points>:<mount_points> <rapthor.sif> rapthor rapthor.parset

In this mode, since Rapthor is running fully inside a container, the
:term:`use_container` parameter should *not* be set, as activating this option
instructs Rapthor to run the operations inside another, additional container
(resulting in it running a container inside a container).


Running only the operations in a container (multinode mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For runs that use multiple nodes of a compute cluster (i.e., when
:term:`batch_system` = ``slurm``), run Rapthor in an environment that already
contains the required Python and external radio-astronomy tools. The legacy
mode that launched operation-level CWL containers is no longer the production
runtime. Slurm/external-Dask validation is deferred until after the Prefect/Dask
migration cutover.


.. _troubleshooting:

Troubleshooting a run
---------------------
See the :ref:`faq_installation` for tips on troubleshooting Rapthor.


.. _resuming_rapthor:

Resuming an interrupted run
---------------------------

Due to the potentially long run times and the consequent non-negligible chance
of some unforeseen failure occurring, Rapthor has been designed to allow easy
resumption of a reduction from a saved state and will skip over any steps that
were successfully completed previously. In this way, one can quickly resume a
reduction that was halted (either by the user or due to some problem) by simply
re-running Rapthor with the same parset. If a step within an operation has failed,
the output of the previous steps are cached and the execution will resume from
that point going forward.


.. _resetting_rapthor:

Resetting an operation
----------------------

Rapthor allows for the processing of an operation to be reset:

.. code-block:: console

    $ rapthor -r rapthor.parset

Upon running this command, a prompt will appear prompting the user to select an
operation to reset:

.. code-block:: console

    INFO - rapthor:state - Reading parset and checking state...

    Current strategy: selfcal

    Operations:
        1) calibrate_1
        2) predict_1
        3) image_1
        4) mosaic_1
        5) calibrate_2
        6) image_2
        7) mosaic_2
        8) calibrate_3
        9) image_3
    Enter number of operation to reset or "q" to quit:

All operations after the selected one will also be reset.
