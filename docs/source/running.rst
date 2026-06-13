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
will determine the DDE calibrators from the input sky model and begin self
calibration and imaging. Rapthor uses Prefect/Dask to handle operation
execution, task orchestration, logging, artifacts, and restart state. Each
Rapthor operation is done in a separate flow. See :ref:`structure` for an overview of the various
operations that Rapthor performs and their relation to one another, and see
:ref:`operations` for details of each operation and their primary data products.


Prefect dashboard demo
----------------------

The public ``rapthor`` command uses the Prefect/Dask process flow. To run a
small demo parset and watch it in the Prefect dashboard, use the local demo
helper:

.. code-block:: console

    $ scripts/dev/run-rapthor-prefect-demo.py examples/prefect_demo.parset

The helper starts a temporary Prefect server when one is not already available,
prints the dashboard URL, materializes relative parset paths to absolute paths,
runs the parset through the Prefect process flow, and leaves the temporary
server running when the run finishes so previous runs remain visible in the
dashboard. By default, the helper also overrides ``global.dir_working`` to a
fresh ``rapthor-work`` directory inside the demo run directory, so repeated runs
do not reuse pipeline state. Pass ``--no-unique-working-dir`` to use the working
directory from the parset, or ``--working-dir /path/to/work`` to choose one
explicitly. Use ``--no-keep-server`` for a one-shot run that stops its temporary
server before exiting. The demo parset uses the small local test Measurement Set
and ``examples/prefect_demo_strategy.py``. To attach to an existing server
instead:

.. code-block:: console

    $ PREFECT_API_URL=http://127.0.0.1:4200/api \
      scripts/dev/run-rapthor-prefect-demo.py --no-start-server /path/to/rapthor.parset

If you are already inside the dev container and want to restart the persistent
Prefect server manually, stop any existing local server process and start a new
one bound to all interfaces:

.. code-block:: console

    $ pkill -f "prefect.server" || true
    $ prefect server start --host 0.0.0.0 --port 4200

This keeps the server in the foreground so its logs remain visible. The
dashboard is then available from the host at ``http://localhost:4200`` when the
container forwards port ``4200``.

Use ``--task-runner local_dask``, ``--task-runner sync``, or
``--task-runner external_dask`` to override the parset for a demo run. The demo
parset uses ``local_dask`` by default. For demo runs, the helper starts a
persistent local Dask cluster, rewrites the runtime parset to use that scheduler,
and serves the Dask dashboard on port ``8787``:

.. code-block:: console

    $ scripts/dev/run-rapthor-prefect-demo.py \
      --task-runner local_dask \
      --dask-dashboard-address :8787 \
      examples/prefect_demo.parset

The helper prints the Dask dashboard URL, usually
``http://127.0.0.1:8787/status``. The Dask dashboard is available only while the
local Dask cluster is running, so keep it open during the Rapthor run. The
Workers tab should show the configured local workers, and Task Stream should
show operation-level Prefect tasks as they are submitted. If the demo runs
inside a dev container, forward port ``8787`` from the container to the host
before opening the URL in your browser. Pass ``--no-start-dask`` to use
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
with formatted JSON content. FITS image products are rendered to PNG previews
and published as image artifacts. At the end of the process flow, Rapthor
publishes an index artifact for everything found under ``dir_working/plots``.
Rapthor also records external command timings in
``dir_working/logs/commands.jsonl`` and publishes a command timing summary as a
Prefect Markdown artifact. With ``prefect_command_profile = auto`` (the
default), streamed external commands are also profiled with GNU
``/usr/bin/time -v`` when available, or with Python's process resource counters
as a portable fallback. The command log and Prefect artifact then include CPU
percentage, user/system time, peak resident memory, filesystem input/output
counts, page faults, and context switches for tools such as DP3 and WSClean.
Rapthor also publishes a compact PNG summary chart showing the slowest commands
and their CPU, memory, and I/O profile. The quick and rich demo parsets set
``prefect_command_profile = time`` because it works in the standard rootless
development container and avoids requiring elevated profiling permissions. Set
``prefect_command_profile = off`` to disable resource profiling.

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

The checked-in demo parset uses a very small test Measurement Set so it starts
quickly. To generate a richer local demo with five bright point-source groups,
48 time slots, multiple frequency bins, and two calibration chunks, run:

.. code-block:: console

    $ scripts/dev/generate-prefect-demo-data.py --force
    $ scripts/dev/run-rapthor-prefect-demo.py \
      examples/generated/prefect_demo_rich/prefect_demo_rich.parset

The generated files are written under ``examples/generated/`` and are ignored by
git. The generator uses DP3 prediction to populate the visibilities, then adds
synthetic time/frequency antenna phases and thermal noise so calibration
solution plots have visible structure. By default the generated parset uses a
generated rich strategy next to the parset; that strategy includes DD
slow-gain calibration with a positive target flux threshold. Pass
``--strategy /path/to/strategy.py`` to reference a different strategy.


Running with Slurm and external Dask
------------------------------------

The Prefect/Dask execution path supports Slurm by starting one Dask scheduler
and one Dask worker per allocated node inside a single Slurm allocation. The
launch scripts export ``DASK_SCHEDULER`` before running ``rapthor`` so the
parset does not need to contain a fixed scheduler address.

The parset should select the Slurm/external-Dask mode:

.. code-block:: ini

    [cluster_specific]
    batch_system = slurm
    prefect_task_runner = external_dask
    max_nodes = 4
    cpus_per_task = 32
    mem_per_node_gb = 256

Use the production template when the allocation should start a temporary
Prefect server:

.. code-block:: console

    $ RAPTHOR_PARSET=/path/to/rapthor.parset sbatch scripts/prod/run-rapthor-slurm.sbatch

Use the development template when a persistent Prefect API is already running:

.. code-block:: console

    $ PREFECT_API_URL=http://prefect.example:4200/api RAPTHOR_PARSET=/path/to/rapthor.parset sbatch scripts/dev/run-rapthor-slurm-dev.sbatch

Both templates can source site-specific environment setup by setting
``RAPTHOR_ENV_SCRIPT``. They reserve one CPU per node for scheduler and wrapper
processes by default; set ``RAPTHOR_DASK_WORKER_THREADS`` to override the Dask
worker thread count for a specific cluster.

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
