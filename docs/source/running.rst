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

    Usage: rapthor <parset>

    Options:
      --version   show program's version number and exit
      -h, --help  show this help message and exit
      -q          enable quiet mode
      -v          enable verbose mode
      -r          reset one or more operations

Rapthor begins a run by checking the input measurement sets. Next, Rapthor
will determine the DD calibrators from the input sky model and begin self
calibration and imaging. See :ref:`structure` for an overview of the various
operations that Rapthor performs and their relation to one another, and see
:ref:`operations` for details of each operation and their primary data products.

Rapthor uses `Prefect <https://docs.prefect.io/>`_ to keep track of the
processing and `Dask <https://docs.dask.org/>`_ to run it. Nothing needs to be
set up for this: by default, Rapthor starts what it needs on the machine it is
run on and stops it again when the run ends. See :ref:`architecture` for a
description of the parts involved.


.. _monitoring_rapthor:

Following a run
---------------

Progress is reported in the terminal and in the main log,
``dir_working/logs/rapthor.log``. The start and end of every operation are
logged, as are the image diagnostics of every cycle (see :ref:`image`).

More detail is available in the ``dir_working/logs`` directory while the run
is in progress:

``logs/<operation>/<task>.log``
    The full output of each DP3, WSClean or other command that was run, for
    example ``logs/image_1/wsclean_image.log``. The output of a command that
    failed is kept as well.

``logs/commands.jsonl``
    One line for each command that was run, with the full command line, the
    operation and task it belongs to, its start and end times, its exit status
    and the CPU time and memory it used.

``logs/diagnostics.txt``
    A summary of the selfcal, calibration and image diagnostics, written when
    the run finishes.

These files do not depend on the dashboards described below. They are written
when :term:`prefect_log_commands` is set, which is the default.


Environment of commands
~~~~~~~~~~~~~~~~~~~~~~~

DP3 calibration and prediction, WSClean imaging (including MPI launches), and
sky-model filtering remove the ``MALLOC_TRIM_THRESHOLD_`` setting inherited
from Dask workers. This restores glibc's adaptive allocation thresholds in
those processes. The removal appears as ``"MALLOC_TRIM_THRESHOLD_": null``
in ``logs/commands.jsonl``. When comparing performance, check peak memory as
well as elapsed time, since allocator reuse can retain more memory.

Sky-model filtering always runs in a fresh Python process, even when
``filter_skymodel_ncores = 1``. Its OpenMP, OpenBLAS, MKL and BLIS thread pools
are limited to one thread per process so that they do not multiply PyBDSF's
requested process count. ``filter_skymodel_ncores`` still controls that count.
These settings affect only the command and its children, and are recorded in
``logs/commands.jsonl``.


.. _persistent_prefect_dashboard:

Persistent Prefect dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Prefect dashboard shows every operation of a run as a *flow* and every
step within it as a *task*, with their state, run time and logs (see
:ref:`architecture_tasks` for the names used). Calibration solution plots and
image diagnostics are attached to the run as artifacts.

By default, Prefect uses a temporary server that exists only for the duration
of the run, and nothing is kept afterwards. To use the dashboard, start a
Prefect server in one terminal:

.. code-block:: console

    $ prefect server start

and, in another terminal, tell Rapthor where it is before starting the run,
either by exporting its API URL:

.. code-block:: console

    $ export PREFECT_API_URL=http://127.0.0.1:4200/api
    $ rapthor rapthor.parset

or by setting :term:`prefect_api_url` in the parset. The dashboard is then
available at ``http://127.0.0.1:4200``. The server keeps the history of all
runs that reported to it until it is stopped and its database is removed.

To make related runs easier to find in the Prefect dashboard, add optional
run tags in the parset:

.. code-block:: ini

    [cluster]
    prefect_run_tags = ngc891, test-robust

Rapthor attaches these tags to all the flows and tasks of the run.

Previews of the images can also be shown in the dashboard. These are switched
off by default, as they add to the run time and to the disk space used:

.. code-block:: ini

    [cluster]
    prefect_publish_fits_previews = True
    prefect_publish_postage_stamp_previews = True

.. note::

    The Prefect server started with ``prefect server start`` stores its data in
    a small local database that is not designed for heavy use. Do not let many
    large runs report to the same server at the same time. Runs that use the
    default temporary server are independent of one another, so any number of
    them can be run side by side.


Dask dashboard
~~~~~~~~~~~~~~

The Dask dashboard shows which tasks are running on which worker, and the
memory and CPU use of each worker. When Rapthor starts its own Dask scheduler,
it reports the address of the dashboard in the terminal at the start of the
run, for example:

.. code-block:: console

    INFO - rapthor:runtime - Dask dashboard: http://127.0.0.1:8787/status

The port can be set with :term:`dask_dashboard_address`.


.. _remote_dashboards:

Viewing the dashboards of a remote run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When Rapthor runs on a remote machine or a compute node, forward the
dashboard ports to your own machine with SSH. Note the name of the machine
that Rapthor runs on:

.. code-block:: console

    $ hostname

and then, on your own machine, forward the Prefect and Dask ports through the
login node of the cluster:

.. code-block:: console

    $ ssh -N \
        -L 127.0.0.1:4200:compute-node:4200 \
        -L 127.0.0.1:8787:compute-node:8787 \
        user@login.cluster.example

Open ``http://127.0.0.1:4200`` for Prefect and
``http://127.0.0.1:8787/status`` for Dask. Replace the host names and ports
with the values used by the run. If Rapthor runs directly on the machine you
log in to, use ``127.0.0.1`` in place of ``compute-node``.


.. _local_dask_runtime:

Running several tasks at once on a single machine
-------------------------------------------------

By default, Rapthor starts one Dask worker, so that the tasks of an operation
are run one after another and each DP3 or WSClean command can use all the
cores of the machine. On a machine with many cores and enough memory, several
tasks can be run at the same time by setting the number of workers, for
example to image several sectors at once:

.. code-block:: ini

    [cluster]
    local_dask_workers = 2
    max_threads = 16

Each worker runs one task at a time, and each task runs one command that uses
up to :term:`max_threads` threads. Rapthor does not limit the total, so choose
the two values so that their product does not exceed the number of cores, and
make sure that the memory of the machine is enough for that many commands at
once.


.. _external_dask_runtime:

Using an existing Dask cluster
------------------------------

Rapthor can use a Dask scheduler and workers that you have started yourself,
instead of starting its own. Start the scheduler and the workers:

.. code-block:: console

    $ dask scheduler
    $ dask worker tcp://127.0.0.1:8786 --nthreads 1

and give the address of the scheduler to Rapthor, either by setting
:term:`dask_scheduler` in the parset or by exporting ``DASK_SCHEDULER``:

.. code-block:: console

    $ export DASK_SCHEDULER=tcp://127.0.0.1:8786
    $ rapthor rapthor.parset

Rapthor checks that the scheduler can be reached and that it has at least one
worker before the run starts. Each worker should run one task at a time, so
start the workers with ``--nthreads 1``.


.. _running_on_cluster:

Running on multiple nodes of a cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use multiple nodes of a Slurm cluster, start a Dask scheduler and one Dask
worker on each node inside your Slurm job, and then start Rapthor in the same
job. Rapthor does not submit Slurm jobs itself. Rapthor, DP3, WSClean and the
other tools must be available on every node, and :term:`dir_working` and the
input data must be on a disk that all the nodes can see.

Set the following in the parset:

.. code-block:: ini

    [cluster]
    batch_system = slurm
    max_nodes = 4
    cpus_per_task = 32
    mem_per_node_gb = 190
    local_scratch_dir = /local/scratch
    global_scratch_dir = /shared/scratch

where :term:`max_nodes` is the number of nodes in the job, and
:term:`cpus_per_task` and :term:`mem_per_node_gb` are the cores and memory of
each node. The scratch directories are optional (see :term:`local_scratch_dir`
and :term:`global_scratch_dir`).

The job script below is an example of how the scheduler, the workers and
Rapthor can be started. It will need to be adapted to your cluster, for
example to load the software or to start the commands inside a container:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --nodes=4
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=32
    #SBATCH --mem=0
    #SBATCH --exclusive

    # Run this job from a directory on a disk that all the nodes can see.

    # Start the Dask scheduler on the first node of the job
    first_node=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
    srun --nodes=1 --ntasks=1 --nodelist="$first_node" \
        dask scheduler --port 8786 --scheduler-file scheduler.json &

    # Wait for the scheduler, then start one worker on each node
    while [ ! -s scheduler.json ]; do sleep 2; done
    srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
        dask worker --scheduler-file scheduler.json --nthreads 1 \
        --local-directory /local/scratch &

    # Run Rapthor, using the scheduler started above
    export DASK_SCHEDULER=tcp://$first_node:8786
    rapthor rapthor.parset

The ``--local-directory`` option sets where a worker keeps its own temporary
data; it should be a disk local to the node.

When Rapthor uses all of the data (a data fraction of 1), each observation is
split in time into about one chunk per node, so that calibration and
prediction can run on all the nodes at once. WSClean can also spread the imaging of a sector
over several nodes if :term:`use_mpi` is set.

.. note::

    Runs on multiple nodes, and in particular imaging with MPI, depend on how
    Slurm and MPI are set up on the cluster. Check that a short run works on
    your cluster before starting a long one.


.. _using_containers:

Using a (u)Docker/Singularity image
-----------------------------------

A Docker image with the latest release of Rapthor and all its dependencies is
available on `Docker Hub <https://hub.docker.com/r/astronrd/rapthor>`_. Using
it means that no local installation of Rapthor, DP3, WSClean or any of the
other dependencies is needed.


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


Using a container on multiple nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For runs that use multiple nodes of a compute cluster (i.e., when
:term:`batch_system` = ``slurm``), every process must be able to find Rapthor
and its dependencies. If they are not installed on the cluster, start each of
the commands of the job script in :ref:`running_on_cluster` (the Dask
scheduler, the Dask workers and Rapthor itself) inside the container, e.g.,
by putting ``singularity exec --bind <mount_points>:<mount_points>
<rapthor.sif>`` in front of each of them.

Rapthor cannot start containers itself: the :term:`use_container` option of
earlier versions is no longer supported, and a run in which it is set to
``True`` stops with an error.


.. _troubleshooting:

Troubleshooting a run
---------------------
See the :ref:`faq_installation` for tips on troubleshooting Rapthor.


.. _resuming_rapthor:

Resuming an interrupted run
---------------------------

Due to the potentially long run times and the consequent non-negligible chance
of some unforeseen failure occurring, Rapthor has been designed to allow easy
resumption of a reduction from a saved state and will skip over any operations
that were successfully completed previously. In this way, one can quickly resume a
reduction that was halted (either by the user or due to some problem) by simply
re-running Rapthor with the same parset.

An operation that had not finished is run again from its start, with one
exception: if WSClean had already finished making the images of a sector,
those images are used again and WSClean is not rerun. The intermediate files
of an operation that failed are kept in ``dir_working/pipelines`` so that they
can be inspected.


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

All operations after the selected one will also be reset. The files that
these operations produced (their images, solutions, sky models, plots and
logs) are removed from the working directory.

Tool thread budgets
-------------------

Set CPU thread budgets by external tool in the parset, independently of the
Dask worker count::

    [cluster]
    max_threads = 192
    cpus_per_task = 192
    dp3_max_threads = 64
    wsclean_max_threads = 192

Both tool options default to 0, which inherits ``max_threads``. MPI imaging
keeps its existing ``cpus_per_task`` default; an explicit WSClean limit is
capped by ``cpus_per_task`` for each rank. The options apply wherever the
tool is used: DP3 imaging preparation uses ``dp3_max_threads``,
and WSClean prediction during calibration uses ``wsclean_max_threads``.
Other tools retain their existing thread settings. These are command thread
budgets, not CPU reservations or Dask scheduling constraints. Keep concurrent
commands within the node's CPU and memory capacity, and ensure Slurm CPU
binding permits each worker to access its command's requested CPUs. Increasing
the worker count does not automatically reduce either tool's budget.
