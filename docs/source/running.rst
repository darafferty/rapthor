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

.. note::
    By default, Rapthor lets Prefect use its temporary local API/server and
    starts one local Dask scheduler for the run. The temporary Prefect server
    and local Dask scheduler stop when Rapthor exits. For how to set up
    a persistent Prefect dashboard, see :ref:`persistent_prefect_dashboard`.

.. note::
    For tips on how to migrate from the CWL version of Rapthor see
    :ref:`migrating_from_cwl`.

Rapthor can be run from the command line as follows:

.. code-block:: console

    $ rapthor3 rapthor.parset

where ``rapthor.parset`` is the parset described in :ref:`rapthor_parset`. A
number of options are available and are described below:

.. code-block:: console

    Usage: rapthor3 parset

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


.. _persistent_prefect_dashboard:

Persistent Prefect dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To keep a persistent Prefect dashboard, start a server in one terminal and
explicitly export its API URL before running Rapthor, or set the
:term:`prefect_api_url` in the parset:

.. code-block:: console

    $ prefect server start

.. code-block:: console

    $ export PREFECT_API_URL=http://127.0.0.1:4200/api
    $ rapthor rapthor.parset

The dashboard is then available at ``http://127.0.0.1:4200``. The Dask dashboard
is available at the address specified by the Dask scheduler, typically
``http://127.0.0.1:8787``.

To make related runs easier to find in the Prefect dashboard, add optional
run tags in the parset:

.. code-block:: ini

    [cluster]
    prefect_run_tags = demo, multi-sector

Rapthor attaches these tags to the pipeline, operation, and task runs launched
with the shared Prefect runner.


.. _external_dask_runtime:

Using an existing Dask cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use an existing Dask cluster, either set ``dask_scheduler`` in the parset or
export ``DASK_SCHEDULER``:

.. code-block:: console

    $ dask scheduler
    $ dask worker tcp://127.0.0.1:8786

.. code-block:: console

    $ export DASK_SCHEDULER=tcp://127.0.0.1:8786
    $ rapthor3 input.parset


.. _prefect_demo_helper:

Quickstart demos
----------------

Run the basic demo script to launch Rapthor with a local Prefect server and Dask cluster:

.. code-block:: console

    $ scripts/dev/run-rapthor-prefect-demo.py examples/prefect_demo.parset

The ``examples/prefect_demo.parset`` parset uses the small local test Measurement Set
and ``examples/prefect_demo_strategy.py``. 

For a more representative local demo with five bright point-source groups,
48 time slots, multiple frequency bins, and two calibration chunks, generate
the data and parsets:

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

Pass ``--strategy /path/to/strategy.py`` to
``generate-prefect-demo-data.py`` when the local demo parset should reference a
different strategy.

To run the generated local demo in the Prefect and Dask dashboards, use the
local demo parset:

.. code-block:: console

    $ scripts/dev/run-rapthor-prefect-demo.py \
      examples/generated/prefect_demo_rich/prefect_demo_rich.parset

You can also override the runtime resources:

.. code-block:: console

    $ scripts/dev/run-rapthor-prefect-demo.py \
      --task-runner local_dask \
      --local-dask-workers 2 \
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


To run a benchmark harness use ``--scenario ci-benchmark``:

.. code-block:: console

    $ scripts/dev/run_benchmark_baseline.py \
      --scenario ci-benchmark \
      --prepare-inputs \
      --repetitions 1 \
      --local-dask-workers 1 \
      --cpus-per-task 4 \
      --max-threads 4



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
