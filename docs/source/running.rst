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
    --version             show program's version number and exit
    -h, --help            show this help message and exit
    -q                    enable quiet mode
    -r                    reset one or more operations so that they will be rerun
    -v                    enable verbose mode

Rapthor begins a run by checking the input measurement sets. Next, Rapthor
will determine the DDE calibrators from the input sky model and begin self
calibration and imaging. Rapthor uses the selected :term:`cwl_runner`
to handle the distribution of jobs and to keep track of the state of a reduction.
By default Rapthor uses Toil as CWL runner; other supported runners are CWLTool and StreamFlow.
Each Rapthor operation is done in a separate workflow. See :ref:`structure` for an overview of the various
operations that Rapthor performs and their relation to one another, and see
:ref:`operations` for details of each operation and their primary data products.


.. _run_rapthor_wrapper:

Using the run wrapper
---------------------

The repository provides ``sbin/run-rapthor.sh`` to load a Rapthor environment,
run ``rapthor -v``, and collect logs and configuration files afterwards. From
the repository root, invoke it with exactly one argument, an existing parset:

.. code-block:: console

    $ bash sbin/run-rapthor.sh /path/to/rapthor.parset

The parset must explicitly set ``dir_working`` to a writable directory, rather
than ``None``. The wrapper creates this directory if necessary. Use absolute
paths for ``dir_working`` and input files to avoid depending on the launch
working directory, especially when submitting a batch job.

Before running Rapthor, the wrapper sources
``/project/rapthor/Software/rapthor/etc/rapthor.rc``. To use another environment
configuration, set ``RAPTHOR_RC``:

.. code-block:: console

    $ RAPTHOR_RC=/path/to/rapthor.rc bash sbin/run-rapthor.sh /path/to/rapthor.parset

This configuration file must be readable, make ``rapthor`` and ``python``
available, and set ``RAPTHOR_INSTALL_DIR`` to an installation directory containing
``sw-versions.txt``. The wrapper prints that file and the installed Python
packages to help diagnose the runtime environment.

To submit the wrapper to Slurm, use ``sbatch`` and pass the parset after the
script path. For example, with a custom environment configuration:

.. code-block:: console

    $ RAPTHOR_RC=/path/to/rapthor.rc sbatch --export=ALL sbin/run-rapthor.sh /path/to/rapthor.parset

The script requests 16 CPUs per task and the ``amd`` node constraint. Adjust
these for your cluster with ``sbatch`` options such as ``--cpus-per-task`` and
``--constraint``, and supply any required partition, account, or time limit.
The ``#SBATCH`` directives only apply when submitting through Slurm; they do
not allocate resources when running the script directly with Bash. Configure
Rapthor's own workflow scheduling separately in the parset (see
:term:`batch_system`).

After Rapthor finishes, the wrapper attempts to write
``rapthor_logs_YYYYMMDD_HHMMSS.tar.gz`` in ``dir_working``. It includes the
``logs`` directory if present, the parset, and available user files referenced
by ``input_skymodel``, ``apparent_skymodel``, and ``strategy`` whose values contain
a path separator. The wrapper may create or update symbolic links to these
files in ``dir_working``; the archive includes their contents.

Log collection is also attempted if Rapthor fails. When setup and archive
creation succeed, the wrapper returns Rapthor's exit status. A setup or archive
creation failure can instead terminate the wrapper with its own error status.


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
:term:`batch_system` = ``slurm``), the recommended method of running Rapthor is
to run the operations (CWL workflows) inside containers, with the parent Rapthor process,
which controls the submission of Slurm jobs, running outside of a container.
Therefore, the use of this mode requires a minimal local installation of Rapthor
on the cluster head node (for details, see the installation instructions on the
`Rapthor GitLab page <https://git.astron.nl/RD/rapthor>`_). Other, non-Python
dependencies (such as DP3 and WSClean) do not need to be installed locally. To
use this mode, activate the :term:`use_container` parameter in the parset. No
further configuration should be necessary, as the CWL runner will handle the
pulling and running of the containers.


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
