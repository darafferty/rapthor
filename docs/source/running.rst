.. _rapthor:

Hardware requirements
---------------------
The minimum recommended hardware is a 20-core machine with 192 GB of
memory and 1 TB of disk space. Rapthor can also take advantage of multiple
nodes of a compute cluster using slurm. In this mode, each node should have
approximately 192 GB of memory, with a shared filesystem with 1 TB of disk space.

Starting a Rapthor run
----------------------

Rapthor can be run with:

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

Rapthor begins a run by checking the input measurement set(s). Next, Rapthor
will determine the DDE calibrators from the input sky model and begin self
calibration and imaging. Rapthor uses Toil+CWL to handle the distribution of
jobs and to keep track of the state of a reduction. Each Rapthor operation is
done in a separate pipeline. See :ref:`structure` for an overview of the various
operations that Rapthor performs and their relation to one another, and see
:ref:`operations` for details of each operation and their primary data products.


Resuming an interrupted run
---------------------------

Due to the potentially long run times and the consequent non-negligible chance
of some unforeseen failure occurring, Rapthor has been designed to allow easy
resumption of a reduction from a saved state and will skip over any steps that
were successfully completed previously. In this way, one can quickly resume a
reduction that was halted (either by the user or due to some problem) by simply
re-running Rapthor with the same parset.


Resetting an operation
----------------------

Rapthor allows for the processing of an operation to be reset:

.. code-block:: console

    $ rapthor -r rapthor.parset

Upon running this command, a prompt will appear prompting the user to select an operation to reset:

.. code-block:: console

    INFO - rapthor:state - Reading parset and checking state...

    Current strategy: selfcal

    Pipelines:
        1) calibrate_1
        2) predict_1
        3) image_1
        4) mosaic_1
        5) calibrate_2
        6) image_2
        7) mosaic_2
        8) calibrate_3
        9) image_3
    Enter number of pipeline to reset or "q" to quit:

All operations after the selected one will also be reset.
