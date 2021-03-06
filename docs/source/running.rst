.. _rapthor:

Starting a Rapthor run
----------------------

Rapthor can be run with::

    $ rapthor rapthor.parset

where ``rapthor.parset`` is the parset described in :ref:`rapthor_parset`. A
number of options are available and are described below::

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

Rapthor allows for the processing of an operation to be reset::

    $ rapthor -r rapthor.parset

Once reset, the operation will start from scratch the next time Rapthor is run.
