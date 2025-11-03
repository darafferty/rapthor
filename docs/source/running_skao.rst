.. _running_skao:

Running Rapthor on the SKAO AWS cluster
=======================================

The recommended way to run rapthor on the SKAO AWS development cluster is to 
use the rapthor spack module that is pre-installed (you can see details of the 
spack package `here 
<https://gitlab.com/ska-telescope/sdp/ska-sdp-spack/-/blob/main/packages/py-rapthor/package.py>`_). 
Loading this module will also load all of rapthor's dependencies, including wsclean and dp3.

.. code-block:: console
    
    $ module use "/shared/fsx1/spack/modules/2025.07.3/linux-ubuntu22.04-x86_64_v3"
    $ module load py-rapthor 

To ensure that PyBDSF can find the correct boost libraries you must also load 
the boost module and add to ``LD_LIBRARY_PATH``:

.. code-block:: console
    
    $ module load boost
    $ export LD_LIBRARY_PATH=$BOOST_ROOT/lib:$LD_LIBRARY_PATH

Rapthor is now ready to run. 

.. note::
    
    We recommend running rapthor as a SLURM job submitted from the headnode. 
    Example SLURM scripts that will set up the required environment variables, 
    run and benchmark rapthor using SKA tools are available for a `single-node run
    <https://git.astron.nl/RD/rapthor/-/blob/master/examples/rapthor_skao_singlenode.slurm>`_ 
    and a `multi-node run 
    <https://git.astron.nl/RD/rapthor/-/blob/master/examples/rapthor_skao_multinode.slurm>`_. 
    See below for details.


.. _starting_rapthor_skao:

Starting a Rapthor run
----------------------

Rapthor can be run from the command line using:

.. code-block:: console

    $ rapthor rapthor.parset

where ``rapthor.parset`` is the parset described in :ref:`rapthor_parset`. A
number of options are available (see :ref:`running` for details).

.. warning::

    Rapthor attempts to resume from a previous state if output files from a 
    previous run are left in the working directory (see 
    :ref:`resuming_rapthor`). This means that changes to your parset may not 
    be respected unless you remove or rename the previous output folder and 
    delete the contents of your scratch/temporary directories.

.. warning::

    Due to storage limits on the default ``/tmp`` directory on AWS, it is best 
    to create a new temporary folder on the shared ``/shared/fsx1`` directory. 
    You will then need to set ``local_scratch_dir`` and ``global_scratch_dir`` 
    in the parset, as well ``TMPDIR`` in the slurm script to this path. The 
    reason is that toil/cwl used by rapthor will create intermediate files in 
    ``TMPDIR``, ``local_scratch_dir`` and ``global_scratch_dir`` during the 
    run which may exceed the available space on ``/tmp``.
    Note, however, that the filter_skymodel step will always set
    ``/tmp`` as the temporary directory. This is a workaround for
    socket file paths having a character limit (107 bytes on unix systems), 
    causing issues with long path names during multiprocessing (used by pybdsf). 
    Since Toil creates path names for temporary storage files using random
    hexadecimal strings, the base location of the temporary storage paths 
    ``global_scratch_dir`` and ``local_scratch_dir`` can be too long, 
    resulting in errors.


Running rapthor on a single node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For runs on a single node (i.e., when
:term:`batch_system` = ``single_machine``), the recommended method of running Rapthor on the 
SKAO cluster is to submit a SLURM job from the headnode. 

An `example SLURM script for a single node run 
<https://git.astron.nl/RD/rapthor/-/blob/master/examples/rapthor_skao_singlenode.slurm>`_
is provided in the examples directory, together with a `corresponding example parset 
<https://git.astron.nl/RD/rapthor/-/blob/master/examples/rapthor_skao_singlenode.parset>`_.

Copy these files and edit as needed (edit the paths to your data set and scratch 
directories and the cluster configuration - make sure the resources requested in 
your slurm script match those in the parset) then submit the job using sbatch.
This will allocate a compute node and run all workflows on this node.

Running rapthor on multiple nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For runs that use multiple nodes of a compute cluster (i.e., when
:term:`batch_system` = ``slurm``), the recommended method of running Rapthor on the 
SKAO cluster is to submit a SLURM job from the headnode. 

An `example SLURM script for a multi-node run 
<https://git.astron.nl/RD/rapthor/-/blob/master/examples/rapthor_skao_multinode.slurm>`_ 
is provided in the examples directory, together with a `corresponding example parset 
<https://git.astron.nl/RD/rapthor/-/blob/master/examples/rapthor_skao_multinode.parset>`_.

Copy these files and edit as needed (edit the paths to your data set and temporary 
directories and the cluster configuration) then submit the job using sbatch. 
This will allocate a compute node to act as the "leader" node which Toil will 
use to orchestrate allocating other nodes for different workflows. 

.. warning::

    Ensure you match the ``max_cores`` and ``max_threads`` to the nodes on the 
    partition(s) you specify in your SLURM script -- if you specify more cores 
    than are available rapthor will fail to run.


Known issues
------------

- Both single node and multi-node runs will be run with benchmarking activated 
  but this will currently not monitor all nodes on a multi-node run if mpi is 
  enabled due to the way rapthor uses ``salloc`` to allocate interactive nodes 
  for ``wsclean-mp``.
    
- The "leader" node will be idle for most of the rapthor run. Toil uses this 
  node to orchestrate the allocation of other nodes. A further node will be 
  idle during imaging steps if mpi is enabled since this node is only used 
  to allocate additional nodes for ``wsclean-mp``.


Troubleshooting a run
---------------------
See the :ref:`faq_installation` for tips on troubleshooting Rapthor.


.. _contributing_skao:

Developing rapthor on the SKAO AWS cluster
------------------------------------------
To test latest changes to the rapthor pipeline or develop on your 
own branch:

1. Clone the rapthor repository
2. Start an interactive compute node on AWS (using ``srun``)
3. Edit and source `this shell script 
   <https://git.astron.nl/RD/rapthor/-/blob/master/examples/setup_skao_aws.sh>`_. 
   This will set up a virtual python environment that with rapthor installed in 
   editable mode.
4. Run ``pytest`` to ensure your environment is setup correctly.

.. note::
    
    To avoid unexpected behaviour while testing code changes by running rapthor,
    always use a fresh output directory and remove all temporary files from 
    previous runs. If rapthor is run using the same parset as previously it 
    will try to resume from the previous state (see :ref:`resuming_rapthor`).

.. note::

    When starting an interactive node for testing, make sure you request 
    enough resources (e.g. ``cpus-per-task``) to satisfy the cluster parameters 
    in your parset (e.g. ``max_cores``).
