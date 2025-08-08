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

To ensure that PyBDSF can find the corrct boost libraries you must also load 
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


Running rapthor on a single node (single-machine mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For runs on a single machine, the recommended method of running Rapthor on the 
SKAO cluster is to submit a SLURM job from the headnode. 

An example SLURM script 
is available `here
<https://git.astron.nl/RD/rapthor/-/blob/master/examples/rapthor_skao_singlenode.slurm>`_ 
and a corresponding example parset is available `here
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

An example slurm script 
is available `here
<https://git.astron.nl/RD/rapthor/-/blob/master/examples/rapthor_skao_multinode.slurm>`_ 
and a corresponding example parset is available `here
<https://git.astron.nl/RD/rapthor/-/blob/master/examples/rapthor_skao_multinode.parset>`_ 

Copy these files and edit as needed (edit the paths to your data set and temporary 
directories and the cluster configuration) then submit the job using sbatch. 
This will allocate a compute node to act as the "leader" node which Toil will 
use to orchestrate allocating other nodes for different workflows. Ensure you 
match the max_cores and max_threads to the nodes on the partition(s) you specify 
in your SLURM script (if you specify more cores than are available rapthor will 
fail to run).

.. note::
    
    Both single node and multi-node runs will be run with benchmarking activated 
    but this will currently not monitor all nodes on a multinode run.

.. note::
    
    The "leader" node will be idle for most of the rapthor run. Toil uses this 
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
2. Start an interactive compute node on AWS 
3. Edit and source `this shell script 
   <https://git.astron.nl/RD/rapthor/-/blob/master/examples/setup_skao_aws.sh>`_ 
   This will set up a virtual python environment that with rapthor installed in 
   editable mode.
4. Run ``pytest`` to ensure your environment is setup correctly).