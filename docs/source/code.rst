.. _code:

Python and CWL code
===================

Rapthor is mainly written in Python and CWL. This section describes the code in detail.


The ``rapthor`` and ``plotrapthor`` executables
-----------------------------------------------

The ``bin/`` directory contains the ``rapthor`` executable, a Python scripts used to run Rapthor (see :ref:`rapthor`). The ``bin/rapthor`` executable calls the ``rapthor.process.run()`` or ``rapthor.modifystate.run()`` functions, described below.

.. automodule:: rapthor.process
   :members:

.. automodule:: rapthor.modifystate
   :members:


Python classes and modules
--------------------------

The following Python classes and modules are defined in Rapthor:

.. toctree::
   :maxdepth: 2

   operation_class
   observation_class
   field_class
   image_class
   sector_class
   cluster_module
   context_module
   miscellaneous_module
   parset_module


Python processing scripts
-------------------------

The Rapthor pipelines call a number of Python scripts to process solutions, images, etc. The scripts are located in the ``rapthor/scripts/`` directory of the code tree. For details of each script's function, see the inline documentation in the script's code.


CWL pipelines
-------------

The CWL pipeline parsets and step definition files are located in the ``rapthor/pipeline/`` directory of the code tree. Each operation in Rapthor has a corresponding pipeline parset (and sometimes subpipeline parset). An overview of each operation is given in :ref:`operations`. For details of each step in the pipelines, see the inline documentation in the pipeline parsets and step files.
