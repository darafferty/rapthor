.. _code:

Python and execution code
=========================

Rapthor is mainly written in Python. The production execution path uses
Prefect/Dask flows. The Rapthor code tree is organized as follows::

   rapthor-master
   ├── bin
   │   └── rapthor
   ├── docs
   ├── examples
   ├── rapthor
   │   ├── lib
   │   ├── operations
   │   ├── execution
   │   └── scripts
   └── test

In the folder structure above:

- ``rapthor-master/bin`` contains the ``rapthor`` executable used to run Rapthor (see :ref:`running`).
- ``rapthor-master/docs`` contains this Sphinx documentation.
- ``rapthor-master/examples`` contains an example parset and strategy file.
- ``rapthor-master/rapthor`` contains the main Rapthor Python package.
- ``rapthor-master/rapthor/lib`` contains the main Rapthor classes and modules (see :ref:`classes_modules`).
- ``rapthor-master/rapthor/operations`` contains the operation subclasses (see :ref:`operation_subclasses`).
- ``rapthor-master/rapthor/execution`` contains the Prefect/Dask execution flows and helpers.
- ``rapthor-master/rapthor/scripts`` contains the processing scripts (see :ref:`scripts`).
- ``rapthor-master/test`` contains files used for testing.


.. _classes_modules:

Python classes and modules
--------------------------

The following Python classes and modules are the principal ones used in Rapthor. The corresponding Python files are located in the ``rapthor/lib`` directory of the code tree.

.. toctree::
   :maxdepth: 2

   operation_class
   observation_class
   field_class
   sector_class
   cluster_module
   context_module
   miscellaneous_module
   parset_module


.. _scripts:

Python processing scripts
-------------------------

The Rapthor operations call a number of Python scripts to process the solutions, images, etc. The scripts are located in the ``rapthor/scripts/`` directory of the code tree. For details of each script's function, see the inline documentation in the script's code. A description of the inputs can also be obtained by running the script with the ``-h`` flag.
