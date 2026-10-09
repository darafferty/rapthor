.. _code:

Python code
===========

Rapthor is written in Python. The Rapthor code tree is organized as follows::

   rapthor
   ├── docs
   ├── examples
   ├── rapthor
   │   ├── cli.py
   │   ├── execution
   │   ├── lib
   │   ├── operations
   │   ├── settings
   │   └── skymodels
   └── tests

In the folder structure above:

- ``docs`` contains this Sphinx documentation.
- ``examples`` contains example parsets and strategy files.
- ``rapthor`` contains the main Rapthor Python package.
- ``rapthor/cli.py`` contains the ``rapthor`` command used to run Rapthor (see :ref:`running`).
- ``rapthor/execution`` contains the code that runs each operation: the steps of the operation, the DP3 and WSClean commands that they run, and the Python processing code (see :ref:`execution_code`).
- ``rapthor/lib`` contains the main Rapthor classes and modules (see :ref:`classes_modules`).
- ``rapthor/operations`` contains the operation subclasses (see :ref:`operation_subclasses`).
- ``rapthor/settings`` contains the default values of the parset options.
- ``rapthor/skymodels`` contains sky models of bright calibrator sources (see :term:`use_included_skymodels`).
- ``tests`` contains files used for testing.

The package also installs the ``concat_linc_files`` command for preparing LINC
measurement sets.

An overview of how these parts work together is given in :ref:`architecture`.


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


.. _execution_code:

Python processing code
----------------------

The code that does the processing of each operation is located in the ``rapthor/execution/`` directory of the code tree, with one subdirectory for each operation: ``calibrate``, ``predict``, ``image``, ``mosaic``, and ``concatenate``. A subdirectory contains:

- ``flow.py``, which defines the steps of the operation and the order in which they are run;
- ``commands.py``, which builds the DP3 and WSClean command lines (for the operations that run these tools);
- further modules with the Python code that processes the solutions, images, sky models, etc. (for example, ``rapthor/execution/image/skymodel_filter.py`` filters the sky model and ``rapthor/execution/calibrate/plotting.py`` plots the calibration solutions).

For details of each function, see the inline documentation in the code. An overview of each operation is given in :ref:`operations`.

A few of the processing modules can also be run from the command line. For example, the following command plots the phase solutions in a solution table:

.. code-block:: console

    $ python -m rapthor.execution.calibrate.plotting_cli field-solutions.h5 phase

A description of the inputs can be obtained by running the module with the ``-h`` flag.
