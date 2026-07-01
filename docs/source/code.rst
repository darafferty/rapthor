.. _code:

Python and execution code
=========================

Rapthor is mainly written in Python. The production execution path uses
Prefect/Dask flows. The Rapthor code tree is organized as follows::

   rapthor-master
   ├── docs
   ├── examples
   ├── rapthor
   │   ├── cli.py
   │   ├── lib
   │   ├── operations
   │   └── execution
   ├── scripts
   │   ├── dev
   │   └── prod
   └── tests

In the folder structure above:

- ``rapthor-master/docs`` contains this Sphinx documentation.
- ``rapthor-master/examples`` contains example parsets and strategy files.
- ``rapthor-master/rapthor`` contains the main Rapthor Python package.
- ``rapthor-master/rapthor/cli.py`` contains the command-line entry point used
  by the installed ``rapthor`` command and by ``python -m rapthor.cli``.
- ``rapthor-master/rapthor/lib`` contains the main Rapthor classes and modules
  (see :ref:`classes_modules`).
- ``rapthor-master/rapthor/operations`` contains operation adapters and
  operation planning helpers (see :ref:`operation_subclasses`).
- ``rapthor-master/rapthor/execution`` contains Prefect/Dask execution flows,
  command builders, payload contracts, output discovery, and importable helper
  modules.
- ``rapthor-master/scripts`` contains development and deployment launch helpers.
  These are not part of the production pipeline layer.
- ``rapthor-master/tests`` contains unit, execution, operation, and integration
  tests.

Installed Python commands are declared in ``pyproject.toml`` as package entry
points. The main ``rapthor`` command uses ``rapthor.cli:main`` and the
``concat_linc_files`` utility uses
``rapthor.execution.concatenate.linc_cli:main``.


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


Execution helpers and module adapters
-------------------------------------

Production pipeline helpers should live under the execution package that owns
the work. For example, image helper code belongs under
``rapthor.execution.image`` and calibration helper code belongs under
``rapthor.execution.calibrate``.

Most helpers are called directly as Python functions from Prefect/Dask task
bodies. When a separate process remains useful for dependency isolation, the
pipeline uses a thin ``python -m`` module adapter under ``rapthor.execution``
rather than a legacy ``rapthor/scripts`` wrapper. The development architecture
guide describes this pattern in more detail.
