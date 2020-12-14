.. _code:

Python and CWL code
===================

Rapthor is mainly written in Python and CWL. The structure of the Rapthor code tree is organized as follows:

| rapthor-master
| ├── bin
|     └── rapthor
| ├── docs
| ├── examples
| ├── rapthor
| │   ├── lib
| │   ├── operations
| │   ├── pipeline
| │   └── scripts
| └── test
|
|

In the folder structure above:

- ``rapthor-master/bin`` contains the ``rapthor`` executable used to run Rapthor (see :ref:`rapthor`).
- ``rapthor-master/docs`` contains this Sphinx documentation.
- ``rapthor-master/examples`` contains an example parset and strategy file.
- ``rapthor-master/rapthor`` contains the main Rapthor Python package and CWL files.
- ``rapthor-master/rapthor/lib`` contains the main Rapthor classes and modules (see :ref:`classes_modules`).
- ``rapthor-master/rapthor/operations`` contains the operation subclasses (see :ref:`operation_subclasses`).
- ``rapthor-master/rapthor/pipelines`` contains the CWL pipeline templates (see :ref:`cwl`).
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

The Rapthor pipelines call a number of Python scripts to process the solutions, images, etc. The scripts are located in the ``rapthor/scripts/`` directory of the code tree. For details of each script's function, see the inline documentation in the script's code.


.. _cwl:

CWL pipelines
-------------

The CWL pipeline parsets and step definition files are located in the ``rapthor/pipeline/`` directory of the code tree. Each operation in Rapthor has a corresponding pipeline parset. An overview of each operation is given in :ref:`operations`. For details of each step of the pipelines, see the inline documentation in the pipeline parset files (in ``rapthor/pipeline/parsets``) and step files (in ``rapthor/pipeline/steps``).

.. note::

   The CWL files in Rapthor are jinja2 templates, and so are not directly parsable by CWL tools. Rapthor uses the templates to generate the actual CWL files that are passed to Toil. These generated CWL files are created in the Rapthor working directory at runtime.

