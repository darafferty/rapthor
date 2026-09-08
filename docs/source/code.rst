.. _code:

Python and CWL code
===================

Rapthor is mainly written in Python and CWL. The Rapthor code tree is organized as follows::

   rapthor
   ├── bin
   │   └── rapthor
   ├── docs
   ├── examples
   ├── rapthor
   │   ├── lib
   │   ├── operations
   │   ├── pipeline
   │   └── scripts
   └── tests

In the folder structure above:

- ``bin`` contains the ``rapthor`` executable used to run Rapthor (see :ref:`running`).
- ``docs`` contains this Sphinx documentation.
- ``examples`` contains example parsets and strategy files.
- ``rapthor`` contains the main Rapthor Python package and CWL files.
- ``rapthor/lib`` contains the main Rapthor classes and modules (see :ref:`classes_modules`).
- ``rapthor/operations`` contains the operation subclasses (see :ref:`operation_subclasses`).
- ``rapthor/pipeline`` contains the CWL workflow templates (see :ref:`cwl`).
- ``rapthor/scripts`` contains the processing scripts (see :ref:`scripts`).
- ``tests`` contains files used for testing.

The package also installs the ``concat_linc_files`` command for preparing LINC
measurement sets and the ``plotrapthor`` command for plotting solution tables.


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


.. _cwl:

CWL workflows
-------------

The CWL workflow parsets and step definition files are located in the ``rapthor/pipeline/`` directory of the code tree. Each operation in Rapthor has a corresponding workflow parset. An overview of each operation is given in :ref:`operations`. For details of each step of the workflows, see the inline documentation in the workflow parset files (in ``rapthor/pipeline/parsets``) and step files (in ``rapthor/pipeline/steps``).

.. note::

   The CWL workflow files in ``rapthor/pipeline/parsets`` are jinja2 templates, and so are not directly parsable by CWL tools. Rapthor uses the templates to generate the actual CWL workflows that are passed to the CWL runner. These generated files are created in the Rapthor working directory in ``workdir/pipelines/operation_name`` at runtime with the following names: the CWL workflow file is named ``pipeline_parset.cwl`` (and ``subpipeline_parset.cwl`` when there is a subworkflow) and the workflow inputs JSON file is named ``pipeline_inputs.json``.

