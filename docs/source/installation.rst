.. _installation:

Downloading and installing
==========================

Instructions for downloading and installing Rapthor are available on the
`Rapthor GitLab page <https://git.astron.nl/RD/rapthor>`_. Below are some
recommended minimum specifications for hardware and a number of frequently asked
questions regarding the installation of Rapthor.

Hardware requirements
---------------------
The minimum recommended hardware is a 20-core machine with 192 GB of memory and
1 TB of disk space. Rapthor can also take advantage of multiple nodes of a
compute cluster using slurm. In this mode, each node should have approximately
192 GB of memory, with a shared filesystem with 1 TB of disk space.


.. _faq_installation:

Installation FAQ
----------------

How can I use containers (Docker or Singularity) with Rapthor?
    Containers can be used with Rapthor in two ways (see :ref:`using_containers`
    for details):

        * by running it completely within the container (for use on a
          single machine, no local installation of Rapthor or its dependencies is
          necessary)
        * by starting Rapthor and the Dask scheduler and workers that it
          uses inside the container on each node (for use on multiple nodes
          of a compute cluster).

    A Docker image with the latest release of Rapthor and all its dependencies
    is available on `Docker Hub <https://hub.docker.com/r/astronrd/rapthor>`_.

How can I troubleshoot a Rapthor problem?
    If you see a message in the terminal or the main log
    (``dir_working/logs/rapthor.log``) like:

    .. code-block:: console

        ERROR - rapthor - Operation image_1 failed due to an error

    then an error was encountered during the running of the ``image_1``
    operation. In this situation, it is usually helpful to check the log files
    for the failed operation, which, in the case above, should be located in
    ``dir_working/logs/image_1``. There is one log file for each command that
    was run (for example, ``wsclean_image.log``), and the file of the command
    that failed will often reveal the reason for the failure. The command
    lines themselves are listed in ``dir_working/logs/commands.jsonl``. If a
    Prefect server is in use (see :ref:`persistent_prefect_dashboard`), the
    Prefect dashboard shows which task failed, together with its log.

    If the error or its cause is not clear from the log files, it may be useful
    to run with the :term:`keep_temporary_files` option enabled. When this option
    is enabled, the temporary and intermediate files of each operation are
    kept. Running with the ``-v`` option gives more detailed log messages.
