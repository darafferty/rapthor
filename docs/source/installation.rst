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

        * by running it completely within the container (only for use on a
          single machine, no local installation of Rapthor or its dependencies is
          necessary)
        * by installing Rapthor locally and running only the operations (CWL workflows)
          within the container (for use on a single machine or on multiple nodes of a
          compute cluster).

    A Docker image with the latest release of Rapthor and all its dependencies
    is available on `Docker Hub <https://hub.docker.com/r/astronrd/rapthor>`_.

How can I troubleshoot a Rapthor problem?
    If you see a message in the terminal or the main log
    (``dir_working/logs/rapthor.log``) like:

    .. code-block:: console

        CRITICAL - rapthor - Operation image_1 failed due to an error

    then an error was encountered during the running of the CWL workflow of the
    ``image_1`` operation. In this situation, it usually helpful to check the
    log files for the failed operation, which, in the case above, should be
    located in ``dir_working/logs/image_1``. When Toil is used for the CWL
    runner (the default; see :term:`cwl_runner`), filenames beginning with the
    word "failed" indicate the logs of the steps, if any, that failed. A search
    in these logs will often reveal the reason for the failure.

    If the error or its cause is not clear from the log files, it may be useful
    to run with the :term:`keep_temporary_files` option enabled. When this option
    is enabled, the working directory will not be cleaned up. In addition, most
    CWL runners also provide extra debugging options. These can be enabled by
    enabling :term:`debug_workflow`. In that case, ``stdout`` and ``stderr``
    will not be redirectied, and the log level of the CWL runner will be set to
    ``DEBUG``.
