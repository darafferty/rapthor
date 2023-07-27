.. _products:

Output
======

Rapthor produces the following output inside the working directory:

``logs/``
    Directory containing the log files. The main log file for the run is called ``rapthor.log``. The detailed logs of each step of the processing can be found in the subdirectories.

``images/``
    Directory containing the FITS images. See :ref:`image` for a detailed description of the images.

``pipelines/``
    Directory containing intermediate files of each operation's CWL workflow. Once a run has finished successfully, this directory can be removed.

``regions/``
    Directory containing ds9 region files. These regions define the imaged areas and the facet layout (if used).

``skymodels/``
    Directory containing sky model files. See :ref:`calibrate` and :ref:`image` for a detailed description of the sky models.

``solutions/``
    Directory containing the calibration solution h5parm files. See :ref:`calibrate` for a detailed description of the solution files.

``plots/``
    Directory containing the PNG plots of the calibration solutions and images. See :ref:`calibrate` for a detailed description of the plots.
