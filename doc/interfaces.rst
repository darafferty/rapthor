===================
Interfaces Overview
===================

There are two levels of interfaces. The lower level Proxy in idg-lib, 
and a higher level BufferSet in idg-api. 

A Proxy represents the accelerator hardware (CPU/GPU/...) for which IDG has been implemented.
The Proxy expects blocks of data grouped per baseline.

The BufferSet accepts data per row in the measurement set, until a block is complete and
then sends the block to the underlying Proxy.
The BufferSet also chooses the configuration of the Proxy (kernel size, taper,...) and
has get_image and set_image methods to do the final/initial FFT.
The BufferSet is the interface that `WSClean <https://wsclean.readthedocs.io/en/latest/>`_ uses for its IDG gridding mode.

Both interfaces are written in C++, but for the Proxy there are C-bindings and Python
bindings available.

.. toctree::
   :maxdepth: 2
   :titlesonly:

   c++-interface
   c-interface
   python-proxy

