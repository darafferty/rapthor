.. toctree::
    :maxdepth: 2
    :numbered:
    :caption: Demo Documentation


===========
Quick start
===========

To make images with IDG one can use `WSClean <https://wsclean.readthedocs.io/en/latest/index.html>`_ in `IDG mode <https://wsclean.readthedocs.io/en/latest/image_domain_gridding.html>`_. To make this mode available :ref:`build the IDG library <build-instructions-label>` before `building WSClean <https://wsclean.readthedocs.io/en/latest/installation.html>`_ and make sure the IDG library can be found when compiling WSClean.
WSClean can then be invoked like this ::

    wclean -use-idg -size 1024 1024 -scale 10asec -idg-mode [cpu/hybrid] example.ms

To learn how to use IDG as a component in your own software the quickest route is to look at the :ref:`python-demo-label`. To run the python demo the IDG library needs to be built with python support and demos ::

    cmake -DBUILD_WITH_PYTHON=ON -DBUILD_WITH_DEMOS=ON
    make
    make install

The demo needs a measurement set. An example measurement set can be downloaded from
`<http://www.astron.nl/citt/ci_data/EveryBeam/MWA-1052736496-averaged.ms.tgz>`_
 
 ::

    idg-demo.py ~/data/MWA-1052736496-averaged.ms --column DATA

