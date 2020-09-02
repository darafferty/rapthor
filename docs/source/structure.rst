.. _structure:

Rapthor structure
=================

Rapthor is effectively a Python wrapper around CWL generic pipelines. The wrapper sets up and executes the pipelines that then perform the actual processing. The overall structure of the processing as done by Rapthor is shown in Figure :num:`rapthor-flowchart` below. The processing is divided into a number of operations, each of which can be run (or not) as needed.
.. _rapthor-flowchart:

.. figure:: rapthor_flow.png
   :figwidth: 90 %
   :align: center

   Rapthor flowchart
