.. _structure:

rapthor structure
================

rapthor effectively sets up and runs generic pipelines that perform the actual processing. The overall structure of facet calibration as done by rapthor is shown in Figure :num:`rapthor-flowchart` below. The processing is divided into a number of operations, the division of which is largely determined by whether or not multiple operations may be run in parallel. In this flowchart, each operation is outlined with a black box.

.. _rapthor-flowchart:

.. figure:: rapthor_flow.png
   :figwidth: 90 %
   :align: center

   rapthor flowchart
