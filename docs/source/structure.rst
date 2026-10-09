.. _structure:

General structure
=================

Rapthor divides the processing into a number of operations. Each operation
does one part of the work, such as calibration or imaging, by running DP3,
WSClean and a number of Python steps. A full processing run consists of one or
more cycles, and in each cycle the operations are run (or not) as needed. The
overall structure of the processing is shown in the figure below.

.. _rapthor-flowchart:

.. mermaid::
   :caption: Rapthor flowchart

   %%{init: {"flowchart": {"nodeSpacing": 35, "rankSpacing": 40, "wrappingWidth": 340}}}%%
   flowchart TB
       input(["Input Measurement Sets,<br/>parset and strategy"])
       concat["<b>Concatenate</b><br/>Join the frequency bands<br/>of each epoch"]
       initial["<b>Initial image</b><br/>Image the field and make<br/>the first sky model"]

       subgraph cycle["One cycle"]
           direction TB
           calibrate["<b>Calibrate</b><br/>Solve for the calibration<br/>solutions with DP3"]
           predict["<b>Predict</b><br/>Subtract the sources that<br/>will not be imaged"]
           image["<b>Image</b><br/>Image each sector with WSClean,<br/>make the new sky model<br/>and the diagnostics"]
           mosaic["<b>Mosaic</b><br/>Join the sector images"]
           calibrate --> predict --> image --> mosaic
       end

       check{"Has selfcal<br/>converged?"}
       final["<b>Final cycle</b><br/>The same operations, on the<br/>final fraction of the data"]
       output(["Images, solutions, sky models<br/>and diagnostics"])

       input --> concat --> initial --> calibrate
       mosaic --> check
       check -- "No: start the next cycle<br/>with the new sky model" --> calibrate
       check -- "Yes" --> final --> output

       classDef operation fill:#438dd5,stroke:#2e6295,color:#ffffff
       classDef data fill:#ffffff,stroke:#444444,color:#000000
       classDef boundary fill:#ffffff,stroke:#444444,stroke-dasharray:5 5,color:#444444
       class concat,initial,calibrate,predict,image,mosaic,final operation
       class input,output,check data
       class cycle boundary

Not every operation is run in every cycle. The operations, in the order in
which they are run, are:

Concatenate
    Run once, at the start, and only if an epoch was supplied as several
    Measurement Sets at different frequencies (as output by LINC). The
    operation is named ``concatenate_1``.

Initial image
    Run once, and only if :term:`generate_initial_skymodel` is set and the
    strategy includes calibration. The full field is imaged from the input
    data, without any further calibration, to make the sky model used in the
    first cycle. The operation is named ``initial_image``.

Calibrate
    Run in each cycle for which the strategy sets :term:`do_calibrate`. The
    :term:`calibration_strategy` of the cycle sets which solves are done and
    whether they are direction dependent (DD), direction independent (DI) or
    both. The DD operation is named ``calibrate_X``, where ``X`` is the cycle
    number. A DI calibration consists of two operations: ``predict_di_X``,
    which predicts the model visibilities, and ``calibrate_di_X``, which does
    the solve.

Predict
    Run when sources have to be subtracted from the data before imaging:
    outlier sources, bright sources, or the sources in other imaging sectors.
    The operation is named ``predict_X``.

Image
    Run in each cycle for which the strategy sets :term:`do_image`. If the
    strategy sets :term:`do_normalize`, an extra imaging operation named
    ``normalize_X`` is run first to derive the flux-scale normalization. The
    main imaging operation is named ``image_X``.

Mosaic
    Run after imaging. If there is more than one imaging sector, the sector
    images are joined into a single image of the field. The operation is named
    ``mosaic_X``.

At the end of each self calibration cycle for which the strategy sets
:term:`do_check`, Rapthor compares the image noise, the dynamic range and the
number of sources with those of the previous cycle. Self calibration stops
when it has converged, or when it has diverged or failed. Once it has
converged, a final cycle is done with the fraction of the data set by
:term:`final_data_fraction`. The final cycle is skipped if self calibration
diverged or failed, or if the last self calibration cycle already used the
same settings and the same fraction of the data.

The operations are described in detail in :ref:`operations`. The way Rapthor
is put together, and where its parts run, is described in :ref:`architecture`.
Details of the Python code are given in :ref:`code`.
