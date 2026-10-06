The Operation class
===================

The Operation class is used to define, set up, and run an operation's CWL workflow. A subclass of the Operation class is defined for each operation. See :ref:`operation_subclasses` for details of each Operation subclass.

.. autoclass:: rapthor.lib.operation.Operation
   :members:


.. _operation_subclasses:

Subclasses of the Operation class
---------------------------------

The operation subclasses implement calibration, prediction, concatenation, imaging,
flux-scale normalization, and mosaicking (see :ref:`operations`).

The Calibrate class
^^^^^^^^^^^^^^^^^^^
The ``mode`` argument selects direction-dependent (``"dd"``) or
direction-independent (``"di"``) calibration.

.. autoclass:: rapthor.operations.calibrate.Calibrate
   :members:

The Predict class
^^^^^^^^^^^^^^^^^
The ``mode`` argument selects prediction for direction-dependent (``"dd"``)
processing or direction-independent (``"di"``) calibration.

.. autoclass:: rapthor.operations.predict.Predict
   :members:

The Concatenate class
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.concatenate.Concatenate
   :members:

The Image class
^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.image.Image
   :members:

The ImageInitial class
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.image.ImageInitial
   :members:

The ImageNormalize class
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.image.ImageNormalize
   :members:

The Mosaic class
^^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.mosaic.Mosaic
   :members:
