The Operation class
===================

The Operation class is used to define, set up, and run a operation's pipeline. A subclass of the Operation class is defined for each operation. See :ref:`operation_subclasses` for details of each Operation subclass.

.. autoclass:: rapthor.lib.operation.Operation
   :members:


.. _operation_subclasses:

Subclasses of the Operation class
---------------------------------

A subclass of the Operation class is defined for each of Rapthor's operations (see :ref:`operations`): calibrate, predict, image, and mosaic. These subclasses are described in detail below.

Calibrate
^^^^^^^^^
.. autoclass:: rapthor.operations.calibrate.Calibrate
   :members:

Predict
^^^^^^^
.. autoclass:: rapthor.operations.predict.Predict
   :members:

Image
^^^^^
.. autoclass:: rapthor.operations.image.Image
   :members:

Mosaic
^^^^^^
.. autoclass:: rapthor.operations.mosaic.Mosaic
   :members:
