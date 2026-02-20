The Operation class
===================

The Operation class is used to define, set up, and run a operation's CWL workflow. A subclass of the Operation class is defined for each operation. See :ref:`operation_subclasses` for details of each Operation subclass.

.. autoclass:: rapthor.lib.operation.Operation
   :members:


.. _operation_subclasses:

Subclasses of the Operation class
---------------------------------

A subclass of the Operation class is defined for each of Rapthor's operations (see :ref:`operations`): calibrate, predict, image, and mosaic. These subclasses are described in detail below.

The CalibrateDD class
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.calibrate.CalibrateDD
   :members:

The CalibrateDI class
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.calibrate.CalibrateDI
   :members:

The PredictDD class
^^^^^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.predict.PredictDD
   :members:

The PredictDI class
^^^^^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.predict.PredictDI
   :members:

The Image class
^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.image.Image
   :members:

The ImageInitial class
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.image.ImageInitial
   :members:

The Mosaic class
^^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.mosaic.Mosaic
   :members:
