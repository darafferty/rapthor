The Operation class
===================

The Operation class is used to define, set up, and run one of Rapthor's
operations. A subclass of the Operation class is defined for each operation.
See :ref:`operation_subclasses` for details of each Operation subclass.

.. autoclass:: rapthor.lib.operation.Operation
   :members:


.. _operation_subclasses:

Subclasses of the Operation class
---------------------------------

A subclass of the Operation class is defined for each of Rapthor's operations (see :ref:`operations`): concatenate, calibrate, predict, image, and mosaic. These subclasses are described in detail below.

The Concatenate class
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.concatenate.Concatenate
   :members:

The Calibrate class
^^^^^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.calibrate.base.Calibrate
   :members:

The Predict class
^^^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.predict.Predict
   :members:

The Image class
^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.image.base.Image
   :members:

The ImageInitial class
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.image.initial.ImageInitial
   :members:

The ImageNormalize class
^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.image.normalize.ImageNormalize
   :members:

The Mosaic class
^^^^^^^^^^^^^^^^
.. autoclass:: rapthor.operations.mosaic.Mosaic
   :members:
