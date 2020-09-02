.. _data_preparation:

Data preparation
================

Rapthor requires that the input data be prepared using the prefactor pipelines. These pipelines perform the calibration of the calibrator data, the removal of instrumental effects (e.g., station clock offsets), the setting of the overall amplitude scale, the calibration of the target data, and the subtraction of sources outside of the target field. The pipelines are available at https://github.com/lofar-astron/prefactor and must be run before Rapthor can be used.
