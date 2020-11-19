module load cmake/3.16.2
module load openblas/0.3.6-gcc-8.3.0
module load hdf5/1.10.6-gcc-8.3.0
module load gcc/8.3.0 
module load cfitsio/3.430-gcc-6.3.0
module load fftw/3.3.8-gcc-8.3.0
module load wcslib/5.18-gcc-6.3.0
module load lua/5.3.5
module load boost/1.73-gcc-8.3.0
module load pybind11/2.4.3
module load cuda101/toolkit

# The openblas module sets BLAS_LIB to /cm/shared/package/openblas/0.3.6-gcc-8.3.0/lib (without the "64"), which does not exist.
export OPENBLAS_LIB="/cm/shared/package/openblas/0.3.6-gcc-8.3.0/lib64"