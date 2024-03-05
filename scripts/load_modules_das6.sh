# Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

module load spack/9.4.0
module load cmake
module load pkg-config
module load openblas
module load hdf5
module load cfitsio
module load fftw
module load boost
module load cuda/12.2.1

module load python
module load py-pytest
module load py-h5py
module load py-casacore
module load py-astropy
module load py-scipy
module load py-matplotlib
module load py-virtualenv

# Modules for integration test
module load py-pip
module load aoflagger
module load casacore
module load everybeam/0.4.0
module load dp3/5.4

