# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

module load spack/9.4.0
module load cmake/3.22.1
module load openblas/0.3.20
module load hdf5/1.10.7
module load cfitsio/3.49
module load fftw/3.3.10
module load boost/1.73.0
module load cuda/11.6.2

module load python/3.9.9
module load py-pip/21.3.1
module load py-pytest/6.2.5
module load py-h5py/3.6.0
module load py-casacore/3.4.0
module load py-astropy/4.0.1.post1
module load py-scipy/1.8.0
module load py-six/1.16.0 # Needed for python-casacore
module load py-python-dateutil/2.8.2
module load py-matplotlib/3.5.1

# Load modulefiles from the "schaap-stack"
module load aoflagger/3.1.0
module load casacore/3.4.0
module load everybeam/0.4.0
module load dp3/5.4
