
FROM ubuntu:22.04 AS base

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		build-essential \
		ca-certificates \
		cmake \
		gfortran \
		git \
		libblas3 \
		libboost-filesystem1.74.0 \
		libboost-numpy-dev \
		libboost-numpy1.74.0 \
		libboost-program-options1.74.0 \
		libboost-python-dev \
		libboost-python1.74.0 \
		libcfitsio-bin \
		libfftw3-double3 \
		libfftw3-single3 \
		libgfortran5 \
		libgomp1 \
		libgsl27 \
		libhdf5-103-1 \
		libhdf5-cpp-103-1 \
		libhdf5-dev \
		liblapack3 \
		liblua5.4-0 \
		libpng16-16 \
		libpython3.10 \
		libstdc++6 \
		libwcs7 \
		ninja-build \
		nodejs \
		python3 \
		python3-dev \
		python3-distutils \
		python3-pip \
		python3-pytest \
		&& rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/lib

FROM ubuntu:22.04 AS build

# Build arguments
ARG CASACORE_COMMIT=master
ARG EVERYBEAM_COMMIT=master
ARG WSCLEAN_COMMIT=master
ARG AOFLAGGER_COMMIT=master
ARG IDG_COMMIT=main
ARG SAGECAL_COMMIT=master
ARG DP3_COMMIT=master
ARG PYTHONCASACORE_COMMIT=master
ARG PORTABLE=FALSE
ARG TARGET_CPU=native

WORKDIR /build

# Use "nl.archive.ubuntu.com", instead of "archive.ubuntu.com".
RUN sed -ri "s/(archive\.ubuntu\.com)/nl.\1/" /etc/apt/sources.list

# Install all build-time dependencies
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y \
        bison \
        flex \
        g++ \
        gfortran \
        git \
        libboost-date-time-dev \
        libboost-filesystem-dev \
        libboost-program-options-dev \
        libboost-python-dev \
        libcfitsio-dev \
        libfftw3-dev \
        libgsl-dev \
        libhdf5-dev \
        liblapack-dev \
        liblua5.4-dev \
        libpng-dev \
        libpython3-dev \
        libreadline-dev \
        ninja-build \
        pkgconf \
        pybind11-dev \
        python3 \
        wcslib-dev \
        wget

# Do not use `pip` from the Debian repository, but fetch it from PyPA.
# This way, we are sure that the latest versions of `pip`, `setuptools`, and
# `wheel` are installed in /usr/local, the only directory we're going to copy
# over to the next build stage.
RUN wget -q https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py

# Install required python packages
RUN python3 -m pip install \
    'cmake<4' \
    numpy

# Build Boost.NumPy from source against pip NumPy 2 (the apt package is compiled
# against NumPy 1.x and causes runtime failures when importing PyBDSF).
# Install it into an isolated prefix so downstream CMake projects continue to
# use distro Boost packages without version-mix issues.
ARG BOOST_VERSION=1.88.0
ARG BOOST_VERSION_UNDERSCORED=1_88_0
RUN wget -q https://archives.boost.io/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_UNDERSCORED}.tar.bz2 && \
    tar -xjf boost_${BOOST_VERSION_UNDERSCORED}.tar.bz2 && \
    cd boost_${BOOST_VERSION_UNDERSCORED} && \
    NUMPY_INC=$(python3 -c "import numpy; print(numpy.get_include())") && \
    ./bootstrap.sh --with-libraries=python --with-python=/usr/bin/python3 && \
    ./b2 --with-python \
        python=3.10 \
        cxxflags="-I${NUMPY_INC}" \
        include="/usr/include/python3.10:${NUMPY_INC}" \
        variant=release \
        threading=multi \
        link=shared \
        --layout=system \
        --prefix=/opt/boost-numpy \
        --libdir=/opt/boost-numpy/lib \
        install && \
    cd .. && rm -rf boost_${BOOST_VERSION_UNDERSCORED}*

# Install the casacore measures data. We purposely do not install these from
# the Ubuntu repository, but download the latest version from the ASTRON site.
# Note: The file on the ftp site is updated daily. When warnings regarding
# leap seconds appear, ignore them or regenerate the docker image.
RUN mkdir -p /usr/local/share/casacore/data && \
    wget -qO - https://www.astron.nl/iers/WSRT_Measures.ztar | \
        tar -C /usr/local/share/casacore/data -xzf -

# Install Casacore
RUN git clone --no-checkout https://github.com/casacore/casacore.git
RUN git -C casacore checkout ${CASACORE_COMMIT}
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
    -DBoost_NO_BOOST_CMAKE=ON \
    -DBUILD_TESTING=OFF \
    -DPORTABLE=${PORTABLE} \
    -DTARGET_CPU=${TARGET_CPU} \
    -H/build/casacore \
    -B/build/casacore/build \
    -G Ninja
RUN ninja -C /build/casacore/build install

# Install IDG
RUN git clone --no-checkout https://git.astron.nl/RD/idg.git
RUN git -C idg checkout ${IDG_COMMIT}
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
    -DBUILD_WITH_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DPORTABLE=${PORTABLE} \
    -DTARGET_CPU=${TARGET_CPU} \
    -H/build/idg \
    -B/build/idg/build \
    -G Ninja
RUN ninja -C /build/idg/build install

# Install EveryBeam. Do not compile python bindings, they will interfere with
# the ones in the binary wheel on PyPI.
RUN git clone --no-checkout https://git.astron.nl/RD/EveryBeam.git
RUN git -C EveryBeam checkout ${EVERYBEAM_COMMIT}
# Initialize git submodules before CMake to avoid fetch failures during configuration
RUN git -C EveryBeam submodule update --init --recursive
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
    -DBUILD_WITH_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DPORTABLE=${PORTABLE} \
    -DTARGET_CPU=${TARGET_CPU} \
    -H/build/EveryBeam \
    -B/build/EveryBeam/build \
    -G Ninja
RUN ninja -C /build/EveryBeam/build install

# Install AOFlagger
RUN git clone --no-checkout https://gitlab.com/aroffringa/aoflagger.git
RUN git -C aoflagger checkout ${AOFLAGGER_COMMIT}
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
    -DPORTABLE=${PORTABLE} \
    -DTARGET_CPU=${TARGET_CPU} \
    -DENABLE_GUI=OFF \
    -H/build/aoflagger \
    -B/build/aoflagger/build \
    -G Ninja
RUN ninja -C /build/aoflagger/build install

# Install WSClean
RUN git clone --no-checkout https://gitlab.com/aroffringa/wsclean.git
RUN git -C wsclean checkout ${WSCLEAN_COMMIT}
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
    -DBUILD_TESTING=OFF \
    -DPORTABLE=${PORTABLE} \
    -DTARGET_CPU=${TARGET_CPU} \
    -H/build/wsclean \
    -B/build/wsclean/build \
    -G Ninja
RUN ninja -C /build/wsclean/build install

# Install SAGECal libdirac
RUN git clone --no-checkout https://github.com/nlesc-dirac/sagecal.git
RUN git -C sagecal checkout ${SAGECAL_COMMIT}
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
    -DLIB_ONLY=1 \
    -H/build/sagecal \
    -B/build/sagecal/build \
    -G Ninja
RUN ninja -C /build/sagecal/build install

# Install DP3
RUN git clone --no-checkout https://git.astron.nl/RD/DP3.git
RUN git -C DP3 checkout ${DP3_COMMIT}
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
    -DBUILD_TESTING=OFF \
    -DPORTABLE=${PORTABLE} \
    -DTARGET_CPU=${TARGET_CPU} \
    -DLIBDIRAC_PREFIX=/usr/ \
    -DMETADATA_COMPRESSION_DEFAULT=ON \
    -H/build/DP3 \
    -B/build/DP3/build \
    -G Ninja
RUN ninja -C /build/DP3/build install

# Install python-casacore
RUN git clone --no-checkout https://github.com/casacore/python-casacore.git
RUN git -C python-casacore checkout ${PYTHONCASACORE_COMMIT}
RUN CASACORE_DATA=/usr/local/share/casacore/data \
    pip install -v /build/python-casacore

# We need to `pip install` EveryBeam from source as well, because the C++
# build does not produce a proper Python package.
RUN pip install -v /build/EveryBeam

FROM base AS dev 

WORKDIR /app
# Copy files compiled in /build
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /usr/local/lib /usr/local/lib
COPY --from=build /usr/local/share /usr/local/share
COPY --from=build /opt/boost-numpy /opt/boost-numpy

# Use legacy FindBoost behavior for pip/scikit-build projects (e.g. PyBDSF).
# This avoids Boost package-config variant matching issues.
ENV BOOST_ROOT=/opt/boost-numpy
ENV BOOST_INCLUDEDIR=/opt/boost-numpy/include
ENV BOOST_LIBRARYDIR=/opt/boost-numpy/lib
ENV CMAKE_ARGS="-DBoost_NO_BOOST_CMAKE=ON -DBOOST_ROOT=/opt/boost-numpy -DBOOST_INCLUDEDIR=/opt/boost-numpy/include -DBOOST_LIBRARYDIR=/opt/boost-numpy/lib -DBoost_PYTHON_LIBRARY=/opt/boost-numpy/lib/libboost_python310.so -DBoost_PYTHON_LIBRARY_RELEASE=/opt/boost-numpy/lib/libboost_python310.so -DBoost_NUMPY_LIBRARY=/opt/boost-numpy/lib/libboost_numpy310.so -DBoost_NUMPY_LIBRARY_RELEASE=/opt/boost-numpy/lib/libboost_numpy310.so"
ENV CPLUS_INCLUDE_PATH=/opt/boost-numpy/include
ENV LD_LIBRARY_PATH=/opt/boost-numpy/lib:/usr/local/lib

# Sanity check that the installed binaries are working
RUN aoflagger --version && \
DP3 --version && \
wsclean --version

CMD ["/bin/bash"]
