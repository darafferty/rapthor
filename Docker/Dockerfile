FROM ubuntu:22.04 as builder

# This Dockerfile builds a Docker image for the Rapthor pipeline
# and all its dependencies.

# By default, build non-portable code, for the oldest CPU that supports AVX2
ARG PORTABLE=FALSE
ARG TARGET_CPU=haswell

# Install all build-time dependencies
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y \
        bison \
        build-essential \
        casacore-dev \
        cmake \
        flex \
        gfortran \
        git \
        libarmadillo-dev \
        libboost-date-time-dev \
        libboost-filesystem-dev \
        libboost-numpy-dev \
        libboost-program-options-dev \
        libboost-python-dev \
        libboost-system-dev \
        libboost-test-dev \
        libcfitsio-dev \
        libfftw3-dev \
        libgsl-dev \
        libgtkmm-3.0-dev \
        libhdf5-serial-dev \
        liblua5.3-dev \
        libpng-dev \
        ninja-build \
        pybind11-dev \
        python3-dev \
        wcslib-dev \
        wget

# Prepare the environment for installing the Rapthor source dependencies.
RUN mkdir /src
RUN git config --global alias.shallow-clone "!git clone --depth 1 --recurse-submodules --shallow-submodules"
WORKDIR /src

# Install IDG
# A shallow clone of IDG breaks its version number. This is an issue for DP3
# since it has a minimal version requirement.
RUN git clone --recurse-submodules https://git.astron.nl/RD/idg.git
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DBUILD_WITH_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DPORTABLE=${PORTABLE} \
    -DTARGET_CPU=${TARGET_CPU} \
    -H/src/idg \
    -B/src/idg/build \
    -G Ninja
RUN ninja -C /src/idg/build install

# Install EveryBeam
# Do not compile python bindings, they will interfere with the ones in the
# binary wheel on PyPI.
RUN git shallow-clone https://git.astron.nl/RD/EveryBeam.git
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DBUILD_WITH_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DPORTABLE=${PORTABLE} \
    -DTARGET_CPU=${TARGET_CPU} \
    -H/src/EveryBeam \
    -B/src/EveryBeam/build \
    -G Ninja
RUN ninja -C /src/EveryBeam/build install

# Install Dysco
RUN git shallow-clone https://github.com/aroffringa/dysco.git
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DPORTABLE=${PORTABLE} \
    -DTARGET_CPU=${TARGET_CPU} \
    -H/src/dysco \
    -B/src/dysco/build \
    -G Ninja
RUN ninja -C /src/dysco/build install

# Install AOFlagger
RUN git shallow-clone https://gitlab.com/aroffringa/aoflagger.git
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DPORTABLE=${PORTABLE} \
    -DTARGET_CPU=${TARGET_CPU} \
    -H/src/aoflagger \
    -B/src/aoflagger/build \
    -G Ninja
RUN ninja -C /src/aoflagger/build install

# Install WSClean
RUN git shallow-clone https://gitlab.com/aroffringa/wsclean.git
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DBUILD_TESTING=OFF \
    -DPORTABLE=${PORTABLE} \
    -DTARGET_CPU=${TARGET_CPU} \
    -H/src/wsclean \
    -B/src/wsclean/build \
    -G Ninja
RUN ninja -C /src/wsclean/build install

# Install SAGECal libdirac
RUN git shallow-clone https://github.com/nlesc-dirac/sagecal.git
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DLIB_ONLY=1 \
    -H/src/sagecal \
    -B/src/sagecal/build \
    -G Ninja
RUN ninja -C /src/sagecal/build install

# Install DP3
RUN git shallow-clone https://git.astron.nl/RD/DP3.git
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DBUILD_TESTING=OFF \
    -DPORTABLE=${PORTABLE} \
    -DTARGET_CPU=${TARGET_CPU} \
    -DLIBDIRAC_PREFIX=/usr/ \
    -H/src/DP3 \
    -B/src/DP3/build \
    -G Ninja
RUN ninja -C /src/DP3/build install

# Generate file with one-liner package version descriptions
RUN mkdir -p /usr/local/share/rapthor; \
for p in $(find /src -type d -name .git | sort); \
do \
  d=$(dirname $p); b=$(basename $d); cd $d; l=$(git log -1 --oneline); \
  echo "$b: $l"; \
done \
> /usr/local/share/rapthor/sw-versions.txt

# Do not use `pip` from the Debian repository, but fetch it from PyPA.
# This way, we are sure that the latest versions of `pip`, `setuptools`, and
# `wheel` are installed in /usr/local, the only directory we're going to copy
# over to the next build stage.
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py

# Install all Python dependencies for Rapthor
COPY ./requirements.txt .
RUN python3 -m pip install -r requirements.txt

#---------------------------------------------------------------------------
# The image will now be rebuilt without adding the sources, in order to
# reduce the size of the image.
#---------------------------------------------------------------------------
FROM ubuntu:22.04 as runner

COPY --from=builder /usr/local /usr/local

# Only install run-time required packages
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        casacore-tools \
        libatkmm-1.6-1v5 \
        libblas3 \
        libboost-filesystem1.74.0 \
        libboost-program-options1.74.0 \
        libcairomm-1.0-1v5 \
        libcasa-casa6 \
        libcasa-fits6 \
        libcasa-measures6 \
        libcasa-ms6 \
        libcasa-scimath6 \
        libcasa-tables6 \
        libcfitsio9 \
        libfftw3-double3 \
        libfftw3-single3 \
        libglib2.0-0 \
        libglibmm-2.4-1v5 \
        libgomp1 \
        libgsl27 \
        libgtkmm-3.0-1v5 \
        libhdf5-103-1 \
        libhdf5-cpp-103-1 \
        liblapack3 \
        liblua5.3-0 \
        libpangomm-1.4-1v5 \
        libpng16-16 \
        libpython3.10 \
        libsigc++-2.0-0v5 \
        libstdc++6 \
        nodejs \
        python3 \
        python3-distutils \
        wget

RUN rm -rf /var/lib/apt/lists/*

# Install the casacore measures data. We purposely do not install these from
# the Ubuntu repository, but download the latest version directly from the
# ASTRON ftp site.
# Note: The file on the ftp site is updated daily. When warnings regarding
# leap seconds appear, ignore them or regenerate the docker image.
RUN mkdir -p /usr/share/casacore/data && \
    ln -s /usr/share/casacore /var/lib/casacore && \
    wget -qO - ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar | \
        tar -C /usr/share/casacore/data -xzf -

# Try to run the compiled tools to make sure they run without
# a problem (e.g. no missing libraries).
RUN aoflagger --version && \
    DP3 --version && \
    wsclean --version

# Install current version of Rapthor.
# Note: VERSION should be provided as build argument.
ARG VERSION=0.0.0
COPY . /tmp/rapthor
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_RAPTHOR=${VERSION} \
    python3 -m pip install --upgrade --no-cache-dir /tmp/rapthor
RUN rm -rf /tmp/*

# Set environment variables needed at run-time
ENV EVERYBEAM_DATADIR=/usr/local/share/everybeam
