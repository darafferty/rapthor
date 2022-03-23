FROM ubuntu:20.04 as builder

# This Docker image builds the dependencies for the Rapthor pipeline.
# It lives on the head of its dependencies.

# Install all build-time dependencies
RUN export DEBIAN_FRONTEND=noninteractive && \ 
    apt-get update && \
    apt-get install -y \
        bison \
        build-essential \
        casacore-data \
        casacore-dev \
        casacore-tools \
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
RUN git clone https://git.astron.nl/RD/idg.git
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DBUILD_WITH_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -H/src/idg \
    -B/src/idg/build \
    -G Ninja
RUN ninja -C /src/idg/build install

# Install EveryBeam
RUN git shallow-clone https://git.astron.nl/RD/EveryBeam.git
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DBUILD_WITH_PYTHON=ON \
    -DBUILD_TESTING=OFF \
    -H/src/EveryBeam \
    -B/src/EveryBeam/build \
    -G Ninja
RUN ninja -C /src/EveryBeam/build install

# Install Dysco
RUN git shallow-clone https://github.com/aroffringa/dysco.git
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -H/src/dysco \
    -B/src/dysco/build \
    -G Ninja
RUN ninja -C /src/dysco/build install

# Install AOFlagger
RUN git shallow-clone https://gitlab.com/aroffringa/aoflagger.git
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -H/src/aoflagger \
    -B/src/aoflagger/build \
    -G Ninja
RUN ninja -C /src/aoflagger/build install

# Install WSClean
RUN git shallow-clone https://gitlab.com/aroffringa/wsclean.git
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DBUILD_TESTING=OFF \
    -H/src/wsclean \
    -B/src/wsclean/build \
    -G Ninja
RUN ninja -C /src/wsclean/build install

# Install DP3
RUN git shallow-clone https://git.astron.nl/RD/DP3.git
RUN cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DBUILD_TESTING=OFF \
    -H/src/DP3 \
    -B/src/DP3/build \
    -G Ninja
RUN ninja -C /src/DP3/build install

# Do not use `pip` from the Debian repository, but fetch it from PyPA.
# This way, we are sure that the latest versions of `pip`, `setuptools`, and
# `wheel` are installed in /usr/local, the only directory we're going to copy
# over to the next build stage.
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py

# Install current version of Rapthor, and then uninstall it again.
COPY . rapthor
RUN python3 -m pip install --upgrade ./rapthor


#---------------------------------------------------------------------------
# The image will now be rebuilt without adding the sources, in order to
# reduce the size of the image.
#---------------------------------------------------------------------------
FROM ubuntu:20.04 as runner

COPY --from=builder /usr/local /usr/local

# Only install run-time required packages 
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y \
        casacore-tools \
        libarmadillo9 \
        libatkmm-1.6-1v5 \
        libboost-date-time1.71.0 \
        libboost-filesystem1.71.0 \
        libboost-program-options1.71.0 \
        libcairomm-1.0-1v5 \
        libcasa-casa4 \
        libcasa-fits4 \
        libcasa-measures4 \
        libcasa-ms4 \
        libcasa-scimath4 \
        libcasa-tables4 \
        libcfitsio8 \
        libfftw3-double3 \
        libfftw3-single3 \
        libglibmm-2.4-1v5 \
        libgomp1 \
        libgsl23 \
        libgtkmm-3.0-1v5 \
        libhdf5-103 \
        libhdf5-cpp-103 \
        liblapack3 \
        liblua5.3-0 \
        libpangomm-1.4-1v5 \
        libpng16-16 \
        libpython3.8 \
        libsigc++-2.0-0v5 \
        libstdc++6 \
        nodejs \
        python3 \
        python3-distutils \
        wget

RUN rm -rf /var/lib/apt/lists/*

# Install WSRT Measures (extra casacore data). We purposely do this in the
# `runner` stage and not in the `builder` stage, because otherwise the
# previous `apt-get install` would clobber the files we had installed in
# in the `builder` stage (as `casacore-data` is installed as a dependency).
# Note: The file on the ftp site is updated daily. When warnings regarding
# leap seconds appear, ignore them or regenerate the docker image.
RUN wget -q -O /WSRT_Measures.ztar \
        ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar && \
    cd /var/lib/casacore/data && \
    tar xfz /WSRT_Measures.ztar && \
    rm /WSRT_Measures.ztar

# Try to run the compiled tools to make sure they run without
# a problem (e.g. no missing libraries).
RUN aoflagger --version && \
    DP3 --version && \
    wsclean --version
