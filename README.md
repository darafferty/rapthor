# Image Domain Gridding (IDG)

The documentation of IDG can be found [here](https://www.astron.nl/citt/IDG/).

This repository will contain the three subprojects of Image Domain Gridding:
 * [idg-lib](https://gitlab.com/astron-idg/idg-lib): The core library
 * [idg-bin](https://gitlab.com/astron-idg/idg-bin): Examples in C++ and Python
 * [idg-api](https://gitlab.com/astron-idg/idg-api): An API which is used in WSClean

Each repository can also be installed separately if desired.

# Installation
```
git clone https://gitlab.com/astron-idg/idg.git
cd idg && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/idg/ ..
ccmake . # For interactively setting parameters, like BUILD_LIB_CUDA
make
make install
```
