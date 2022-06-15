# Image Domain Gridding (IDG)

The documentation of IDG can be found [here](https://idg.readthedocs.io).

This repository contains the three subprojects of Image Domain Gridding:
 * [idg-lib](https://git.astron.nl/RD/idg/-/tree/master/idg-lib): The core library
 * [idg-bin](https://git.astron.nl/RD/idg/-/tree/master/idg-bin): Examples in C++ and Python
 * [idg-api](https://git.astron.nl/RD/idg/-/tree/master/idg-api): An API which is used in WSClean

# Installation
```
git clone https://git.astron.nl/RD/idg.git
cd idg && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/idg/ ..
ccmake . # For interactively setting parameters, like BUILD_LIB_CUDA
make
make install
```
