# Image Domain Gridding (IDG)

[![pipeline status](https://gitlab.com/astron-idg/idg/badges/master/pipeline.svg)](https://gitlab.com/astron-idg/idg/commits/master)


This repository will contain the three subprojects of Image Domain Gridding:
 * [idg-lib](https://gitlab.com/astron-idg/idg-lib): The core library
 * [idg-bin](https://gitlab.com/astron-idg/idg-bin): Examples in C++ and Python
 * [idg-api](https://gitlab.com/astron-idg/idg-api): An API which is used in WSClean

Each repository can also be installed separately if desired.

# Installation
```
git clone --recursive https://gitlab.com/astron-idg/idg.git
cd idg && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/idg/ ..
ccmake . # For interactively setting parameters, like BUILD_LIB_CUDA
make
make install
```

# Developing with submodules
If you intend to develop with this repository, it is useful to have the submodules point to the latest versions, instead of a frozen commit. To this purpose, please use `git clone --recursive --remote https://gitlab.com/astron-idg/idg` on newer git versions, or `git clone --recursive https://gitlab.com/astron-idg/idg`. On older git versions, update the submodules manually e.g. by typing `cd idg-api && git pull origin master`.
