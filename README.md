# Image Domain Gridding (IDG)
This repository will contain the three subprojects of Image Domain Gridding:
 * [idg-lib](https://gitlab.com/astron-idg/idg-lib): The core library
 * [idg-bin](https://gitlab.com/astron-idg/idg-bin): Examples in C++ and Python
 * [idg-api](https://gitlab.com/astron-idg/idg-api): An API which is used in WSClean

Each repository can also be installed separately if desired.

# Installation
```
git clone --recursive https://gitlab.com/astron-idg/idg
cd idg && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/idg/ ..
```

# Work-in-progress!
For now, this head repository is not completely tested.
When in doubt, please install the submodules separately.
