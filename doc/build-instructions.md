# Build instructions

## Quick installation guide:      
```
git clone --recursive -j4 https://gitlab.com/astron-idg/idg.git      
cd idg     
mkdir build; cd build           
cmake -DCMAKE_INSTALL_PREFIX=<idg_install_path> ..       
make install
```

## Installation options:      
(Best: use `ccmake` or `cmake -i` to configure all options.)       
```
* BUILD_STATIC_LIBS: build static libraries, instead of shared ones       
* BUILD_LIB_CPU: build library 'libidg-cpu' for usage on CPU's      
* BUILD_LIB_CUDA: build library 'libidg-cuda' for usage on Nvidia GPU's      
* BUILD_LIB_OPENCL: build library 'libidg-opencl' for usage of OpenCL     
* BUILD_WITH_PYTHON: build Python module 'idg' to use IDG from Python       
```
All other build options are for development purposes only, and should be
left at the default values by a regular user.      

All libraries are installed in `'<installpath>/lib'`. The header files in
`'<installpath>/include'`. The Python module in
`'<installpath>/lib/python3.8/site-packages'`. Make sure that your
`LD_LIBRARY_PATH` and `PYTHONPATH` are set as appropiate.      



