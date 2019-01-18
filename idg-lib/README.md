# Introduction
This library implements the image domain gridding method for gridding and degridding of visibilities from an aperture synthesis radio telescope.

# Quick installation guide:      
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
`'<installpath>/lib/python2.7/dist-packages'`. Make sure that your
`LD_LIBRARY_PATH` and `PYTHONPATH` are set as appropiate.      


# Using IDG in your own cmake project:

## Finding IDG    

In your CMakeLists.txt file, use for instance:     

`find_package(IDG NO_MODULE)`

(See more on the find_package command in the cmake documentation.)      

When building your project, use `CMAKE_PREFIX_PATH` or `IDG_DIR`    

`cmake  -DCMAKE_PREFIX_PATH=<idg_install_path>  <project_src_path>`
     
or       

`cmake -DIDG_DIR=<idg_install_path> <project_src_path>`

## Predefined varibles from find_package

*  `IDG_INCLUDE_DIR`: general include directories       
*  `IDG_LIB`: IDG library to link against      
*  `IDG_PYTHON_MODULE_PATH`: path to 'idg' python module    

## Linking against IDG

For instance, for the CPU library, include the `idg-cpu.` in your source file. In cmake, set the include path and link to the libarary:     
```
*include_directories (${IDG_INCLUDE_DIR})*    
*target_link_libraries (a.out ${IDG_LIB})*      
```
# Legal
ASTRON has a patent pending on the Image Domain Gridding method implemented by
this software.  This software is licensed under the GNU General Public License
v3.0 (GPLv3.0).  That means that you are allowed to use, modify, distribute and 
sell this software under the conditions of GPLv3.0. For the exact terms of
GPLv3.0, see the LICENSE file.  Making, using, distributing, or offering for 
sale of independent implementations of the Image Domain Gridding method, other
than under GPLv3.0, might infringe upon patent claims by ASTRON. In that case
you must obtain a license from ASTRON.

ASTRON (Netherlands Institute for Radio Astronomy)  
Attn. IPR Department  
P.O.Box 2, 7990 AA Dwingeloo, The Netherlands  
T +31 521 595100
