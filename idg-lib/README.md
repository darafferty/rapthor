# Quick installation guide:      

mkdir idg; cd idg     
git clone https://gitlab.com/astron-idg/code.git      
cd code     
mkdir build; cd build           
cmake -DCMAKE_INSTALL_PREFIX=\<idg\_install\_path\> ..       
make install            
[ source init-environment.sh ]         

## Installation options:      
(Best: use 'ccmake' or 'cmake -i' to configure all options.)       

* BUILD_STATIC_LIBS: build static libraries, instead of shared ones       
* BUILD_LIB_CPU: build library 'libidg-cpu' for usage on CPU's      
* BUILD_LIB_CUDA: build library 'libidg-cuda' for usage on Nvidia GPU's      
* BUILD_LIB_OPENCL: build library 'libidg-opencl' for usage of OpenCL     
* BUILD_WITH_PYTHON: build Python module 'idg' to use IDG from Python       
* BUILD_WITH_BOOST_PYTHON: ...       

All other build options are for development purposes only, and should be
left at the default values by a regular user.      

All libraries are installed in '\<installpath\>/lib'. The header files in
'\<installpath\>/include'. The Python module in
'\<installpath\>/lib/python2.7/dist-packages'. Make sure that your
LD_LIBRARY_PATH and PYTHONPATH are set as appropiate.      


# Using IDG in your own cmake project:

## Finding IDG    

In your CMakeLists.txt file, use for instance:     

*find_package(IDG  NO_MODULE)*         

(See more on the find_package command in the cmake documentation.)      

When building your project, use CMAKE_PREFIX_PATH or IDG_DIR    

*cmake  -DCMAKE_PREFIX_PATH=\<idg\_install\_path\>  \<project\_src\_path\>*
     
or       

*cmake  -DIDG_DIR=\<idg\_install\_path\>  \<project\_src\_path\>*

## Predefined varibles from find_package

*  IDG_INCLUDE_DIR: general include directories       
*  IDG_LIB: IDG library to link against      
*  IDG_PYTHON_MODULE_PATH: path to 'idg' python module    

## Linking against IDG

For instance, for the CPU library, include the "idg-cpu." in your source file. In cmake, set the include path and link to the libarary:     

*include_directories (${IDG_INCLUDE_DIR})*    
*target_link_libraries (a.out ${IDG_LIB})*      

