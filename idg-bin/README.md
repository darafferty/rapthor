## Quick installation guide:    

git clone https://gitlab.com/astron-dome/image-domain-gridder.git     
cd image-domain-gridder     
mkdir build; cd build           
cmake -DCMAKE_INSTALL_PREFIX=\<installpath\> ..       
make install            
source init-environment.sh         

## Installation options:      
(Best: use 'ccmake' to configure all options.)       

* BUILD_STATIC_LIBS: build static libraries, instead of shared ones       
* BUILD_LIB_CPU: build library 'libidg-cpu' for usage on CPU's      
* BUILD_LIB_CUDA: build library 'libidg-cuda' for usage on Nvidia GPU's      
* BUILD_LIB_OPENCL: build library 'libidg-opencl' for usage of OpenCL     
* BUILD_LIB_KNC: build library 'libidg-knc' for usage of Intel Xeon Phi KNC      
* BUILD_WITH_PYTHON: build Python module 'idg' to use IDG from Python       
* BUILD_WITH_BOOST_PYTHON: ...       
* BUILD_WITH_EXAMPLES: small executables to demonstrate the use of 'libidg-*'        

All other build options are for development purposes only, and should be
left at the default values by a regular user.      

All libraries are installed in '\<installpath\>/lib'. The header files in
'\<installpath\>/include'. The Python module in
'\<installpath\>/lib/python2.7/dist-packages'. Make sure that your
LD_LIBRARY_PATH and PYTHONPATH are set as appropiate.      
