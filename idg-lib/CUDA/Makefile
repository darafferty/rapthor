#
#   Parameters
#
NR_STATIONS=70
NR_TIME=2048
NR_CHANNELS=16
NR_POLARIZATIONS=4
GRIDSIZE=512
SUBGRIDSIZE=32
JOBSIZE=300
IMAGESIZE=0.1
W_PLANES=10
CHUNKSIZE=16

PARAMETERS=-DIMAGESIZE=${IMAGESIZE} -DW_PLANES=${W_PLANES} -DNR_STATIONS=${NR_STATIONS} -DNR_TIME=${NR_TIME} -DNR_CHANNELS=${NR_CHANNELS} -DNR_POLARIZATIONS=${NR_POLARIZATIONS} -DGRIDSIZE=${GRIDSIZE} -DSUBGRIDSIZE=${SUBGRIDSIZE} -DJOBSIZE=${JOBSIZE} -DCHUNKSIZE=${CHUNKSIZE}

#
#   Benchmarking
#
MEASURE_POWER=1

#
#   Dependencies
#
CUDA_INC=/usr/local/cuda-7.0/include
CUDA_LIB=/usr/local/cuda-7.0/lib64

#
#   IDG	
#
default: IDG-CUDA IDG-CUDA-WRAPPER

IDG-CUDA: IDG.cpp Kernels.cpp Power.cpp ../Data/Init.cpp CU.cpp CUFFT.cpp  ../Common/RW.cpp ../Lib/uvwsim.c
	g++ -o $@ $^ -ldl -lcuda -lcufft -fopenmp -I../Common -I../Data -I../Lib -I${CUDA_INC} -L${CUDA_LIB} ${PARAMETERS} -DMEASURE_POWER=${MEASURE_POWER}

IDG-CUDA-WRAPPER: IDG-WRAPPER.cpp Proxies.cpp CU.cpp ../Data/Init.cpp ../Common/RW.cpp ../Lib/uvwsim.c
	g++ -o $@ $^ -ldl -lcuda -lcufft -fopenmp -I../Common -I../Data -I../Lib -I${CUDA_INC} -L${CUDA_LIB}

EXEC=IDG-CUDA IDG-CUDA-WRAPPER

clean:
	@rm -f ${EXEC} *.so */*.so *.gch *.ptx */*~ *~ powerdump
