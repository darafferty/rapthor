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
MEASURE_POWER=0
USE_LIKWID=1

#
#   Architecture specific
#
ARCH_AVX=-xAVX -lmkl_avx -lmkl_vml_avx
ARCH_AVX2=-xCORE-AVX2 -lmkl_avx2 -lmkl_vml_avx2
ARCH=${ARCH_AVX}

#
#   Dependencies
#
CUDA_INC=/usr/local/cuda-7.0/include
CUDA_LIB=/usr/local/cuda-7.0/lib64
LIKWID_INC=/usr/local/include/likwid
FFTW3_INC=/opt/intel/composer_xe_2015/mkl/include/fftw/

#
#   IDG	
#
IDG-XEON: XEON/IDG.cpp XEON/Power.cpp Data/Init.cpp Common/Write.cpp Lib/uvwsim.c
	icpc -o $@ $^ -ldl -fopenmp -ICommon -IData -ILib -I${LIKWID_INC} -mkl -llikwid -lnuma ${PARAMETERS} -DMEASURE_POWER=${MEASURE_POWER} -DUSE_LIKWID=${USE_LIKWID} ${ARCH}

IDG-XEON-WRAPPER: XEON/IDG-WRAPPER.cpp XEON/Proxies.cpp Data/Init.cpp Common/RW.cpp Lib/uvwsim.c
	icpc -o $@ $^ -ldl -fopenmp -ICommon -IData -ILib

IDG-CUDA: CUDA/IDG.cpp CUDA/Kernels.cpp CUDA/Power.cpp Data/Init.cpp CUDA/CU.cpp CUDA/CUFFT.cpp  Common/RW.cpp Lib/uvwsim.c
	g++ -o $@ $^ -ldl -lcuda -lcufft -fopenmp -ICommon -IData -ILib -I${CUDA_INC} -L${CUDA_LIB} ${PARAMETERS} -DMEASURE_POWER=${MEASURE_POWER}

IDG-CUDA-WRAPPER: CUDA/IDG-WRAPPER.cpp CUDA/Proxies.cpp CUDA/CU.cpp Data/Init.cpp Common/RW.cpp Lib/uvwsim.c
	g++ -o $@ $^ -ldl -lcuda -lcufft -fopenmp -ICommon -IData -ILib -I${CUDA_INC} -L${CUDA_LIB}

#
#	Util
#
BRANCH=$(shell git rev-parse --abbrev-ref HEAD)
HASH=$(shell git rev-parse HEAD | head -c 10)
archive:
	git archive -o idg-${BRANCH}-${HASH}.tar HEAD

EXEC=IDG-XEON IDG-XEON-WRAPPER IDG-CUDA IDG-CUDA-WRAPPER IDG-MIC
clean:
	@rm -f ${EXEC} *.so */*.so *.gch */*.ptx */*~ *~ *.optrpt */*.optrpt subgrid* grid* visibilities* powerdump
