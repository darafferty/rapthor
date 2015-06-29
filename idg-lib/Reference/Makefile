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
USE_LIKWID=0

#
#   Architecture specific
#
ARCH_AVX=-xAVX -lmkl_avx -lmkl_vml_avx
ARCH_AVX2=-xCORE-AVX2 -lmkl_avx2 -lmkl_vml_avx2
ARCH=${ARCH_AVX}

#
#   Dependencies
#
LIKWID_INC=-I/usr/local/include/likwid
FFTW3_INC=-I/opt/intel/composer_xe_2015/mkl/include/fftw/

#
#   IDG	
#
all: IDG.x IDG-WRAPPER.x

IDG.x: IDG.cpp Power.cpp ../Data/Init.cpp ../Common/Write.cpp ../Lib/uvwsim.c
	icpc -o $@ $^ -ldl -fopenmp -I../Common -I../Data -I../Lib ${LIKWID_INC} -mkl -llikwid -lnuma ${PARAMETERS} -DMEASURE_POWER=${MEASURE_POWER} -DUSE_LIKWID=${USE_LIKWID} ${ARCH}

IDG-WRAPPER.x: IDG-WRAPPER.cpp Proxies.cpp ../Data/Init.cpp ../Common/RW.cpp ../Lib/uvwsim.c
	icpc -o $@ $^ -ldl -fopenmp -I../Common -I../Data -I../Lib

clean:
	@rm -f *.x *.so */*.so *.gch */*.ptx */*~ *~ *.optrpt */*.optrpt 
