#ifndef IDG_HEADER_
#define IDG_HEADER_

#include "idg-config.h"
#if defined(BUILD_LIB_CPU)
#include "idg-cpu.h"
#endif
#if defined(BUILD_LIB_CUDA)
#include "idg-cuda.h"
#endif
#if defined(BUILD_LIB_OPENCL)
#include "idg-opencl.h"
#endif

#endif
