#ifndef IDG_HEADER_
#define IDG_HEADER_

#include "idg-config.h"
#include "api/GridderPlan.h"
#include "api/DegridderPlan.h"
#include "api/Datatypes.h"
#include "api/Scheme.h"
#if defined(BUILD_LIB_CPU)
#include "idg-cpu.h"
#endif
#if defined(BUILD_LIB_CUDA)
// TODO: fix
//#include "idg-cuda.h"
#endif
#if defined(BUILD_LIB_OPENCL)
// TODO: fix
//#include "idg-opencl.h"
#endif

#endif
