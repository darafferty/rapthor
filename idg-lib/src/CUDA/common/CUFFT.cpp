#include "CUFFT.h"

#include <iostream>

#define checkCuFFTcall(val)	__checkCuFFTcall((val), #val, __FILE__, __LINE__)

namespace cufft {

	inline void __checkCuFFTcall(
        cufftResult result,
        char const *const func,
        const char *const file,
        int const line)
    {
		if (result != CUFFT_SUCCESS) {
            std::cerr << "CUFFT Error at " << file;
            std::cerr << ":" << line;
            std::cerr << " in function " << func;
            std::cerr << ": " << Error(result).what();
            std::cerr << std::endl;
	        throw Error(result);
		}
	}

    const char *Error::what() const throw()
    {
    	switch (_result) {
    		case  CUFFT_SUCCESS:
    		return "success";
    	case  CUFFT_INVALID_PLAN:
    		return "invalid plan";
    	case  CUFFT_ALLOC_FAILED:
    		return "alloc failed";
    	case  CUFFT_INVALID_TYPE:
    		return "invalid type";
    	case  CUFFT_INVALID_VALUE:
    		return "invalid value";
    	case  CUFFT_INTERNAL_ERROR:
    		return "internal error";
    	case  CUFFT_EXEC_FAILED:
    		return "exec failed";
    	case  CUFFT_SETUP_FAILED:
    		return "setup failed";
    	case  CUFFT_INVALID_SIZE:
    		return "invalid size";
    	case  CUFFT_UNALIGNED_DATA:
    		return "unaligned data";
    	#if defined CUFFT_INCOMPLETE_PARAMETER_LIST
    	case  CUFFT_INCOMPLETE_PARAMETER_LIST:
    		return "incomplete parameter list";
    	#endif
    	#if defined CUFFT_INVALID_DEVICE
    	case  CUFFT_INVALID_DEVICE:
    		return "invalid device";
    	#endif
    	#if defined CUFFT_PARSE_ERROR
    	case  CUFFT_PARSE_ERROR:
    		return "parse error";
    	#endif
    	#if defined CUFFT_NO_WORKSPACE
    	case  CUFFT_NO_WORKSPACE:
    		return "no workspace";
    	#endif
        default:
            return "unknown error";
    	}
    } // end what


    /*
        C2C_1D
    */
    C2C_1D::C2C_1D(unsigned n, unsigned count)
    {
    	checkCuFFTcall(cufftPlan1d(&plan, n, CUFFT_C2C, count));
    }

    C2C_1D::C2C_1D(unsigned n, unsigned stride, unsigned dist, unsigned count)
    {
        checkCuFFTcall(cufftPlanMany(&plan, 1, (int *) &n, (int *) &n, stride, dist, (int *) &n, stride, dist, CUFFT_C2C, count));
    }

    C2C_1D::~C2C_1D()
    {
    	checkCuFFTcall(cufftDestroy(plan));
    }

    void C2C_1D::setStream(CUstream stream)
    {
    	checkCuFFTcall(cufftSetStream(plan, stream));
    }

    void C2C_1D::execute(cufftComplex *in, cufftComplex *out, int direction)
    {
    	checkCuFFTcall(cufftExecC2C(plan, in, out, direction));
    }


    /*
        C2C_2D
    */
    C2C_2D::C2C_2D(unsigned nx, unsigned ny)
    {
    	checkCuFFTcall(cufftPlan2d(&plan, nx, ny, CUFFT_C2C));
    }

    C2C_2D::C2C_2D(unsigned nx, unsigned ny, unsigned stride, unsigned dist, unsigned count)
    {
    	int n[] = {(int) ny, (int) nx};
        checkCuFFTcall(cufftPlanMany(&plan, 2, n, n, stride, dist, n, stride, dist, CUFFT_C2C, count));
    }

    C2C_2D::~C2C_2D()
    {
        checkCuFFTcall(cufftDestroy(plan));
    }

    void C2C_2D::setStream(CUstream stream)
    {
    	checkCuFFTcall(cufftSetStream(plan, stream));
    }

    void C2C_2D::execute(cufftComplex *in, cufftComplex *out, int direction)
    {
    	checkCuFFTcall(cufftExecC2C(plan, in, out, direction));
    }


} // end namespace cufft
