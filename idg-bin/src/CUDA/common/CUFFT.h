#include <exception>

#include <cuda.h>
#include <cufft.h>
#include <iostream>

#define checkCuFFTcall(val)	__checkCuFFTcall((val), #val, __FILE__, __LINE__)

namespace cufft {
	class Error : public std::exception {
		public:
			Error(cufftResult result):
			_result(result)
			{}   

			virtual const char *what() const throw();

			operator cufftResult () const {   
				return _result;
			}   

		private:
			cufftResult _result;
	};  

	#if 1
	inline void __checkCuFFTcall(cufftResult result, char const *const func, const char *const file, int const line) {
		if (result != CUFFT_SUCCESS) {
			std::cerr << "CUFFT Error at " << file;
			std::cerr << ":" << line;
			std::cerr << " in function " << func;
			std::cerr << ": " << Error(result).what();
			std::cerr << std::endl;
			abort();
			//throw Error(result);
		}
	}
	#else
	inline void checkCuFFTcall(cufftResult result) {
		if (result != CUFFT_SUCCESS) {
			throw Error(result);
		}
	}
	#endif

	class C2C_1D {
		public:
			C2C_1D(unsigned n, unsigned count) {   
				checkCuFFTcall(cufftPlan1d(&plan, n, CUFFT_C2C, count));
			}   

			~C2C_1D() {   
				checkCuFFTcall(cufftDestroy(plan));
			}   

			void setStream(CUstream stream) {   
				checkCuFFTcall(cufftSetStream(plan, stream));
			}   

			void execute(cufftComplex *in, cufftComplex *out, int direction = CUFFT_FORWARD) {   
				checkCuFFTcall(cufftExecC2C(plan, in, out, direction));
			}   

		private:
			cufftHandle plan;
	};
	
	class C2C_2D {
		public:
			C2C_2D(unsigned nx, unsigned ny) {
				checkCuFFTcall(cufftPlan2d(&plan, nx, ny, CUFFT_C2C));
			}
			
			C2C_2D(unsigned nx, unsigned ny, unsigned stride, unsigned dist, unsigned count) {
				int n[] = {(int) ny, (int) nx};
				checkCuFFTcall(cufftPlanMany(&plan, 2, n, n, stride, dist, n, stride, dist, CUFFT_C2C, count));
			}
			
			~C2C_2D() {
			    checkCuFFTcall(cufftDestroy(plan));
			}
				
			void setStream(CUstream stream) {   
				checkCuFFTcall(cufftSetStream(plan, stream));
			}
			
			void execute(cufftComplex *in, cufftComplex *out, int direction = CUFFT_FORWARD) {
				checkCuFFTcall(cufftExecC2C(plan, in, out, direction));
			}
			
		private:
			cufftHandle plan;
	};
	
}
