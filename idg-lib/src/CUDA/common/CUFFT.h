#ifndef IDG_CUFFT_H_
#define IDG_CUFFT_H_

#include <stdexcept>

#include <cuda.h>
#include <cufft.h>


namespace cufft {

	class Error : public std::exception {
		public:
			Error(cufftResult result):
			_result(result) {}

			virtual const char *what() const throw();

			operator cufftResult () const {
				return _result;
			}

		private:
			cufftResult _result;
	};

	class C2C_1D {
		public:
			C2C_1D(unsigned n, unsigned count);
			C2C_1D(unsigned n, unsigned stride, unsigned dist, unsigned count);
			~C2C_1D();
			void setStream(CUstream stream);
			void execute(cufftComplex *in, cufftComplex *out, int direction = CUFFT_FORWARD);

		private:
			cufftHandle plan;
	};

	class C2C_2D {
		public:
			C2C_2D(unsigned nx, unsigned ny);
			C2C_2D(unsigned nx, unsigned ny, unsigned stride, unsigned dist, unsigned count);
			~C2C_2D();
			void setStream(CUstream stream);
			void execute(cufftComplex *in, cufftComplex *out, int direction = CUFFT_FORWARD);

		private:
			cufftHandle plan;
	};

} // end namespace cufft

#endif
