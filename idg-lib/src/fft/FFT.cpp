#include "FFT.h"

using namespace std;

namespace idg {

    void fft2f(int m, int n, complex<float> *data)
    {
        fftwf_complex *tmp = (fftwf_complex *) data;
        fftwf_plan plan;
        plan = fftwf_plan_dft_2d(m, n,
                                 tmp, tmp,
                                 FFTW_FORWARD,
                                 FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftshift(m, n, data);
        fftwf_destroy_plan(plan);
    }


    void fft2f(int n, complex<float> *data)
    {
        fft2f(n, n, data);
    }


    void ifft2f(int m, int n, complex<float> *data)
    {
        fftwf_complex *tmp = (fftwf_complex *) data;
        fftwf_plan plan;
        plan = fftwf_plan_dft_2d(m, n,
                                 tmp, tmp,
                                 FFTW_BACKWARD,
                                 FFTW_ESTIMATE);
        ifftshift(m, n, data);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
    }


    void ifft2f(int n, complex<float> *data)
    {
        ifft2f(n, n, data);
    }


    void fft2f_r2c(int m, int n, float *data_in, complex<float> *data_out)
    {
        fftwf_complex *tmp = (fftwf_complex *) data_out;
        fftwf_plan plan;
        plan = fftwf_plan_dft_r2c_2d(m, n,
                                     data_in, tmp,
                                     FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
    }


    void fft2f_r2c(int n, float *data_in, complex<float> *data_out)
    {
        fft2f_r2c(n, n, data_in, data_out);
    }


    void ifft2f_c2r(int m, int n, complex<float> *data_in, float *data_out)
    {
        fftwf_complex *tmp = (fftwf_complex *) data_in;
        fftwf_plan plan;
        plan = fftwf_plan_dft_c2r_2d(m, n,
                                     tmp, data_out,
                                     FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
    }


    void ifft2f_c2r(int n, complex<float> *data_in, float *data_out)
    {
        ifft2f_c2r(n, n, data_in, data_out);
    }


    void fft2(int m, int n, complex<double> *data)
    {
        fftw_complex *tmp = (fftw_complex *) data;
        fftw_plan plan;
        plan = fftw_plan_dft_2d(m, n,
                                tmp, tmp,
                                FFTW_FORWARD,
                                FFTW_ESTIMATE);
        fftw_execute(plan);
        fftshift(m, n, data);
        fftw_destroy_plan(plan);
    }


    void fft2(int n, complex<double> *data)
    {
        fft2(n, n, data);
    }


    void ifft2(int m, int n, complex<double> *data)
    {
        fftw_complex *tmp = (fftw_complex *) data;
        fftw_plan plan;
        plan = fftw_plan_dft_2d(m, n,
                                tmp, tmp,
                                FFTW_BACKWARD,
                                FFTW_ESTIMATE);
        ifftshift(m, n, data);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }


    void ifft2(int n, complex<double> *data)
    {
        ifft2(n, n, data);
    }

}




// C interface:
// Rationale: calling the code from C code and Fortran easier,
// and bases to create interface to scripting languages such as
// Python, Julia, Matlab, ...
extern "C" {

    void fft2f(int m, int n, void* data) {
        idg::fft2f(m, n, (complex<float>*) data);
    }

    void ifft2f(int m, int n, void* data) {
        idg::ifft2f(m, n, (complex<float>*) data);
    }

    void fft2f_r2c(int m, int n, void* data_in, void* data_out) {
        idg::fft2f_r2c(m, n, (float*) data_in, (complex<float>*) data_out);
    }

    void ifft2f_c2r(int m, int n, void* data_in, void* data_out) {
        idg::ifft2f_c2r(m, n, (complex<float>*) data_in, (float*) data_out);
    }

    void fftshift2f(int m, int n, void* array) {
        idg::fftshift(m, n, (complex<float>*) array);
    }

    void ifftshift2f(int m, int n, void* array) {
        idg::ifftshift(m, n, (complex<float>*) array);
    }

    void fft2(int m, int n, void* data) {
        idg::fft2(m, n, (complex<double>*) data);
    }

    void ifft2(int m, int n, void* data) {
        idg::ifft2(m, n, (complex<double>*) data);
    }

}
