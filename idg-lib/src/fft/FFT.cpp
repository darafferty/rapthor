#include "FFT.h"

#include <fftw3.h>

using namespace std;

namespace idg {

    void fft2f(int m, int n, complex<float> *data)
    {
        fftwf_complex *tmp = (fftwf_complex *) data;
        fftwf_plan plan;

        #pragma omp critical
        {
        fftwf_plan_with_nthreads(4);
        plan = fftwf_plan_dft_2d(m, n,
                                 tmp, tmp,
                                 FFTW_FORWARD,
                                 FFTW_ESTIMATE);
        }
        ifftshift(m, n, data);
        fftwf_execute(plan);
        fftshift(m, n, data);
        fftwf_destroy_plan(plan);
    }


    void fft2f(int n, complex<float> *data)
    {
        fft2f(n, n, data);
    }


    void fft2f(unsigned batch, int m, int n, complex<float> *data)
    {
        if (batch == 1) {
            fft2f(m, n, data);
            return;
        }

        fftwf_complex *tmp = (fftwf_complex *) data;
        fftwf_plan plan;

        #pragma omp critical
        {
        fftwf_plan_with_nthreads(1);
        plan = fftwf_plan_dft_2d(m, n,
                                 tmp, tmp,
                                 FFTW_FORWARD,
                                 FFTW_ESTIMATE);
        }

        #pragma omp parallel for private(tmp)
	    for (unsigned i = 0; i < batch; i++) {
            tmp = (fftwf_complex *) data + i * m * n;
            ifftshift(m, n, tmp);
            fftwf_execute_dft(plan, tmp, tmp);
            fftshift(m, n, tmp);
        }

        fftwf_destroy_plan(plan);
    }

    void ifft2f(int m, int n, complex<float> *data)
    {
        fftwf_complex *tmp = (fftwf_complex *) data;
        fftwf_plan plan;

        #pragma omp critical
        {
        fftwf_plan_with_nthreads(4);
        plan = fftwf_plan_dft_2d(m, n,
                                 tmp, tmp,
                                 FFTW_BACKWARD,
                                 FFTW_ESTIMATE);
        }
        ifftshift(m, n, data);
        fftwf_execute(plan);
        fftshift(m, n, data);
        fftwf_destroy_plan(plan);
    }

    void ifft2f(unsigned batch, int m, int n, complex<float> *data)
    {
        if (batch == 1) {
            ifft2f(m, n, data);
            return;
        }

        fftwf_complex *tmp = (fftwf_complex *) data;
        fftwf_plan plan;

        #pragma omp critical
        {
        fftwf_plan_with_nthreads(1);
        plan = fftwf_plan_dft_2d(m, n,
                                 tmp, tmp,
                                 FFTW_BACKWARD,
                                 FFTW_ESTIMATE);
        }

        #pragma omp parallel for private(tmp)
	    for (unsigned i = 0; i < batch; i++) {
            tmp = (fftwf_complex *) data + i * m * n;
            ifftshift(m, n, tmp);
            fftwf_execute_dft(plan, tmp, tmp);
            fftshift(m, n, tmp);
        }

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

        #pragma omp critical
        plan = fftwf_plan_dft_r2c_2d(m, n,
                                     data_in, tmp,
                                     FFTW_ESTIMATE);
        ifftshift(m, n, tmp);
        fftwf_execute(plan);
        fftshift(m, n, tmp);
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

        #pragma omp critical
        plan = fftwf_plan_dft_c2r_2d(m, n,
                                     tmp, data_out,
                                     FFTW_ESTIMATE);
        ifftshift(m, n, tmp);
        fftwf_execute(plan);
        fftshift(m, n, tmp);
        fftwf_destroy_plan(plan);
    }


    void ifft2f_c2r(int n, complex<float> *data_in, float *data_out)
    {
        ifft2f_c2r(n, n, data_in, data_out);
    }


    void resize2f(
        int   m_in,
        int   n_in,
        complex<float> *data_in,
        int   m_out,
        int   n_out,
        complex<float> *data_out)
    {
        // scale before FFT
        float s = 1.0f / (m_in * n_in);
        for (int i = 0; i < m_in; i++) {
            for (int j = 0; j < n_in; j++) {
                data_in[i*n_in + j] *= s;
            }
        }

        // in-place FFT
        fft2f(m_in, n_in, data_in);

        // put FFTed data in center
        int m_offset = int((m_out - m_in)/2);
        int n_offset = int((n_out - n_in)/2);
        if (m_offset >= 0 && n_offset >= 0) {
            for (int i = 0; i < m_in; i++) {
                for (int j = 0; j < n_in; j++) {
                    data_out[(i+m_offset)*n_out + (j+n_offset)] = data_in[i*n_in + j];
                }
            }
        } else if (m_offset < 0 && n_offset < 0) {
            m_offset = int((m_in - m_out)/2);
            n_offset = int((n_in - n_out)/2);
            for (int i = 0; i < m_out; i++) {
                for (int j = 0; j < n_out; j++) {
                    data_out[i*n_out + j] = data_in[(i+m_offset)*n_in + (j+n_offset)];
                }
            }
        } else if (m_offset >= 0 && n_offset < 0) {
            n_offset = int((n_in - n_out)/2);
            for (int i = 0; i < m_in; i++) {
                for (int j = 0; j < n_out; j++) {
                    data_out[(i+m_offset)*n_out + j] = data_in[i*n_in + (j+n_offset)];
                }
            }
        } else if (m_offset < 0 && n_offset >= 0) {
            m_offset = int((m_in - m_out)/2);
            for (int i = 0; i < m_out; i++) {
                for (int j = 0; j < n_in; j++) {
                    data_out[i*n_out + (j+n_offset)] = data_in[(i+m_offset)*n_in + j];
                }
            }
        }

        // in-place inverse FFT
        ifft2f(m_out, n_out, data_out);
    }


    void resize2f(
        int   m_in,
        int   n_in,
        float *data_in,
        int   m_out,
        int   n_out,
        float *data_out)
    {
        auto copy_in  = new std::complex<float>[m_in *n_in ];
        auto copy_out = new std::complex<float>[m_out*n_out];

        for (int i = 0; i < m_in; i++) {
            for (int j = 0; j < n_in; j++) {
                copy_in[i*n_in + j] = data_in[i*n_in + j];
            }
        }

        resize2f(m_in, n_in, copy_in, m_out, n_out, copy_out);

        for (int i = 0; i < m_out; i++) {
            for (int j = 0; j < n_out; j++) {
                data_out[i*n_out + j] = copy_out[i*n_out + j].real();
            }
        }

        delete [] copy_in;
        delete [] copy_out;
    }

} // end namespace idg

#include "FFTC.h"
