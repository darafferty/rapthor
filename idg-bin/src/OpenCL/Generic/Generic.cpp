#include "Generic.h"

using namespace std;
using namespace idg::kernel::opencl;

namespace idg {
    namespace proxy {
        namespace opencl {
            Generic::Generic(
                Parameters params) :
                OpenCLNew(params)
            {
                #if defined(DEBUG)
                cout << "Generic::" << __func__ << endl;
                #endif
            }


            /* High level routines */
            void Generic::grid_visibilities(
                const complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }


            void Generic::degrid_visibilities(
                std::complex<float> *visibilities,
                const float *uvw,
                const float *wavenumbers,
                const int *baselines,
                const std::complex<float> *grid,
                const float w_offset,
                const int kernel_size,
                const std::complex<float> *aterm,
                const int *aterm_offsets,
                const float *spheroidal)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif
            }

            void Generic::transform(
                DomainAtoDomainB direction,
                complex<float>* grid)
            {
                #if defined(DEBUG)
                cout << __func__ << endl;
                #endif

                // Load device
                DeviceInstance *device    = devices[0];
                PowerSensor *power_sensor = device->get_powersensor();
				cl::Context &context      = device->get_context();

                // Constants
                auto nr_polarizations = mParams.get_nr_polarizations();
                auto gridsize = mParams.get_grid_size();
                clfftDirection sign = (direction == FourierDomainToImageDomain) ? CLFFT_BACKWARD : CLFFT_FORWARD;

                // Command queue
                cl::CommandQueue &queue = device->get_execute_queue();

                // Events
                vector<cl::Event> inputReady(1);
                vector<cl::Event> fftFinished(1);
                vector<cl::Event> outputReady(1);

                // Device memory
                cl::Buffer d_grid = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof_grid());

                // Performance counter
                PerformanceCounter counter_fft;
                #if defined(MEASURE_POWER_ARDUINO)
                counter_fft.setPowerSensor(&powerSensor);
                #endif

                // Load kernel function
                unique_ptr<GridFFT> kernel_fft = device->get_kernel_fft();

                // Perform fft shift
                double time_shift = -omp_get_wtime();
                kernel_fft->shift(grid);
                time_shift += omp_get_wtime();

                // Copy grid to device
                double time_input = -omp_get_wtime();
                queue.enqueueWriteBuffer(d_grid, CL_FALSE, 0, sizeof_grid(), grid, NULL, &inputReady[0]);
                inputReady[0].wait();
                time_input += omp_get_wtime();

                // Create FFT plan
                kernel_fft->plan(context, queue, gridsize, 1);

				// Launch FFT
                kernel_fft->launchAsync(queue, d_grid, sign);
                queue.enqueueMarkerWithWaitList(NULL, &fftFinished[0]);
                fftFinished[0].wait();

                // Copy grid to host
                double time_output = -omp_get_wtime();
                queue.enqueueReadBuffer(d_grid, CL_FALSE, 0, sizeof_grid(), grid, &fftFinished, &outputReady[0]);
                outputReady[0].wait();
                time_output += omp_get_wtime();

                // Perform fft shift
                time_shift = -omp_get_wtime();
                kernel_fft->shift(grid);
                time_shift += omp_get_wtime();

                // Perform fft scaling
                double time_scale = -omp_get_wtime();
                complex<float> scale = complex<float>(2, 0);
                if (direction == FourierDomainToImageDomain) {
                    kernel_fft->scale(grid, scale);
                }
                time_scale += omp_get_wtime();

                #if defined(REPORT_TOTAL)
                auxiliary::report("   input", time_input, 0, sizeof_grid(), 0);
                auxiliary::report("     fft",
                                  PerformanceCounter::get_runtime((cl_event) inputReady[0](), (cl_event) fftFinished[0]()),
                                  kernel_fft->flops(gridsize, 1),
                                  kernel_fft->bytes(gridsize, 1),
                                  0);
                auxiliary::report("  output", time_output, 0, sizeof_grid(), 0);
                auxiliary::report("fftshift", time_shift/2, 0, sizeof_grid() * 2, 0);
                if (direction == FourierDomainToImageDomain) {
                    auxiliary::report(" scaling", time_scale, 0, sizeof_grid() * 2, 0);
                }
                clog << endl;
                #endif
            } // end transform

        } // namespace opencl
    } // namespace proxy
} // namespace idg
