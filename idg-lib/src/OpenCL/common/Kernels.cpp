#include <cstdint> // unint64_t

#include "idg-config.h"
#include "Kernels.h"

using namespace std;

namespace idg {
    namespace kernel {
        namespace opencl {

            // Gridder class
            Gridder::Gridder(cl::Program &program, const Parameters &parameters) :
                kernel(program, name_gridder.c_str()),
                parameters(parameters) {}

            void Gridder::launchAsync(
                cl::CommandQueue &queue,
                int nr_baselines,
                int nr_subgrids,
                float w_offset,
                cl::Buffer &d_uvw, cl::Buffer &d_wavenumbers,
                cl::Buffer &d_visibilities, cl::Buffer &d_spheroidal,
                cl::Buffer &d_aterm, cl::Buffer &d_metadata,
                cl::Buffer &d_subgrid,
                PerformanceCounter &counter) {
                int subgridsize = parameters.get_subgrid_size();
                int localSizeX = 256;
                int localSizeY = 1;
                cl::NDRange globalSize(localSizeX * nr_subgrids, localSizeY);
                cl::NDRange localSize(localSizeX, localSizeY);
                kernel.setArg(0, w_offset);
                kernel.setArg(1, d_uvw);
                kernel.setArg(2, d_wavenumbers);
                kernel.setArg(3, d_visibilities);
                kernel.setArg(4, d_spheroidal);
                kernel.setArg(5, d_aterm);
                kernel.setArg(6, d_metadata);
                kernel.setArg(7, d_subgrid);
                try {
                    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, &event);
                    counter.doOperation(event, "gridder", flops(nr_baselines, nr_subgrids), bytes(nr_baselines, nr_subgrids));
                } catch (cl::Error &error) {
                    std::cerr << "Error launching gridder: " << error.what() << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            uint64_t Gridder::flops(int nr_baselines, int nr_subgrids) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_time = parameters.get_nr_time();
                int nr_channels = parameters.get_nr_channels();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t flops = 0;
                flops += 1ULL * nr_baselines * nr_time * subgridsize * subgridsize * 5; // phase index
                flops += 1ULL * nr_baselines * nr_time * subgridsize * subgridsize * 5; // phase offset
                flops += 1ULL * nr_baselines * nr_time * subgridsize * subgridsize * nr_channels * 2; // phase
                flops += 1ULL * nr_baselines * nr_time * subgridsize * subgridsize * nr_channels * (nr_polarizations * 8); // update
                flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 30; // aterm
                flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 2; // spheroidal
                flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 6; // shift
                return flops;
            }

            uint64_t Gridder::bytes(int nr_baselines, int nr_subgrids) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_time = parameters.get_nr_time();
                int nr_channels = parameters.get_nr_channels();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t bytes = 0;
                bytes += 1ULL * nr_baselines * nr_time * 3 * sizeof(float); // uvw
                bytes += 1ULL * nr_baselines * nr_time * nr_channels * nr_polarizations * sizeof(std::complex<float>); // visibilities
                bytes += 1ULL * nr_subgrids * nr_polarizations * subgridsize * subgridsize  * sizeof(std::complex<float>); // subgrids
                return bytes;
            }


            // Degridder class
            Degridder::Degridder(cl::Program &program, const Parameters &parameters) :
                kernel(program, name_degridder.c_str()),
                parameters(parameters) {}

            void Degridder::launchAsync(
                cl::CommandQueue &queue,
                int nr_baselines,
                int nr_subgrids,
                float w_offset,
                cl::Buffer &d_uvw, cl::Buffer &d_wavenumbers,
                cl::Buffer &d_visibilities, cl::Buffer &d_spheroidal,
                cl::Buffer &d_aterm, cl::Buffer &d_metadata,
                cl::Buffer &d_subgrid,
                PerformanceCounter &counter) {
                // IF wgSize IS MODIFIED, ALSO MODIFY NR_THREADS in KernelDegridder.cl
                int subgridsize = parameters.get_subgrid_size();
                int wgSize = subgridsize % 16 == 0 ? 256 : 64;
                cl::NDRange globalSize(wgSize * nr_subgrids);
                cl::NDRange localSize(wgSize);
                kernel.setArg(0, w_offset);
                kernel.setArg(1, d_uvw);
                kernel.setArg(2, d_wavenumbers);
                kernel.setArg(3, d_visibilities);
                kernel.setArg(4, d_spheroidal);
                kernel.setArg(5, d_aterm);
                kernel.setArg(6, d_metadata);
                kernel.setArg(7, d_subgrid);
                try {
                    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, &event);
                    counter.doOperation(event, "degridder", flops(nr_baselines, nr_subgrids), bytes(nr_baselines, nr_subgrids));
                } catch (cl::Error &error) {
                    std::cerr << "Error launching degridder: " << error.what() << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            uint64_t Degridder::flops(int nr_baselines, int nr_subgrids) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_time = parameters.get_nr_time();
                int nr_channels = parameters.get_nr_channels();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t flops = 0;
                flops += 1ULL * nr_baselines * nr_time * subgridsize * subgridsize * 5; // phase index
                flops += 1ULL * nr_baselines * nr_time * subgridsize * subgridsize * 5; // phase offset
                flops += 1ULL * nr_baselines * nr_time * subgridsize * subgridsize * nr_channels * 2; // phase
                flops += 1ULL * nr_baselines * nr_time * subgridsize * subgridsize * nr_channels * (nr_polarizations * 8); // update
                flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 30; // aterm
                flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 2; // spheroidal
                flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 6; // shift
                return flops;
            }

            uint64_t Degridder::bytes(int nr_baselines, int nr_subgrids) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_time = parameters.get_nr_time();
                int nr_channels = parameters.get_nr_channels();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t bytes = 0;
                bytes += 1ULL * nr_baselines * nr_time * 3 * sizeof(float); // uvw
                bytes += 1ULL * nr_baselines * nr_time * nr_channels * nr_polarizations * sizeof(std::complex<float>); // visibilities
                bytes += 1ULL * nr_subgrids * nr_polarizations * subgridsize * subgridsize  * sizeof(std::complex<float>); // subgrids
                return bytes;
            }


            // GridFFT class
            GridFFT::GridFFT(const Parameters &parameters) : parameters(parameters) {
                uninitialized = true;
            }

            GridFFT::~GridFFT() {
                clfftDestroyPlan(&fft);
            }

            void GridFFT::plan(cl::Context &context, cl::CommandQueue &queue, int size, int batch) {
                // Check wheter a new plan has to be created
                if (uninitialized ||
                    size  != planned_size ||
                    batch != planned_batch) {
                    // Destroy old plan (if any)
                    if (!uninitialized) {
                        clfftDestroyPlan(&fft);
                    }

                    // Create new plan
                    size_t lengths[2] = {(size_t) size, (size_t) size};
                    clfftCreateDefaultPlan(&fft, context(), CLFFT_2D, lengths);

                    // Set plan parameters
                    clfftSetPlanPrecision(fft, CLFFT_SINGLE);
                    clfftSetLayout(fft, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
                    clfftSetResultLocation(fft, CLFFT_INPLACE);
                    int distance = size*size;
                    clfftSetPlanDistance(fft, distance, distance);
                    int nr_polarizations = parameters.get_nr_polarizations();
                    clfftSetPlanBatchSize(fft, batch * nr_polarizations);

                    // Update parameters
                    planned_size = size;
                    planned_batch = batch;

                    // Bake plan
                    clfftStatus status = clfftBakePlan(fft, 1, &queue(), NULL, NULL);
                    if (status != CL_SUCCESS) {
                        std::cerr << "Error baking fft plan" << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
                uninitialized = false;
            }

            void GridFFT::launchAsync(
                cl::CommandQueue &queue, cl::Buffer &d_data, clfftDirection direction) {
                clfftStatus status = clfftEnqueueTransform(fft, direction, 1, &queue(), 0, NULL, NULL, &d_data(), NULL, NULL);
                if (status != CL_SUCCESS) {
                    std::cerr << "Error enqueing fft plan" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            void GridFFT::shift(std::complex<float> *data) {
                int gridsize = parameters.get_grid_size();
                int nr_polarizations = parameters.get_nr_polarizations();

                std::complex<float> tmp13, tmp24;

                // Dimensions
                int n = gridsize;
                int n2 = n / 2;

                // Pointer
                typedef std::complex<float> GridType[nr_polarizations][gridsize][gridsize];
                GridType *x = (GridType *) data;

                // Interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4
                #pragma omp parallel for
                for (int pol = 0; pol < nr_polarizations; pol++) {
                    for (int i = 0; i < n2; i++) {
                        for (int k = 0; k < n2; k++) {
                            tmp13                 = (*x)[pol][i][k];
                            (*x)[pol][i][k]       = (*x)[pol][i+n2][k+n2];
                            (*x)[pol][i+n2][k+n2] = tmp13;

                            tmp24              = (*x)[pol][i+n2][k];
                            (*x)[pol][i+n2][k] = (*x)[pol][i][k+n2];
                            (*x)[pol][i][k+n2] = tmp24;
                         }
                    }
                }
            }

            void GridFFT::scale(std::complex<float> *data, std::complex<float> scale) {
                int gridsize = parameters.get_grid_size();
                int nr_polarizations = parameters.get_nr_polarizations();

                // Pointer
                typedef std::complex<float> GridType[nr_polarizations][gridsize][gridsize];
                GridType *x = (GridType *) data;

                #pragma omp parallel for collapse(2)
                for (int pol = 0; pol < nr_polarizations; pol++) {
                    for (int i = 0; i < gridsize * gridsize; i++) {
                        std::complex<float> value = (*x)[pol][0][i];
                        (*x)[pol][0][i] = std::complex<float>(
                            value.real() * scale.real(),
                            value.imag() * scale.imag());
                    }
                }
            }

            uint64_t GridFFT::flops(int size, int batch) {
                int nr_polarizations = parameters.get_nr_polarizations();
                return 1ULL * batch * nr_polarizations * 5 * size * size * log(size * size) / log(2.0);
            }

            uint64_t GridFFT::bytes(int size, int batch) {
                int nr_polarizations = parameters.get_nr_polarizations();
                return 1ULL * 2 * batch * size * size * nr_polarizations * sizeof(std::complex<float>);
            }

            // Adder class
            Adder::Adder(cl::Program &program, const Parameters &parameters) :
                kernel(program, name_adder.c_str()),
                parameters(parameters) {}

            void Adder::launchAsync(
                cl::CommandQueue &queue,
                int nr_subgrids,
                cl::Buffer d_metadata,
                cl::Buffer d_subgrid,
                cl::Buffer d_grid,
                PerformanceCounter &counter) {
                cl::NDRange globalSize(128 * nr_subgrids, 1);
                cl::NDRange localSize(128, 1);
                kernel.setArg(0, d_metadata);
                kernel.setArg(1, d_subgrid);
                kernel.setArg(2, d_grid);
                try {
                    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, &event);
                    counter.doOperation(event, "adder", flops(nr_subgrids), bytes(nr_subgrids));
                } catch (cl::Error &error) {
                    std::cerr << "Error launching adder: " << error.what() << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            uint64_t Adder::flops(int nr_subgrids) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t flops = 0;
                flops += 1ULL * nr_subgrids * subgridsize * subgridsize * 8; // shift
                flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 2; // add
                return flops;
            }

            uint64_t Adder::bytes(int nr_subgrids) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t bytes = 0;
                bytes += 1ULL * nr_subgrids * 2 * sizeof(int); // coordinate
                bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // grid in
                bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // subgrid in
                bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // subgrid out
                return bytes;
            }

            // Splitter class
            Splitter::Splitter(cl::Program &program, const Parameters &parameters) :
                kernel(program, name_splitter.c_str()),
                parameters(parameters) {}

            void Splitter::launchAsync(
                cl::CommandQueue &queue,
                int nr_subgrids,
                cl::Buffer d_metadata,
                cl::Buffer d_subgrid,
                cl::Buffer d_grid,
                PerformanceCounter &counter) {
                cl::NDRange globalSize(128 * nr_subgrids, 1);
                cl::NDRange localSize(128, 1);
                kernel.setArg(0, d_metadata);
                kernel.setArg(1, d_subgrid);
                kernel.setArg(2, d_grid);
                try {
                    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, &event);
                    counter.doOperation(event, "splitter", flops(nr_subgrids), bytes(nr_subgrids));
                } catch (cl::Error &error) {
                    std::cerr << "Error launching splitter: " << error.what() << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            uint64_t Splitter::flops(int nr_subgrids) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t flops = 0;
                flops += 1ULL * nr_subgrids * subgridsize * subgridsize * 8; // shift
                return flops;
            }

            uint64_t Splitter::bytes(int nr_subgrids) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t bytes = 0;
                bytes += 1ULL * nr_subgrids * 2 * sizeof(int); // coordinate
                bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // grid in
                bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * 2 * sizeof(float); // subgrid out
                return bytes;
            }

            // Scaler class
            Scaler::Scaler(cl::Program &program, const Parameters &parameters) :
                kernel(program, name_scaler.c_str()),
                parameters(parameters) {}

            void Scaler::launchAsync(
                cl::CommandQueue &queue,
                int nr_subgrids,
                cl::Buffer d_subgrid,
                PerformanceCounter &counter) {
                cl::NDRange globalSize(128 * nr_subgrids, 1);
                cl::NDRange localSize(128, 1);
                kernel.setArg(0, d_subgrid);
                try {
                    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, &event);
                    counter.doOperation(event, "scaler", flops(nr_subgrids), bytes(nr_subgrids));
                } catch (cl::Error &error) {
                    std::cerr << "Error launching gridder: " << error.what() << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            uint64_t Scaler::flops(int nr_subgrids) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t flops = 0;
                flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 2; // scale
                flops += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 6; // shift
                return flops;
            }

            uint64_t Scaler::bytes(int nr_subgrids) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t bytes = 0;
                bytes += 1ULL * nr_subgrids * subgridsize * subgridsize * nr_polarizations * 2 * sizeof(float); // scale
                return bytes;
            }

        } // namespace opencl
    } // namespace kernel
} // namespace idg
