#ifndef IDG_KERNELS_H_
#define IDG_KERNELS_H_

#include <cstdint>
#include <ccomplex>
#include <cmath>
#include <string>

#include "CU.h"
#include "CUFFT.h"

#include "Parameters.h"

namespace idg {
    namespace kernel {

    // define the kernel function names
    static const std::string name_gridder   = "kernel_gridder";
    static const std::string name_degridder = "kernel_degridder";
    static const std::string name_adder     = "kernel_adder";
    static const std::string name_splitter  = "kernel_splitter";
    static const std::string name_fft       = "kernel_fft";

    class Gridder {
        public:
            Gridder(cu::Module &module, Parameters &parameters);
            void launchAsync(
                cu::Stream &stream, int jobsize, float w_offset,
                cu::DeviceMemory &d_uvw, cu::DeviceMemory &d_wavenumbers,
                cu::DeviceMemory &d_visibilities, cu::DeviceMemory &d_spheroidal,
                cu::DeviceMemory &d_aterm, cu::DeviceMemory &d_metadata,
                cu::DeviceMemory &d_subgrid);

            uint64_t flops(int jobsize) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_timesteps = parameters.get_nr_timesteps();
                int nr_channels = parameters.get_nr_channels();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t flops = 0;
                flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * 5; // phase index
                flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * 5; // phase offset
                flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * nr_channels * 2; // phase
                flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * nr_channels * (nr_polarizations * 8); // update
                flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 30; // aterm
                flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 2; // spheroidal
                flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 6; // shift
                return flops;
            }

            uint64_t bytes(int jobsize) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_timesteps = parameters.get_nr_timesteps();
                int nr_channels = parameters.get_nr_channels();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t bytes = 0;
                bytes += 1ULL * jobsize * nr_timesteps * 3 * sizeof(float); // uvw
                bytes += 1ULL * jobsize * nr_timesteps * nr_channels * nr_polarizations * sizeof(cuFloatComplex); // visibilities
                bytes += 1ULL * jobsize * nr_polarizations * subgridsize * subgridsize  * sizeof(cuFloatComplex); // subgrids
                return bytes;
            }

    	private:
    	    cu::Function function;
            Parameters &parameters;
    };


    class Degridder {
        public:
            Degridder(cu::Module &module, Parameters &parameters);
            void launchAsync(
                cu::Stream &stream, int jobsize, float w_offset,
                cu::DeviceMemory &d_uvw, cu::DeviceMemory &d_wavenumbers,
                cu::DeviceMemory &d_visibilities, cu::DeviceMemory &d_spheroidal,
                cu::DeviceMemory &d_aterm, cu::DeviceMemory &d_metadata,
                cu::DeviceMemory &d_subgrid);

            uint64_t flops(int jobsize) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_timesteps = parameters.get_nr_timesteps();
                int nr_channels = parameters.get_nr_channels();
                int nr_polarizations = parameters.get_nr_polarizations();
                uint64_t flops = 0;
                flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * 5; // phase index
                flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * 5; // phase offset
                flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * nr_channels * 2; // phase
                flops += 1ULL * jobsize * nr_timesteps * subgridsize * subgridsize * nr_channels * (nr_polarizations * 8); // update
                flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 30; // aterm
                flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 2; // spheroidal
                flops += 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 6; // shift
                return flops;
            }
        
            uint64_t bytes(int jobsize) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_time = parameters.get_nr_timesteps();
                int nr_channels = parameters.get_nr_channels();
                int nr_polarizations = parameters.get_nr_polarizations();
                return 1ULL * jobsize * (
                // ATerm
                2 * subgridsize * subgridsize * nr_polarizations * sizeof(cuFloatComplex) +
                // UV grid
                subgridsize * subgridsize * nr_polarizations * sizeof(cuFloatComplex) +
                // Visibilities
                nr_time * nr_channels * nr_polarizations * sizeof(cuFloatComplex));
            }
    	private:
    	    cu::Function function;
            Parameters &parameters;
    };


    class GridFFT {
    	public:
            GridFFT(cu::Module &module, Parameters &parameters);
            void plan(int size, int batch);
            void launchAsync(cu::Stream &stream, cu::DeviceMemory &data, int direction);

            uint64_t flops(int size, int batch) {
                int nr_polarizations = parameters.get_nr_polarizations();
            	return 1ULL * batch * nr_polarizations * 5 * size * size * log(size * size);
            }

            uint64_t bytes(int size, int batch) {
                int nr_polarizations = parameters.get_nr_polarizations();
            	return 1ULL * 2 * batch * size * size * nr_polarizations * sizeof(cuFloatComplex);
            }

        private:
            cu::Function function;
            Parameters &parameters;
            int planned_size;
            int planned_batch;
            const int bulk_size = 8192;
            cufft::C2C_2D *fft_bulk;
            cufft::C2C_2D *fft_remainder;
    };


    class Adder {
    	public:
    	    Adder(cu::Module &module, Parameters &parameters);
    		void launchAsync(
    			cu::Stream &stream, int jobsize,
    			cu::DeviceMemory &d_metadata,
    			cu::DeviceMemory &d_subgrid,
    			cu::DeviceMemory &d_grid);
            uint64_t flops(int jobsize) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_polarizations = parameters.get_nr_polarizations();
            	return 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 2;
            }
        
            uint64_t bytes(int jobsize) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_polarizations = parameters.get_nr_polarizations();
            	return
                // Coordinate
                1ULL * jobsize * subgridsize * subgridsize * 2 * sizeof(int) +
                // Grid
                1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * sizeof(cuFloatComplex);
            }

    	private:
    		cu::Function function;
            Parameters &parameters;
    };


    /*
        Splitter
    */
    class Splitter {
    	public:
    		Splitter(cu::Module &module, Parameters &parameters);
    		void launchAsync(
    			cu::Stream &stream, int jobsize,
    			cu::DeviceMemory &d_metadata,
    			cu::DeviceMemory &d_subgrid,
    			cu::DeviceMemory &d_grid);

            uint64_t flops(int jobsize) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_polarizations = parameters.get_nr_polarizations();
            	return 1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * 2;
            }

            uint64_t bytes(int jobsize) {
                int subgridsize = parameters.get_subgrid_size();
                int nr_polarizations = parameters.get_nr_polarizations();
            	return
                // Coordinate
                1ULL * jobsize * subgridsize * subgridsize * 2 * sizeof(int) +
                // Grid
                1ULL * jobsize * subgridsize * subgridsize * nr_polarizations * sizeof(cuFloatComplex);
            }

    	private:
    		cu::Function function;
            Parameters &parameters;
    };
    } // namespace kernel
} // namespace idg
#endif
