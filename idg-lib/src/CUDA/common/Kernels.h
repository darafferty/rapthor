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

    class Gridder {
        public:
            Gridder(cu::Module &module, Parameters &parameters);
            void launchAsync(
                cu::Stream &stream, int jobsize, float w_offset,
                cu::DeviceMemory &d_uvw, cu::DeviceMemory &d_wavenumbers,
                cu::DeviceMemory &d_visibilities, cu::DeviceMemory &d_spheroidal,
                cu::DeviceMemory &d_aterm, cu::DeviceMemory &d_metadata,
                cu::DeviceMemory &d_subgrid);
        	uint64_t flops(int jobsize);
    		uint64_t bytes(int jobsize);
    	
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
           	uint64_t flops(int jobsize);
    		uint64_t bytes(int jobsize);
    	
    	private:
    	    cu::Function function;
            Parameters &parameters;
    };
    
    
    class GridFFT {
    	public:
            GridFFT(Parameters &parameters);
            void plan(int size, int batch);
            void launchAsync(cu::Stream &stream, cu::DeviceMemory &data, int direction);
    		uint64_t flops(int size, int batch);
    		uint64_t bytes(int size, int batch);
    
        private:
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
    		uint64_t flops(int jobsize);
    		uint64_t bytes(int jobsize);
    		
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
    		uint64_t flops(int jobsize);
    		uint64_t bytes(int jobsize);
    		
    	private:
    		cu::Function function;
            Parameters &parameters;
    };
    
  } // namespace kernel

} // namespace idg

#endif
