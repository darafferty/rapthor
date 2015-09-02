#ifndef IDG_KERNELS_H_
#define IDG_KERNELS_H_

#include <cstdint>
#include <ccomplex>
#include <cmath>
#include <string>

#include "CU.h"
#include "CUFFT.h"

namespace idg {

  namespace kernel {

    // define the kernel function names
    static const std::string name_gridder   = "kernel_gridder";
    static const std::string name_degridder = "kernel_degridder";
    static const std::string name_adder     = "kernel_adder";
    static const std::string name_splitter  = "kernel_splitter";

    
// TODO: remove #define and add arguments
#define SUBGRIDSIZE 32
#define NR_TIME 16
#define NR_TIMESTEPS 128
#define NR_CHANNELS 16
#define NR_POLARIZATIONS 4


    class Gridder {
        public:
            Gridder(cu::Module &module);
            void launchAsync(
                cu::Stream &stream, int jobsize, float w_offset,
                cu::DeviceMemory &d_uvw, cu::DeviceMemory &d_wavenumbers,
                cu::DeviceMemory &d_visibilities, cu::DeviceMemory &d_spheroidal,
                cu::DeviceMemory &d_aterm, cu::DeviceMemory &d_metadata,
                cu::DeviceMemory &d_subgrid);
        	static uint64_t flops(int jobsize);
    		static uint64_t bytes(int jobsize);
    	
    	private:
    	    cu::Function function;
    };
    
    
    class Degridder {
        public:
            Degridder(cu::Module &module);
            void launchAsync(
                cu::Stream &stream, int jobsize, float w_offset,
                cu::DeviceMemory &d_uvw, cu::DeviceMemory &d_wavenumbers,
                cu::DeviceMemory &d_visibilities, cu::DeviceMemory &d_spheroidal,
                cu::DeviceMemory &d_aterm, cu::DeviceMemory &d_metadata,
                cu::DeviceMemory &d_subgrid);
           	static uint64_t flops(int jobsize);
    		static uint64_t bytes(int jobsize);
    	
    	private:
    	    cu::Function function;
    };
    
    
    class GridFFT {
    	public:
            GridFFT();
            void plan(int size, int batch);
            void launchAsync(cu::Stream &stream, cu::DeviceMemory &data, int direction);
    		static uint64_t flops(int size, int batch);
    		static uint64_t bytes(int size, int batch);
    
        private:
            int planned_size;
            int planned_batch;
            cufft::C2C_2D *fft;
    };
    
    
    class Adder {
    	public:
    	    Adder(cu::Module &module);
    		void launchAsync(
    			cu::Stream &stream, int jobsize,
    			cu::DeviceMemory &d_metadata,
    			cu::DeviceMemory &d_subgrid,
    			cu::DeviceMemory &d_grid);
    		static uint64_t flops(int jobsize);
    		static uint64_t bytes(int jobsize);
    		
    	private:
    		cu::Function function;
    };
    
    
    /*
        Splitter
    */
    class Splitter {
    	public:
    		Splitter(cu::Module &module);
    		void launchAsync(
    			cu::Stream &stream, int jobsize,
    			cu::DeviceMemory &d_metadata,
    			cu::DeviceMemory &d_subgrid,
    			cu::DeviceMemory &d_grid);
    		static uint64_t flops(int jobsize);
    		static uint64_t bytes(int jobsize);
    		
    	private:
    		cu::Function function;
    };
    
  } // namespace kernel

} // namespace idg

#endif
