#include "RW.h"
#include "CU.h"

/*
    Interface to kernels
*/
class CUDA {
	public:
        CUDA(
            const char *cc, const char *cflags, int deviceNumber,
            int nr_stations, int nr_baselines, int nr_time,
        	int nr_channels, int nr_polarizations, int subgridsize,
        	int gridsize, float imagesize, int chunksize);
		
		void gridder(
            cu::Context &context, int nr_streams, int jobsize,
			cu::HostMemory &visibilities, cu::HostMemory &uvw, cu::HostMemory &subgrid,
            cu::DeviceMemory &wavenumbers, cu::DeviceMemory &aterm,
            cu::DeviceMemory &spheroidal, cu::DeviceMemory &baselines);
		
		void adder(
            cu::Context &context, int nr_streams, int jobsize,
			cu::HostMemory &subgrid, cu::HostMemory &uvw, cu::HostMemory &grid);
		
        void splitter(
            cu::Context &context, int nr_streams, int jobsize,
		    cu::HostMemory &subgrid, cu::HostMemory &uvw, cu::HostMemory &grid);
		
		void degridder(
            cu::Context &context, int nr_streams, int jobsize,
			cu::HostMemory &visibilities, cu::HostMemory &uvw, cu::HostMemory &subgrid,
            cu::DeviceMemory &wavenumbers, cu::DeviceMemory &spheroidal,
            cu::DeviceMemory &aterm, cu::DeviceMemory &baselines);
		
		void fft(
            cu::Context &context, cu::HostMemory &grid, int sign);
	private:
		rw::Module *module;
		
};

/*
    Definitions
*/
std::string definitions(
	int nr_stations, int nr_baselines, int nr_time,
	int nr_channels, int nr_polarizations, int subgridsize,
	int gridsize, float imagesize, int chunksize); 
