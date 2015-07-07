#include "Proxies.h"

/*
    Library and source names
*/
#define SO_WRAPPER      "Wrapper.so"
#define SRC_WRAPPER     "Wrapper.cpp"
#define SRC_CUDA        "CU.cpp CUFFT.cpp Kernels.cpp"
#define SRC_POWER       "Power.cpp"

/*
	Function names
*/
#define FUNCTION_GRIDDER    "run_gridder"
#define FUNCTION_DEGRIDDER  "run_degridder"
#define FUNCTION_ADDER      "run_adder"
#define FUNCTION_SPLITTER   "run_splitter"
#define FUNCTION_FFT        "run_fft"
#define FUNCTION_COMPILE    "compile"

/*
    CUDA library location
*/
#define CUDA_INCLUDE    " -I/usr/local/cuda-7.0/include"
#define CUDA_LIB        " -L/usr/local/cuda-7.0/lib64"

std::string definitions(
	int nr_stations, int nr_baselines, int nr_time,
	int nr_channels, int nr_polarizations, int subgridsize,
	int gridsize, float imagesize, int chunksize) {
	std::stringstream parameters;
	parameters << " -DNR_STATIONS="		 << nr_stations;
	parameters << " -DNR_BASELINES="	 << nr_baselines;
	parameters << " -DNR_TIME="			 << nr_time;
	parameters << " -DNR_CHANNELS="		 << nr_channels;
	parameters << " -DNR_POLARIZATIONS=" << nr_polarizations;
	parameters << " -DSUBGRIDSIZE="		 << subgridsize;
	parameters << " -DGRIDSIZE="		 << gridsize;
	parameters << " -DIMAGESIZE="	     << imagesize;
	parameters << " -DCHUNKSIZE="        << chunksize;
	return parameters.str();
}


CUDA::CUDA(
    const char *cc, const char *cflags, int deviceNumber,
    int nr_stations, int nr_baselines, int nr_time,
	int nr_channels, int nr_polarizations, int subgridsize,
    int gridsize, float imagesize, int chunksize) {
    // Get compile options
	std::string parameters = definitions(
		nr_stations, nr_baselines, nr_time, nr_channels,
		nr_polarizations, subgridsize, gridsize, imagesize, chunksize);

    // Compile CUDA wrapper
	std::string options_cuda = parameters + " " +
                               cflags     + " " +
                               "-std=c++11" + " " +
                               "-I../Common" + " " +
                               " -fopenmp -lcuda -lcufft " +
                               CUDA_INCLUDE + " " +
                               CUDA_LIB + " " +
                               SRC_CUDA + " " +
                               SRC_POWER;
	rw::Source(SRC_WRAPPER).compile(cc, SO_WRAPPER, options_cuda.c_str());

    // Load module
    module = new rw::Module(SO_WRAPPER);

    // Compile kernels
	((void (*)(int)) (void *) rw::Function(*module, FUNCTION_COMPILE))(deviceNumber);
}

void CUDA::gridder(
            cu::Context &context, int nr_streams, int jobsize,
			cu::HostMemory &visibilities, cu::HostMemory &uvw, cu::HostMemory &subgrid,
            cu::DeviceMemory &wavenumbers, cu::DeviceMemory &aterm,
            cu::DeviceMemory &spheroidal, cu::DeviceMemory &baselines) {
    ((void (*)(cu::Context&,int,int,cu::HostMemory&,cu::HostMemory&,cu::HostMemory&,cu::DeviceMemory&,cu::DeviceMemory&,cu::DeviceMemory&,cu::DeviceMemory&)) (void *)
	rw::Function(*module, FUNCTION_GRIDDER))(context, nr_streams, jobsize, visibilities, uvw, subgrid, wavenumbers, aterm, spheroidal, baselines);
}

void CUDA::adder(
            cu::Context &context, int nr_streams, int jobsize,
			cu::HostMemory &subgrid, cu::HostMemory &uvw, cu::HostMemory &grid) {
	((void (*)(cu::Context&,int,int,cu::HostMemory&,cu::HostMemory&,cu::HostMemory&)) (void *)
	rw::Function(*module, FUNCTION_ADDER))(context, nr_streams, jobsize, subgrid, uvw, grid);
}

void CUDA::splitter(
            cu::Context &context, int nr_streams, int jobsize,
		    cu::HostMemory &subgrid, cu::HostMemory &uvw, cu::HostMemory &grid) {
	((void (*)(cu::Context&,int,int,cu::HostMemory&,cu::HostMemory&,cu::HostMemory&)) (void *)
	rw::Function(*module, FUNCTION_SPLITTER))(context, nr_streams, jobsize, subgrid, uvw, grid);
}

void CUDA::degridder(
            cu::Context &context, int nr_streams, int jobsize,
			cu::HostMemory &visibilities, cu::HostMemory &uvw, cu::HostMemory &subgrid,
            cu::DeviceMemory &wavenumbers, cu::DeviceMemory &spheroidal,
            cu::DeviceMemory &aterm, cu::DeviceMemory &baselines) {
    ((void (*)(cu::Context&,int,int,cu::HostMemory&,cu::HostMemory&,cu::HostMemory&,cu::DeviceMemory&,cu::DeviceMemory&,cu::DeviceMemory&,cu::DeviceMemory&)) (void *)
	rw::Function(*module, FUNCTION_DEGRIDDER))(context, nr_streams, jobsize, visibilities, uvw, subgrid, wavenumbers, spheroidal, aterm, baselines);
}

void CUDA::fft(
            cu::Context &context, cu::HostMemory &grid, int sign) {
	((void (*)(cu::Context&,cu::HostMemory&,int)) (void *)
	rw::Function(*module, FUNCTION_FFT))(context, grid, sign);
}
