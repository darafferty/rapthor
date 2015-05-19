#include "Proxies.h"

std::string definitions(
	int nr_stations, int nr_baselines, int nr_time,
	int nr_channels, int nr_polarizations, int blocksize,
	int gridsize, float imagesize) {
	std::stringstream parameters;
	parameters << " -DNR_STATIONS="		    << nr_stations;
	parameters << " -DNR_BASELINES="		<< nr_baselines;
	parameters << " -DNR_TIME="			    << nr_time;
	parameters << " -DNR_CHANNELS="		    << nr_channels;
	parameters << " -DNR_POLARIZATIONS="	<< nr_polarizations;
	parameters << " -DBLOCKSIZE="			<< blocksize;
	parameters << " -DGRIDSIZE="			<< gridsize;
	parameters << " -DIMAGESIZE="			<< imagesize;
	return parameters.str();
}

CUDA::CUDA(
    const char *cc, const char *cflags,
    int nr_stations, int nr_baselines, int nr_time,
	int nr_channels, int nr_polarizations, int blocksize,
	int gridsize, float imagesize, bool measure_power) {

    // Get compile options
	std::string parameters = definitions(
		nr_stations, nr_baselines, nr_time, nr_channels,
		nr_polarizations, blocksize, gridsize, imagesize);

    // Compile CUDA wrapper
    const char *options_power = measure_power ? "-DMEASURE_POWER libPowerSensor.o" : "";
	std::string options_cuda = parameters + " " +
                               cflags     + " " +
                               options_power +
                               CUDA_INCLUDE + " " +
                               CUDA_LIB + " " +
                               SRC_CUDA;
	rw::Source(SRC_WRAPPER).compile(cc, SO_WRAPPER, options_cuda.c_str());

    // Load module
    module = new rw::Module(SO_WRAPPER);
}

void CUDA::init(int deviceNumber, const char *powerSensor) {
	((void (*)(int, const char *)) rw::Function(*module, "init").get())(deviceNumber, powerSensor);

}

void CUDA::gridder(
	int deviceNumber, int nr_streams, int jobsize,
	void *visibilities, void *uvw, void *offset, void *wavenumbers,
	void *aterm, void *spheroidal, void *baselines, void *uvgrid) {
	((void (*)(int,int,int,void*,void*,void*,void*,void*,void*,void*,void*))
	rw::Function(*module, FUNCTION_GRIDDER).get())(
	deviceNumber, nr_streams, jobsize, visibilities, uvw, offset, wavenumbers, aterm, spheroidal, baselines, uvgrid);
}

void CUDA::adder(
	int deviceNumber, int nr_streams, int jobsize,
	void *coordinates, void *uvgrid, void *grid) {
	((void (*)(int,int,int,void*,void*,void*))
	rw::Function(*module, FUNCTION_ADDER).get())(
	deviceNumber, nr_streams, jobsize, coordinates, uvgrid, grid);
}

void CUDA::degridder(
	int deviceNumber, int nr_streams, int jobsize,
	void *offset, void *wavenumbers, void *aterm, void *baselines,
	void *visibilities, void *uvw, void *spheroidal, void *uvgrid) {
	((void (*)(int,int,int,void*,void*,void*,void*,void*,void*,void*,void*))
	rw::Function(*module, FUNCTION_DEGRIDDER).get())(
	deviceNumber, nr_streams, jobsize, offset, wavenumbers, aterm, baselines, visibilities, uvw, spheroidal, uvgrid);
}
