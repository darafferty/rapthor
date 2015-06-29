#include "Proxies.h"

/*
	 Library and source names
*/
#define SO_WRAPPER      "./Wrapper.so"
#define SRC_WRAPPER     "Wrapper.cpp"
#define SRC_RW          "../Common/RW.cpp"
#define SRC_KERNELS     "Kernels.cpp"

/*
	Function names
*/
#define FUNCTION_GRIDDER    "run_gridder"
#define FUNCTION_DEGRIDDER  "run_degridder"
#define FUNCTION_ADDER      "run_adder"
#define FUNCTION_SPLITER    "run_splitter"
#define FUNCTION_FFT        "run_fft"
#define FUNCTION_INIT       "init"

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
    parameters << " -DNR_CHUNKS="        << nr_time / chunksize;
	return parameters.str();
}

Xeon::Xeon(
    const char *cc, const char *cflags,
    int nr_stations, int nr_baselines, int nr_time,
	int nr_channels, int nr_polarizations, int subgridsize,
	int gridsize, float imagesize, int chunksize) {
    // Get compile options
	std::string parameters = definitions(
		nr_stations, nr_baselines, nr_time, nr_channels,
		nr_polarizations, subgridsize, gridsize, imagesize, chunksize);

    // Compile XEON wrapper
	std::string options_xeon = parameters + " " +
                               cflags     + " " +
                               "-I../Common" + " " +
                               SRC_RW     + " " + SRC_KERNELS;
	rw::Source(SRC_WRAPPER).compile(cc, SO_WRAPPER, options_xeon.c_str());
    
    // Load module
    module = new rw::Module(SO_WRAPPER);

    // Initialize module
	((void (*)(const char*, const char *)) (void *) rw::Function(*module, FUNCTION_INIT))(cc, cflags);
}

void Xeon::gridder(
	int jobsize,
	void *visibilities, void *uvw, void *wavenumbers,
	void *aterm, void *spheroidal, void *baselines, void *subgrid) {
	((void (*)(int,void*,void*,void*,void*,void*,void*,void*)) (void *)
	rw::Function(*module, FUNCTION_GRIDDER))(
	jobsize, visibilities, uvw, wavenumbers, aterm, spheroidal, baselines, subgrid);
}

void Xeon::adder(
	int jobsize,
	void *coordinates, void *subgrid, void *grid) {
	((void (*)(int,void*,void*,void*)) (void *)
	rw::Function(*module, FUNCTION_ADDER))(
	jobsize, coordinates, subgrid, grid);
}

void Xeon::splitter(
	int jobsize,
	void *coordinates, void *subgrid, void *grid) {
	((void (*)(int,void*,void*,void*)) (void *)
	rw::Function(*module, FUNCTION_SPLITER))(
	jobsize, coordinates, subgrid, grid);
}

void Xeon::degridder(
	int jobsize,
	void *wavenumbers, void *aterm, void *baselines,
	void *visibilities, void *uvw, void *spheroidal, void *subgrid) {
	((void (*)(int,void*,void*,void*,void*,void*,void*,void*)) (void *)
	rw::Function(*module, FUNCTION_DEGRIDDER))(
	jobsize, wavenumbers, aterm, baselines, visibilities, uvw, spheroidal, subgrid);
}

void Xeon::fft(void *grid, int sign) {
    ((void (*)(void*,int)) (void *)
	rw::Function(*module, FUNCTION_FFT))(
	grid, sign);
}
