#include <idg/XEON/Proxies.h>
#include <idg/Common/RW.h>


#include <stdlib.h>
#include <boost/filesystem.hpp>
#include <iostream>

/*
	 Library and source names
*/
#define SO_WRAPPER      "Wrapper.so"
#define SRC_WRAPPER     "XEON/Wrapper.cpp"
#define SRC_RW          "Common/RW.cpp"
#define SRC_KERNELS     "XEON/Kernels.cpp"

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

Xeon::Xeon(
    const char *cc, const char *cflags,
    int nr_stations, int nr_baselines, int nr_time,
	int nr_channels, int nr_polarizations, int blocksize,
	int gridsize, float imagesize) {
  
    Dl_info dl_info;
    dladdr((void *)definitions, &dl_info);
    boost::filesystem::path pathname(dl_info.dli_fname);
    
    std::string libdir  = pathname.parent_path().string();
    std::string basename = pathname.filename().string();
 
    std::cout << "module " << basename << " loaded." << std::endl;
    std::cout << "is in directory " << libdir << std::endl;
    
    idgdir = libdir + "/idg";
    
    std::cout << "I will look for source files in " << idgdir << std::endl;
    
    char _tmpdir[] = "/tmp/idg-XXXXXX";
    mkdtemp(_tmpdir);
    
    tmpdir = _tmpdir;
    
    std::cout << "temporary files will be stored in " << tmpdir << std::endl;
    
    // Get compile options
	std::string parameters = definitions(
		nr_stations, nr_baselines, nr_time, nr_channels,
		nr_polarizations, blocksize, gridsize, imagesize);

    // Compile XEON wrapper
	std::string options_xeon = parameters + " " +
                               cflags     + " " +
                               "-I" + libdir + " " +
                               "-DSOURCE_DIR=\\\"" + idgdir + "\\\" " +
                               "-DBINARY_DIR=\\\"" + tmpdir + "\\\" " +
                               "-DINCLUDE_DIR=\\\"" + libdir + "\\\" " +
                               idgdir + "/" + SRC_RW     + " " + idgdir + "/" + SRC_KERNELS;
	rw::Source((idgdir+"/"+SRC_WRAPPER).c_str()).compile(cc, (tmpdir + "/" + SO_WRAPPER).c_str(), options_xeon.c_str());
    
    // Load module
    module = new rw::Module((tmpdir + "/" + SO_WRAPPER).c_str());

    // Initialize module
	((void (*)(const char*, const char *)) rw::Function(*module, FUNCTION_INIT).get())(cc, cflags);
}

void Xeon::gridder(
	int jobsize,
	void *visibilities, void *uvw, void *offset, void *wavenumbers,
	void *aterm, void *spheroidal, void *baselines, void *uvgrid) {
	((void (*)(unsigned,void*,void*,void*,void*,void*,void*,void*,void*))
	rw::Function(*module, FUNCTION_GRIDDER).get())(
	jobsize, visibilities, uvw, offset, wavenumbers, aterm, spheroidal, baselines, uvgrid);
}

void Xeon::adder(
	int jobsize,
	void *coordinates, void *uvgrid, void *grid) {
	((void (*)(unsigned,void*,void*,void*))
	rw::Function(*module, FUNCTION_ADDER).get())(
	jobsize, coordinates, uvgrid, grid);
}

void Xeon::splitter(
	int jobsize,
	void *coordinates, void *uvgrid, void *grid) {
	((void (*)(unsigned,void*,void*,void*))
	rw::Function(*module, FUNCTION_SPLITER).get())(
	jobsize, coordinates, uvgrid, grid);
}

void Xeon::degridder(
	int jobsize,
	void *offset, void *wavenumbers, void *aterm, void *baselines,
	void *visibilities, void *uvw, void *spheroidal, void *uvgrid) {
	((void (*)(unsigned,void*,void*,void*,void*,void*,void*,void*,void*))
	rw::Function(*module, FUNCTION_DEGRIDDER).get())(
	jobsize, offset, wavenumbers, aterm, baselines, visibilities, uvw, spheroidal, uvgrid);
}

void Xeon::fft(void *grid, int sign) {
    ((void (*)(void*,int))
	rw::Function(*module, FUNCTION_FFT).get())(
	grid, sign);
}
