#include "RW.h"

/*
    Library and source names
*/
#define SO_WRAPPER      "./Wrapper.so"
#define SRC_WRAPPER     "CUDA/Wrapper.cpp"
#define SRC_CUDA        "CUDA/CU.cpp CUDA/CUFFT.cpp CUDA/Kernels.cpp"

/*
	Function names
*/
#define FUNCTION_GRIDDER    "run_gridder"
#define FUNCTION_DEGRIDDER  "run_degridder"
#define FUNCTION_ADDER      "run_adder"

/*
    CUDA library location
*/
#define CUDA_INCLUDE    " -I/usr/local/cuda-7.0/include"
#define CUDA_LIB        " -L/usr/local/cuda-7.0/lib64"

/*
    Interface to kernels
*/
class CUDA {
	public:
        CUDA(
            const char *cc, const char *cflags,
            int nr_stations, int nr_baselines, int nr_time,
        	int nr_channels, int nr_polarizations, int blocksize,
        	int gridsize, float imagesize, bool measure_power);
		
		void init(int deviceNumber, const char *powerSensor=NULL);
		
		void gridder(
			int deviceNumber, int nr_streams, int jobsize,
			void *visibilities, void *uvw, void *offset, void *wavenumbers,
			void *aterm, void *spheroidal, void *baselines, void *uvgrid);
		
		void adder(
			int deviceNumber, int nr_streams, int jobsize,
			void *coordinates, void *uvgrid, void *grid);
		
		void degridder(
			int deviceNumber, int nr_streams, int jobsize,
			void *offset, void *wavenumbers, void *aterm, void *baselines,
			void *visibilities, void *uvw, void *spheroidal, void *uvgrid);
		
	private:
		rw::Module *module;
		
};

/*
    Definitions
*/
std::string definitions(
	int nr_stations, int nr_baselines, int nr_time,
	int nr_channels, int nr_polarizations, int blocksize,
	int gridsize, float imagesize);
