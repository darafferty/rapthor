#include <string>

namespace rw 
{
  class Module;
}


/*
    Interface to kernels
*/
class Xeon {
	public:
        Xeon(
            const char *cc, const char *cflags,
            int nr_stations, int nr_baselines, int nr_time,
        	int nr_channels, int nr_polarizations, int blocksize,
        	int gridsize, float imagesize);		

		void gridder(
			int jobsize,
			void *visibilities, void *uvw, void *offset, void *wavenumbers,
			void *aterm, void *spheroidal, void *baselines, void *uvgrid);
		
		void adder(
			int jobsize,
			void *coordinates, void *uvgrid, void *grid);
			
		void splitter(
		    int jobsize,
		    void *coordinates, void *uvgrid, void *grid);
		
		void degridder(
			int jobsize,
			void *offset, void *wavenumbers, void *aterm, void *baselines,
			void *visibilities, void *uvw, void *spheroidal, void *uvgrid);
			
		void fft(
		    void *grid, int sign);
		
	private:
		rw::Module *module;
        std::string idgdir;
        std::string tmpdir;
		
};

/*
    Definitions
*/
std::string definitions(
	int nr_stations, int nr_baselines, int nr_time,
	int nr_channels, int nr_polarizations, int blocksize,
	int gridsize, float imagesize);
