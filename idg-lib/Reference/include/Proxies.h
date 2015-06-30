/// Proxies.h: brief description
/**
 * Here goes some more detailed description
 */

#include "../Common/RW.h"

/* Interface to kernels */
// Q: not to kernels, but to gridding routines, etc... 
class Xeon {
	public:
        Xeon(const char *cc, 
             const char *cflags,
             int nr_stations, 
             int nr_baselines, 
             int nr_time,
             int nr_channels, 
             int nr_polarizations, 
             int subgridsize,
             int gridsize, 
             float imagesize, 
             int chunksize);		

		void gridder(int jobsize,
                     void *visibilities, 
                     void *uvw, 
                     void *wavenumbers,
                     void *aterm, 
                     void *spheroidal, 
                     void *baselines, 
                     void *subgrid);
		
		void adder(int jobsize,
                   void *uvw, 
                   void *subgrid, 
                   void *grid);
			
		void splitter(int jobsize,
                      void *uvw, 
                      void *subgrid, 
                      void *grid);
		
		void degridder(int jobsize,
                       void *wavenumbers, 
                       void *aterm, 
                       void *baselines,
                       void *visibilities, 
                       void *uvw, 
                       void *spheroidal, 
                       void *subgrid);
			
		void fft(void *grid, 
                 int sign);
		
	private:
		rw::Module *module;
};

/* Definitions */
std::string definitions(int nr_stations, int nr_baselines, int nr_time,
                        int nr_channels, int nr_polarizations, 
                        int subgridsize, int gridsize, float imagesize, 
                        int chunksize);
