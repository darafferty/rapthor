#ifndef IDG_OFFLOAD_KERNELS_H_
#define IDG_OFFLOAD_KERNELS_H_

#include <cstdint>

namespace idg {

#pragma omp declare target
void kernel_gridder (
	const int jobsize, const float w_offset,
   	const void *uvw,
	const void *wavenumbers,
	const void *visibilities,
	const void *spheroidal,
	const void *aterm,
	const void *metadata,
	      void *subgrid,
    const int nr_stations,
    const int nr_timesteps,
    const int nr_timeslots,
    const int nr_channels,
    const int subgridsize,
    const float imagesize,
    const int nr_polarizations
	);

void kernel_fft(
	const int size, 
	const int batch,
	void *data,
	const int sign,
    const int nr_polarizations
    );
    

uint64_t kernel_gridder_flops(int jobsize, int nr_timesteps, int nr_channels, int subgridsize, int nr_polarizations);
uint64_t kernel_gridder_bytes(int jobsize, int nr_timesteps, int nr_channels, int subgridsize, int nr_polarizations);
uint64_t kernel_fft_flops(int size, int batch, int nr_polarizations);
uint64_t kernel_fft_bytes(int size, int batch, int nr_polarizations);
#pragma omp end declare target

} // namespace idg

#endif
