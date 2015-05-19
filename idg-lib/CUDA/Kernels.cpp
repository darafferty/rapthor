#include "Kernels.h"

KernelGridder::KernelGridder(cu::Module &module, const char *kernel) : function(module, kernel) {}

void KernelGridder::launchAsync(
    cu::Stream &stream, int jobsize, int bl_offset,
    cu::DeviceMemory &d_uvw, cu::DeviceMemory &d_offset,
    cu::DeviceMemory &d_wavenumbers, cu::DeviceMemory &d_visibilities,
    cu::DeviceMemory &d_spheroidal, cu::DeviceMemory &d_aterm,
    cu::DeviceMemory &d_baselines, cu::DeviceMemory &d_uvgrid) {
    const void *parameters[] = {
        &bl_offset, d_uvw, d_offset, d_wavenumbers, d_visibilities,
        d_spheroidal, d_aterm, d_baselines, d_uvgrid };
	int grid_x = jobsize / GRID_DISTRIBUTION;
	int grid_y = GRID_DISTRIBUTION;
	stream.launchKernel(function, grid_x, grid_y, 1, 8, 8, 1, 0, parameters);
}   

uint64_t KernelGridder::flops(int jobsize) {
    return 1ULL * jobsize * BLOCKSIZE * BLOCKSIZE * (
    // LMN
    14 +
    // Phase
    NR_TIME * 10 +
    // Phasor
    NR_TIME * NR_CHANNELS * 4 +
    // Grid
    NR_TIME * NR_CHANNELS * (NR_POLARIZATIONS * 8) +
    // ATerm
    NR_POLARIZATIONS * 32 +
    // Spheroidal
    NR_POLARIZATIONS * 2);
}

uint64_t KernelGridder::bytes(int jobsize) {
	return 1ULL * jobsize * BLOCKSIZE * BLOCKSIZE *(
    // Grid
    (NR_POLARIZATIONS * sizeof(cuFloatComplex) + sizeof(float)) +
    // ATerm
    ((2 * sizeof(int)) + (2 * NR_POLARIZATIONS * sizeof(cuFloatComplex))) +
    // Spheroidal
	NR_POLARIZATIONS * sizeof(cuFloatComplex));
}


KernelDegridder::KernelDegridder(cu::Module &module, const char *kernel) : function(module, kernel) {}

void KernelDegridder::launchAsync(
    cu::Stream &stream, int jobsize, int bl_offset,
    cu::DeviceMemory &d_uvgrid, cu::DeviceMemory &d_uvw,
    cu::DeviceMemory &d_offset, cu::DeviceMemory &d_wavenumbers,
    cu::DeviceMemory &d_aterm, cu::DeviceMemory &d_baselines,
    cu::DeviceMemory &d_spheroidal, cu::DeviceMemory &d_visibilities) {
    const void *parameters[] = {
        &bl_offset, d_uvgrid, d_uvw, d_offset, d_wavenumbers,
        d_aterm, d_baselines, d_spheroidal, d_visibilities};
	int grid_x = jobsize / GRID_DISTRIBUTION;
	int grid_y = GRID_DISTRIBUTION;
	stream.launchKernel(function, grid_x, grid_y, 1, 128, 1, 1, 0, parameters);
}

uint64_t KernelDegridder::flops(int jobsize) {
    return 1ULL * jobsize * BLOCKSIZE * BLOCKSIZE * (
    // ATerm
    NR_POLARIZATIONS * 32 +
    // Spheroidal
    NR_POLARIZATIONS * 2 +
    // LMN
    14 +
    // Phase
    10 +
    // Phasor
    NR_TIME * NR_CHANNELS * 4 +
    // Degrid
    NR_TIME * NR_CHANNELS * NR_POLARIZATIONS * 8);
}

uint64_t KernelDegridder::bytes(int jobsize) {
    return 1ULL * jobsize * (
    // ATerm
    2 * BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS * sizeof(cuFloatComplex) +
    // UV grid
    BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS * sizeof(cuFloatComplex) +
    // Visibilities
    NR_TIME * NR_CHANNELS * NR_POLARIZATIONS * sizeof(cuFloatComplex));
}

#if ORDER == ORDER_BL_V_U_P
void KernelFFT::launchAsync(cu::Stream &stream, int jobsize, cu::DeviceMemory &d_uvgrid, int direction) {
    cufftComplex *d_uvgrid_ptr = reinterpret_cast<cufftComplex *>(static_cast<CUdeviceptr>(d_uvgrid));
    int n = BLOCKSIZE;
    int stride = NR_POLARIZATIONS;
    int dist = n * n * NR_POLARIZATIONS;
    int count = jobsize;
    cufft::C2C_2D fft(n, n, stride, dist, count);
    fft.setStream(stream);
    cufftComplex *d_uvgrid_xx = d_uvgrid_ptr + 0;
    cufftComplex *d_uvgrid_xy = d_uvgrid_ptr + 1;
    cufftComplex *d_uvgrid_yx = d_uvgrid_ptr + 2;
    cufftComplex *d_uvgrid_yy = d_uvgrid_ptr + 3;
    fft.execute(d_uvgrid_xx, d_uvgrid_xx, direction);
    fft.execute(d_uvgrid_xy, d_uvgrid_xy, direction);
    fft.execute(d_uvgrid_yx, d_uvgrid_yx, direction);
    fft.execute(d_uvgrid_yy, d_uvgrid_yy, direction);
}
#elif ORDER == ORDER_BL_P_V_U
void KernelFFT::launchAsync(cu::Stream &stream, int jobsize, cu::DeviceMemory &d_uvgrid, int direction) {
    cufftComplex *d_uvgrid_ptr = reinterpret_cast<cufftComplex *>(static_cast<CUdeviceptr>(d_uvgrid));
    int n = BLOCKSIZE;
    int stride = 1;
    int dist = n * n;
    int count = jobsize * NR_POLARIZATIONS;
    cufft::C2C_2D fft(n, n, stride, dist, count);
    fft.setStream(stream);
    fft.execute(d_uvgrid_ptr, d_uvgrid_ptr, direction);
}
#endif

uint64_t KernelFFT::flops(int jobsize) {
	return 1ULL * jobsize * NR_POLARIZATIONS * 5 * BLOCKSIZE * BLOCKSIZE * log(BLOCKSIZE * BLOCKSIZE);
}

uint64_t KernelFFT::bytes(int jobsize) {
	return 1ULL * 2 * jobsize * BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS * sizeof(cuFloatComplex);
}

KernelAdder::KernelAdder(cu::Module &module, const char *kernel) : function(module, kernel) {}

void KernelAdder::launchAsync(
	cu::Stream &stream, int jobsize, int bl_offset,
	cu::DeviceMemory &d_coordinates,
	cu::DeviceMemory &d_uvgrid,
	cu::DeviceMemory &d_grid) {
	const void *parameters[] = { &jobsize, &bl_offset, d_coordinates, d_uvgrid, d_grid };
	int grid_x = jobsize / GRID_DISTRIBUTION;
	int grid_y = GRID_DISTRIBUTION;
	stream.launchKernel(function, grid_x, grid_y, 1, BLOCKSIZE, BLOCKSIZE, 1, 0, parameters);
}

uint64_t KernelAdder::flops(int jobsize) {
	return 1ULL * jobsize * BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS * 2;
}

uint64_t KernelAdder::bytes(int jobsize) {
	return
    // Coordinate
    1ULL * jobsize * BLOCKSIZE * BLOCKSIZE * 2 * sizeof(int) +
    // Grid
    1ULL * jobsize * BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS * sizeof(cuFloatComplex);
}


KernelShifter::KernelShifter(cu::Module &module, const char *kernel) : function(module, kernel) {}

void KernelShifter::launchAsync(
    cu::Stream &stream, int jobsize,
	cu::DeviceMemory &d_uvgrid) {
    const void *parameters[] = { &jobsize, d_uvgrid };
    stream.launchKernel(function, jobsize, 1, 1, BLOCKSIZE, NR_POLARIZATIONS, 1, 0, parameters);
}

uint64_t KernelShifter::flops(int jobsize) {
    return 0;;
}

uint64_t KernelShifter::bytes(int jobsize) {
    return 1ULL * jobsize * BLOCKSIZE * BLOCKSIZE * 3 * sizeof(cuFloatComplex);
}
