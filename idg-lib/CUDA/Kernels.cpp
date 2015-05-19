#include "Kernels.h"

KernelGridder::KernelGridder(cu::Module &module, const char *kernel) : function(module, kernel) {}

void KernelGridder::launchAsync(
    cu::Stream &stream, int jobsize, int bl_offset,
    cu::DeviceMemory &d_uvw,  cu::DeviceMemory &d_wavenumbers,
    cu::DeviceMemory &d_visibilities, cu::DeviceMemory &d_spheroidal,
    cu::DeviceMemory &d_aterm, cu::DeviceMemory &d_baselines,
    cu::DeviceMemory &d_subgrid) {
    const void *parameters[] = {
        &bl_offset, d_uvw, d_wavenumbers, d_visibilities,
        d_spheroidal, d_aterm, d_baselines, d_subgrid };
    stream.launchKernel(function, jobsize, 1, 1, 8, 8, 1, 0, parameters);
}   

uint64_t KernelGridder::flops(int jobsize) {
    return 1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * (
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
	return 1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE *(
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
    cu::DeviceMemory &d_subgrid, cu::DeviceMemory &d_uvw,
    cu::DeviceMemory &d_wavenumbers, cu::DeviceMemory &d_aterm,
    cu::DeviceMemory &d_baselines, cu::DeviceMemory &d_spheroidal,
    cu::DeviceMemory &d_visibilities) {
    const void *parameters[] = {
        &bl_offset, d_subgrid, d_uvw, d_wavenumbers,
        d_aterm, d_baselines, d_spheroidal, d_visibilities};
	stream.launchKernel(function, jobsize, 1, 1, 64, 1, 1, 0, parameters);
}

uint64_t KernelDegridder::flops(int jobsize) {
    return 1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * (
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
    2 * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(cuFloatComplex) +
    // UV grid
    SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(cuFloatComplex) +
    // Visibilities
    NR_TIME * NR_CHANNELS * NR_POLARIZATIONS * sizeof(cuFloatComplex));
}

KernelFFT::KernelFFT() {
    fft = NULL;
}

void KernelFFT::plan(int size, int batch, int layout) {
    // Check wheter a new plan has to be created
    if (fft == NULL ||
        size == planned_size ||
        batch == planned_batch ||
        layout == planned_layout) {
        // Create new plan
        if (layout == FFT_LAYOUT_YXP) {
            // Polarizations in inner dimension
            int stride = NR_POLARIZATIONS;
            int dist = 1;
            fft = new cufft::C2C_2D(size, size, stride, dist, batch * NR_POLARIZATIONS);
        } else if (layout == FFT_LAYOUT_PYX) {
            int stride = 1;
            int dist = size * size;
            fft = new cufft::C2C_2D(size, size, stride, dist, batch * NR_POLARIZATIONS);
        }
        
        // Update parameters
        planned_size = size;
        planned_batch = batch;
        planned_layout = layout;
    }
}

void KernelFFT::launchAsync(cu::Stream &stream, cu::DeviceMemory &data, int direction) {
    cufftComplex *data_ptr = reinterpret_cast<cufftComplex *>(static_cast<CUdeviceptr>(data));
    (*fft).setStream(stream);
    (*fft).execute(data_ptr, data_ptr, direction);
}

uint64_t KernelFFT::flops(int size, int batch) {
	return 1ULL * batch * NR_POLARIZATIONS * 5 * size * size * log(size * size);
}

uint64_t KernelFFT::bytes(int size, int batch) {
	return 1ULL * 2 * batch * size * size * NR_POLARIZATIONS * sizeof(cuFloatComplex);
}


KernelAdder::KernelAdder(cu::Module &module, const char *kernel) : function(module, kernel) {}

void KernelAdder::launchAsync(
	cu::Stream &stream, int jobsize, int bl_offset,
	cu::DeviceMemory &d_uvw,
	cu::DeviceMemory &d_subgrid,
	cu::DeviceMemory &d_grid) {
	const void *parameters[] = { &jobsize, &bl_offset, d_uvw, d_subgrid, d_grid };
	stream.launchKernel(function, jobsize, 1, 1, 64, 1, 1, 0, parameters);
}

uint64_t KernelAdder::flops(int jobsize) {
	return 1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * 2;
}

uint64_t KernelAdder::bytes(int jobsize) {
	return
    // Coordinate
    1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * 2 * sizeof(int) +
    // Grid
    1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(cuFloatComplex);
}


KernelSplitter::KernelSplitter(cu::Module &module, const char *kernel) : function(module, kernel) {}

void KernelSplitter::launchAsync(
	cu::Stream &stream, int jobsize, int bl_offset,
	cu::DeviceMemory &d_uvw,
	cu::DeviceMemory &d_subgrid,
	cu::DeviceMemory &d_grid) {
	const void *parameters[] = { &jobsize, &bl_offset, d_uvw, d_subgrid, d_grid };
	stream.launchKernel(function, jobsize, 1, 1, 64, 1, 1, 0, parameters);
}

uint64_t KernelSplitter::flops(int jobsize) {
	return 1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * 2;
}

uint64_t KernelSplitter::bytes(int jobsize) {
	return
    // Coordinate
    1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * 2 * sizeof(int) +
    // Grid
    1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * NR_POLARIZATIONS * sizeof(cuFloatComplex);
}


KernelShifter::KernelShifter(cu::Module &module, const char *kernel) : function(module, kernel) {}

void KernelShifter::launchAsync(
    cu::Stream &stream, int jobsize,
	cu::DeviceMemory &d_subgrid) {
    const void *parameters[] = { &jobsize, d_subgrid };
    stream.launchKernel(function, jobsize, 1, 1, SUBGRIDSIZE, NR_POLARIZATIONS, 1, 0, parameters);
}

uint64_t KernelShifter::flops(int jobsize) {
    return 0;
}

uint64_t KernelShifter::bytes(int jobsize) {
    return 1ULL * jobsize * SUBGRIDSIZE * SUBGRIDSIZE * 3 * sizeof(cuFloatComplex);
}
