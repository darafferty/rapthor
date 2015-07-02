#include "Kernels.h"

KernelGridder::KernelGridder(cl::Program &program, const char *kernel_name) : kernel(program, kernel_name) {}

void KernelGridder::launchAsync(
    cl::CommandQueue &queue, cl::Event &event, int jobsize, int bl_offset,
    cl::Buffer &d_uvw, cl::Buffer &d_wavenumbers,
    cl::Buffer &d_visibilities, cl::Buffer &d_spheroidal,
    cl::Buffer &d_aterm, cl::Buffer &d_baselines,
    cl::Buffer &d_subgrid) {
    int wgSize = 8;
    cl::NDRange globalSize(jobsize * wgSize, wgSize);
    cl::NDRange localSize(wgSize, wgSize);
    kernel.setArg(0, bl_offset);
    kernel.setArg(1, d_uvw);
    kernel.setArg(2, d_wavenumbers);
    kernel.setArg(3, d_visibilities);
    kernel.setArg(4, d_spheroidal);
    kernel.setArg(5, d_aterm);
    kernel.setArg(6, d_baselines);
    kernel.setArg(7, d_subgrid);
    try {
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, &event);
    } catch (cl::Error &error) {
        std::cerr << "Error launching gridder: " << error.what() << std::endl;
        exit(EXIT_FAILURE);
    }
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
    (NR_POLARIZATIONS * sizeof(float complex) + sizeof(float)) +
    // ATerm
    ((2 * sizeof(int)) + (2 * NR_POLARIZATIONS * sizeof(float complex))) +
    // Spheroidal
	NR_POLARIZATIONS * sizeof(float complex));
}


KernelFFT::KernelFFT() {
    uninitialized = true;
}

void KernelFFT::plan(cl::Context &context, int size, int batch, int layout) {
    // Check wheter a new plan has to be created
    if (uninitialized ||
        size   != planned_size ||
        batch  != planned_batch ||
        layout != planned_layout) {
        // Create new plan
        size_t lengths[2] = {size, size};
        clfftCreateDefaultPlan(&fft, context(), CLFFT_2D, lengths);
        clfftSetPlanBatchSize(fft, batch);
        if (layout == FFT_LAYOUT_YXP) {
            // Polarizations in inner dimension
            size_t stride[2] = {NR_POLARIZATIONS, NR_POLARIZATIONS};
            clfftSetPlanInStride(fft, CLFFT_2D, stride);
            clfftSetPlanOutStride(fft, CLFFT_2D, stride);
        } else if (layout == FFT_LAYOUT_PYX) {
            size_t dist = size * size;
            clfftSetPlanDistance(fft, dist, dist);
        }
        
        // Update parameters
        planned_size = size;
        planned_batch = batch;
        planned_layout = layout;
    }
}

void KernelFFT::launchAsync(cl::CommandQueue &queue, cl::Event &event, cl::Buffer &data, clfftDirection direction) {
    cl_event waitEvents[0];
    cl_event outEvents[1];
    outEvents[0] = event();
    cl_command_queue queues[1];
    queues[0] = queue();
    cl_mem input[1];
    input[0] = data();
    clfftEnqueueTransform(fft, direction, 1, queues, 0, waitEvents, outEvents, input, NULL, NULL);
}

uint64_t KernelFFT::flops(int size, int batch) {
	return 1ULL * batch * NR_POLARIZATIONS * 5 * size * size * log(size * size);
}

uint64_t KernelFFT::bytes(int size, int batch) {
	return 1ULL * 2 * batch * size * size * NR_POLARIZATIONS * sizeof(float complex);
}
