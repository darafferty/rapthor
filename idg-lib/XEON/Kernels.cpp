#include "Kernels.h"


// Function signatures
#define sig_degridder (void (*)(unsigned,void*,void*,void*,void*,void*,void*,void*,void*))
#define sig_gridder   (void (*)(unsigned,void*,void*,void*,void*,void*,void*,void*,void*))
#define sig_fft		  (void (*)(int,int,void*,int))
#define sig_adder	  (void (*)(int,void*,void*,void*))
#define sig_shifter   (void (*)(int,void*))


KernelGridder::KernelGridder(rw::Module &module, const char *kernel) : function(module, kernel) {}

void KernelGridder::run(
    int jobsize,
    void *uvw, void *offset,
    void *wavenumbers, void *visibilities,
    void *spheroidal, void *aterm,
    void *baselines, void *uvgrid) {
    (sig_gridder function.get())(
        jobsize, uvw, offset, wavenumbers, visibilities, spheroidal, aterm, baselines, uvgrid);
}

uint64_t KernelGridder::flops(int jobsize) {
    return
    // Grid
    1ULL * jobsize * NR_TIME * BLOCKSIZE * BLOCKSIZE * NR_CHANNELS * (
        // Phasor        
        2 + 92 + 
        // UV
        NR_POLARIZATIONS * 8) +
    // ATerm
    1ULL * jobsize * BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS * 30 +
    // Spheroidal
    1ULL * jobsize * BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS * 2;
}

uint64_t KernelGridder::bytes(int jobsize) {
    return
    // Grid
    1ULL * jobsize * NR_TIME * BLOCKSIZE * BLOCKSIZE * NR_CHANNELS * (NR_POLARIZATIONS * sizeof(float complex) + sizeof(float)) +
    // ATerm
    1ULL * jobsize * BLOCKSIZE * BLOCKSIZE * (2 * sizeof(unsigned)) + (2 * NR_POLARIZATIONS * sizeof(float complex) + sizeof(float)) +
    // Spheroidal
    1ULL * jobsize * BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS * sizeof(float complex);
}


KernelDegridder::KernelDegridder(rw::Module &module, const char *kernel) : function(module, kernel) {}

void KernelDegridder::run(
    int jobsize,
    void *uvgrid, void *uvw,
    void *offset, void *wavenumbers,
    void *aterm, void *baselines,
    void *spheroidal, void *visibilities) {
    (sig_degridder function.get())(
        jobsize, uvgrid, uvw, offset, wavenumbers, aterm, baselines, spheroidal, visibilities);
}

uint64_t KernelDegridder::flops(int jobsize) {
    return
    // ATerm
    1ULL * jobsize * BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS * 32 +
    // Spheroidal
    1ULL * jobsize * BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS * 2 +
    // Degrid
    1ULL * jobsize * NR_TIME * NR_CHANNELS * BLOCKSIZE * BLOCKSIZE * (
        // LMN
        12 + 
        // Offset
        5 +
        // Phasor
        7 + 97 +
        // UV
        NR_POLARIZATIONS * 8);
}

uint64_t KernelDegridder::bytes(int jobsize) {
    return
    // ATerm
    1ULL * jobsize * BLOCKSIZE * BLOCKSIZE * 2 * NR_POLARIZATIONS * sizeof(float complex) +
    // Spheroidal
    1ULL * jobsize * BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS * sizeof(float) +
    // Degrid
    1ULL * jobsize * NR_TIME * NR_CHANNELS * (
        // Offset
        BLOCKSIZE * BLOCKSIZE * 3 * sizeof(float) +
        // UV
        BLOCKSIZE * BLOCKSIZE * NR_POLARIZATIONS * sizeof(float complex) +
        // Visibilities            
        NR_POLARIZATIONS * sizeof(float complex));
}

KernelFFT::KernelFFT(rw::Module &module, const char *kernel) : function(module, kernel) {}

void KernelFFT::run(int size, int batch, void *data, int direction) {
	(sig_fft function.get())(
		size, batch, data, direction);
}

uint64_t KernelFFT::flops(int size, int batch) {
	return 1ULL * batch * 5 * size * size * log(size * size);
}

uint64_t KernelFFT::bytes(int size, int batch) {
	return 1ULL * 2 * batch * size * size * sizeof(float complex);
}


KernelAdder::KernelAdder(rw::Module &module, const char *kernel) : function(module, kernel) {}

void KernelAdder::run(
	int jobsize,
	void *coordinates,
	void *uvgrid,
	void *grid) {
	(sig_adder function.get())(
		jobsize, coordinates, uvgrid, grid);
}

uint64_t KernelAdder::flops(int jobsize) {
    return 1ULL * BLOCKSIZE * BLOCKSIZE * jobsize;
}

uint64_t KernelAdder::bytes(int jobsize) {
	return 1ULL * BLOCKSIZE * BLOCKSIZE * jobsize * (
    // Coordinate
    2 * sizeof(unsigned) +
    // Pixels
    3 * NR_POLARIZATIONS * sizeof(float complex));
}

KernelSplitter::KernelSplitter(rw::Module &module, const char *kernel) : function(module, kernel) {}

void KernelSplitter::run(
	int jobsize,
	void *coordinates,
	void *uvgrid,
	void *grid) {
	(sig_adder function.get())(
		jobsize, coordinates, uvgrid, grid);
}

uint64_t KernelSplitter::flops(int jobsize) {
    return 1ULL * BLOCKSIZE * BLOCKSIZE * jobsize;
}

uint64_t KernelSplitter::bytes(int jobsize) {
	return 1ULL * BLOCKSIZE * BLOCKSIZE * jobsize * (
    // Coordinate
    2 * sizeof(unsigned) +
    // Pixels
    3 * NR_POLARIZATIONS * sizeof(float complex));
}

KernelShifter::KernelShifter(rw::Module &module, const char *kernel) : function(module, kernel) {}

void KernelShifter::run(
	int jobsize,
	void *uvgrid) {
	(sig_shifter function.get())(
		jobsize, uvgrid);
}

uint64_t KernelShifter::flops(int jobsize) {
    return 1ULL * jobsize * BLOCKSIZE * BLOCKSIZE * 6;
}

uint64_t KernelShifter::bytes(int jobsize) {
    return 1ULL * jobsize * BLOCKSIZE * BLOCKSIZE * 3 * sizeof(float complex);
}
