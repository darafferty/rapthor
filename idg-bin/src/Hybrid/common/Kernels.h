#include <cstdint>

uint64_t kernel_gridder_flops(
    int jobsize, int nr_timesteps, int nr_channels, int subgridsize, int nr_polarizations);
uint64_t kernel_gridder_bytes(
    int jobsize, int nr_timesteps, int nr_channels, int subgridsize, int nr_polarizations);

uint64_t kernel_degridder_flops(
    int jobsize,  int nr_timesteps,
    int nr_channels, int subgridsize,
    int nr_polarizations);
uint64_t kernel_degridder_bytes(
    int jobsize, int nr_timesteps,
    int nr_channels, int subgridsize,
    int nr_polarizations);

uint64_t kernel_fft_flops(
    int size, int batch, int nr_polarizations);
uint64_t kernel_fft_bytes(
    int size, int batch, int nr_polarizations);

uint64_t kernel_adder_flops(
    int jobsize, int subgridsize);
uint64_t kernel_adder_bytes(
    int jobsize, int subgridsize, int nr_polarizations);

uint64_t kernel_splitter_flops(
    int jobsize, int subgridsize);
uint64_t kernel_splitter_bytes(
    int jobsize, int subgridsize,
    int nr_polarizations);
