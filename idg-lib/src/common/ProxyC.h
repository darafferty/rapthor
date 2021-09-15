// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Grid visibilities
 * 
 * This wrapper creates a Plan, and Arrays from the raw data pointers and then passes these on 
 * to the @ref idg::proxy::Proxy::gridding "gridding" method of @p p
 *
 * @param p Pointer to Proxy object, previously obtained by one of the create_<proxy_name>() functions. 
 * @param kernel_size Size of the kernel, 
 * see the documentation on @verbatim embed:rst:inline :doc:`kernelsize` @endverbatim
 * @param subgrid_size Size of the subgrids
 * @param nr_channels Number of channels
 * @param nr_baselines Number of baselines
 * @param nr_timesteps Number of time steps
 * @param nr_correlations Number of correlations
 * @param nr_timeslots Number of a-term time slots
 * see the documentation on @verbatim embed:rst:inline :doc:`aterms` @endverbatim
 * @param nr_stations Number of stations
 * @param frequencies Pointer to @p nr_channels frequencies (floats)
 * @param visibilities Pointer to @p nr_baselines * @p nr_timesteps * @p nr_correlations visibilities (complex float*)
 * @param uvw Pointer to @p nr_baselines * @p nr_timesteps u,v,w triplets (floats)
 * @param baselines Pointer to @p nr_baselines pairs of station indices (ints)
 * @param aterms Pointer to @p nr_timeslots x @p nr_stations x 2 x 2 Jones matrix entries (float complex*) see aterms
 * @param aterms_offsets Pointer to @p nr_timesteps + 1  time step indices, 
 * see the documentation on @verbatim embed:rst:inline :doc:`aterms` @endverbatim
 * @param taper Pointer to @p subgrid_size x @p subgrid_size floats, 
 * see the documentation on @verbatim embed:rst:inline :doc:`taper` @endverbatim
 */
void Proxy_gridding(struct Proxy* p, int kernel_size, int subgrid_size,
                    int nr_channels, int nr_baselines, int nr_timesteps,
                    int nr_correlations, int nr_timeslots, int nr_stations,
                    float* frequencies, float complex* visibilities, float* uvw,
                    unsigned int* baselines, float complex* aterms,
                    unsigned int* aterms_offsets, float* taper);

/**
 * Degrid visibilities
 *
 * This wrapper creates a @ref idg::Plan "Plan" for the given uvw coordinates,
 * and @ref ArrayTypes.h "Arrays" from the raw data pointers and then passes these on 
 * to the @ref idg::proxy::Proxy::degridding "degridding" method of @p p
 *
 * @param p Pointer to Proxy object, previously obtained by one of the create_<proxy_name>() functions. 
 * @param kernel_size Size of the kernel, 
 * see the documentation on @verbatim embed:rst:inline :doc:`kernelsize` @endverbatim
 * @param subgrid_size Size of the subgrids
 * @param nr_channels Number of channels
 * @param nr_baselines Number of baselines
 * @param nr_timesteps Number of time steps
 * @param nr_correlations Number of correlations
 * @param nr_timeslots Number of a-term time slots
 * see the documentation on @verbatim embed:rst:inline :doc:`aterms` @endverbatim
 * @param nr_stations Number of stations
 * @param frequencies Pointer to @p nr_channels frequencies (floats)
 * @param visibilities Pointer to @p nr_baselines * @p nr_timesteps * @p nr_correlations visibilities (complex float*)
 * @param uvw Pointer to @p nr_baselines * @p nr_timesteps u,v,w triplets (floats)
 * @param baselines Pointer to @p nr_baselines pairs of station indices (ints)
 * @param aterms Pointer to @p nr_timeslots x @p nr_stations x 2 x 2 Jones matrix entries (float complex*) see aterms
 * @param aterms_offsets Pointer to @p nr_timesteps + 1  time step indices, 
 * see the documentation on @verbatim embed:rst:inline :doc:`aterms` @endverbatim
 * @param taper Pointer to @p subgrid_size x @p subgrid_size floats, 
 * see the documentation on @verbatim embed:rst:inline :doc:`taper` @endverbatim
 */

void Proxy_degridding(struct Proxy* p, int kernel_size, int subgrid_size,
                      int nr_channels, int nr_baselines, int nr_timesteps,
                      int nr_correlations, int nr_timeslots, int nr_stations,
                      float* frequencies, float complex* visibilities,
                      float* uvw, unsigned int* baselines,
                      float complex* aterms, unsigned int* aterms_offsets,
                      float* taper);

/**
 * Init cache
 *
 * @param p
 * @param subgrid_size
 * @param cell_size size of a cell (pixel) in radians
 * @param w_step 
 * @param shift
 */
void Proxy_init_cache(struct Proxy* p, unsigned int subgrid_size,
                      const float cell_size, float w_step, float* shift);

/**
 * Init calibration
 *
 * @param p
 * @param kernel_size
 * @param subgrid_size
 * @param nr_channels
 * @param nr_baselines
 * @param nr_timesteps
 * @param nr_timeslots
 * @param frequencies
 * @param visibilities
 * @param weights
 * @param uvw
 * @param baselines
 * @param aterm_offsets
 * @param taper
 */
void Proxy_calibrate_init(struct Proxy* p, unsigned int kernel_size,
                          unsigned int subgrid_size, unsigned int nr_channels,
                          unsigned int nr_baselines, unsigned int nr_timesteps,
                          unsigned int nr_timeslots, float* frequencies,
                          float complex* visibilities, float* weights,
                          float* uvw, unsigned int* baselines,
                          unsigned int* aterms_offsets, float* taper);

/**
 * Init calibration
 *
 * @param p
 * @param kernel_size
 * @param subgrid_size
 * @param nr_channels
 * @param nr_baselines
 * @param nr_timesteps
 * @param nr_timeslots
 * @param frequencies
 * @param visibilities
 * @param weights
 * @param uvw
 * @param baselines
 * @param aterm_offsets
 * @param taper
 */
void Proxy_calibrate_update(struct Proxy* p, const unsigned int station_nr,
                            const unsigned int subgrid_size,
                            const unsigned int nr_stations,
                            const unsigned int nr_time_slots,
                            const unsigned int nr_terms, float complex* aterms,
                            float complex* aterm_derivatives, double* hessian,
                            double* gradient, double* residual);

/**
 * Init calibration
 *
 * @param p
 */
void Proxy_calibrate_finish(struct Proxy* p);

/**
 * Fourier transformation
 *
 * @param p
 * @param direction
 */
void Proxy_transform(struct Proxy* p, int direction);

/**
 * Destroy proxy
 *
 * @param p
 */
void Proxy_destroy(struct Proxy* p);

void* Proxy_allocate_grid(struct Proxy* p, unsigned int nr_correlations,
                          unsigned int grid_size);

/**
 * set grid
 *
 * @param p
 */
void Proxy_set_grid(struct Proxy* p, float complex* grid_ptr,
                    unsigned int nr_w_layers, unsigned int nr_correlations,
                    unsigned int height, unsigned int width);

void Proxy_get_final_grid(struct Proxy* p, std::complex<float>* grid_ptr,
                          unsigned int nr_w_layers,
                          unsigned int nr_correlations, unsigned int height,
                          unsigned int width);
