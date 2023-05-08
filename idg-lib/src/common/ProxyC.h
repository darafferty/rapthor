// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/** @file */

/**
 * Grid visibilities
 *
 * This wrapper creates a Plan, and
 * @verbatim embed:rst:inline :doc:`Arrays <arraytypes>` @endverbatim
 * from the raw data pointers and then
 * passes these on to the @ref idg::proxy::Proxy::gridding "gridding" method of
 * @p p
 *
 * @param p Pointer to Proxy object, previously obtained by one of the
 * create_<proxy_name>() functions.
 * @param kernel_size Size of the kernel,
 * see the documentation on
 * @verbatim embed:rst:inline :doc:`kernelsize` @endverbatim
 * @param subgrid_size Size of the subgrids
 * @param nr_channels Number of channels
 * @param nr_baselines Number of baselines
 * @param nr_timesteps Number of time steps
 * @param nr_correlations Number of correlations
 * @param nr_timeslots Number of a-term time slots
 * see the documentation on
 * @verbatim embed:rst:inline :doc:`aterms` @endverbatim
 * @param nr_stations Number of stations
 * @param frequencies Pointer to @p nr_channels frequencies (floats)
 * @param visibilities Pointer to @p nr_baselines * @p nr_timesteps * @p
 * nr_correlations visibilities (complex float*)
 * @param uvw Pointer to @p nr_baselines * @p nr_timesteps u,v,w triplets
 * (floats)
 * @param baselines Pointer to @p nr_baselines pairs of station indices (ints)
 * @param aterms Pointer to @p nr_timeslots x @p nr_stations x 2 x 2 Jones
 * matrix entries (float complex*) see
 * @verbatim embed:rst:inline :doc:`aterms` @endverbatim
 * @param aterm_offsets Pointer to @p nr_timesteps + 1  time step indices,
 * see the documentation on
 * @verbatim embed:rst:inline :doc:`aterms` @endverbatim
 * @param taper Pointer to @p subgrid_size x @p subgrid_size floats,
 * see the documentation on @verbatim embed:rst:inline :doc:`taper` @endverbatim
 */
void Proxy_gridding(struct Proxy* p, int kernel_size, int subgrid_size,
                    int nr_channels, int nr_baselines, int nr_timesteps,
                    int nr_correlations, int nr_timeslots, int nr_stations,
                    float* frequencies, float complex* visibilities, float* uvw,
                    unsigned int* baselines, float complex* aterms,
                    unsigned int* aterm_offsets, float* taper);

/**
 * Degrid visibilities
 *
 * This wrapper creates a @ref idg::Plan "Plan" for the given uvw coordinates,
 * and
 * @verbatim embed:rst:inline :doc:`Arrays <arraytypes>` @endverbatim
 * from the raw data pointers and then passes
 * these on to the @ref idg::proxy::Proxy::degridding "degridding" method of @p
 * p
 *
 * @param p[in] Pointer to Proxy object, previously obtained by one of the
 * create_<proxy_name>() functions.
 * @param kernel_size[in] Size of the kernel,
 * see the documentation on
 * @verbatim embed:rst:inline :doc:`kernelsize` @endverbatim
 * @param subgrid_size[in] Size of the subgrids
 * @param nr_channels[in] Number of channels
 * @param nr_baselines[in] Number of baselines
 * @param nr_timesteps[in] Number of time steps
 * @param nr_correlations[in] Number of correlations
 * @param nr_timeslots[in] Number of a-term time slots
 * see the documentation on
 * @verbatim embed:rst:inline :doc:`aterms` @endverbatim
 * @param nr_stations[in] Number of stations
 * @param frequencies[in] Pointer to @p nr_channels frequencies (floats)
 * @param[out] visibilities Pointer to @p nr_baselines * @p nr_timesteps * @p
 * nr_correlations visibilities (complex float*)
 * @param uvw[in] Pointer to @p nr_baselines * @p nr_timesteps u,v,w triplets
 * (floats)
 * @param baselines[in] Pointer to @p nr_baselines pairs of station indices
 * (ints)
 * @param aterms[in] Pointer to @p nr_timeslots x @p nr_stations x 2 x 2 Jones
 * matrix entries (float complex*) see
 * @verbatim embed:rst:inline :doc:`aterms` @endverbatim
 * @param aterm_offsets[in] Pointer to @p nr_timesteps + 1  time step indices,
 * see the documentation on
 * @verbatim embed:rst:inline :doc:`aterms` @endverbatim
 * @param taper[in] Pointer to @p subgrid_size x @p subgrid_size floats,
 * see the documentation on
 * @verbatim embed:rst:inline :doc:`taper` @endverbatim
 */

void Proxy_degridding(struct Proxy* p, int kernel_size, int subgrid_size,
                      int nr_channels, int nr_baselines, int nr_timesteps,
                      int nr_correlations, int nr_timeslots, int nr_stations,
                      float* frequencies, float complex* visibilities,
                      float* uvw, unsigned int* baselines,
                      float complex* aterms, unsigned int* aterm_offsets,
                      float* taper);

/**
 * Initialize the cache
 *
 * Call is forwarded to the @ref idg::proxy::Proxy::init_cache "init_cache"
 * method of @p.
 *
 * @param p pointer to Proxy object
 * @param subgrid_size size of the subgrids in pixels in one dimension
 * @param cell_size size of a cell (pixel) in radians
 * @param w_step distance between w-layers in wavelengths
 * @param shift l,m pointing offset (facet position) in radians
 */
void Proxy_init_cache(struct Proxy* p, unsigned int subgrid_size,
                      const float cell_size, float w_step, float* shift);

/**
 * Initialize calibration
 *
 * This wrapper creates
 * @verbatim embed:rst:inline :doc:`Arrays <arraytypes>` @endverbatim
 * from the raw data pointers and then forwards the call to
 * the @ref idg::proxy::Proxy::calibrate_init "calibrate_init" method of @p
 * p
 *
 * @param p Pointer to Proxy object, previously obtained by one of the
 * create_<proxy_name>() functions.
 * @param kernel_size Size of the kernel,
 * see the documentation on
 * @verbatim embed:rst:inline :doc:`kernelsize` @endverbatim
 * @param subgrid_size Size of the subgrids
 * @param nr_channels Number of channels
 * @param nr_baselines Number of baselines
 * @param nr_timesteps Number of time steps
 * @param nr_timeslots Number of a-term time slots
 * see the documentation on
 * @verbatim embed:rst:inline :doc:`aterms` @endverbatim
 * @param frequencies Pointer to @p nr_channels frequencies (floats)
 * @param visibilities Pointer to @p nr_baselines * @p nr_timesteps * @p
 * nr_correlations visibilities (complex float*)
 * @param weights
 * @param uvw Pointer to @p nr_baselines * @p nr_timesteps u,v,w triplets
 * (floats)
 * @param baselines Pointer to @p nr_baselines pairs of station indices (ints)
 * @param aterm_offsets Pointer to @p nr_timesteps + 1  time step indices,
 * see the documentation on
 * @verbatim embed:rst:inline :doc:`aterms` @endverbatim
 * @param taper Pointer to @p subgrid_size x @p subgrid_size floats,
 * see the documentation on
 * @verbatim embed:rst:inline :doc:`taper` @endverbatim
 */
void Proxy_calibrate_init(struct Proxy* p, unsigned int kernel_size,
                          unsigned int subgrid_size,
                          unsigned int nr_channel_blocks,
                          unsigned int nr_channels_per_block,
                          unsigned int nr_baselines, unsigned int nr_timesteps,
                          unsigned int nr_timeslots, float* frequencies,
                          float complex* visibilities, float* weights,
                          float* uvw, unsigned int* baselines,
                          unsigned int* aterm_offsets, float* taper);

/**
 * @brief Calibration update step
 *
 * Computes Hessian, gradient and residual for the current working point
 * This wrapper creates
 * @verbatim embed:rst:inline :doc:`Arrays <arraytypes>` @endverbatim
 * from the raw data pointers and then forwards the call to
 * the @ref idg::proxy::Proxy::calibrate_update "calibrate_update" method of @p
 * p
 *
 * @param p[in] Pointer to Proxy object, previously obtained by one of the
 *              create_<proxy_name>() functions.
 * @param[in] antenna_nr Antenna for which the update is computed
 * @param[in] subgrid_size Size of the subgrid
 * @param[in] nr_stations Number of stations
 * @param nr_timeslots[in] Number of a-term time slots
 * see the documentation on @verbatim embed:rst:inline :doc:`aterms`
 * @endverbatim
 * @param[in] nr_terms Number of unknowns
 * @param aterms[in] Pointer to @p nr_timeslots x @p nr_stations x 2 x 2 Jones
 * matrix entries (float complex*) see
 * @verbatim embed:rst:inline :doc:`aterms` @endverbatim
 * @param[in] aterm_derivatives Pointer to @p nr_timeslots x @p nr_terms x @p
 * subgrid_size x @subgrid_size x 2 x 2 derivative Jones matrix entries (float
 * complex*)
 * @param[out] hessian
 * @param[out] gradient
 * @param[out] residual
 */
void Proxy_calibrate_update(struct Proxy* p, const unsigned int antenna_nr,
                            const unsigned int nr_channel_blocks,
                            const unsigned int subgrid_size,
                            const unsigned int nr_stations,
                            const unsigned int nr_time_slots,
                            const unsigned int nr_terms, float complex* aterms,
                            float complex* aterm_derivatives, double* hessian,
                            double* gradient, double* residual);

/**
 * Finish  calibration, free internal buffers
 *
 * Call is forwarded to
 * the @ref idg::proxy::Proxy::calibrate_finish "calibrate_finish" method of @p
 * p
 * @param p
 */
void Proxy_calibrate_finish(struct Proxy* p);

/**
 * Fourier transformation
 *
 * Applies the Fourier transform to the grid that has been set previously
 * through the Proxy_set_grid method
 * Call is forwarded to
 * the @ref idg::proxy::Proxy::transform "transform" method of @p p
 *
 * @param p
 * @param direction
 */
void Proxy_transform(struct Proxy* p, int direction);

/**
 * @brief Destroy proxy
 *
 * Call the destructor of p, and deallocates memory
 *
 * @param p
 */
void Proxy_destroy(struct Proxy* p);

void* Proxy_allocate_grid(struct Proxy* p, unsigned int nr_correlations,
                          unsigned int grid_size);

/**
 * @brief Set the grid to be used in subsequent data processing calls
 *
 * Call is forwarded to
 * the @ref idg::proxy::Proxy::set_grid "set_grid" method of @p p
 *
 * @param p pointer to proxy
 * @param grid_ptr pointer to grid data
 * @param nr_w_layers number of w layers in grid
 * @param nr_correlations number of correlations in grid
 * @param height height of grid
 * @param width width of grid
 */
void Proxy_set_grid(struct Proxy* p, float complex* grid_ptr,
                    unsigned int nr_w_layers, unsigned int nr_correlations,
                    unsigned int height, unsigned int width);

/**
 * @brief Get the final grid after gridding
 *
 * This call flushes any pending operations
 *
 * Call is forwarded to
 * the @ref idg::proxy::Proxy::get_final_grid "get_final_grid" method of @p p
 *
 * @param p pointer to proxy
 * @param grid_ptr pointer to grid data
 * @param nr_w_layers number of w layers in grid
 * @param nr_correlations number of correlations in grid
 * @param height height of grid
 * @param width width of grid
 */
void Proxy_get_final_grid(struct Proxy* p, std::complex<float>* grid_ptr,
                          unsigned int nr_w_layers,
                          unsigned int nr_correlations, unsigned int height,
                          unsigned int width);
