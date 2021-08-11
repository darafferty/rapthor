// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Grid visibilities
 *
 * @param p
 * @param kernel_size
 * @param subgrid_size
 * @param nr_channels
 * @param nr_baselines
 * @param nr_timesteps
 * @param nr_correlations
 * @param nr_timeslots
 * @param nr_stations
 * @param frequencies
 * @param visibilities
 * @param uvw
 * @param baselines
 * @param aterms
 * @param aterms_offsets
 * @param taper
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
 * @param p
 * @param kernel_size
 * @param subgrid_size
 * @param nr_channels
 * @param nr_baselines
 * @param nr_timesteps
 * @param nr_correlations
 * @param nr_timeslots
 * @param nr_stations
 * @param frequencies
 * @param visibilities
 * @param uvw
 * @param baselines
 * @param aterms
 * @param aterms_offsets
 * @param taper
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
 * @param cell_size
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
