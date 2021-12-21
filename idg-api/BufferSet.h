// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_API_BUFFERSET_H_
#define IDG_API_BUFFERSET_H_

#include <vector>
#include <string>
#include <map>
#include <memory>

#include "Buffer.h"
#include "Value.h"

namespace idg {
namespace api {

class BulkDegridder;
class DegridderBuffer;
class GridderBuffer;

enum class BufferSetType {
  kGridding,
  kDegridding,
  kBulkDegridding,
  gridding = kGridding,  // Keep legacy names for backward compatibility.
  degridding = kDegridding
};

enum compute_flags { compute_only = 1, compute_and_grid = 2 };

typedef std::map<std::string, Value> options_type;

class BufferSet {
 public:
  static BufferSet* create(Type architecture);

  virtual ~BufferSet(){};

  static uint64_t get_memory_per_timestep(size_t nStations, size_t nChannels,
                                          size_t nCorrelations = 4);

  /**
   * @brief Initialize bufferset for the image properties
   *
   * @param[in] width
   * @param[in] cellsize
   * @param[in] max_w
   * @param[in] shiftl
   * @param[in] shiftm
   * @param[in] options Map from strings to Values, specifying additional
   * options The following options are recognized: "aterm_kernel_size"
   *                       "max_threads"
   *                       "max_nr_w_layers"
   *                       "padded_size"
   *                       "padding"
   *
   */
  virtual void init(size_t width, float cellsize, float max_w, float shiftl,
                    float shiftm, options_type& options) = 0;

  // Legacy API. To be removed when DP3 / wsclean do not use it anymore
  void init(size_t width, float cellsize, float max_w, float shiftl,
            float shiftm, float /*shiftp*/, options_type& options) {
    init(width, cellsize, max_w, shiftl, shiftm, options);
  }

  /**
   * @brief Initialize buffers for the data to be processed
   *
   * @param bufferTimesteps Size of the buffers in number of timesteps
   * @param bands Vector of frequency bands. Each band is a vector of
   *              channels frequencies (floats). For each band a buffer is
   * allocated.
   * @param nr_stations Number of stations
   * @param max_baseline unused
   * @param options unused
   * @param buffer_set_type Type of buffer to allocate
   */
  virtual void init_buffers(size_t bufferTimesteps,
                            std::vector<std::vector<double>> bands,
                            int nr_stations, float max_baseline,
                            options_type& options,
                            BufferSetType buffer_set_type) = 0;

  /**
   * @brief Get the bulk degridder object for a given frequency band
   *
   * @param i Buffer Id, (DataDescId in Measurement Set)
   * @return const BulkDegridder*
   */
  virtual const BulkDegridder* get_bulk_degridder(int i) = 0;

  /**
   * @brief Get the degridder buffer for a given frequency band
   *
   * @param i Buffer Id, (DataDescId in Measurement Set)
   * @return DegridderBuffer*
   */
  virtual DegridderBuffer* get_degridder(int i) = 0;

  /**
   * @brief Get the gridder buffer for a given frequency band
   *
   * @param i Buffer Id, (DataDescId in Measurement Set)
   * @return GridderBuffer*
   */
  virtual GridderBuffer* get_gridder(int i) = 0;

  /**
   * @brief Fourier tranform the image and keep the resulting grid
   *
   * Running this function is required to set the model image before starting
   * degridding.
   *
   * @param[in] image Pointer to image data.
   *                  Image properties must already have been set by a call to
   *                  init()
   * @param[in] do_scale If true, scale image from apparent to intrinsic flux.
   * Default: false
   */
  virtual void set_image(const double* image, bool do_scale = false) = 0;

  /**
   * @brief Fourier transform the grid and copy the resulting image to the
   * output parameter
   *
   * Running this function is required to get the image after gridding.
   *
   * @param[out] image Pointer to image data.
   *                   Image properties must already have been set by a call to
   *                   init()
   */
  virtual void get_image(double* image) = 0;

  /**
   * @brief Free memory after gridding/degridding has finished
   */
  virtual void finished() = 0;

  /**
   * @brief Get the current subgridsize
   *
   * @return Size of the subgrid in pixels in one dimension
   */
  virtual size_t get_subgridsize() const = 0;

  /**
   * @brief Get the current subgrid pixelsize
   *
   * @return Size of a subgrid pixel in radians
   */
  virtual float get_subgrid_pixelsize() const = 0;

  /**
   * @brief Set the application of  aterms on or off
   *
   * @param do_apply Apply aterms or not?
   */
  virtual void set_apply_aterm(bool do_apply) = 0;

  /**
   * @brief Initialize the computation of the average beam.
   *
   * The actual computation of the average beam happens along gridding
   *
   * @param flag
   */
  virtual void init_compute_avg_beam(compute_flags flag) = 0;

  /**
   * @brief Finalize the computation of the average beam
   *
   * The resulting (inverse) beams are stored as members that can be retrieved
   * through the get_scalar_beam() and get_matrix_inverse_beam() functions.
   */
  virtual void finalize_compute_avg_beam() = 0;

  /**
   * @brief Get the scalar beam object
   */
  virtual std::shared_ptr<std::vector<float>> get_scalar_beam() const = 0;

  /**
   * @brief Get the matrix inverse beam object
   */
  virtual std::shared_ptr<std::vector<std::complex<float>>>
  get_matrix_inverse_beam() const = 0;

  /**
   * @brief Set the scalar beam object
   */
  virtual void set_scalar_beam(std::shared_ptr<std::vector<float>>) = 0;

  /**
   * @brief Set the matrix inverse beam object
   *
   * The matrix inverse beam is applied while gridding
   */
  virtual void set_matrix_inverse_beam(
      std::shared_ptr<std::vector<std::complex<float>>>) = 0;

  /**
   * @brief
   */
  virtual void unset_matrix_inverse_beam() = 0;

 protected:
  BufferSet() {}
};

}  // namespace api
}  // namespace idg

#endif
