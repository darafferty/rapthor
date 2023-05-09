// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_AUX_H_
#define IDG_AUX_H_

#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include <sys/mman.h>

inline int min(int a, int b) { return a < b ? a : b; }

inline int max(int a, int b) { return a > b ? a : b; }

namespace idg {
namespace auxiliary {

/*
    Operation and byte count
*/
uint64_t flops_gridder(uint64_t nr_channels, uint64_t nr_timesteps,
                       uint64_t nr_subgrids, uint64_t subgrid_size,
                       uint64_t nr_correlations);

uint64_t bytes_gridder(uint64_t nr_channels, uint64_t nr_timesteps,
                       uint64_t nr_subgrids, uint64_t subgrid_size,
                       uint64_t nr_correlations);

uint64_t flops_degridder(uint64_t nr_channels, uint64_t nr_timesteps,
                         uint64_t nr_subgrids, uint64_t subgrid_size,
                         uint64_t nr_correlations);

uint64_t bytes_degridder(uint64_t nr_channels, uint64_t nr_timesteps,
                         uint64_t nr_subgrids, uint64_t subgrid_size,
                         uint64_t nr_correlations);

uint64_t flops_calibrate(uint64_t nr_terms, uint64_t nr_channels,
                         uint64_t nr_timesteps, uint64_t nr_subgrids,
                         uint64_t subgrid_size, uint64_t nr_correlations = 4);

uint64_t bytes_calibrate();

uint64_t flops_fft(uint64_t size, uint64_t batch, uint64_t nr_correlations = 4);

uint64_t bytes_fft(uint64_t size, uint64_t batch, uint64_t nr_correlations = 4);

uint64_t flops_adder(uint64_t nr_subgrids, uint64_t subgrid_size,
                     uint64_t nr_correlations);

uint64_t bytes_adder(uint64_t nr_subgrids, uint64_t subgrid_size,
                     uint64_t nr_correlations);

uint64_t flops_splitter(uint64_t nr_subgrids, uint64_t subgrid_size,
                        uint64_t nr_correlations);

uint64_t bytes_splitter(uint64_t nr_subgrids, uint64_t subgrid_size,
                        uint64_t nr_correlations);

uint64_t flops_scaler(uint64_t nr_subgrids, uint64_t subgrid_size,
                      uint64_t nr_correlations = 4);

uint64_t bytes_scaler(uint64_t nr_subgrids, uint64_t subgrid_size,
                      uint64_t nr_correlations = 4);

/*
    Sizeof routines
*/
uint64_t sizeof_visibilities(unsigned int nr_baselines,
                             unsigned int nr_timesteps,
                             unsigned int nr_channels,
                             unsigned int nr_correlations);

uint64_t sizeof_uvw(unsigned int nr_baselines, unsigned int nr_timesteps);

uint64_t sizeof_subgrids(unsigned int nr_subgrids, unsigned int subgrid_size,
                         unsigned int nr_correlations);

uint64_t sizeof_metadata(unsigned int nr_subgrids);

uint64_t sizeof_wavenumbers(unsigned int nr_channels);

uint64_t sizeof_aterms(unsigned int nr_stations, unsigned int nr_timeslots,
                       unsigned int subgrid_size, uint64_t nr_correlations = 4);

uint64_t sizeof_aterm_indices(unsigned int nr_baselines,
                              unsigned int nr_timesteps);

uint64_t sizeof_taper(unsigned int subgrid_size);

uint64_t sizeof_avg_aterm_correction(unsigned int subgrid_size,
                                     uint64_t nr_correlations = 4);

uint64_t sizeof_baselines(unsigned int nr_baselines);

uint64_t sizeof_aterm_offsets(unsigned int nr_timeslots);

uint64_t sizeof_weights(unsigned int nr_baselines, unsigned int nr_timesteps,
                        unsigned int nr_channels,
                        unsigned int nr_correlations = 4);

/*
    Misc
*/
std::vector<int> split_int(const char* string, const char* delimiter);
std::vector<std::string> split_string(char* string, const char* delimiter);

std::string get_inc_dir();
std::string get_lib_dir();

size_t get_nr_threads();

void print_version();

/*
    Memory
*/
size_t get_total_memory();
size_t get_used_memory();
size_t get_free_memory();

class Memory {
 public:
  virtual ~Memory() {}
  Memory(const Memory&) = delete;
  Memory(Memory&&) = delete;
  Memory& operator=(const Memory&) = delete;
  Memory& operator=(Memory&&) = delete;

  void* data() { return ptr_; };
  size_t size() { return size_; };
  virtual void zero() { std::fill_n(static_cast<char*>(data()), size(), 0); };

 protected:
  explicit Memory(size_t size) : size_(size) {}
  Memory(void* ptr, size_t size) : ptr_(ptr) {}
  void set(void* ptr) { ptr_ = ptr; }

 private:
  void* ptr_ = nullptr;
  size_t size_ = 0;
};

class DefaultMemory : public Memory {
 public:
  DefaultMemory(size_t size);
  ~DefaultMemory() override;
};

class AlignedMemory : public Memory {
 public:
  AlignedMemory(size_t size);
  ~AlignedMemory() override;

 private:
  static const unsigned int alignment_ = 64;
};

}  // namespace auxiliary
}  // namespace idg

#endif
