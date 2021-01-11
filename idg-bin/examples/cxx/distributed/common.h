// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <iostream>
#include <iomanip>
#include <cstdlib>  // size_t
#include <complex>
#include <tuple>
#include <typeinfo>
#include <vector>
#include <algorithm>  // max_element
#include <numeric>    // accumulate
#include <mutex>
#include <thread>

#include <mpi.h>

#include "idg-util.h"  // Data init routines

using namespace std;

std::tuple<int, int, int, int, int, int, int, int> read_parameters() {
  const unsigned int DEFAULT_NR_STATIONS = 52;      // all LOFAR LBA stations
  const unsigned int DEFAULT_NR_CHANNELS = 16 * 4;  // 16 channels, 4 subbands
  const unsigned int DEFAULT_NR_TIMESTEPS =
      (3600 * 4);  // 4 hours of observation
  const unsigned int DEFAULT_NR_TIMESLOTS =
      DEFAULT_NR_TIMESTEPS / (60 * 30);  // update every 30 minutes
  const unsigned int DEFAULT_GRIDSIZE = 4096;
  const unsigned int DEFAULT_SUBGRIDSIZE = 32;
  const unsigned int DEFAULT_NR_CYCLES = 1;

  char *cstr_nr_stations = getenv("NR_STATIONS");
  auto nr_stations =
      cstr_nr_stations ? atoi(cstr_nr_stations) : DEFAULT_NR_STATIONS;

  char *cstr_nr_channels = getenv("NR_CHANNELS");
  auto nr_channels =
      cstr_nr_channels ? atoi(cstr_nr_channels) : DEFAULT_NR_CHANNELS;

  char *cstr_nr_timesteps = getenv("NR_TIMESTEPS");
  auto nr_timesteps =
      cstr_nr_timesteps ? atoi(cstr_nr_timesteps) : DEFAULT_NR_TIMESTEPS;

  char *cstr_nr_timeslots = getenv("NR_TIMESLOTS");
  auto nr_timeslots =
      cstr_nr_timeslots ? atoi(cstr_nr_timeslots) : DEFAULT_NR_TIMESLOTS;

  char *cstr_grid_size = getenv("GRIDSIZE");
  auto grid_size = cstr_grid_size ? atoi(cstr_grid_size) : DEFAULT_GRIDSIZE;

  char *cstr_subgrid_size = getenv("SUBGRIDSIZE");
  auto subgrid_size =
      cstr_subgrid_size ? atoi(cstr_subgrid_size) : DEFAULT_SUBGRIDSIZE;

  char *cstr_kernel_size = getenv("KERNELSIZE");
  auto kernel_size =
      cstr_kernel_size ? atoi(cstr_kernel_size) : (subgrid_size / 4) + 1;

  char *cstr_nr_cycles = getenv("NR_CYCLES");
  auto nr_cycles = cstr_nr_cycles ? atoi(cstr_nr_cycles) : DEFAULT_NR_CYCLES;

  return std::make_tuple(nr_stations, nr_channels, nr_timesteps, nr_timeslots,
                         grid_size, subgrid_size, kernel_size, nr_cycles);
}

void print_parameters(unsigned int nr_stations, unsigned int nr_channels,
                      unsigned int nr_timesteps, unsigned int nr_timeslots,
                      float image_size, unsigned int grid_size,
                      unsigned int subgrid_size, unsigned int kernel_size) {
  const int fw1 = 30;
  const int fw2 = 10;
  ostream &os = clog;

  os << "-----------" << endl;
  os << "PARAMETERS:" << endl;

  os << setw(fw1) << left << "Number of stations"
     << "== " << setw(fw2) << right << nr_stations << endl;

  os << setw(fw1) << left << "Number of channels"
     << "== " << setw(fw2) << right << nr_channels << endl;

  os << setw(fw1) << left << "Number of timesteps"
     << "== " << setw(fw2) << right << nr_timesteps << endl;

  os << setw(fw1) << left << "Number of timeslots"
     << "== " << setw(fw2) << right << nr_timeslots << endl;

  os << setw(fw1) << left << "Imagesize"
     << "== " << setw(fw2) << right << image_size << endl;

  os << setw(fw1) << left << "Grid size"
     << "== " << setw(fw2) << right << grid_size << endl;

  os << setw(fw1) << left << "Subgrid size"
     << "== " << setw(fw2) << right << subgrid_size << endl;

  os << setw(fw1) << left << "Kernel size"
     << "== " << setw(fw2) << right << kernel_size << endl;

  os << "-----------" << endl;
}

void send_int(int dst, int value) {
  MPI_Send(&value, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
}

void send_float(int dst, float value) {
  MPI_Send(&value, 1, MPI_FLOAT, dst, 0, MPI_COMM_WORLD);
}

int receive_int(int src = 0) {
  int result = 0;
  MPI_Recv(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  return result;
}

float receive_float(int src = 0) {
  float result = 0;
  MPI_Recv(&result, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  return result;
}

template <typename T>
void send_array(int dst, T &array) {
  MPI_Send(array.data(), array.bytes(), MPI_BYTE, dst, 0, MPI_COMM_WORLD);
}

template <typename T>
void receive_array(int src, T &array) {
  MPI_Recv(array.data(), array.bytes(), MPI_BYTE, src, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
}

void send_bytes(int dst, void *buf, size_t bytes) {
  MPI_Send(buf, bytes, MPI_BYTE, dst, 0, MPI_COMM_WORLD);
}

void receive_bytes(int src, void *buf, size_t bytes) {
  MPI_Recv(buf, bytes, MPI_BYTE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

class MPIRequest
{
public:

  MPIRequest(bool blocking = false) :
    m_blocking(blocking)
  {
  }

  void send(
    const void *buf,
    int bytes,
    int dest,
    int tag = 0)
  {
    MPI_Isend(buf, bytes, MPI_BYTE, dest, tag, MPI_COMM_WORLD, &m_request);
    if (m_blocking)
    {
      wait();
    }
  }

  void receive(
    void *buf,
    int bytes,
    int source,
    int tag = 0)
  {
    MPI_Irecv(buf, bytes, MPI_BYTE, source, tag, MPI_COMM_WORLD, &m_request);
    if (m_blocking)
    {
      wait();
    }
  }

  void wait()
  {
    MPI_Wait(&m_request, MPI_STATUS_IGNORE);
  }

private:
  MPI_Request m_request;
  bool m_blocking;
};

class MPIRequestList
{
public:
  MPIRequestList() :
    m_requests(0)
  {
  }

  ~MPIRequestList()
  {
    wait();
  }

  std::shared_ptr<MPIRequest> create(bool blocking = false)
  {
    m_requests.emplace_back(new MPIRequest(blocking));
    return m_requests.back();
  }

  void wait()
  {
    for (auto &request : m_requests)
    {
      request->wait();
    }
  }

private:
  std::vector<std::shared_ptr<MPIRequest>> m_requests;
};

void synchronize() {
  MPI_Barrier(MPI_COMM_WORLD);
}

void print(int rank, const char *message) {
  std::clog << "[" << rank << "] " << message << std::endl;
}

void print(int rank, const std::string& message) {
  print(rank, message.c_str());
}

idg::Plan::Options get_plan_options() {
  idg::Plan::Options options;
  options.plan_strict = true;
  options.max_nr_timesteps_per_subgrid = 128;
  options.max_nr_channels_per_subgrid = 8;
  return options;
}

void reduce_grids(
  std::shared_ptr<idg::Grid> grid,
  unsigned int rank,
  unsigned int world_size)
{
  unsigned int nr_w_layers = grid->get_w_dim();
  unsigned int grid_size = grid->get_y_dim();

  #pragma omp parallel for
  for (unsigned int y = 0; y < grid_size; y++)
  {
    idg::Array1D<std::complex<float>> row(grid_size);

    for (unsigned int w = 0; w < nr_w_layers; w++)
    {
      for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++)
      {
        for (unsigned int i = (world_size+1)/2; i > 0; i /= 2)
        {
          if ((unsigned int) rank < i)
          {
            #pragma omp critical
            {
              MPIRequest request(true);
              request.receive(row.data(), row.bytes(), i + rank);
            }

            for (unsigned int x = 0; x < grid_size; x++)
            {
              auto& grid_ = *grid;
              row(x) += grid_(w, pol, y, x);
            }
          } else if (rank < (2 * i)) {
            #pragma omp critical
            {
              MPIRequest request(true);
              request.send(row.data(), row.bytes(), rank - i);
            }
          }
        }
      }
    }
  }
}

void broadcast_grid(
  std::shared_ptr<idg::Grid> grid,
  int root)
{
  unsigned int grid_size = grid->get_y_dim();
  unsigned int w = 0; // W-stacking is handled by the workers
  for (unsigned int y = 0; y < grid_size; y++)
  {
    for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++)
    {
      std::complex<float> *row_ptr = grid->data(w, pol, y, 0);
      size_t sizeof_row = grid_size * sizeof(std::complex<float>);
      MPI_Bcast(row_ptr, sizeof_row, MPI_BYTE, root, MPI_COMM_WORLD);
    }
  }
}

void run_master() {
  idg::auxiliary::print_version();

  // Constants
  unsigned int nr_w_layers = 1;
  unsigned int nr_correlations = 4;
  float w_offset = 0.0;
  unsigned int nr_stations;
  unsigned int nr_channels;
  unsigned int nr_timesteps;
  unsigned int nr_timeslots;
  float integration_time = 1.0;
  unsigned int grid_size;
  unsigned int subgrid_size;
  unsigned int kernel_size;
  unsigned int nr_cycles;

  // Read parameters from environment
  std::tie(nr_stations, nr_channels, nr_timesteps, nr_timeslots, grid_size,
           subgrid_size, kernel_size, nr_cycles) =
      read_parameters();
  unsigned int nr_baselines = (nr_stations * (nr_stations - 1)) / 2;

  // Initialize Data object
  idg::Data data =
      idg::get_example_data(nr_baselines, grid_size, integration_time);

  // Print data info
  data.print_info();

  // Get remaining parameters
  nr_baselines = data.get_nr_baselines();
  float image_size = data.compute_image_size(grid_size);
  float cell_size = image_size / grid_size;

  // Print parameters
  print_parameters(nr_stations, nr_channels, nr_timesteps, nr_timeslots,
                   image_size, grid_size, subgrid_size, kernel_size);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Distribute the work over frequency
  nr_channels /= world_size;

  // Distribute parameters
  for (int dst = 0; dst < world_size; dst++) {
    send_int(dst, nr_stations);
    send_int(dst, nr_baselines);
    send_int(dst, nr_timesteps);
    send_int(dst, nr_timeslots);
    send_float(dst, integration_time);
    send_int(dst, nr_channels);
    send_int(dst, nr_correlations);
    send_int(dst, nr_w_layers);
    send_float(dst, w_offset);
    send_int(dst, grid_size);
    send_int(dst, subgrid_size);
    send_int(dst, kernel_size);
    send_float(dst, cell_size);
    send_int(dst, nr_cycles);
  }

  // Initialize frequency data for master
  idg::Array1D<float> frequencies(nr_channels);
  data.get_frequencies(frequencies, image_size);

  // Distribute frequencies to workers
  // Every worker processes a different subband, thus
  // take a channel offset into account when initializing
  // frequencies.
  for (int dst = 1; dst < world_size; dst++) {
    int channel_offset = (dst - 1) * nr_channels;
    idg::Array1D<float> frequencies_(nr_channels);
    data.get_frequencies(frequencies_, image_size, channel_offset);
    send_array(dst, frequencies_);
  }

  // Distribute data
  for (int dst = 1; dst < world_size; dst++) {
    send_bytes(dst, &data, sizeof(data));
  }

  // Initialize proxy
  ProxyType proxy;

  // Allocate and initialize static data structures
  idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms =
      idg::get_identity_aterms(proxy, nr_timeslots, nr_stations, subgrid_size,
                               subgrid_size);
  idg::Array1D<unsigned int> aterms_offsets =
      idg::get_example_aterms_offsets(proxy, nr_timeslots, nr_timesteps);
  idg::Array2D<float> spheroidal =
      idg::get_example_spheroidal(proxy, subgrid_size, subgrid_size);
  auto grid =
      proxy.allocate_grid(nr_w_layers, nr_correlations, grid_size, grid_size);
  idg::Array1D<float> shift = idg::get_zero_shift();
  idg::Array1D<std::pair<unsigned int, unsigned int>> baselines =
      idg::get_example_baselines(proxy, nr_stations, nr_baselines);

  // Plan options
  idg::Plan::Options options = get_plan_options();
  omp_set_nested(true);

  // Input buffers
  idg::Array2D<idg::UVW<float>> uvw(nr_baselines, nr_timesteps);
  idg::Array3D<idg::Visibility<std::complex<float>>> visibilities =
      idg::get_dummy_visibilities(nr_baselines, nr_timesteps, nr_channels);
  int time_offset = 0;
  int bl_offset = 0;

  // Set grid
  proxy.set_grid(grid);

  // Performance measurement
  std::vector<double> runtimes_gridding(nr_cycles);
  std::vector<double> runtimes_degridding(nr_cycles);
  std::vector<double> runtimes_grid_reduce(nr_cycles);
  std::vector<double> runtimes_grid_fft(nr_cycles);
  std::vector<double> runtimes_grid_broadcast(nr_cycles);

  // Iterate all cycles
  for (unsigned cycle = 0; cycle < nr_cycles; cycle++) {
    // Get UVW coordinates for current cycle
    data.get_uvw(uvw, bl_offset, time_offset, integration_time);

    // Create plan
    auto plan = std::unique_ptr<idg::Plan>(new idg::Plan(
       kernel_size, subgrid_size, grid_size, cell_size, frequencies, uvw,
       baselines, aterms_offsets, options));

    // Run gridding
    runtimes_gridding[cycle] = -omp_get_wtime();
    proxy.gridding(*plan, w_offset, shift, cell_size, kernel_size,
        subgrid_size, frequencies, visibilities, uvw,
        baselines, aterms, aterms_offsets, spheroidal);
    synchronize();
    runtimes_gridding[cycle] += omp_get_wtime();

    // Get grid
    grid = proxy.get_grid();

    // Run FFT
    runtimes_grid_fft[cycle] = -omp_get_wtime();
    proxy.transform(idg::FourierDomainToImageDomain, *grid);
    runtimes_grid_fft[cycle] += omp_get_wtime();

    // Reduce grids
    if (world_size > 1)
    {
      runtimes_grid_reduce[cycle] = -omp_get_wtime();
      reduce_grids(grid, 0, world_size);
      runtimes_grid_reduce[cycle] += omp_get_wtime();
    }

    // Deconvolution
    // not implemented

    // Broadcast model image to workers
    if (world_size > 1)
    {
      runtimes_grid_broadcast[cycle] = -omp_get_wtime();
      broadcast_grid(grid, 0);
      runtimes_grid_broadcast[cycle] += omp_get_wtime();
    }

    // Set grid
    proxy.set_grid(grid);

    // Run FFT
    runtimes_grid_fft[cycle] -= omp_get_wtime();
    proxy.transform(idg::ImageDomainToFourierDomain, *grid);
    runtimes_grid_fft[cycle] += omp_get_wtime();

    // Run degridding
    runtimes_degridding[cycle] = -omp_get_wtime();
    proxy.degridding(*plan, w_offset, shift, cell_size, kernel_size,
        subgrid_size, frequencies, visibilities, uvw,
        baselines, aterms, aterms_offsets, spheroidal);
    synchronize();
    runtimes_degridding[cycle] += omp_get_wtime();

    // Go the the next batch of timesteps
    time_offset += nr_timesteps;
  }

  // Report timings
  std::clog << std::endl;
  double runtime_gridding = std::accumulate(runtimes_gridding.begin(), runtimes_gridding.end(), 0.0);
  double runtime_degridding = std::accumulate(runtimes_degridding.begin(), runtimes_degridding.end(), 0.0);
  double runtime_grid_fft = std::accumulate(runtimes_grid_fft.begin(), runtimes_grid_fft.end(), 0.0);
  double runtime_grid_reduce = std::accumulate(runtimes_grid_reduce.begin(), runtimes_grid_reduce.end(), 0.0);
  double runtime_grid_broadcast = std::accumulate(runtimes_grid_broadcast.begin(), runtimes_grid_broadcast.end(), 0.0);
  idg::report("gridding", runtime_gridding);
  idg::report("grid fft", runtime_grid_fft);
  idg::report("degridding", runtime_degridding);
  idg::report("grid reduce", runtime_grid_reduce);
  idg::report("grid broadcast", runtime_grid_broadcast);
  double runtime_imaging =
    runtime_gridding + runtime_degridding + runtime_grid_fft +
    runtime_grid_reduce + runtime_grid_broadcast;
  idg::report("runtime imaging", runtime_imaging);
  std::clog << std::endl;

  // Report throughput
  uint64_t nr_visibilities = 1ULL * nr_baselines * nr_timesteps * nr_channels * world_size;
  idg::report_visibilities("gridding", runtime_gridding, nr_visibilities);
  idg::report_visibilities("degridding", runtime_degridding, nr_visibilities);
  idg::report_visibilities("imaging", runtime_imaging, nr_visibilities);
} // end run_master

void run_worker() {
  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Receive parameters
  unsigned int nr_stations = receive_int();
  unsigned int nr_baselines = receive_int();
  unsigned int nr_timesteps = receive_int();
  unsigned int nr_timeslots = receive_int();
  float integration_time = receive_float();
  unsigned int nr_channels = receive_int();
  unsigned int nr_correlations = receive_int();
  unsigned int nr_w_layers = receive_int();
  float w_offset = receive_float();
  unsigned int grid_size = receive_int();
  unsigned int subgrid_size = receive_int();
  unsigned int kernel_size = receive_int();
  float cell_size = receive_float();
  unsigned int nr_cycles = receive_int();

  // Initialize proxy
  ProxyType proxy;

  // Allocate and initialize static data structures
  idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms =
      idg::get_identity_aterms(proxy, nr_timeslots, nr_stations, subgrid_size,
                               subgrid_size);
  idg::Array1D<unsigned int> aterms_offsets =
      idg::get_example_aterms_offsets(proxy, nr_timeslots, nr_timesteps);
  idg::Array2D<float> spheroidal =
      idg::get_example_spheroidal(proxy, subgrid_size, subgrid_size);
  auto grid =
      proxy.allocate_grid(nr_w_layers, nr_correlations, grid_size, grid_size);
  idg::Array1D<float> shift = idg::get_zero_shift();
  idg::Array1D<std::pair<unsigned int, unsigned int>> baselines =
      idg::get_example_baselines(proxy, nr_stations, nr_baselines);

  // Receive frequencies
  idg::Array1D<float> frequencies = proxy.allocate_array1d<float>(nr_channels);
  receive_array(0, frequencies);

  // Receive data
  idg::Data data =
      idg::get_example_data(nr_baselines, grid_size, integration_time);

  // Plan options
  idg::Plan::Options options = get_plan_options();
  omp_set_nested(true);

  // Buffers for input data
  idg::Array2D<idg::UVW<float>> uvw(nr_baselines, nr_timesteps);
  idg::Array3D<idg::Visibility<std::complex<float>>> visibilities =
      idg::get_dummy_visibilities(nr_baselines, nr_timesteps, nr_channels);
  int time_offset = 0;
  int bl_offset = 0;

  // Set grid
  proxy.set_grid(grid);

  // Iterate all cycles
  for (unsigned cycle = 0; cycle < nr_cycles; cycle++) {

    // Get UVW coordinates for current cycle
    data.get_uvw(uvw, bl_offset, time_offset, integration_time);

    // Create plan
    auto plan = std::unique_ptr<idg::Plan>(new idg::Plan(
       kernel_size, subgrid_size, grid_size, cell_size, frequencies, uvw,
       baselines, aterms_offsets, options));

    // Run gridding
    proxy.gridding(*plan, w_offset, shift, cell_size, kernel_size,
        subgrid_size, frequencies, visibilities, uvw,
        baselines, aterms, aterms_offsets, spheroidal);
    synchronize();

    // Get grid
    grid = proxy.get_grid();

    // Run FFT
    proxy.transform(idg::FourierDomainToImageDomain, *grid);

    // Reduce grids
    reduce_grids(grid, rank, world_size);

    // Master performs deconvolution and constructs model image

    // Receive model image from master
    broadcast_grid(grid, 0);

    // Set grid
    proxy.set_grid(grid);

    // Run gridding #2 (create model image)
    proxy.gridding(*plan, w_offset, shift, cell_size, kernel_size,
        subgrid_size, frequencies, visibilities, uvw,
        baselines, aterms, aterms_offsets, spheroidal);

    // Run FFT
    proxy.transform(idg::ImageDomainToFourierDomain, *grid);

    // Run degridding
    proxy.degridding(*plan, w_offset, shift, cell_size, kernel_size,
        subgrid_size, frequencies, visibilities, uvw,
        baselines, aterms, aterms_offsets, spheroidal);
    synchronize();

    // Subtract model visibilities
    // not implemented

    // Go the the next batch of timesteps
    time_offset += nr_timesteps;
  }
} // end run_worker


void run() {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::thread master_thread, worker_thread;
  if (rank == 0) {
    print(rank, ">>> Running master");
    run_master();
  } else {
    print(rank, ">>> Running worker");
    run_worker();
  }

  print(rank, ">>> Finalize");

  // Finalize the MPI environment.
  MPI_Finalize();
}
