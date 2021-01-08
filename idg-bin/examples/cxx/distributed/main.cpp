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

#include "idg-cpu.h"
#include "idg-cuda.h"
#include "idg-util.h"  // Data init routines

using namespace std;

// Option to let the master distribute input data (uvw coordinates
// and visibilities) to all workers. If set 0, the workers will
// initialize their own data, taking their baseline offset into account.
#define DISTRIBUTE_INPUT 0

// using ProxyType = idg::proxy::cuda::Generic;
using ProxyType = idg::proxy::cpu::Optimized;

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

void run_master(int argc, char *argv[]) {
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

  // Determine number of baselines per worker
  unsigned int nr_baselines_per_worker = nr_baselines / world_size;
  unsigned int nr_baselines_all_workers = (world_size - 1) * nr_baselines_per_worker;
  unsigned int nr_baselines_master = nr_baselines - nr_baselines_all_workers;

  // Distribute parameters
  for (int dst = 0; dst < world_size; dst++) {
    send_int(dst, nr_stations);
    send_int(dst, nr_baselines);
    send_int(dst, nr_baselines_per_worker);
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

  // Initialize frequency data
  idg::Array1D<float> frequencies(nr_channels);
  data.get_frequencies(frequencies, image_size);

  // Distribute frequencies
  for (int dst = 1; dst < world_size; dst++) {
    send_array(dst, frequencies);
  }

  // Distribute data
  #if !DISTRIBUTE_INPUT
  for (int dst = 1; dst < world_size; dst++) {
    send_bytes(dst, &data, sizeof(data));
  }
  #endif

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
  idg::Array1D<std::pair<unsigned int, unsigned int>> baselines_all =
      idg::get_example_baselines(proxy, nr_stations, nr_baselines);

  // Plan options
  idg::Plan::Options options = get_plan_options();
  omp_set_nested(true);

  // Input buffers for all workers
  #if DISTRIBUTE_INPUT
  idg::Array2D<idg::UVW<float>> uvw_all(nr_baselines, nr_timesteps);
  idg::Array3D<idg::Visibility<std::complex<float>>> visibilities_all =
      idg::get_dummy_visibilities(nr_baselines, nr_timesteps, nr_channels);
  #else
  idg::Array2D<idg::UVW<float>> uvw(nr_baselines_master, nr_timesteps);
  idg::Array3D<idg::Visibility<std::complex<float>>> visibilities =
      idg::get_dummy_visibilities(nr_baselines_master, nr_timesteps, nr_channels);
  #endif
  int time_offset = 0;

  // Set grid
  proxy.set_grid(grid);

  // Performance measurement
  double runtime_input = 0;
  double runtime_gridding = 0;
  double bytes_input = 0;

  // Iterate all cycles
  for (unsigned cycle = 0; cycle < nr_cycles; cycle++) {
    #if DISTRIBUTE_INPUT
    // Get UVW coordinates for current cycle
    data.get_uvw(uvw_all, 0, time_offset, integration_time);

    // Distribute input data
    MPIRequestList requests;
    runtime_input -= omp_get_wtime();
    for (unsigned int bl = 0; bl < nr_baselines_all_workers; bl++) {
      unsigned int dest = 1 + (bl / nr_baselines_per_worker);

      // Send visibilities
      void *visibilities_ptr = (void *) visibilities_all.data(bl, 0, 0);
      size_t sizeof_visibilities =
          nr_timesteps * nr_channels *
          sizeof(idg::Visibility<std::complex<float>>);
      requests.create()->send(visibilities_ptr, sizeof_visibilities, dest);

      // Send uvw coordinates
      void *uvw_ptr = (void *) uvw_all.data(bl, 0);
      size_t sizeof_uvw = nr_timesteps * sizeof(idg::UVW<float>);
      requests.create()->send(uvw_ptr, sizeof_uvw, dest);

      // Update bytes_input
      bytes_input += sizeof_visibilities;
      bytes_input += sizeof_uvw;
    }

    requests.wait();
    runtime_input += omp_get_wtime();

    // Get master buffers
    idg::Array2D<idg::UVW<float>> uvw(uvw_all.data(nr_baselines_all_workers, 0), nr_baselines_master, nr_timesteps);
    idg::Array3D<idg::Visibility<std::complex<float>>> visibilities(visibilities_all.data(nr_baselines_all_workers, 0, 0), nr_baselines_master, nr_timesteps, nr_channels);
    #else
    // Get UVW coordinates for current cycle
    data.get_uvw(uvw, nr_baselines_all_workers, time_offset, integration_time);
    #endif

    idg::Array1D<std::pair<unsigned int, unsigned int>> baselines(baselines_all.data(nr_baselines_all_workers), nr_baselines_master);

    // Create plan
    auto plan = std::unique_ptr<idg::Plan>(new idg::Plan(
       kernel_size, subgrid_size, grid_size, cell_size, frequencies, uvw,
       baselines, aterms_offsets, options));

    // Run gridding
    runtime_gridding = -omp_get_wtime();
    proxy.gridding(*plan, w_offset, shift, cell_size, kernel_size,
        subgrid_size, frequencies, visibilities, uvw,
        baselines, aterms, aterms_offsets, spheroidal);
        
    synchronize();
    runtime_gridding += omp_get_wtime();

    // Go the the next batch of timesteps
    time_offset += nr_timesteps;
  }

  // Get grid
  grid = proxy.get_grid();

  // Receive grids from workers and add to master grid
  double runtime_output = 0;
  double runtime_add_grid = -omp_get_wtime();
  #pragma omp parallel for
  for (unsigned int y = 0; y < grid_size; y++)
  {
    idg::Array1D<std::complex<float>> row(grid_size);

    for (unsigned int w = 0; w < nr_w_layers; w++)
    {
      for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++)
      {
        for (unsigned int src = 1; src < (unsigned int) world_size; src++)
        {
          #pragma omp critical
          {
            runtime_output -= omp_get_wtime();
            MPIRequest request(true);
            request.receive(row.data(), row.bytes(), src);
            runtime_output += omp_get_wtime();
          }

          for (unsigned int x = 0; x < grid_size; x++)
          {
            auto& grid_ = *grid;
            grid_(w, pol, y, x) = row(x);
          }
        }
      }
    }
  }
  runtime_add_grid += omp_get_wtime();

  synchronize();

  // Run fft
  double runtime_fft = -omp_get_wtime();
  proxy.transform(idg::FourierDomainToImageDomain, *grid);
  runtime_fft += omp_get_wtime();

  std::cout << std::scientific;

  // Report input
  double input_bw = bytes_input ? bytes_input / runtime_input * 1e-9 : 0;
  std::stringstream report_input;
  report_input << "input: " << runtime_input
               << " s , " << input_bw << " GB/s";
  print(0, report_input.str());

  // Report gridding
  std::stringstream report_gridding;
  report_gridding << "gridding: " << runtime_gridding << "s";
  print(0, report_gridding.str());

  // Report output time
  size_t output_bytes = world_size > 1 ? grid->bytes() * (world_size - 1) : 0;
  double output_bw = output_bytes / runtime_output * 1e-9;
  std::stringstream report_output;
  report_output << "output: " << runtime_output
                << " s, " << output_bw << " GB/s";
  print(0, report_output.str());

  // Report grid addition time
  std::stringstream report_add_grid;
  report_add_grid << "add grid: " << runtime_add_grid << " s";
  print(0, report_add_grid.str());

  // Report fft runtime
  std::stringstream report_fft;
  report_fft << "fft: " << runtime_fft << " s";
  print(0, report_fft.str());

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
  unsigned int total_nr_baselines = receive_int();
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
  print(rank, "initializing proxy");
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
      idg::get_example_data(total_nr_baselines, grid_size, integration_time);

  // Plan options
  idg::Plan::Options options = get_plan_options();
  omp_set_nested(true);

  // Buffers for input data
  idg::Array2D<idg::UVW<float>> uvw(nr_baselines, nr_timesteps);
  #if DISTRIBUTE_INPUT
  idg::Array3D<idg::Visibility<std::complex<float>>> visibilities(
      nr_baselines, nr_timesteps, nr_channels);
  #else
  idg::Array3D<idg::Visibility<std::complex<float>>> visibilities =
      idg::get_dummy_visibilities(nr_baselines, nr_timesteps, nr_channels);
  int time_offset = 0;
  int bl_offset = (rank - 1) * nr_baselines;
  #endif

  // Set grid
  proxy.set_grid(grid);

  // Iterate all cycles
  for (unsigned cycle = 0; cycle < nr_cycles; cycle++) {

    #if DISTRIBUTE_INPUT
    // Receive input data
    MPIRequestList requests;
    for (unsigned bl = 0; bl < nr_baselines; bl++) {
      // Receive visibilities
      void *visibilities_ptr = (void *) visibilities.data(bl, 0, 0);
      size_t sizeof_visibilities = nr_timesteps * nr_channels *
                                   sizeof(idg::Visibility<std::complex<float>>);
      requests.create()->receive(visibilities_ptr, sizeof_visibilities, 0);

      // Receive uvw coordinates
      void *uvw_ptr = (void *) uvw.data(bl, 0);
      size_t sizeof_uvw = nr_timesteps * sizeof(idg::UVW<float>);
      requests.create()->receive(uvw_ptr, sizeof_uvw, 0);
    }
    requests.wait();
    #else
    // Get UVW coordinates for current cycle
    data.get_uvw(uvw, bl_offset, time_offset, integration_time);
    #endif

    // Create plan
    auto plan = std::unique_ptr<idg::Plan>(new idg::Plan(
       kernel_size, subgrid_size, grid_size, cell_size, frequencies, uvw,
       baselines, aterms_offsets, options));

    // Run gridding
    proxy.gridding(*plan, w_offset, shift, cell_size, kernel_size,
        subgrid_size, frequencies, visibilities, uvw,
        baselines, aterms, aterms_offsets, spheroidal);

    synchronize();
  }

  // Get grid
  grid = proxy.get_grid();

  // Send grid to master
  MPIRequestList requests;
  for (unsigned int y = 0; y < grid_size; y++)
  {
    for (unsigned int w = 0; w < nr_w_layers; w++)
    {
      for (unsigned int pol = 0; pol < NR_POLARIZATIONS; pol++)
      {
        std::complex<float> *row_ptr = grid->data(w, pol, y, 0);
        size_t sizeof_row = grid_size * sizeof(std::complex<float>);
        requests.create()->send(row_ptr, sizeof_row, 0);
      }
    }
  }
  requests.wait();

  synchronize();
} // end run_worker

int main(int argc, char *argv[]) {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::thread master_thread, worker_thread;
  if (rank == 0) {
    print(rank, ">>> Running master");
    run_master(argc, argv);
  } else {
    print(rank, ">>> Running worker");
    run_worker();
  }

  print(rank, ">>> Finalize");

  // Finalize the MPI environment.
  MPI_Finalize();
}
