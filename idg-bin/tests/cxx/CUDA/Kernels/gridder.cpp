#define BOOST_TEST_MODULE

#include <boost/test/unit_test.hpp>
#include <cudawrappers/cu.hpp>
#include <xtensor/xcomplex.hpp>

#include "CPU/Optimized/OptimizedKernels.h"
#include "CUDA/common/InstanceCUDA.h"

#include "idg-util.h"
#include "common.h"
#include "TestFixture.h"

BOOST_AUTO_TEST_CASE(cuda_kernel_gridder) {
  const unsigned int kNrSubgrids = kNrBaselines;
  const unsigned int kTimeOffset = 0;
  const float kWStepInLambda = 0;

  TestFixture fixture;
  float image_size = fixture.GetImageSize();
  float cell_size = fixture.GetCellSize();
  std::array<float, 2> shift = fixture.GetShift();
  auto frequencies = fixture.GetFrequencies();
  auto uvw = fixture.GetUVW();
  auto baselines = fixture.GetBaselines();
  auto aterm_offsets = fixture.GetAtermOffsets();
  auto wavenumbers = fixture.GetWavenumbers();
  auto visibilities = fixture.GetVisibilities();
  auto taper = fixture.GetTaper();
  auto aterms = fixture.GetAterms();

  idg::Plan plan(kKernelSize, kSubgridSize, kGridSize, cell_size, shift,
                 frequencies, uvw, baselines, aterm_offsets);

  const std::complex<float>* avg_aterm_ptr = nullptr;

  const unsigned int* aterm_indices_ptr = plan.get_aterm_indices_ptr();

  const idg::Metadata* metadata_ptr = plan.get_metadata_ptr();

  const std::array<size_t, 4> subgrids_shape{kNrSubgrids, kNrCorrelations,
                                             kSubgridSize, kSubgridSize};
  auto subgrids = xt::xtensor<std::complex<float>, 4>(subgrids_shape);
  subgrids.fill(0);

  std::cout << ">>> Initialize CUDA" << std::endl;
  cu::init();
  idg::kernel::cuda::InstanceCUDA cuda;

  std::cout << ">>> Allocate device memory" << std::endl;
  const size_t sizeof_uvw = uvw.size() * sizeof(*uvw.data());
  cu::DeviceMemory d_uvw(sizeof_uvw, CU_MEMORYTYPE_DEVICE);

  const size_t sizeof_wavenumbers =
      wavenumbers.size() * sizeof(*wavenumbers.data());
  cu::DeviceMemory d_wavenumbers(sizeof_wavenumbers, CU_MEMORYTYPE_DEVICE);

  const size_t sizeof_visibilities =
      visibilities.size() * sizeof(*visibilities.data());
  cu::DeviceMemory d_visibilities(sizeof_visibilities, CU_MEMORYTYPE_DEVICE);

  const size_t sizeof_taper = taper.size() * sizeof(float);
  cu::DeviceMemory d_taper(sizeof_taper, CU_MEMORYTYPE_DEVICE);

  const size_t sizeof_aterms = aterms.size() * sizeof(*aterms.data());
  cu::DeviceMemory d_aterms(sizeof_aterms, CU_MEMORYTYPE_DEVICE);

  const size_t sizeof_aterm_indices = kNrBaselines * kNrTimesteps * sizeof(int);
  cu::DeviceMemory d_aterm_indices(sizeof_aterm_indices, CU_MEMORYTYPE_DEVICE);

  const size_t sizeof_metadata = kNrSubgrids * sizeof(*metadata_ptr);
  cu::DeviceMemory d_metadata(sizeof_metadata, CU_MEMORYTYPE_DEVICE);

  cu::DeviceMemory d_avg_aterm(0, CU_MEMORYTYPE_DEVICE);

  const size_t sizeof_subgrids = subgrids.size() * sizeof(*subgrids.data());
  cu::DeviceMemory d_subgrids(sizeof_subgrids, CU_MEMORYTYPE_DEVICE);

  std::cout << ">>> Copy host memory to device" << std::endl;
  cu::Stream stream_htod = cuda.get_htod_stream();
  stream_htod.memcpyHtoDAsync(d_uvw, uvw.data(), sizeof_uvw);
  stream_htod.memcpyHtoDAsync(d_wavenumbers, wavenumbers.data(),
                              sizeof_wavenumbers);
  stream_htod.memcpyHtoDAsync(d_visibilities, visibilities.data(),
                              sizeof_visibilities);
  stream_htod.memcpyHtoDAsync(d_taper, taper.data(), sizeof_taper);
  stream_htod.memcpyHtoDAsync(d_aterms, aterms.data(), sizeof_aterms);
  stream_htod.memcpyHtoDAsync(d_aterm_indices, aterm_indices_ptr,
                              sizeof_aterm_indices);
  stream_htod.memcpyHtoDAsync(d_metadata, metadata_ptr, sizeof_metadata);
  stream_htod.synchronize();

  std::cout << ">>> Launch gridder kernel" << std::endl;
  cuda.launch_gridder(kTimeOffset, kNrSubgrids, kNrPolarizations, kGridSize,
                      kSubgridSize, image_size, kWStepInLambda, kNrChannels,
                      kNrStations, shift[0], shift[1], d_uvw, d_wavenumbers,
                      d_visibilities, d_taper, d_aterms, d_aterm_indices,
                      d_metadata, d_avg_aterm, d_subgrids);
  cuda.get_execute_stream().synchronize();

  std::cout << ">>> Copy subgrids to host" << std::endl;
  cu::Stream stream_dtoh = cuda.get_dtoh_stream();
  stream_dtoh.memcpyDtoHAsync(subgrids.data(), d_subgrids, sizeof_subgrids);
  stream_dtoh.synchronize();

  // Run reference
  std::cout << ">>> Run CPU gridding" << std::endl;
  idg::kernel::cpu::OptimizedKernels cpu;
  auto subgrids_ref = xt::xtensor<std::complex<float>, 4>(subgrids_shape);
  subgrids_ref.fill(0);
  cpu.run_gridder(
      kNrSubgrids, kNrPolarizations, kGridSize, kSubgridSize, image_size,
      kWStepInLambda, shift.data(), kNrChannels, kNrCorrelations, kNrStations,
      uvw.data(), wavenumbers.data(), visibilities.data(), taper.data(),
      reinterpret_cast<std::complex<float>*>(aterms.data()), aterm_indices_ptr,
      avg_aterm_ptr, metadata_ptr, subgrids_ref.data());

  std::cout << ">>> Compare subgrids" << std::endl;
  xt::xtensor<std::complex<float>, 1> a = xt::flatten(subgrids);
  xt::xtensor<std::complex<float>, 1> b = xt::flatten(subgrids_ref);
  check_close(a, b, 1e-3);
}