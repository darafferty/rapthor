#define BOOST_TEST_MODULE

#include <boost/test/unit_test.hpp>
#include <cudawrappers/cu.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xrandom.hpp>

#include "CPU/Optimized/OptimizedKernels.h"
#include "CUDA/common/InstanceCUDA.h"

#include "common.h"

BOOST_AUTO_TEST_SUITE(cuda_kernel_fft)
BOOST_AUTO_TEST_CASE(subgrid_fft) {
  const unsigned int kNrPolarizations = 4;
  const unsigned int kGridSize = 2048;
  const unsigned int kSubgridSize = 32;
  const unsigned int kNrSubgrids =
      idg::kernel::cuda::InstanceCUDA::kFftBatch * 1.2;

  std::cout << ">>> Initialize CUDA" << std::endl;
  cu::init();
  idg::kernel::cuda::InstanceCUDA cuda;

  const std::array<size_t, 4> subgrids_shape{kNrSubgrids, kNrPolarizations,
                                             kSubgridSize, kSubgridSize};
  auto subgrids_ref = xt::xtensor<std::complex<float>, 4>(subgrids_shape);
  auto subgrids = xt::xtensor<std::complex<float>, 4>(subgrids_shape);

  for (idg::DomainAtoDomainB sign :
       {idg::FourierDomainToImageDomain, idg::ImageDomainToFourierDomain}) {
    std::cout << ">>> Testing "
              << (sign == idg::FourierDomainToImageDomain ? "inverse"
                                                          : "forward")
              << " FFT" << std::endl;

    // Set input
    xt::real(subgrids_ref) =
        xt::random::rand<float>(subgrids_shape, 0.0f, 0.1f);
    xt::imag(subgrids_ref) =
        xt::random::rand<float>(subgrids_shape, 0.0f, 0.1f);
    subgrids.fill(0);

    std::cout << ">>> Allocate device memory" << std::endl;
    const size_t sizeof_subgrids =
        subgrids_ref.size() * sizeof(std::complex<float>);
    cu::DeviceMemory d_subgrids(sizeof_subgrids, CU_MEMORYTYPE_DEVICE);

    std::cout << ">>> Copy host memory to device" << std::endl;
    cu::Stream stream_htod = cuda.get_htod_stream();
    stream_htod.memcpyHtoDAsync(d_subgrids, subgrids_ref.data(),
                                sizeof_subgrids);
    stream_htod.synchronize();

    std::cout << ">>> Launch batched fft kernel" << std::endl;
    std::unique_ptr<idg::kernel::cuda::KernelFFT> fft_kernel =
        cuda.plan_batched_fft(kSubgridSize, kNrSubgrids * kNrPolarizations);
    cuda.launch_batched_fft(*fft_kernel, d_subgrids,
                            kNrSubgrids * kNrPolarizations, sign);

    if (sign == idg::ImageDomainToFourierDomain) {
      std::cout << ">>> Launch scaler kernel" << std::endl;
      cuda.launch_scaler(kNrSubgrids, kNrPolarizations, kSubgridSize,
                         d_subgrids);
    }
    cuda.get_execute_stream().synchronize();

    std::cout << ">>> Copy subgrids to host" << std::endl;
    cu::Stream stream_dtoh = cuda.get_dtoh_stream();
    stream_dtoh.memcpyDtoHAsync(subgrids.data(), d_subgrids, sizeof_subgrids);
    stream_dtoh.synchronize();

#if 0
    // Equivalent to applying the scaler kernel on the device
    if (sign == idg::ImageDomainToFourierDomain) {
      std::cout << ">>> Scale subgrids" << std::endl;
      float scale = 1.0 / (kSubgridSize * kSubgridSize);
      subgrids *= scale;
    }
#endif

    // Run reference
    std::cout << ">>> Run CPU subgrid fft" << std::endl;
    idg::kernel::cpu::OptimizedKernels cpu;
    cpu.run_subgrid_fft(kGridSize, kSubgridSize, kNrSubgrids * kNrPolarizations,
                        subgrids_ref.data(), sign);

    std::cout << ">>> Compare subgrids" << std::endl;
    xt::xtensor<std::complex<float>, 1> a = xt::flatten(subgrids);
    xt::xtensor<std::complex<float>, 1> b = xt::flatten(subgrids_ref);
    check_close(a, b, 1e-2);
  }
}

BOOST_AUTO_TEST_CASE(grid_fft) {
  const unsigned int kNrPolarizations = 4;
  const unsigned int kGridSize = 2048;

  std::cout << ">>> Initialize CUDA" << std::endl;
  cu::init();
  idg::kernel::cuda::InstanceCUDA cuda;

  const std::array<size_t, 3> grid_shape{kNrPolarizations, kGridSize,
                                         kGridSize};
  auto grid_ref = xt::xtensor<std::complex<float>, 3>(grid_shape);
  auto grid = xt::xtensor<std::complex<float>, 3>(grid_shape);

  std::vector<idg::DomainAtoDomainB> signs = {idg::FourierDomainToImageDomain,
                                              idg::ImageDomainToFourierDomain};
  for (idg::DomainAtoDomainB sign : signs) {
    std::cout << ">>> Testing "
              << (sign == idg::FourierDomainToImageDomain ? "inverse"
                                                          : "forward")
              << " FFT" << std::endl;

    // Set input
    xt::real(grid_ref) = xt::random::rand<float>(grid_shape, 0.0f, 1.0f);
    xt::imag(grid_ref) = xt::random::rand<float>(grid_shape, 0.0f, 1.0f);
    grid.fill(0);

    std::cout << ">>> Allocate device memory" << std::endl;
    const size_t sizeof_grid = grid_ref.size() * sizeof(*grid_ref.data());
    cu::DeviceMemory d_grid(sizeof_grid, CU_MEMORYTYPE_DEVICE);

    std::cout << ">>> Copy host memory to device" << std::endl;
    cu::Stream stream_htod = cuda.get_htod_stream();
    stream_htod.memcpyHtoDAsync(d_grid, grid_ref.data(), sizeof_grid);
    stream_htod.synchronize();

    std::cout << ">>> Launch grid fft kernel" << std::endl;
    cuda.launch_grid_fft(d_grid, kNrPolarizations, kGridSize, sign);
    cuda.get_execute_stream().synchronize();

    std::cout << ">>> Copy grid to host" << std::endl;
    cu::Stream stream_dtoh = cuda.get_dtoh_stream();
    stream_dtoh.memcpyDtoHAsync(grid.data(), d_grid, sizeof_grid);
    stream_dtoh.synchronize();

    if (sign == idg::ImageDomainToFourierDomain) {
      std::cout << ">>> Scale grid" << std::endl;
      const std::complex<float> scale{2.0f / (kGridSize * kGridSize), 0.0f};
      xt::real(grid) *= scale.real();
      xt::imag(grid) *= scale.imag();
    }

    // Run reference
    std::cout << ">>> Run CPU grid fft" << std::endl;
    idg::kernel::cpu::OptimizedKernels cpu;
    cpu.run_fft(kGridSize, kGridSize, kNrPolarizations, grid_ref.data(), sign);

    std::cout << ">>> Compare grids" << std::endl;
    xt::xtensor<std::complex<float>, 1> a = xt::flatten(grid);
    xt::xtensor<std::complex<float>, 1> b = xt::flatten(grid_ref);
    check_close(a, b, 1e-3);
  }
}
BOOST_AUTO_TEST_SUITE_END()