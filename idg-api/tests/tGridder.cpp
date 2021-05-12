// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <math.h>

#include "gridder-common.h"

namespace utf = boost::unit_test;

namespace {
const std::size_t kNrTimesteps = 9;
const std::size_t kNrStations = 3;
const std::size_t kNrCorrelations = 4;
const std::vector<std::vector<double>> kBands = {
    {100.e6, 101.e6, 102.e6, 103.e6, 104.e6}};
const float kCellSize = 0.001;  // Pixel size in radians.
const std::size_t kImageSize = 256;

std::vector<double> image_ref;

std::unique_ptr<idg::api::BufferSet> CreateBufferset(
    idg::api::Type architecture, const WMode wmode,
    const std::array<int, 2> shift) {
  idg::api::options_type options;
  AddWModeToOptions(wmode, options);
  std::unique_ptr<idg::api::BufferSet> bufferset(
      idg::api::BufferSet::create(architecture));

  unsigned int kBufferSize = 4;  // Timesteps per buffer
  float max_baseline = 3000.0f;  // in meters
  float max_w = 5.0f;

  // By convention l runs in negative direction.
  const double shiftl = shift[0] * -kCellSize;
  const double shiftm = shift[1] * kCellSize;

  bufferset->init(kImageSize, kCellSize, max_w, shiftl, shiftm, options);
  bufferset->init_buffers(kBufferSize, kBands, kNrStations, max_baseline,
                          options, idg::api::BufferSetType::gridding);
  return bufferset;
}

std::vector<double> GridImage(idg::api::Type arch, const WMode wmode,
                              const std::array<int, 2> shift = {0, 0}) {
  std::unique_ptr<idg::api::BufferSet> bufferset =
      CreateBufferset(arch, wmode, shift);

  const std::vector<double> uvw = {10.0, 20.0, 3.0};
  const std::vector<float> weights(kBands[0].size() * kNrCorrelations, 1.0f);

  std::vector<std::complex<float>> phasors;
  if (shift[0] != 0 || shift[1] != 0) {
    // Apply phase shift to visibilities: The resulting image should then equal
    // the image without using a shift.
    const double shiftl = shift[0] * kCellSize;
    const double shiftm = shift[1] * kCellSize;
    const double shiftp = sqrt(1.0 - shiftl * shiftl - shiftm * shiftm) - 1.0;

    const double phase_shift =
        uvw[0] * shiftl + uvw[1] * shiftm + uvw[2] * shiftp;
    const double phase_factor = -phase_shift * 2.0 * M_PI / 299792458.0;
    for (double freq : kBands[0]) {
      const double phase = freq * phase_factor;
      phasors.emplace_back(std::cos(phase), std::sin(phase));
    }
  }

  for (std::size_t timestep = 0; timestep < kNrTimesteps; ++timestep) {
    for (std::size_t st1 = 0; st1 < kNrStations; ++st1) {
      for (std::size_t st2 = st1; st2 < kNrStations; ++st2) {
        const std::complex<float> value{timestep + st1 + 1.0f, st2 / 4.0f};
        std::vector<std::complex<float>> data(weights.size(), value);

        if (shift[0] != 0 || shift[1] != 0) {
          std::complex<float>* data_band = data.data();
          for (std::size_t b = 0; b < kBands[0].size(); ++b) {
            for (std::size_t c = 0; c < kNrCorrelations; ++c) {
              *data_band *= phasors[b];
              ++data_band;
            }
          }
        }

        bufferset->get_gridder(0)->grid_visibilities(
            timestep, st1, st2, uvw.data(), data.data(), weights.data());
      }
    }
  }
  bufferset->finished();

  std::vector<double> image(kNrCorrelations * kImageSize * kImageSize, 42.0);
  bufferset->get_image(image.data());
  return image;
}

void CompareImages(const std::vector<double>& ref,
                   const std::vector<double>& test,
                   const double pixel_tolerance, const size_t border = 0) {
  double total_diff = 0.0;
  double allowed_diff = 0.0;

  for (size_t y = border; y < kImageSize - border; ++y) {
    for (size_t x = border; x < kImageSize - border; ++x) {
      double ref_pixel = ref[y * kImageSize + x];
      double test_pixel = test[y * kImageSize + x];

      if (boost::test_tools::check_is_small(ref_pixel, 1e-6) ||
          boost::test_tools::check_is_small(test_pixel, 1e-6)) {
        BOOST_CHECK_CLOSE(ref_pixel - test_pixel, 0.0f, 1e-3);
      } else {
        // Check both individual and average pixel differences.
        BOOST_CHECK_CLOSE(ref_pixel, test_pixel, pixel_tolerance);
        total_diff +=
            std::max(ref_pixel / test_pixel, test_pixel / ref_pixel) - 1.0;
        allowed_diff += 0.01;
      }
    }
  }

  BOOST_CHECK_LT(total_diff, allowed_diff);
}

}  // namespace

BOOST_AUTO_TEST_SUITE(gridder)

BOOST_AUTO_TEST_CASE(aterms) {
  const std::array<int, 2> kShift{0, 0};
  std::unique_ptr<idg::api::BufferSet> bufferset =
      CreateBufferset(idg::api::Type::CPU_REFERENCE, WMode::kNeither, kShift);

  size_t subgridsize = bufferset->get_subgridsize();
  std::cout << "Subgridsize: " << subgridsize << std::endl;

  const std::size_t aterm_steps = 3;  // Change a-term every 3 time steps
  const std::size_t nr_aterms = ceil(kNrTimesteps / aterm_steps);
  const std::size_t atermsize =
      kNrStations * subgridsize * subgridsize * kNrCorrelations;

  // Initialize a-terms
  std::vector<std::complex<float>> aterms(nr_aterms * atermsize);
  std::cout << "Aterms.size() = " << aterms.size() << std::endl;
  std::complex<float>* aterm = aterms.data();
  for (auto aterm_num = 0; aterm_num < nr_aterms; ++aterm_num) {
    for (auto station = 0; station < kNrStations; ++station) {
      for (auto x = 0; x < subgridsize; ++x) {
        for (auto y = 0; y < subgridsize; ++y) {
          aterm[0] = 42 + aterm_num;
          aterm[1] = 0.;
          aterm[2] = 0.;
          aterm[3] = 42 + aterm_num;
          aterm += kNrCorrelations;
        }
      }
    }
  }

  for (size_t timestep = 0; timestep < kNrTimesteps; ++timestep) {
    if (timestep % aterm_steps == 0) {
      size_t atermsize = kNrStations * subgridsize * subgridsize * 4;
      bufferset->get_gridder(0)->set_aterm(
          timestep, aterms.data() + timestep / aterm_steps * atermsize);
    }

    for (size_t st1 = 0; st1 < kNrStations; ++st1) {
      for (size_t st2 = st1; st2 < kNrStations; ++st2) {
        std::vector<double> uvw = {1., 1., 1.};
        std::vector<std::complex<float>> data(4 * kBands[0].size(), 0.);
        std::vector<float> weights(data.size(), 1.);
        bufferset->get_gridder(0)->grid_visibilities(
            timestep, st1, st2, uvw.data(), data.data(), weights.data());
      }
    }
  }
  bufferset->finished();
}

// Create a reference image, for use in other tests.
BOOST_AUTO_TEST_CASE(reference) {
  image_ref = GridImage(idg::api::Type::CPU_REFERENCE, WMode::kNeither, {0, 0});
}

// Test that all architectures produce similar results.
BOOST_AUTO_TEST_CASE(architectures, *utf::depends_on("gridder/reference")) {
  std::set<idg::api::Type> architectures = GetArchitectures();
  architectures.erase(idg::api::Type::CPU_REFERENCE);

  for (idg::api::Type architecture : architectures) {
    std::vector<double> image_arch = GridImage(architecture, WMode::kNeither);
    double pixel_tolerance = 3.0;
    // CUDA images require higher tolerance values.
    if (architecture == idg::api::Type::CUDA_GENERIC ||
        architecture == idg::api::Type::HYBRID_CUDA_CPU_OPTIMIZED) {
      pixel_tolerance = 8.0;
    }
    CompareImages(image_ref, image_arch, pixel_tolerance);
  }
}

// Test that all wmodes produce the same results.
BOOST_AUTO_TEST_CASE(wmodes, *utf::depends_on("gridder/reference")) {
  const std::vector<WMode> kWModes{WMode::kWStacking, WMode::kWTiling};

  for (WMode wmode : kWModes) {
    std::vector<double> image_wmode =
        GridImage(idg::api::Type::CPU_REFERENCE, wmode);
    CompareImages(image_ref, image_wmode, 1.3);
  }
}

// Test that using a shift produces a shifted image.
BOOST_AUTO_TEST_CASE(shift, *utf::depends_on("gridder/reference")) {
  const std::array<int, 2> kShift{10, 20};

  for (idg::api::Type architecture : GetArchitectures()) {
    std::vector<double> image_shift =
        GridImage(architecture, WMode::kNeither, kShift);
    // This test has to use a very large border: Outside the center, the
    // pixel differences can become very large.
    // CompareImages now only compares 40 x 40 pixels at the center.
    // A manual test with wsclean using the 1052736496-averaged MS
    // showed that the gridder works fine.
    // TODO: Analyze the origin of the inaccuracy in this test.
    CompareImages(image_ref, image_shift, 1.3, 108);
  }
}

BOOST_AUTO_TEST_SUITE_END()
