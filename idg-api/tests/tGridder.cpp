// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <math.h>

#include "gridder-common.h"

namespace utf = boost::unit_test;

namespace {
const double kPixelTolerance = 1.0e-3;
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
    const std::array<int, 2> shift, const bool stokes_i_only = false) {
  idg::api::options_type options;
  AddWModeToOptions(wmode, options);
  options["stokes_I_only"] = stokes_i_only;
  std::unique_ptr<idg::api::BufferSet> bufferset(
      idg::api::BufferSet::create(architecture));

  unsigned int kBufferSize = 4;  // Timesteps per buffer
  float max_baseline = 3000.0f;  // in meters
  float max_w = 100.0f;

  // By convention l runs in negative direction.
  const double shiftl = shift[0] * -kCellSize;
  const double shiftm = shift[1] * kCellSize;

  bufferset->init(kImageSize, kCellSize, max_w, shiftl, shiftm, options);
  bufferset->init_buffers(kBufferSize, kBands, kNrStations, max_baseline,
                          options, idg::api::BufferSetType::gridding);
  return bufferset;
}

std::vector<double> GridImage(idg::api::Type arch, const WMode wmode,
                              const std::array<int, 2> shift = {0, 0},
                              const bool stokes_i_only = false) {
  std::unique_ptr<idg::api::BufferSet> bufferset =
      CreateBufferset(arch, wmode, shift, stokes_i_only);

  const std::vector<double> uvw = {100.0, 200.0, 30.0};
  const std::vector<float> weights(kBands[0].size() * kNrCorrelations, 1.0f);

  for (std::size_t timestep = 0; timestep < kNrTimesteps; ++timestep) {
    for (std::size_t st1 = 0; st1 < kNrStations; ++st1) {
      for (std::size_t st2 = st1; st2 < kNrStations; ++st2) {
        const std::complex<float> value{timestep + st1 + 1.0f, st2 / 4.0f};
        std::vector<std::complex<float>> data(weights.size(), value);

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
                   const double pixel_tolerance,
                   const std::array<int, 2> shift = {0, 0}) {
  double total_diff = 0.0;
  double allowed_diff = 0.0;

  size_t x_start = std::max(0, shift[0]);
  size_t x_end = std::min(kImageSize, kImageSize + shift[0]);
  size_t y_start = std::max(0, shift[1]);
  size_t y_end = std::min(kImageSize, kImageSize + shift[1]);

  double peak_value = 0.0;
  for (size_t y = y_start; y < y_end; ++y) {
    for (size_t x = x_start; x < x_end; ++x) {
      peak_value = std::max(peak_value, ref[y * kImageSize + x]);
    }
  }

  double sum_squared_difference = 0.0;
  double max_difference = 0.0;
  for (size_t y = y_start; y < y_end; ++y) {
    for (size_t x = x_start; x < x_end; ++x) {
      double ref_pixel = ref[y * kImageSize + x];
      double test_pixel = test[(y - shift[1]) * kImageSize + (x - shift[0])];
      double normalized_difference =
          std::abs(ref_pixel - test_pixel) / peak_value;
      sum_squared_difference += normalized_difference * normalized_difference;
      max_difference = std::max(max_difference, normalized_difference);
    }
  }

  double rms_difference =
      std::sqrt(sum_squared_difference /
                ((kImageSize - shift[0]) * (kImageSize - shift[1])));
  BOOST_CHECK_SMALL(max_difference, pixel_tolerance);
  BOOST_CHECK_SMALL(rms_difference, pixel_tolerance);
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
    CompareImages(image_ref, image_arch, kPixelTolerance);
  }
}

// Test that in Stokes I only and WTiling mode
// all architectures produce similar results.
BOOST_AUTO_TEST_CASE(stokes_I_only, *utf::depends_on("gridder/reference")) {
  std::set<idg::api::Type> architectures = GetArchitectures();
  architectures.erase(idg::api::Type::CPU_REFERENCE);

  for (idg::api::Type architecture : architectures) {
    std::vector<double> image_arch =
        GridImage(architecture, WMode::kWTiling, {0, 0}, true);
    CompareImages(image_ref, image_arch, kPixelTolerance);
  }
}

// Test that all wmodes produce the same results.
BOOST_AUTO_TEST_CASE(wmodes, *utf::depends_on("gridder/reference")) {
  const std::vector<WMode> kWModes{WMode::kWStacking, WMode::kWTiling};

  for (WMode wmode : kWModes) {
    std::vector<double> image_wmode =
        GridImage(idg::api::Type::CPU_REFERENCE, wmode);
    CompareImages(image_ref, image_wmode, kPixelTolerance);
  }
}

// Test that using a shift produces a shifted image.
BOOST_AUTO_TEST_CASE(shift, *utf::depends_on("gridder/reference")) {
  const std::array<int, 2> kShift{10, 20};

  for (idg::api::Type architecture : GetArchitectures()) {
    std::vector<double> image_shift =
        GridImage(architecture, WMode::kNeither, kShift);
    CompareImages(image_ref, image_shift, kPixelTolerance, kShift);
  }
}

BOOST_AUTO_TEST_SUITE_END()
