// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <idg-api.h>

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>

namespace {

const std::size_t kNrTimesteps = 9;
const std::size_t kNrStations = 4;
const std::size_t kNrCorrelations = 4;
const std::vector<std::vector<double>> kBands = {
    {100.e6, 101.e6, 102.e6, 103.e6, 104.e6}};
const std::size_t kNrBaselines = (kNrStations + 1) * kNrStations / 2;
const std::size_t kNrRows = kNrTimesteps * kNrBaselines;
const std::size_t kRowSize = kBands.front().size() * kNrCorrelations;
const std::complex<float> kDummyData{42.0, -42.0};
const float kTolerance = 3e-3;  // Maximum difference % between visibilities.
const float kCellSize = 0.001;  // Pixel size in radians.

enum class WMode { kNeither, kWStacking, kWTiling };

std::ostream& operator<<(std::ostream& stream, WMode wmode) {
  switch (wmode) {
    case WMode::kNeither:
      stream << "No W-Tiling/W-Stacking";
      break;
    case WMode::kWStacking:
      stream << "W-Stacking";
      break;
    case WMode::kWTiling:
      stream << "W-Tiling";
      break;
  }
  return stream;
}

std::unique_ptr<idg::api::BufferSet> CreateBufferset(
    idg::api::BufferSetType type, const std::array<int, 2> shift = {0, 0},
    const WMode wmode = WMode::kNeither) {
  idg::api::options_type options;
  std::unique_ptr<idg::api::BufferSet> bufferset(
      idg::api::BufferSet::create(idg::api::Type::CPU_OPTIMIZED));

  unsigned int buffersize = 4000;  // Timesteps per buffer
  float max_baseline = 3000.;      // in meters

  unsigned int imagesize = 256;
  float max_w = 5;

  std::vector<double> image(kNrCorrelations * imagesize * imagesize, 0.0);

  // Create a few very artificial sources.
  for (int x = -6; x <= 6; ++x) {
    for (int y = -4; y <= 4; ++y) {
      std::size_t x1 = 42 + x - shift[0];
      std::size_t y1 = 42 + y - shift[1];
      std::size_t x2 = 142 + x - shift[0];
      std::size_t y2 = 142 + y - shift[1];
      std::size_t x3 = 200 + x - shift[0];
      std::size_t y3 = 200 + y - shift[1];
      for (int c = 0; c < kNrCorrelations; c++) {
        std::size_t c_offset = c * imagesize * imagesize;
        image[c_offset + imagesize * y1 + x1] += 42 - std::abs(x * y) + c;
        image[c_offset + imagesize * y2 + x2] += 16 + 36 - x * x - y * y + c;
        image[c_offset + imagesize * y3 + x3] += 42 - std::abs(x) - std::abs(y);
      }
    }
  }

  // By convention l runs in negative direction.
  const double shift_l = shift[0] * -kCellSize;
  const double shift_m = shift[1] * kCellSize;
  const double shift_p =
      sqrt(1.0 - shift_l * shift_l - shift_m * shift_m) - 1.0;

  switch (wmode) {
    case WMode::kNeither:
      options["disable_wtiling"] = true;
      options["disable_wstacking"] = true;
      break;
    case WMode::kWStacking:
      options["disable_wtiling"] = true;
      break;
    case WMode::kWTiling:
      // Do not disable anything.
      break;
  }

  bufferset->init(imagesize, kCellSize, max_w, shift_l, shift_m, shift_p,
                  options);
  bufferset->init_buffers(buffersize, kBands, kNrStations, max_baseline,
                          options, type);
  bufferset->set_image(image.data());
  return bufferset;
}

std::vector<double> CreateUVW(double u, double v, double w) {
  std::vector<double> uvw;
  uvw.reserve(3 * kNrBaselines);
  for (std::size_t bl = 0; bl < kNrBaselines; ++bl) {
    uvw.push_back(u);
    uvw.push_back(v);
    uvw.push_back(w);
  }
  return uvw;
}

std::pair<std::vector<std::size_t>, std::vector<std::size_t>> CreateAntennas() {
  std::pair<std::vector<std::size_t>, std::vector<std::size_t>> antennas;
  for (size_t st1 = 0; st1 < kNrStations; ++st1) {
    for (size_t st2 = st1; st2 < kNrStations; ++st2) {
      antennas.first.push_back(st1);
      antennas.second.push_back(st2);
    }
  }
  BOOST_REQUIRE_EQUAL(antennas.first.size(), kNrBaselines);
  return antennas;
}

void CompareResults(std::vector<std::complex<float>*>& result1,
                    std::vector<std::complex<float>*>& result2) {
  BOOST_REQUIRE_EQUAL(result1.size(), kNrRows);
  BOOST_REQUIRE_EQUAL(result2.size(), kNrRows);

  for (std::size_t row = 0; row < kNrRows; ++row) {
    const std::complex<float>* data1 = result1[row];
    const std::complex<float>* data2 = result2[row];
    BOOST_REQUIRE(data1);
    BOOST_REQUIRE(data2);
    while (data1 != result1[row] + kRowSize) {
      BOOST_CHECK_CLOSE(std::abs(*data1), std::abs(*data2), kTolerance);
      BOOST_CHECK_SMALL(std::arg(*data1) - std::arg(*data2), kTolerance);
      ++data1;
      ++data2;
    }
  }
}

void CompareResults(const std::complex<float>* ref,
                    const std::complex<float>* custom,
                    const std::size_t time_steps = kNrTimesteps,
                    const std::complex<float> aterm = {1.0}) {
  // IDG applies the aterm for two stations, and uses the complex conjugate
  // for one station.
  const std::complex<float> aterm_factor = aterm * std::conj(aterm);

  for (std::size_t t = 0; t < time_steps; ++t) {
    for (std::size_t st1 = 0; st1 < kNrStations; ++st1) {
      // Verify auto-correlations, where the aterm does not apply.
      for (std::size_t i = 0; i < kRowSize; ++i, ++ref, ++custom) {
        BOOST_CHECK_EQUAL(*ref, kDummyData);
        BOOST_CHECK_EQUAL(*custom, kDummyData);
      }

      // Verify normal correlations, where the aterm does apply.
      for (std::size_t st2 = st1 + 1; st2 < kNrStations; ++st2) {
        for (std::size_t i = 0; i < kRowSize; ++i, ++ref, ++custom) {
          std::complex<float> ref_value = *ref * aterm_factor;
          BOOST_CHECK_CLOSE(std::abs(ref_value), std::abs(*custom), kTolerance);
          BOOST_CHECK_SMALL(std::arg(ref_value) - std::arg(*custom),
                            kTolerance);
        }
      }
    }
  }
}

/**
 * Apply phase shift angle to a result.
 * @param data Pointer to result visibilities ( kNrRows * kRowSize values ).
 * @param uvw Pointer to uvw values. This function uses the first set
 *        of uvw values for all baselines.
 * @param shift Phase shift, in pixels.
 */
void ApplyPhaseShift(std::complex<float>* data, const double* uvw,
                     const std::array<int, 2>& shift) {
  // By convention l runs in negative direction.
  const double shift_l = shift[0] * -kCellSize;
  const double shift_m = shift[1] * kCellSize;
  const double shift_p =
      sqrt(1.0 - shift_l * shift_l - shift_m * shift_m) - 1.0;

  // Sign flip of uvw because subgrid coordinates are flipped.
  // Additional sign flip of u because subgrid l runs in positive direction
  const double uvw_factors[3] = {1.0, -1.0, -1.0};

  // The angle is: 2*pi * (u*dl + v*dm + w*dp) * (frequency / speed_of_light)
  // phase_factor contains all parts except the frequency.
  const double phase_factor =
      (uvw[0] * uvw_factors[0] * shift_l + uvw[1] * uvw_factors[1] * shift_m +
       uvw[2] * uvw_factors[2] * shift_p) *
      (2.0 * M_PI / 299792458.0);
  for (std::size_t t = 0; t < kNrTimesteps; ++t) {
    for (std::size_t st1 = 0; st1 < kNrStations; ++st1) {
      data += kRowSize;  // Do not touch auto-correlations.
      for (std::size_t st2 = st1 + 1; st2 < kNrStations; ++st2) {
        for (const double& frequency : kBands.front()) {
          const double angle = frequency * phase_factor;
          const std::complex<float> phasor(cos(angle), sin(angle));
          for (std::size_t corr = 0; corr < kNrCorrelations; ++corr) {
            *data *= phasor;
            ++data;
          }
        }
      }
    }
  }
}

}  // namespace

BOOST_AUTO_TEST_SUITE(degridder)

BOOST_AUTO_TEST_CASE(strategies) {
  // Use different strategies for predicting visibilities, and compare results:
  // 1) req: Do a compute() call after each request_visibilities() call.
  // 2) multi: Do one compute() after multiple request_visibilities() calls.
  // 3) comp: Do a single compute_visibilities() call, with default uvw_factors.

  std::unique_ptr<idg::api::BufferSet> bs_req =
      CreateBufferset(idg::api::BufferSetType::kDegridding);
  std::unique_ptr<idg::api::BufferSet> bs_multi =
      CreateBufferset(idg::api::BufferSetType::kDegridding);
  std::unique_ptr<idg::api::BufferSet> bs_comp =
      CreateBufferset(idg::api::BufferSetType::kBulkDegridding);

  idg::api::DegridderBuffer* dg_req = bs_req->get_degridder(0);
  idg::api::DegridderBuffer* dg_multi = bs_multi->get_degridder(0);
  const idg::api::BulkDegridder* dg_comp = bs_comp->get_bulk_degridder(0);

  std::vector<std::complex<float>> dummy_row(kRowSize, kDummyData);

  // Pointers to result visibilities. Each baseline has one pointer.
  // For _req and _multi, they will point to the internal IDG buffers that
  // compute() returs. For autocorrelations, they will point to dummy values.
  // For _comp and _uvwf, they will point to *_data buffers in this test that
  // are initialized to dummy values. For autocorrelations, IDG should not
  // touch the dummy values, so those rows also have dummy values in the end.
  std::vector<std::complex<float>*> result_req;
  std::vector<std::complex<float>*> result_multi;
  std::vector<std::complex<float>*> result_comp;

  const std::vector<double> uvw = CreateUVW(1.0, 2.0, 3.0);
  const std::vector<const double*> uvws(kNrTimesteps, uvw.data());

  // Note that auto-correlations are included in the baselines.
  const auto antennas = CreateAntennas();

  // Predict visibilities using request_visibilities() and compute().
  const std::size_t kFirstRow = 42;
  std::size_t row_id = kFirstRow;

  for (size_t t = 0; t < kNrTimesteps; ++t) {
    for (size_t st1 = 0; st1 < kNrStations; ++st1) {
      for (size_t st2 = st1; st2 < kNrStations; ++st2) {
        bool full_req =
            dg_req->request_visibilities(row_id, t, st1, st2, uvw.data());
        BOOST_REQUIRE_EQUAL(full_req, false);

        auto compute_req = dg_req->compute();
        if (st1 == st2) {  // Degridder ignores auto correlations.
          BOOST_CHECK_EQUAL(compute_req.size(), 0);
          result_req.push_back(dummy_row.data());
        } else {
          BOOST_REQUIRE_EQUAL(compute_req.size(), 1);
          BOOST_CHECK_EQUAL(compute_req.front().first, row_id);
          result_req.push_back(compute_req.front().second);
        }

        bool full_multi =
            dg_multi->request_visibilities(row_id, t, st1, st2, uvw.data());
        BOOST_REQUIRE_EQUAL(full_multi, false);

        ++row_id;
      }
    }
  }
  BOOST_REQUIRE_EQUAL(kNrRows, row_id - kFirstRow);

  auto compute_multi = dg_multi->compute();

  // Verify compute_multi and copy the result pointers to result_multi.
  const std::size_t kNrBaselinesNoAuto = kNrBaselines - kNrStations;
  BOOST_REQUIRE_EQUAL(compute_multi.size(), kNrTimesteps * kNrBaselinesNoAuto);
  row_id = kFirstRow;
  auto row = compute_multi.begin();
  for (size_t t = 0; t < kNrTimesteps; ++t) {
    for (size_t bl = 0; bl < kNrBaselines; ++bl) {
      if (antennas.first[bl] == antennas.second[bl]) {
        result_multi.push_back(dummy_row.data());
      } else {
        BOOST_REQUIRE(row != compute_multi.end());
        BOOST_CHECK_EQUAL(row->first, row_id);
        result_multi.push_back(row->second);
        ++row;
      }

      ++row_id;
    }
  }
  BOOST_CHECK(row == compute_multi.end());

  // Predict visibilities using compute_visibilities().

  // Buffer and input data pointers for result values from compute_visibilities.
  std::vector<std::complex<float>> comp_data(kNrRows * kRowSize, kDummyData);
  std::vector<std::complex<float>*> comp_data_ptrs;

  for (std::size_t t = 0; t < kNrTimesteps; ++t) {
    comp_data_ptrs.push_back(comp_data.data() + t * kNrBaselines * kRowSize);
    for (std::size_t bl = 0; bl < kNrBaselines; ++bl) {
      result_comp.push_back(comp_data_ptrs.back() + bl * kRowSize);
    }
  }

  dg_comp->compute_visibilities(antennas.first, antennas.second, uvws,
                                comp_data_ptrs);

  // Finally, compare the predictions.
  CompareResults(result_req, result_multi);
  CompareResults(result_multi, result_comp);

  dg_req->finished_reading();
  dg_multi->finished_reading();

  bs_req->finished();
  bs_multi->finished();
  bs_comp->finished();
}

BOOST_AUTO_TEST_CASE(custom_factors) {
  std::unique_ptr<idg::api::BufferSet> bs_ref =
      CreateBufferset(idg::api::BufferSetType::kBulkDegridding);
  std::unique_ptr<idg::api::BufferSet> bs_uvw =
      CreateBufferset(idg::api::BufferSetType::kBulkDegridding);
  std::unique_ptr<idg::api::BufferSet> bs_aterm =
      CreateBufferset(idg::api::BufferSetType::kBulkDegridding);

  const idg::api::BulkDegridder* dg_ref = bs_ref->get_bulk_degridder(0);
  const idg::api::BulkDegridder* dg_uvw = bs_uvw->get_bulk_degridder(0);
  const idg::api::BulkDegridder* dg_aterm = bs_aterm->get_bulk_degridder(0);

  const std::vector<double> uvw = CreateUVW(1.0, 2.0, 3.0);
  const std::vector<double> uvw_custom = CreateUVW(1.0, -1.0, 6.0);
  // uvw_factors * uvw_custom[x] should equal uvw[x].
  const std::vector<double> uvw_factors{1.0, -2.0, 0.5};
  const std::vector<const double*> uvws(kNrTimesteps, uvw.data());
  const std::vector<const double*> uvws_custom(kNrTimesteps, uvw_custom.data());

  // Generate artficial aterms.
  const std::complex<float> aterm_012{2.0, 0.0};   // For timesteps 0, 1, 2.
  const std::complex<float> aterm_34{1.0, 1.0};    // For timesteps 3, 4.
  const std::complex<float> aterm_5678{0.0, 3.0};  // For timesteps 5, 6, 7, 8.
  const std::vector<unsigned int> aterm_offsets{0, 3, 5};

  const std::size_t subgrid_size = bs_aterm->get_subgridsize();
  const std::size_t aterm_block_size =
      kNrStations * subgrid_size * subgrid_size * kNrCorrelations;
  std::vector<std::complex<float>> aterms(3 * aterm_block_size, {0.0});
  for (std::size_t i = 0; i < aterm_block_size; i += 4) {
    aterms[0 * aterm_block_size + i + 0] = aterm_012;
    aterms[0 * aterm_block_size + i + 3] = aterm_012;
    aterms[1 * aterm_block_size + i + 0] = aterm_34;
    aterms[1 * aterm_block_size + i + 3] = aterm_34;
    aterms[2 * aterm_block_size + i + 0] = aterm_5678;
    aterms[2 * aterm_block_size + i + 3] = aterm_5678;
  }

  std::vector<std::complex<float>> data_ref(kNrRows * kRowSize, kDummyData);
  std::vector<std::complex<float>> data_uvw(kNrRows * kRowSize, kDummyData);
  std::vector<std::complex<float>> data_aterm(kNrRows * kRowSize, kDummyData);

  // Input data pointers for compute_visibilities: One for each timestep.
  std::vector<std::complex<float>*> ptrs_ref;
  std::vector<std::complex<float>*> ptrs_uvw;
  std::vector<std::complex<float>*> ptrs_aterm;

  for (std::size_t t = 0; t < kNrTimesteps; ++t) {
    ptrs_ref.push_back(data_ref.data() + t * kNrBaselines * kRowSize);
    ptrs_uvw.push_back(data_uvw.data() + t * kNrBaselines * kRowSize);
    ptrs_aterm.push_back(data_aterm.data() + t * kNrBaselines * kRowSize);
  }

  const auto antennas = CreateAntennas();
  dg_ref->compute_visibilities(antennas.first, antennas.second, uvws, ptrs_ref);
  dg_uvw->compute_visibilities(antennas.first, antennas.second, uvws_custom,
                               ptrs_uvw, uvw_factors.data());
  dg_aterm->compute_visibilities(antennas.first, antennas.second, uvws,
                                 ptrs_aterm, nullptr, aterms.data(),
                                 aterm_offsets);

  CompareResults(data_ref.data(), data_uvw.data());

  const std::size_t timestep_size = kNrBaselines * kRowSize;
  CompareResults(data_ref.data(), data_aterm.data(), 3, aterm_012);
  CompareResults(data_ref.data() + timestep_size * 3,
                 data_aterm.data() + timestep_size * 3, 2, aterm_34);
  CompareResults(data_ref.data() + timestep_size * 5,
                 data_aterm.data() + timestep_size * 5, 4, aterm_5678);
}

// TODO: Fix shift in w-tiling and add WMode::kWTiling test.
BOOST_DATA_TEST_CASE(shift,
                     boost::unit_test::data::make({WMode::kNeither,
                                                   WMode::kWStacking}),
                     wmode) {
  const std::array<int, 2> kShift{10, 20};

  std::unique_ptr<idg::api::BufferSet> bs_ref =
      CreateBufferset(idg::api::BufferSetType::kBulkDegridding);
  std::unique_ptr<idg::api::BufferSet> bs_shift =
      CreateBufferset(idg::api::BufferSetType::kBulkDegridding, kShift, wmode);

  const idg::api::BulkDegridder* dg_ref = bs_ref->get_bulk_degridder(0);
  const idg::api::BulkDegridder* dg_shift = bs_shift->get_bulk_degridder(0);

  const std::vector<double> uvw = CreateUVW(1.0, 2.0, 3.0);
  const std::vector<const double*> uvws(kNrTimesteps, uvw.data());

  std::vector<std::complex<float>> data_ref(kNrRows * kRowSize, kDummyData);
  std::vector<std::complex<float>> data_shift(kNrRows * kRowSize, kDummyData);
  std::vector<std::complex<float>*> ptrs_ref;
  std::vector<std::complex<float>*> ptrs_shift;

  for (std::size_t t = 0; t < kNrTimesteps; ++t) {
    ptrs_ref.push_back(data_ref.data() + t * kNrBaselines * kRowSize);
    ptrs_shift.push_back(data_shift.data() + t * kNrBaselines * kRowSize);
  }

  const auto antennas = CreateAntennas();
  dg_ref->compute_visibilities(antennas.first, antennas.second, uvws, ptrs_ref);
  dg_shift->compute_visibilities(antennas.first, antennas.second, uvws,
                                 ptrs_shift);

  ApplyPhaseShift(data_shift.data(), uvw.data(), kShift);

  CompareResults(data_ref.data(), data_shift.data());
}

BOOST_AUTO_TEST_CASE(bulk_invalid_arguments) {
  std::unique_ptr<idg::api::BufferSet> bs =
      CreateBufferset(idg::api::BufferSetType::kBulkDegridding);
  const idg::api::BulkDegridder* dg = bs->get_bulk_degridder(0);

  // A single auto-correlation is valid input.
  BOOST_CHECK_NO_THROW(
      dg->compute_visibilities({0}, {0}, {nullptr}, {nullptr}));

  BOOST_CHECK_THROW(dg->compute_visibilities({0}, {0, 0}, {nullptr}, {nullptr}),
                    std::invalid_argument);
  BOOST_CHECK_THROW(dg->compute_visibilities({0}, {0}, {nullptr}, {}),
                    std::invalid_argument);
  BOOST_CHECK_THROW(dg->compute_visibilities({0}, {0}, {}, {nullptr}),
                    std::invalid_argument);

  // Test aterm arguments with two (fake) time steps.
  // Since we still use a single auto-correlation, the function should ignore
  // the null pointers.

  // Empty aterm offsets.
  BOOST_CHECK_THROW(
      dg->compute_visibilities({0}, {0}, {nullptr, nullptr}, {nullptr, nullptr},
                               nullptr, nullptr, {}),
      std::invalid_argument);

  // Invalid first aterm offset (should be zero).
  BOOST_CHECK_THROW(
      dg->compute_visibilities({0}, {0}, {nullptr, nullptr}, {nullptr, nullptr},
                               nullptr, nullptr, {1}),
      std::invalid_argument);

  // Use default aterms with multiple offsets.
  BOOST_CHECK_THROW(
      dg->compute_visibilities({0}, {0}, {nullptr, nullptr}, {nullptr, nullptr},
                               nullptr, nullptr, {0, 1}),
      std::invalid_argument);

  // Non-default aterms with multiple offsets are ok.
  const std::complex<float> dummy_aterm;
  BOOST_CHECK_NO_THROW(dg->compute_visibilities({0}, {0}, {nullptr, nullptr},
                                                {nullptr, nullptr}, nullptr,
                                                &dummy_aterm, {0, 1}));
}

BOOST_AUTO_TEST_SUITE_END()
