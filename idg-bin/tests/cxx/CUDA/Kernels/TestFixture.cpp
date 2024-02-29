#include "TestFixture.h"

#include "idg-util.h"

TestFixture::TestFixture() {
  idg::Data data = idg::get_example_data(
      kNrBaselines, kGridSize, kIntegrationTime, kNrChannels, kLayoutFile);

  image_size_ = data.compute_image_size(kGridSize, kNrChannels);
  cell_size_ = image_size_ / kGridSize;

  uvw_ = xt::xtensor<idg::UVW<float>, 2>(uvw_shape_);
  auto uvw_span = aocommon::xt::CreateSpan(uvw_);
  data.get_uvw(uvw_span);

  frequencies_ = xt::xtensor<float, 1>(frequencies_shape_);
  auto frequencies_span = aocommon::xt::CreateSpan(frequencies_);
  data.get_frequencies(frequencies_span, image_size_);

  wavenumbers_ = xt::xtensor<float, 1>(frequencies_shape_);
  wavenumbers_ = 2 * M_PI * frequencies_ / kSpeedOfLight;
  auto wavenumbers_span = aocommon::xt::CreateSpan(wavenumbers_);

  visibilities_ = xt::xtensor<std::complex<float>, 4>(visibilities_shape_);
  auto visibilities_span = aocommon::xt::CreateSpan(visibilities_);
  idg::init_example_visibilities(visibilities_span, uvw_span, frequencies_span,
                                 image_size_, kGridSize);

  taper_ = xt::xtensor<float, 2>(taper_shape_);
  auto taper_span = aocommon::xt::CreateSpan(taper_);
  idg::init_example_taper(taper_span);

  aterms_ = xt::xtensor<idg::Matrix2x2<std::complex<float>>, 4>(aterms_shape_);
  auto aterms_span = aocommon::xt::CreateSpan(aterms_);
  idg::init_example_aterms(aterms_span);

  aterm_offsets_ = xt::xtensor<unsigned int, 1>(aterm_offsets_shape_);
  auto aterm_offsets_span = aocommon::xt::CreateSpan(aterm_offsets_);
  idg::init_example_aterm_offsets(aterm_offsets_span, kNrTimesteps);

  baselines_ =
      xt::xtensor<std::pair<unsigned int, unsigned int>, 1>(baselines_shape_);
  auto baselines_span = aocommon::xt::CreateSpan(baselines_);
  idg::init_example_baselines(baselines_span, kNrStations);
}

aocommon::xt::Span<idg::UVW<float>, 2> TestFixture::GetUVW() {
  assert(uvw_.size());
  return aocommon::xt::CreateSpan(uvw_);
}

aocommon::xt::Span<float, 1> TestFixture::GetFrequencies() {
  assert(frequencies_.size());
  return aocommon::xt::CreateSpan(frequencies_);
}

aocommon::xt::Span<float, 1> TestFixture::GetWavenumbers() {
  assert(wavenumbers_.size());
  return aocommon::xt::CreateSpan(wavenumbers_);
}

aocommon::xt::Span<std::complex<float>, 4> TestFixture::GetVisibilities() {
  assert(visibilities_.size());
  return aocommon::xt::CreateSpan(visibilities_);
}

aocommon::xt::Span<float, 2> TestFixture::GetTaper() {
  assert(taper_.size());
  return aocommon::xt::CreateSpan(taper_);
}

aocommon::xt::Span<idg::Matrix2x2<std::complex<float>>, 4>
TestFixture::GetAterms() {
  assert(aterms_.size());
  return aocommon::xt::CreateSpan(aterms_);
}

aocommon::xt::Span<unsigned int, 1> TestFixture::GetAtermOffsets() {
  assert(aterm_offsets_.size());
  return aocommon::xt::CreateSpan(aterm_offsets_);
}

aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>
TestFixture::GetBaselines() {
  assert(baselines_.size());
  return aocommon::xt::CreateSpan(baselines_);
}
