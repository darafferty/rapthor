#include <array>
#include <string>

#include <aocommon/xt/span.h>
#include <idg-common.h>

namespace {
constexpr unsigned int kNrCorrelations = 4;
constexpr unsigned int kNrPolarizations = 4;
constexpr unsigned int kNrStations = 9;
constexpr unsigned int kNrChannels = 9;
constexpr unsigned int kNrTimesteps = 2048;
constexpr unsigned int kNrTimeslots = 7;
constexpr unsigned int kGridSize = 2048;
constexpr unsigned int kSubgridSize = 32;
constexpr unsigned int kKernelSize = 9;
constexpr unsigned int kNrBaselines = (kNrStations * (kNrStations - 1)) / 2;
constexpr float kIntegrationTime = 1.0f;
constexpr double kSpeedOfLight = 299792458.0;
const std::string kLayoutFile = "LOFAR_lba.txt";
const std::array<float, 2> kShift{0.02f, -0.03f};
}  // namespace

class TestFixture {
 public:
  TestFixture();

  inline std::array<float, 2> GetShift() const { return kShift; }
  inline float GetImageSize() const { return image_size_; }
  inline float GetCellSize() const { return cell_size_; }

  aocommon::xt::Span<idg::UVW<float>, 2> GetUVW();
  aocommon::xt::Span<float, 1> GetFrequencies();
  aocommon::xt::Span<float, 1> GetWavenumbers();
  aocommon::xt::Span<std::complex<float>, 4> GetVisibilities();
  aocommon::xt::Span<float, 2> GetTaper();
  aocommon::xt::Span<idg::Matrix2x2<std::complex<float>>, 4> GetAterms();
  aocommon::xt::Span<unsigned int, 1> GetAtermOffsets();
  aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1> GetBaselines();

 private:
  float image_size_;
  float cell_size_;

  const std::array<size_t, 2> uvw_shape_{kNrBaselines, kNrTimesteps};
  const std::array<size_t, 1> frequencies_shape_{kNrChannels};
  const std::array<size_t, 4> visibilities_shape_{kNrBaselines, kNrTimesteps,
                                                  kNrChannels, kNrCorrelations};
  const std::array<size_t, 2> taper_shape_{kSubgridSize, kSubgridSize};
  const std::array<size_t, 4> aterms_shape_{kNrTimeslots, kNrStations,
                                            kSubgridSize, kSubgridSize};
  const std::array<size_t, 1> aterm_offsets_shape_{kNrTimeslots};
  const std::array<size_t, 1> baselines_shape_{kNrBaselines};

  xt::xtensor<idg::UVW<float>, 2> uvw_;
  xt::xtensor<float, 1> frequencies_;
  xt::xtensor<float, 1> wavenumbers_;
  xt::xtensor<std::complex<float>, 4> visibilities_;
  xt::xtensor<float, 2> taper_;
  xt::xtensor<idg::Matrix2x2<std::complex<float>>, 4> aterms_;
  xt::xtensor<unsigned int, 1> aterm_offsets_;
  xt::xtensor<std::pair<unsigned int, unsigned int>, 1> baselines_;
};