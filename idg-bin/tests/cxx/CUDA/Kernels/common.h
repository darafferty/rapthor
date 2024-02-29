#include <boost/test/tools/floating_point_comparison.hpp>

void check_close(const xt::xtensor<float, 1>& a, const xt::xtensor<float, 1>& b,
                 double tolerance) {
  const float max = std::max(1.0f, xt::amax(b, {0})[0]);
  const auto diff = xt::square(a - b) / max;
  const size_t nnz = std::max(1ul, xt::count_nonzero(b)[0]);
  const double error = sqrt(xt::sum(diff)[0] / nnz);

#if defined(DEBUG)
  std::cout << "max: " << max << std::endl;
  std::cout << "nnz: " << nnz << std::endl;
  std::cout << "error: " << error << std::endl;
#endif

  BOOST_CHECK(error < tolerance);

  if (error > tolerance) {
    const size_t n = a.size();
    for (size_t i = 0; i < n; i++) {
      BOOST_REQUIRE_CLOSE_FRACTION(a[i], b[i], 1);
    }
  }
}

void check_close(const xt::xtensor<std::complex<float>, 1>& a,
                 const xt::xtensor<std::complex<float>, 1>& b,
                 double tolerance) {
  const xt::xtensor<float, 1> a_real = xt::real(a);
  const xt::xtensor<float, 1> a_imag = xt::imag(a);
  const xt::xtensor<float, 1> b_real = xt::real(b);
  const xt::xtensor<float, 1> b_imag = xt::imag(b);
  check_close(a_real, b_real, tolerance);
  check_close(a_imag, b_imag, tolerance);
}