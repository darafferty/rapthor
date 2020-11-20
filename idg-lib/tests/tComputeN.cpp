// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <boost/test/unit_test.hpp>

#include "common/Types.h"
#include "common/Math.h"

BOOST_AUTO_TEST_SUITE(test_compute_n)

BOOST_AUTO_TEST_CASE(without_shift) {
  // Check if n reproduces correct value for ||[l,m]^T|| < 1
  float n_lt1 = compute_n(0.3, 0.4);
  BOOST_CHECK_CLOSE(n_lt1, 0.13397459f, 1e-6);

  // Check if n=1 if ||[l, m]^T|| > 1
  float n_gt1 = compute_n(1.25, 1.5);
  BOOST_CHECK_CLOSE(n_gt1, 1.0f, 1e-6);
}

BOOST_AUTO_TEST_CASE(with_shift) {
  std::array<float, 3> shift = {0.9, 1.3, 0.4};

  // Check if n reproduces correct value for ||[l - shift[0], m - shift[1]]^T||
  // > 1
  float n_gt1 = compute_n(0.3, 0.4, shift.data());
  BOOST_CHECK_CLOSE(n_gt1, 1.0f, 1e-6);

  // Check if n=1 if ||[l - shift[0], m - shift[1]]^T|| < 1
  float n_lt1 = compute_n(1.25, 1.5, shift.data());
  BOOST_CHECK_CLOSE(n_lt1, 0.48484975f, 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()