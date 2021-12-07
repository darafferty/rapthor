// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_CPU_REFERENCE2_H_
#define IDG_CPU_REFERENCE2_H_

#include "idg-cpu.h"

namespace idg {
namespace proxy {
namespace cpu {

/**
 * @brief Reference CPU implementation, not optimized for speed
 *
 */
class Reference : public CPU {
 public:
  // Constructor
  Reference();

};  // class Reference

}  // namespace cpu
}  // namespace proxy
}  // namespace idg

#endif
