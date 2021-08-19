// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_CPU_OPTIMIZED_H_
#define IDG_CPU_OPTIMIZED_H_

#include "idg-cpu.h"

namespace idg {
namespace proxy {
namespace cpu {

/*! CPU implementation, optimized for performance
 */
class Optimized : public CPU {
 public:
  // Constructor
  Optimized();

};  // class Optimized

}  // namespace cpu
}  // namespace proxy
}  // namespace idg

#endif
