// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_CPU_OPTIMIZED_H_
#define IDG_CPU_OPTIMIZED_H_

#include "idg-cpu.h"

namespace idg {
namespace proxy {
namespace cpu {

class Optimized : public CPU {
 public:
  // Constructor
  Optimized(std::vector<std::string> libraries = default_libraries());

 private:
  static std::vector<std::string> default_libraries();

};  // class Optimized

}  // namespace cpu
}  // namespace proxy
}  // namespace idg

#endif
