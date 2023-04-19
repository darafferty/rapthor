// Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <aocommon/xt/span.h>

#include "auxiliary.h"

namespace idg {

/// Structure that holds a span and the memory allocated for that span.
template <typename T, size_t Dimensions>
class Tensor {
 public:
  Tensor(std::unique_ptr<auxiliary::Memory> memory,
         const std::array<size_t, Dimensions>& shape)
      : memory_(std::move(memory)),
        span_(aocommon::xt::CreateSpan<T, Dimensions>(
            reinterpret_cast<T*>(memory_->data()), shape)){};

  aocommon::xt::Span<T, Dimensions>& Span() { return span_; }

 private:
  std::unique_ptr<auxiliary::Memory> memory_;
  aocommon::xt::Span<T, Dimensions> span_;
};

}  // namespace idg