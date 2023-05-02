// Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <aocommon/xt/span.h>

#include "auxiliary.h"

namespace idg {

/// Structure that holds a span and the memory allocated for that span.
template <typename T, size_t Dimensions>
class Tensor {
 public:
  Tensor()
      : span_(aocommon::xt::CreateSpan<T, Dimensions>(
            nullptr, std::array<size_t, Dimensions>{})) {}

  Tensor(std::unique_ptr<auxiliary::Memory> memory,
         const std::array<size_t, Dimensions>& shape)
      : memory_(std::move(memory)),
        span_(aocommon::xt::CreateSpan<T, Dimensions>(
            memory_ ? reinterpret_cast<T*>(memory_->data()) : nullptr, shape)) {
  }

  aocommon::xt::Span<T, Dimensions>& Span() { return span_; }

  void Reset() {
    memory_.reset();
    span_ = aocommon::xt::CreateSpan<T, Dimensions>(
        nullptr, std::array<size_t, Dimensions>{});
  }

 private:
  std::unique_ptr<auxiliary::Memory> memory_;
  aocommon::xt::Span<T, Dimensions> span_;
};

}  // namespace idg