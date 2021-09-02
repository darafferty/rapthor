// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_CUFFT_H_
#define IDG_CUFFT_H_

#include <stdexcept>

#include <cuda.h>
#include <cufft.h>

#include "CU.h"

namespace cufft {

class Error : public std::exception {
 public:
  Error(cufftResult result, std::string& message)
      : _result(result), _message(message) {}

  const char* what() const throw() { return _message.c_str(); };

  static const char* what(cufftResult result);

  operator cufftResult() const { return _result; }

 private:
  cufftResult _result;
  std::string _message;
};

class C2C_1D {
 public:
  C2C_1D(const cu::Context& context, unsigned n, unsigned count);
  C2C_1D(const cu::Context& context, unsigned n, unsigned stride, unsigned dist,
         unsigned count);
  ~C2C_1D();
  void setStream(CUstream stream);
  void execute(cufftComplex* in, cufftComplex* out,
               int direction = CUFFT_FORWARD);

 private:
  cufftHandle plan;
  const cu::Context& context;
};

class C2C_2D {
 public:
  C2C_2D(const cu::Context& context, unsigned nx, unsigned ny);
  C2C_2D(const cu::Context& context, unsigned nx, unsigned ny, unsigned stride,
         unsigned dist, unsigned count);
  ~C2C_2D();
  void setStream(CUstream stream);
  void execute(cufftComplex* in, cufftComplex* out,
               int direction = CUFFT_FORWARD);

 private:
  cufftHandle plan;
  const cu::Context& context;
};

}  // end namespace cufft

#endif
