// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_RUNTIME_WRAPPER_H
#define IDG_RUNTIME_WRAPPER_H

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <dlfcn.h>
#include <stdio.h>
#include <errno.h>

#include "idg-config.h"

namespace idg {

namespace runtime {

class Error : public std::exception {
 public:
  Error(const char* what) : _what(what) {}

  const char* what() const throw();

 private:
  const char* _what;
};

class Source {
 public:
  Source();

  Source(const char* input_file_name);

  void compile(const char* compiler, const char* output_file_name,
               const char* compiler_options = 0);

 private:
  const char* input_file_name;
};

class Module {
 public:
  Module(const char* file_name) {
    _module = dlopen(file_name, RTLD_NOW);
    if (!_module) {
      std::cerr << "Error loading: " << file_name << ": ";
      std::cerr << dlerror() << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  Module(Module& module) { _module = module; }

  operator void*() { return _module; }

  ~Module() { dlclose(_module); }

 private:
  void* _module;
};

class Function {
 public:
  Function(Module& module, const char* name) {
    _function = dlsym(module, name);
    if (!_function) {
      std::cerr << "Error loading: " << name << std::endl;
      std::cerr << dlerror() << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  operator void*() { return _function; }

 private:
  void* _function;
};

}  // namespace runtime

}  // namespace idg

#endif
