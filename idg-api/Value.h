// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_API_VALUE_H_
#define IDG_API_VALUE_H_

#include <typeinfo>
#include <memory>

namespace idg {
namespace api {

class ValueHolder {
 public:
  virtual ~ValueHolder() {}
};

template <typename T>
class TypedValueHolder : public ValueHolder {
 public:
  TypedValueHolder(T x) : _x(x) {}

  T get() { return _x; }

 private:
  T _x;
};

// the Value class is a container for any type
// Intended use is for data collections of mixed type
// Example:

//     std::map<std::string, Value> vm;
//
//     vm["aaa"] = 20;
//     vm["bbb"] = std::string("Hello");
//     int a = vm["aaa"];
//     std::string b = vm["bbb"];

class Value {
 public:
  Value() : _v(nullptr) {}

  template <typename T>
  Value(T x) : _v(new TypedValueHolder<T>(x)) {}
  template <typename T>
  operator T() const {
    TypedValueHolder<T>* tv(dynamic_cast<TypedValueHolder<T>*>(_v.get()));
    if (tv) {
      return tv->get();
    } else {
      throw std::bad_cast();
    }
  }

  template <typename T>
  T as() const {
    return operator T();
  }

 private:
  std::unique_ptr<ValueHolder> _v;
};

}  // namespace api
}  // namespace idg

#endif
