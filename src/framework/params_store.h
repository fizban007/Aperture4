/*
 * Copyright (c) 2020 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __PARAMS_STORE_H_
#define __PARAMS_STORE_H_

// #include "core/params.h"
#include "utils/logger.h"
#include "utils/vec.hpp"
#include "visit_struct/visit_struct.hpp"
#include <string>

namespace Aperture {

////////////////////////////////////////////////////////////////////////////////
///  `params_store` maintains a hashtable that points to different types of
///  parameters. One can save new parameters to the store or retrieve from it at
///  any time. Parameter can have one of the following types: `bool`, `int64_t`,
///  `double`, `string`, and `vector` of the above types.
////////////////////////////////////////////////////////////////////////////////
class params_store {
 private:
  // Because we need to use std::variant which is not supported in CUDA code, we
  // have to use p_impl idiom to hide the implementation detail
  class params_store_impl;
  params_store_impl* p_impl;

  // Private struct to facilitate visiting of the different parameter types
  struct visit_param {
    const params_store& store;

    visit_param(const params_store& s) : store(s) {}

    void operator()(const char* name, float& x) {
      x = store.get_as<double>(name, (double)x);
    }

    void operator()(const char* name, double& x) {
      x = store.get_as<double>(name, x);
    }

    void operator()(const char* name, bool& x) {
      x = store.get_as<bool>(name, x);
    }

    void operator()(const char* name, int& x) {
      x = store.get_as<int64_t>(name, (int64_t)x);
    }

    void operator()(const char* name, long& x) {
      x = store.get_as<int64_t>(name, (int64_t)x);
    }

    void operator()(const char* name, uint32_t& x) {
      x = store.get_as<int64_t>(name, (int64_t)x);
    }

    void operator()(const char* name, uint64_t& x) {
      x = store.get_as<int64_t>(name, (int64_t)x);
    }

    void operator()(const char* name, std::string& x) {
      x = store.get_as<std::string>(name, x);
    }

    template <size_t N>
    void operator()(const char* name, float (&x)[N]) {
      auto v = store.get_as<std::vector<double>>(name);
      for (int i = 0; i < std::min(N, v.size()); i++) x[i] = v[i];
    }

    template <size_t N>
    void operator()(const char* name, double (&x)[N]) {
      auto v = store.get_as<std::vector<double>>(name);
      for (int i = 0; i < std::min(N, v.size()); i++) x[i] = v[i];
    }

    template <size_t N>
    void operator()(const char* name, int (&x)[N]) {
      auto v = store.get_as<std::vector<int64_t>>(name);
      for (int i = 0; i < std::min(N, v.size()); i++) x[i] = v[i];
    }

    template <size_t N>
    void operator()(const char* name, uint32_t (&x)[N]) {
      auto v = store.get_as<std::vector<int64_t>>(name);
      for (int i = 0; i < std::min(N, v.size()); i++) x[i] = v[i];
    }

    template <size_t N>
    void operator()(const char* name, uint64_t (&x)[N]) {
      auto v = store.get_as<std::vector<int64_t>>(name);
      for (int i = 0; i < std::min(N, v.size()); i++) x[i] = v[i];
    }

    template <size_t N>
    void operator()(const char* name, bool (&x)[N]) {
      auto v = store.get_as<std::vector<bool>>(name);
      for (int i = 0; i < std::min(N, v.size()); i++) x[i] = v[i];
    }

    template <size_t N>
    void operator()(const char* name, std::string (&x)[N]) {
      auto v = store.get_as<std::vector<std::string>>(name);
      for (int i = 0; i < std::min(N, v.size()); i++) x[i] = v[i];
    }

    template <int Dim>
    void operator()(const char* name, vec_t<uint32_t, Dim>& x) {
      auto v = store.get_as<std::vector<int64_t>>(name);
      for (int i = 0; i < std::min((size_t)Dim, v.size()); i++) x[i] = v[i];
    }

    template <int Dim>
    void operator()(const char* name, vec_t<float, Dim>& x) {
      auto v = store.get_as<std::vector<double>>(name);
      for (int i = 0; i < std::min((size_t)Dim, v.size()); i++) x[i] = v[i];
    }

    template <int Dim>
    void operator()(const char* name, vec_t<double, Dim>& x) {
      auto v = store.get_as<std::vector<double>>(name);
      for (int i = 0; i < std::min((size_t)Dim, v.size()); i++) x[i] = v[i];
    }
  };

 public:
  params_store();
  params_store(const params_store& other) = delete;
  params_store(params_store&& other) = delete;
  ~params_store();

  /// Parse a config file to populate the parameters storage
  void parse(const std::string& filename);
  // const params_struct& params() const;

  bool has(const std::string& name);

  /// Get a parameter from the store. This version is overloaded for `bool`,
  /// `int64_t`, `double`, and `std::string`. A @default_value needs to be
  /// provided since otherwise the function does not know what to return in case
  /// the parameter is not found in the store.
  template <typename T>
  T get_as(const std::string& name, T default_value) const;

  /// Get a parameter from the store. This version is overloaded for `vector`s
  /// of `bool`, `int64_t`, `double`, and `std::string`. No default value needs
  /// to be provided since an empty vector will be returned if not found.
  template <typename T>
  T get_as(const std::string& name) const;

  /// Add a parameter to the store. If the parameter already exists in the
  /// store, update it to the new value provided. Type of the parameter is
  /// automatically deduced. However, since the only integer type supported
  /// internally is `int64_t`, the compiler will be confused if a integer of
  /// another type is sent in. In that case, cast the type to `int64_t`
  /// explicitly, or use a literal to enforce the type.
  template <typename T>
  void add(const std::string& name, const T& value);

  /// Parse the parameters directly into a visitable struct, populating each
  /// struct entry by applying `get` on each member on the params store.
  template <typename ParamStruct>
  void parse_struct(ParamStruct& params) {
    visit_struct::for_each(params, visit_param{*this});
  }

  /// Get an array from the parameter store. The output is stored in the
  /// parameter x.
  template <typename T, size_t N>
  void get_array(const std::string& name, T (&x)[N]) const {
    visit_param{*this}(name.c_str(), x);
  }

  /// Get single value from the parameter store. The output is stored in the
  /// parameter x.
  template <typename T>
  void get_value(const std::string& name, T& x) const {
    visit_param{*this}(name.c_str(), x);
  }

  /// Get a vec_t from the parameter store. The output is stored in the
  /// parameter x.
  template <typename T, int Dim>
  void get_vec_t(const std::string& name, vec_t<T, Dim>& x) const {
    visit_param{*this}(name.c_str(), x);
  }

  void clear();

  void write(const std::string& path) const;
};

}  // namespace Aperture

#endif  // __PARAMS_STORE_H_
