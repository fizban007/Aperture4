#ifndef __PARSE_PARAMS_H_
#define __PARSE_PARAMS_H_

#include "framework/params_store.hpp"
#include "visit_struct/visit_struct.hpp"

namespace Aperture {

namespace detail {

struct visit_param {
  const params_store& store;

  visit_param(const params_store& s) : store(s) {}

  void operator()(const char* name, float& x) {
    x = store.get<double>(name, (double)x);
  }

  void operator()(const char* name, double& x) {
    x = store.get<double>(name, x);
  }

  void operator()(const char* name, bool& x) {
    x = store.get<bool>(name, x);
  }

  void operator()(const char* name, int& x) {
    x = store.get<int64_t>(name, (int64_t)x);
  }

  void operator()(const char* name, long& x) {
    x = store.get<int64_t>(name, (int64_t)x);
  }

  void operator()(const char* name, uint32_t& x) {
    x = store.get<int64_t>(name, (int64_t)x);
  }

  void operator()(const char* name, uint64_t& x) {
    x = store.get<int64_t>(name, (int64_t)x);
  }

  void operator()(const char* name, std::string& x) {
    x = store.get<std::string>(name, x);
  }

  template <size_t N>
  void operator()(const char* name, float (&x)[N]) {
    auto v = store.get<std::vector<double>>(name);
    for (int i = 0; i < std::min(N, v.size()); i++)
      x[i] = v[i];
  }

  template <size_t N>
  void operator()(const char* name, double (&x)[N]) {
    auto v = store.get<std::vector<double>>(name);
    for (int i = 0; i < std::min(N, v.size()); i++)
      x[i] = v[i];
  }

  template <size_t N>
  void operator()(const char* name, int (&x)[N]) {
    auto v = store.get<std::vector<int64_t>>(name);
    for (int i = 0; i < std::min(N, v.size()); i++)
      x[i] = v[i];
  }

  template <size_t N>
  void operator()(const char* name, uint32_t (&x)[N]) {
    auto v = store.get<std::vector<int64_t>>(name);
    for (int i = 0; i < std::min(N, v.size()); i++)
      x[i] = v[i];
  }

  template <size_t N>
  void operator()(const char* name, uint64_t (&x)[N]) {
    auto v = store.get<std::vector<int64_t>>(name);
    for (int i = 0; i < std::min(N, v.size()); i++)
      x[i] = v[i];
  }

  template <size_t N>
  void operator()(const char* name, bool (&x)[N]) {
    auto v = store.get<std::vector<bool>>(name);
    for (int i = 0; i < std::min(N, v.size()); i++)
      x[i] = v[i];
  }

  template <size_t N>
  void operator()(const char* name, std::string (&x)[N]) {
    auto v = store.get<std::vector<std::string>>(name);
    for (int i = 0; i < std::min(N, v.size()); i++)
      x[i] = v[i];
  }
};

}

// Take in a visitable struct and populate every entry if possible
template <typename ParamStruct>
void parse_struct(ParamStruct& params, const params_store& store) {
  visit_struct::for_each(params, detail::visit_param(store));
}

template <typename T, size_t N>
void parse_array(const std::string& name, T (&x)[N], const params_store& store) {
  detail::visit_param{store}(name.c_str(), x);
}

}

#endif // __PARSE_PARAMS_H_
