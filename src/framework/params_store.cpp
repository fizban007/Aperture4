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

#include "params_store.h"
#include "cpptoml.h"
#if __GNUC__ >= 8 || __clang_major__ >= 7
#include <filesystem>
#else
#include <boost/filesystem.hpp>
#endif
#include <fstream>
#include <fmt/format.h>
#include <memory>
#include <type_traits>
#include <variant>

#if __GNUC__ >= 8 || __clang_major__ >= 7
namespace fs = std::filesystem;
#else
namespace fs = boost::filesystem;
#endif

template <typename T>
struct is_vector : public std::false_type {};

template <typename T, typename A>
struct is_vector<std::vector<T, A>> : public std::true_type {};

namespace Aperture {

typedef std::variant<bool, int64_t, double, std::string, std::vector<bool>,
                     std::vector<int64_t>, std::vector<double>,
                     std::vector<std::string>>
    param_type;

class params_store::params_store_impl {
 public:
  void parse(const std::string& filename) {
    // parse_config(filename, m_params);
    fs::path config_path(filename);
    if (!fs::exists(config_path)) {
      Logger::print_info("Config file not found, using all defaults!");
      return;
    }
    Logger::print_debug("=== Begin parsing parameter file ===");
    auto table = cpptoml::parse_file(filename);
    read_table(*table, m_param_map);
    Logger::print_debug("=== End parsing parameter file ===");
  }

  bool has(const std::string& name) {
    auto it = m_param_map.find(name);
    return (it != m_param_map.end());
  }

  template <typename T>
  T get(const std::string& name, T default_value) {
    auto it = m_param_map.find(name);
    if (it != m_param_map.end()) {
      try {
        T f = std::get<T>(it->second);
        return f;
      } catch (std::bad_variant_access&) {
        Logger::print_err("> Parameter '{}' has the incorrect type!", name);
        return default_value;
      }
    } else {
      m_param_map.insert({name, default_value});
      if constexpr (is_vector<T>::value) {
        Logger::print_err(
            "> Parameter '{}' not found in store, using default [{}]", name,
            fmt::join(default_value, ","));
      } else {
        Logger::print_err(
            "> Parameter '{}' not found in store, using default {}", name,
            default_value);
      }
      return default_value;
    }
  }

  template <typename T>
  void add(const std::string& name, const T& value) {
    // m_param_map.insert({name, param_type(value)});
    auto it = m_param_map.find(name);
    if (it != m_param_map.end()) {
      it->second = value;
    } else {
      m_param_map.insert({name, param_type(value)});
    }
  }

  void read_table(const cpptoml::table& table,
                  std::unordered_map<std::string, param_type>& params) {
    for (auto& it : table) {
      auto& name = it.first;
      auto& ptr = it.second;
      if (ptr->is_table_array()) {
        auto& array =
            std::dynamic_pointer_cast<cpptoml::table_array>(ptr)->get();
        Logger::print_debug("Entering table array '{}'", name);
        for (auto& p : array) {
          read_table(*p, params);
        }
      } else if (ptr->is_table()) {
        Logger::print_debug("Entering table '{}'", name);
        read_table(*std::dynamic_pointer_cast<cpptoml::table>(ptr), params);
      } else if (ptr->is_array()) {
        auto array_ptr = std::dynamic_pointer_cast<cpptoml::array>(ptr);
        if (auto v = array_ptr->get_array_of<bool>()) {
          Logger::print_debug("parsed bool array '{}'", name);
          params.insert({name, param_type(*v)});
        } else if (auto v = array_ptr->get_array_of<int64_t>()) {
          Logger::print_debug("parsed int array '{}'", name);
          params.insert({name, param_type(*v)});
        } else if (auto v = array_ptr->get_array_of<double>()) {
          Logger::print_debug("parsed double array '{}'", name);
          params.insert({name, param_type(*v)});
        } else if (auto v = array_ptr->get_array_of<std::string>()) {
          Logger::print_debug("parsed string array '{}'", name);
          params.insert({name, param_type(*v)});
        }
      } else if (ptr->is_value()) {
        if (auto v = ptr->as<bool>()) {
          Logger::print_debug("parsed bool '{}'", name);
          params.insert({name, param_type(v->get())});
        } else if (auto v = ptr->as<int64_t>()) {
          Logger::print_debug("parsed int '{}'", name);
          params.insert({name, param_type(v->get())});
        } else if (auto v = ptr->as<double>()) {
          Logger::print_debug("parsed double '{}'", name);
          params.insert({name, param_type(v->get())});
        } else if (auto v = ptr->as<std::string>()) {
          Logger::print_debug("parsed string '{}'", name);
          params.insert({name, param_type(v->get())});
        }
      }
    }
  }

  // params_struct m_params;
  std::unordered_map<std::string, param_type> m_param_map;
};

params_store::params_store() { p_impl = new params_store_impl; }

params_store::~params_store() { delete p_impl; }

void
params_store::clear() {
  p_impl->m_param_map.clear();
}

void
params_store::parse(const std::string& filename) {
  p_impl->parse(filename);
}

bool
params_store::has(const std::string& name) {
  return p_impl->has(name);
}

// const params_struct&
// params_store::params() const {
//   return p_impl->m_params;
// }

template <typename T>
T
params_store::get_as(const std::string& name, T default_value) const {
  return p_impl->get(name, default_value);
}

template <typename T>
T
params_store::get_as(const std::string& name) const {
  return p_impl->get(name, T{});
}

template <typename T>
void
params_store::add(const std::string& name, const T& value) {
  p_impl->add(name, value);
}

void
params_store::write(const std::string &path) const {
  auto root = cpptoml::make_table();

  for (auto el : p_impl->m_param_map) {
    if (std::holds_alternative<bool>(el.second)) {
      root->insert(el.first, std::get<bool>(el.second));
    } else if (std::holds_alternative<int64_t>(el.second)) {
      root->insert(el.first, std::get<int64_t>(el.second));
    } else if (std::holds_alternative<double>(el.second)) {
      root->insert(el.first, std::get<double>(el.second));
    } else if (std::holds_alternative<std::string>(el.second)) {
      root->insert(el.first, std::get<std::string>(el.second));
    } else if (std::holds_alternative<std::vector<bool>>(el.second)) {
      auto array = cpptoml::make_array();
      for (bool v : std::get<std::vector<bool>>(el.second)) {
        array->push_back(v);
      }
      root->insert(el.first, array);
    } else if (std::holds_alternative<std::vector<int64_t>>(el.second)) {
      auto array = cpptoml::make_array();
      for (int64_t v : std::get<std::vector<int64_t>>(el.second)) {
        array->push_back(v);
      }
      root->insert(el.first, array);
    } else if (std::holds_alternative<std::vector<double>>(el.second)) {
      auto array = cpptoml::make_array();
      for (double v : std::get<std::vector<double>>(el.second)) {
        array->push_back(v);
      }
      root->insert(el.first, array);
    } else if (std::holds_alternative<std::vector<std::string>>(el.second)) {
      auto array = cpptoml::make_array();
      for (std::string v : std::get<std::vector<std::string>>(el.second)) {
        array->push_back(v);
      }
      root->insert(el.first, array);
    }
  }

  std::ofstream out(path);
  out << *root << std::endl;
  out.close();
}

////////////////////////////////////////////////////////////////
template bool params_store::get_as(const std::string& name,
                                   bool default_value) const;
template int64_t params_store::get_as(const std::string& name,
                                      int64_t default_value) const;
// template uint64_t params_store::get_as(const std::string& name,
//                                    uint64_t default_value) const;
template double params_store::get_as(const std::string& name,
                                     double default_value) const;
template std::string params_store::get_as(const std::string& name,
                                          std::string default_value) const;

template std::string params_store::get_as<std::string>(
    const std::string& name) const;
template std::vector<bool> params_store::get_as<std::vector<bool>>(
    const std::string& name) const;
template std::vector<int64_t> params_store::get_as<std::vector<int64_t>>(
    const std::string& name) const;
// template std::vector<uint64_t> params_store::get<std::vector<uint64_t>>(
//     const std::string& name) const;
template std::vector<double> params_store::get_as<std::vector<double>>(
    const std::string& name) const;
template std::vector<std::string>
params_store::get_as<std::vector<std::string>>(const std::string& name) const;

template void params_store::add(const std::string& name, const bool& value);
template void params_store::add(const std::string& name, const int64_t& value);
// template void params_store::add(const std::string& name,
//                                const uint64_t& value);
template void params_store::add(const std::string& name, const double& value);
template void params_store::add(const std::string& name,
                                const std::string& value);
template void params_store::add(const std::string& name,
                                const std::vector<bool>& value);
template void params_store::add(const std::string& name,
                                const std::vector<int64_t>& value);
// template void params_store::add(const std::string& name,
//                                const std::vector<uint64_t>& value);
template void params_store::add(const std::string& name,
                                const std::vector<double>& value);
template void params_store::add(const std::string& name,
                                const std::vector<std::string>& value);

}  // namespace Aperture
