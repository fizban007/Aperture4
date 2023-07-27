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

#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include "data.h"
#include "framework/params_store.h"
#include "gsl/pointers"
#include "system.h"
#include "utils/logger.h"
#include "utils/nonown_ptr.hpp"
#include "utils/singleton_holder.h"

namespace cxxopts {
class Options;
class ParseResult;
}  // namespace cxxopts

namespace Aperture {

class sim_environment_impl {
 private:
  // Registry for systems and data
  std::unordered_map<std::string, std::unique_ptr<system_t>> m_system_map;
  std::unordered_map<std::string, std::unique_ptr<data_t>> m_data_map;
  std::vector<std::string> m_system_order;
  std::vector<std::string> m_data_order;
  // std::vector<float> m_system_time;
  std::unordered_map<std::string, float> m_system_time;

  // Modules that manage callback, shared data pointers, and parameters
  // callback_handler_t m_callback_handler;
  // data_store_t m_shared_data;
  params_store m_params;

  // Information about commandline arguments
  std::unique_ptr<cxxopts::Options> m_options;
  std::unique_ptr<cxxopts::ParseResult> m_commandline_args;

  void parse_options(int argc, char** argv);

  // These are variables governing the lifetime of the simulation
  bool m_use_mpi = true;
  bool is_dry_run = false;
  double dt;
  double time;
  float step_time = 0.0f;
  int m_rank = 0;
  uint32_t step;
  uint32_t max_steps;
  uint32_t perf_interval;

  bool m_is_restart = false;
  std::string m_restart_file = "";
  std::function<void()> m_load_snapshot;

 public:
  typedef std::unordered_map<std::string, std::unique_ptr<data_t>> data_map_t;

  sim_environment_impl(bool use_mpi = true);
  sim_environment_impl(int* argc, char*** argv, bool use_mpi = true);
  ~sim_environment_impl();

  sim_environment_impl(const sim_environment_impl& other) = delete;
  sim_environment_impl(sim_environment_impl&& other) = delete;
  sim_environment_impl& operator=(const sim_environment_impl& other) = delete;
  sim_environment_impl& operator=(sim_environment_impl&& other) = delete;

  void reset(int* argc = nullptr, char*** argv = nullptr);
  ////////////////////////////////////////////////////////////////////////////////
  ///  Register a system class with the environment. This will either construct
  ///  a `unique_ptr` of the given `System` and insert it into the registry, or
  ///  return a reference to an existing one. The parameters are simply
  ///  constructor parameters. System names are given by their static `name()`
  ///  method. There cannot be more than one system registered under the same
  ///  name.
  ///
  ///  \tparam System  Type of the system to be registered. This template
  ///  parameter
  ///                 has to be specified.
  ///  \tparam Args    Types of the parameters to be sent to the constructor
  ///  \param args    The parameters to be sent to the system constructor
  ////////////////////////////////////////////////////////////////////////////////
  template <typename System, typename... Args>
  // auto register_system(Args&&... args) -> gsl::not_null<System*> {
  auto register_system(Args&&... args) -> nonown_ptr<System> {
    const std::string& name = System::name();
    // Check if the system has already been installed. If so, return it
    // directly
    auto it = m_system_map.find(name);
    if (it != m_system_map.end())
      return nonown_ptr<System>(dynamic_cast<System*>(it->second.get()));

    // Otherwise, make the system, and return the pointer
    std::unique_ptr<Aperture::system_t> ptr =
        std::make_unique<System>(std::forward<Args>(args)...);
    ptr->register_data_components();
    m_system_map.insert({name, std::move(ptr)});
    m_system_order.push_back(name);
    return nonown_ptr<System>(dynamic_cast<System*>(m_system_map[name].get()));
  }

  ////////////////////////////////////////////////////////////////////////////////
  ///  Register a data class with the environment. Multiple instances of the
  ///  same data type can be registered, as long as they have different names.
  ///
  ///  \tparam Data   Type of the data component to be registered. This template
  ///                 parameter has to be specified.
  ///  \tparam Args   Types of the parameters to be sent to the constructor
  ///  \param args    The parameters to be sent to the data constructor
  ////////////////////////////////////////////////////////////////////////////////
  template <typename Data, typename... Args>
  auto register_data(const std::string& name, Args&&... args)
      // -> gsl::not_null<Data*> {
      -> nonown_ptr<Data> {
    // Check if the data component has already been installed
    auto it = m_data_map.find(name);
    if (it != m_data_map.end())
      // return std::dynamic_pointer_cast<Data>(it->second);
      return nonown_ptr<Data>(dynamic_cast<Data*>(it->second.get()));

    // Otherwise, make the data, and return the pointer
    auto ptr = std::make_unique<Data>(std::forward<Args>(args)...);
    m_data_map.insert({name, std::move(ptr)});
    m_data_order.push_back(name);
    return nonown_ptr<Data>(dynamic_cast<Data*>(m_data_map[name].get()));
  }

  ////////////////////////////////////////////////////////////////////////////////
  ///  Obtain a reference to the named system. If the system is not found
  ///  then a `nullptr` is returned.
  ///
  ///  \param name  Name of the system.
  ///  \return A raw pointer to the system
  ////////////////////////////////////////////////////////////////////////////////
  nonown_ptr<system_t> get_system(const std::string& name) {
    auto it = m_system_map.find(name);
    if (it != m_system_map.end()) {
      return nonown_ptr<system_t>(it->second.get());
    } else {
      Logger::print_err("Failed to get system '{}'", name);
      return nullptr;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  ///  Obtain a const raw pointer to the named system. If the system is not
  ///  found then a `nullptr` is returned.
  ///
  ///  \param name  Name of the system.
  ///  \return A const raw pointer to the system
  ////////////////////////////////////////////////////////////////////////////////
  const nonown_ptr<system_t> get_system(const std::string& name) const {
    auto it = m_system_map.find(name);
    if (it != m_system_map.end()) {
      return nonown_ptr<system_t>(it->second.get());
    } else {
      Logger::print_err("Failed to get system '{}'", name);
      return nullptr;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  ///  Obtain an optional raw pointer to the named data component. If the data
  ///  component is not found then a `nullptr` is returned.
  ///
  ///  \param name  Name of the data component.
  ///  \return A raw pointer to the data component
  ////////////////////////////////////////////////////////////////////////////////
  nonown_ptr<data_t> get_data_optional(const std::string& name) {
    auto it = m_data_map.find(name);
    if (it != m_data_map.end()) {
      return nonown_ptr<data_t>(it->second.get());
    } else {
      Logger::print_info("Failed to get optional data component '{}'", name);
      return nullptr;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  ///  Obtain a raw pointer to the named data component. If the data component
  ///  is not found then an exception is thrown.
  ///
  ///  \param name  Name of the data component.
  ///  \return A raw pointer to the data component
  ////////////////////////////////////////////////////////////////////////////////
  nonown_ptr<data_t> get_data(const std::string& name) {
    auto it = m_data_map.find(name);
    if (it != m_data_map.end()) {
      return nonown_ptr<data_t>(it->second.get());
    } else {
      throw std::runtime_error("Data component not found: " + name);
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  ///  Obtain a raw pointer to the named data component. If the
  ///  data component is not found then an exception is thrown.
  ///
  ///  \param name  Name of the data component.
  ///  \param ptr   A raw pointer to the data component, supplied to be
  ///  the output
  ////////////////////////////////////////////////////////////////////////////////
  template <typename T>
  void get_data(const std::string& name, T** ptr) {
    auto it = m_data_map.find(name);
    if (it != m_data_map.end()) {
      // ptr = std::dynamic_pointer_cast<T>(it->second);
      *ptr = dynamic_cast<T*>(it->second.get());
    } else {
      throw std::runtime_error("Data component not found: " + name);
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  ///  Obtain a non-owning pointer to the named data component. If the
  ///  data component is not found then an exception is thrown.
  ///
  ///  \param name  Name of the data component.
  ///  \param ptr   A non-owning pointer to the data component, supplied to be
  ///  the output
  ////////////////////////////////////////////////////////////////////////////////
  template <typename T>
  void get_data(const std::string& name, nonown_ptr<T>& ptr) {
    auto it = m_data_map.find(name);
    if (it != m_data_map.end()) {
      // ptr = std::dynamic_pointer_cast<T>(it->second);
      ptr.reset(dynamic_cast<T*>(it->second.get()));
    } else {
      throw std::runtime_error("Data component not found: " + name);
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  ///  Obtain an optional raw pointer to the named data component. If the data
  ///  component is not found then a nullptr is returned.
  ///
  ///  \param name  Name of the data component.
  ///  \param ptr   A raw pointer to the data component, supplied to be
  ///  the output
  ////////////////////////////////////////////////////////////////////////////////
  template <typename T>
  void get_data_optional(const std::string& name, T** ptr) {
    auto it = m_data_map.find(name);
    if (it != m_data_map.end()) {
      *ptr = dynamic_cast<T*>(it->second.get());
    } else {
      Logger::print_info("Failed to get optional data component '{}', ignoring",
                         name);
      *ptr = nullptr;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  ///  Obtain an optional raw pointer to the named data component. If the data
  ///  component is not found then a nullptr is returned.
  ///
  ///  \param name  Name of the data component.
  ///  \param ptr   A raw pointer to the data component, supplied to be
  ///  the output
  ////////////////////////////////////////////////////////////////////////////////
  template <typename T>
  void get_data_optional(const std::string& name, nonown_ptr<T>& ptr) {
    auto it = m_data_map.find(name);
    if (it != m_data_map.end()) {
      ptr.reset(dynamic_cast<T*>(it->second.get()));
    } else {
      Logger::print_info("Failed to get optional data component '{}', ignoring",
                         name);
      ptr.reset(nullptr);
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  ///  Initialize all systems in order defined. This calls all the `init()`
  ///  functions of the systems one by one.
  ////////////////////////////////////////////////////////////////////////////////
  void init();

  ////////////////////////////////////////////////////////////////////////////////
  ///  Update all system by one timestep. This is the method called in the main
  ///  loop. One can call this directly in a main file to customize the
  ///  simulation loop.
  ////////////////////////////////////////////////////////////////////////////////
  void update();

  ////////////////////////////////////////////////////////////////////////////////
  ///  Start the main simulation loop. This enters a loop that calls the
  ///  `update()` function of each system in order at every time step.
  ////////////////////////////////////////////////////////////////////////////////
  void run();

  ////////////////////////////////////////////////////////////////////////////////
  ///  End the program and finalize MPI, among other things.
  ////////////////////////////////////////////////////////////////////////////////
  void end();

  // data_store_t& shared_data() { return m_shared_data; }
  // const data_store_t& shared_data() const { return m_shared_data; }
  // callback_handler_t& callback_handler() { return m_callback_handler; }
  // const callback_handler_t& callback_handler() const { return
  // m_callback_handler; }
  params_store& params() { return m_params; }
  const params_store& params() const { return m_params; }
  const cxxopts::ParseResult* commandline_args() const {
    return m_commandline_args.get();
  }

  bool use_mpi() const { return m_use_mpi; }
  int get_rank() const { return m_rank; }
  uint32_t get_step() const { return step; }
  uint32_t get_max_steps() const { return max_steps; }
  double get_time() const { return time; }
  uint32_t get_perf_interval() const { return perf_interval; }

  bool is_restart() const { return m_is_restart; }
  const std::string& restart_file() const { return m_restart_file; }
  void finish_restart() { m_is_restart = false; }

  void set_step(uint32_t s) { step = s; }
  void set_time(double t) { time = t; }
  const data_map_t& data_map() { return m_data_map; }
};

using sim_environment = singleton_holder<sim_environment_impl>;

inline sim_environment_impl&
sim_env(int* argc = nullptr, char*** argv = nullptr, bool use_mpi = true) {
  return sim_environment::instance(argc, argv, use_mpi);
}

}  // namespace Aperture
