#ifndef __ENVIRONMENT_H_
#define __ENVIRONMENT_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include "data.h"
#include "framework/params_store.hpp"
#include "system.h"
#include "utils/logger.h"
#include <boost/any.hpp>

namespace cxxopts {
class Options;
class ParseResult;
}  // namespace cxxopts

namespace Aperture {

class event_handler_t {
 public:
  template <typename Func>
  void register_callback(const std::string& name, const Func& func) {
    // event_map[name].push_back(func);
    event_map.insert({name, func});
  }

  template <typename... Args>
  void invoke_callback(const std::string& name, Args&... args) const {
    auto it = event_map.find(name);
    if (it == event_map.end()) {
      Logger::print_err("Failed to find callback '{}'", name);
      return;
    } else {
      auto& any_p = it->second;
      std::function<void(Args & ...)> f;
      try {
        f = boost::any_cast<std::function<void(Args & ...)>>(any_p);
      } catch (const boost::bad_any_cast& e) {
        Logger::print_err("Failed to cast callback '{}': {}", name, e.what());
        return;
      }
      f(args...);
    }
  }

 private:
  // std::unordered_map<std::string, std::vector<std::any>> event_map;
  std::unordered_map<std::string, boost::any> event_map;
};

class data_store_t {
 public:
  template <typename Data>
  void save(const std::string& name, const Data& data) {
    store_map[name] = &data;
  }

  template <typename T>
  const T* get(const std::string& name) const {
    auto it = store_map.find(name);
    if (it != store_map.end()) {
      auto& any_p = it->second;
      if (any_p != nullptr) {
        return reinterpret_cast<const T*>(any_p);
      }
    }
    return nullptr;
  }

 private:
  std::unordered_map<std::string, const void*> store_map;
};

class sim_environment {
 private:
  // Registry for systems and data
  std::unordered_map<std::string, std::shared_ptr<data_t>> m_data_map;
  std::unordered_map<std::string, std::shared_ptr<system_t>> m_system_map;
  std::vector<std::string> m_system_order;
  std::vector<std::string> m_data_order;

  // Modules that manage events, shared pointers, and parameters
  event_handler_t m_event_handler;
  data_store_t m_shared_data;
  params_store m_params;

  // Information about commandline arguments
  std::unique_ptr<cxxopts::Options> m_options;
  std::unique_ptr<cxxopts::ParseResult> m_commandline_args;
  int* m_argc;
  char*** m_argv;

  // Data related to dependency resolution
  std::vector<std::string> m_init_order;
  std::set<std::string> m_unresolved;

  void resolve_dependencies(const system_t& system, const std::string& name);

 public:
  sim_environment();
  sim_environment(int* argc, char*** argv);
  ~sim_environment();

  template <typename System, typename... Args>
  void register_system(Args&&... args) {
    const std::string& name = System::name();
    // Check if the system has already been installed
    if (m_system_map.find(name) != m_system_map.end()) return;
    m_system_map.insert(
        {name, std::make_shared<System>(std::forward<Args>(args)...)});
    m_system_map[name]->register_dependencies(*this);
    m_system_order.push_back(name);
  }

  template <template <typename...> typename System, typename Conf,
            typename... Args>
  void register_system(const Conf& conf, Args&&... args) {
    const std::string& name = System<Conf, Args...>::name();
    // Check if the system has already been installed
    if (m_system_map.find(name) != m_system_map.end()) return;
    m_system_map.insert({name, std::make_shared<System<Conf, Args...>>(
                                   conf, std::forward<Args>(args)...)});
    m_system_map[name]->register_dependencies(*this);
    m_system_order.push_back(name);
  }

  template <typename Data, typename... Args>
  void register_data(const std::string& name, Args&&... args) {
    // Check if the data component has already been installed
    if (m_data_map.find(name) != m_data_map.end()) return;
    m_data_map.insert(
        {name, std::make_shared<Data>(std::forward<Args>(args)...)});
    m_data_order.push_back(name);
  }

  template <template <typename...> typename Data, typename Conf,
            typename... Args>
  void register_data(const Conf& conf, const std::string& name,
                     Args&&... args) {
    // Check if the data component has already been installed
    if (m_data_map.find(name) != m_data_map.end()) return;
    m_data_map.insert({name, std::make_shared<Data<Conf>>(
                                 conf, std::forward<Args>(args)...)});
    m_data_order.push_back(name);
  }

  void parse_options();

  std::shared_ptr<system_t> get_system(const std::string& name) {
    auto it = m_system_map.find(name);
    if (it != m_system_map.end()) {
      return it->second;
    } else {
      Logger::print_err("Failed to get system '{}'", name);
      return nullptr;
    }
  }

  std::shared_ptr<const system_t> get_system(const std::string& name) const {
    auto it = m_system_map.find(name);
    if (it != m_system_map.end()) {
      return it->second;
    } else {
      Logger::print_err("Failed to get system '{}'", name);
      return nullptr;
    }
  }

  std::shared_ptr<data_t> get_data(const std::string& name) {
    auto it = m_data_map.find(name);
    if (it != m_data_map.end()) {
      return it->second;
    } else {
      Logger::print_err("Failed to get data component '{}'", name);
      return nullptr;
    }
  }

  template <typename T>
  void get_data(const std::string& name, std::shared_ptr<T>& ptr) {
    auto it = m_data_map.find(name);
    if (it != m_data_map.end()) {
      ptr = std::dynamic_pointer_cast<T>(it->second);
    } else {
      Logger::print_err("Failed to get data component '{}'", name);
      ptr = nullptr;
    }
  }

  data_store_t& shared_data() { return m_shared_data; }
  const data_store_t& shared_data() const { return m_shared_data; }
  event_handler_t& event_handler() { return m_event_handler; }
  const event_handler_t& event_handler() const { return m_event_handler; }
  params_store& params() { return m_params; }
  const params_store& params() const { return m_params; }
  const cxxopts::ParseResult& commandline_args() const {
    return *m_commandline_args;
  }

  void init();
  void destroy();
  void run();
};

}  // namespace Aperture

#endif
