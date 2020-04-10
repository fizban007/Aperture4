#ifndef __ENVIRONMENT_H_
#define __ENVIRONMENT_H_

#include "data.h"
#include "system.h"
#include "utils/logger.h"
#include <any>
#include <map>
#include <memory>
#include <string>

namespace Aperture {

class event_handler_t {
 public:
  template <typename Func>
  void push_callback(const std::string& name, const Func& func) {
    event_map[name].push_back(func);
  }

  template <typename... Args>
  void invoke_callback(const std::string& name, Args&... args) const {
    auto it = event_map.find(name);
    if (it == event_map.end()) {
      return;
    } else {
      for (auto& any_p : it->second) {
        try {
          auto f = std::any_cast<std::function<void(Args&...)>>(any_p);
          f(args...);
        } catch (const std::bad_any_cast& e) {
          Logger::print_err("Failed to find callback '{}': {}", name, e.what());
        }
      }
    }
  }

 private:
  std::unordered_map<std::string, std::vector<std::any>>
      event_map;
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
      if (any_p.has_value()) {
        const T* ptr = nullptr;
        try {
          ptr = std::any_cast<const T*>(any_p);
        } catch (const std::bad_any_cast& e) {
          Logger::print_err("Failed to find shared_data '{}': {}", name, e.what());
        }
        return ptr;
      }
    }
    return nullptr;
  }

 private:
  std::unordered_map<std::string, std::any> store_map;
};

class sim_environment {
 private:
  std::unordered_map<std::string, std::shared_ptr<data_t>>
      m_data_map;
  std::unordered_map<std::string, std::shared_ptr<system_t>>
      m_system_map;
  std::vector<std::string> m_system_order;
  std::vector<std::string> m_data_order;

  event_handler_t m_event_handler;
  data_store_t m_shared_data;

 public:
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
    m_data_map.insert({name, std::make_shared<Data>(
                                      std::forward<Args>(args)...)});
    m_data_order.push_back(name);
  }

  template <template <typename...> typename Data, typename Conf,
            typename... Args>
  void register_data(const Conf& conf, const std::string& name,
                     Args&&... args) {
    // Check if the data component has already been installed
    if (m_data_map.find(name) != m_data_map.end()) return;
    m_data_map.insert(
        {name, std::make_shared<Data<Conf, Args...>>(
                   conf, std::forward<Args>(args)...)});
    m_data_order.push_back(name);
  }

  std::shared_ptr<system_t> get_system(const std::string& name) {
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

  data_store_t& shared_data() { return m_shared_data; }
  event_handler_t& event_handler() { return m_event_handler; }

  void init();
  void destroy();
  void run();
};

}  // namespace Aperture

#endif
