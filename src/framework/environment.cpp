#include "environment.hpp"

namespace Aperture {

sim_environment::sim_environment(int* argc, char*** argv) {}

sim_environment::~sim_environment() {
  destroy();
}

void
sim_environment::init() {
  // for (auto& c : m_data_map) {
  for (auto name : m_data_order) {
    auto& c = m_data_map[name];
    Logger::print_info("Initializing data '{}'", name);
    c->init(*this);
  }
  for (auto& name : m_system_order) {
    auto& s = m_system_map[name];
    Logger::print_info("Initializing system '{}'", name);
    s->init_system(*this);
  }
}

void
sim_environment::destroy()  {
  for (auto t = m_system_order.rbegin(); t != m_system_order.rend(); ++t) {
    m_system_map[*t]->destroy();
  }
}

void
sim_environment::run() {
  for (int i = 0; i < 1; i++) {
    Logger::print_info("=== Time step {}", i);
    for (auto& name : m_system_order) {
      m_system_map[name]->update(0.1);
    }
  }
}

}  // namespace Aperture
