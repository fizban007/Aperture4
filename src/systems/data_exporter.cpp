#include "data_exporter.h"
#include "framework/environment.hpp"

namespace Aperture {

data_exporter::data_exporter(sim_environment& env) :
    system_t(env) {}

data_exporter::~data_exporter() {}

void
data_exporter::init() {
  m_env.params().get_value("data_interval", m_data_interval);
  m_env.params().get_value("snapshot_interval", m_snapshot_interval);
}

void
data_exporter::update(double time, uint32_t step) {
  if (step % m_data_interval == 0) {
    // Do data output!
    for (auto& it : m_env.data_map()) {
      Logger::print_info("Working on {}", it.first);
    }
  }

  if (step % m_snapshot_interval == 0) {
    // Take snapshot!
  }
}

}
