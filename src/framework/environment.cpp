#include "environment.hpp"
#include "cxxopts.hpp"
#include <algorithm>
#include <iostream>
#include <mpi.h>

namespace Aperture {

sim_environment::sim_environment()
    : sim_environment(nullptr, nullptr) {}

sim_environment::sim_environment(int* argc, char*** argv) {
  // Parse options
  m_options = std::unique_ptr<cxxopts::Options>(
      new cxxopts::Options("aperture", "Aperture PIC code"));
  m_options->add_options()("h,help", "Prints this help message.")(
      "c,config", "Configuration file for the simulation.",
      cxxopts::value<std::string>()->default_value("config.toml"));

  int is_initialized = 0;
  MPI_Initialized(&is_initialized);

  if (!is_initialized) {
    if (argc == nullptr && argv == nullptr) {
      MPI_Init(NULL, NULL);
    } else {
      MPI_Init(argc, argv);
    }
  }

  // Parse options and store the results
  if (argc != nullptr && argv != nullptr) {
    parse_options(*argc, *argv);
  } else {
    m_commandline_args = nullptr;
  }
  m_params.parse(
      m_params.get<std::string>("config_file", "config.toml"));
}

sim_environment::~sim_environment() {
  int is_finalized = 0;
  MPI_Finalized(&is_finalized);

  if (!is_finalized) MPI_Finalize();
}

void
sim_environment::parse_options(int argc, char** argv) {
  // Read command line arguments
  try {
    m_commandline_args.reset(
        new cxxopts::ParseResult(m_options->parse(argc, argv)));
    auto& result = *m_commandline_args;
    m_shared_data.save("commandline_args", result);

    if (result["help"].as<bool>()) {
      std::cout << m_options->help() << std::endl;
      exit(0);
    }
    auto conf_file = result["config"].as<std::string>();
    m_params.add("config_file", conf_file);

  } catch (std::exception& e) {
    Logger::print_err("Error: {}", e.what());
    std::cout << m_options->help() << std::endl;
    exit(1);
  }
}

void
sim_environment::init() {
  // First resolve initialization order
  // for (auto& it : m_system_map) {
  //   resolve_dependencies(*it.second, it.first);
  // }

  // Initialize systems following declaration order
  for (auto& name : m_system_order) {
    auto& s = m_system_map[name];
    Logger::print_info("Initializing system '{}'", name);
    s->init();
  }

  // Initialize all data
  for (auto name : m_data_order) {
    auto& c = m_data_map[name];
    Logger::print_info("Initializing data '{}'", name);
    c->init();
  }
}

void
sim_environment::run() {
  uint32_t max_steps = m_params.get<int64_t>("max_steps", 1l);
  double dt = m_params.get<double>("dt", 0.01);

  for (uint32_t step = 0; step < max_steps; step++) {
    Logger::print_info("=== Time step {}", step);
    for (auto& name : m_system_order) {
      m_system_map[name]->update(dt, step);
    }
  }
}

void
sim_environment::resolve_dependencies(const system_t& system,
                                      const std::string& name) {
  m_unresolved.insert(name);
  for (auto& n : system.dependencies()) {
    if (std::find(m_init_order.begin(), m_init_order.end(), n) ==
        m_init_order.end()) {
      if (m_unresolved.count(n) > 0)
        throw std::runtime_error(
            "Circular dependency in the given list of systems!");
      if (m_system_map.find(n) != m_system_map.end())
        resolve_dependencies(*m_system_map[n], n);
    }
  }
  if (std::find(m_init_order.begin(), m_init_order.end(), name) ==
      m_init_order.end())
    m_init_order.push_back(name);
  m_unresolved.erase(name);
}

}  // namespace Aperture
