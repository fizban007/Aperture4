#include "environment.hpp"
#include "cxxopts.hpp"
#include <iostream>
#include <mpi.h>

namespace Aperture {

sim_environment::sim_environment(int* argc, char*** argv) {
  // Parse options
  m_options = std::unique_ptr<cxxopts::Options>(
      new cxxopts::Options("aperture", "Aperture PIC code"));
  m_options->add_options()("h,help", "Prints this help message.")(
      "c,config", "Configuration file for the simulation.",
      cxxopts::value<std::string>()->default_value("config.toml"));
  // ("r,restart_file", "The restart file used in this run.",
  // cxxopts::value<std::string>()->default_value(""));

  int is_initialized = 0;
  MPI_Initialized(&is_initialized);

  // RANGE_PUSH("Initialization", CLR_BLUE);
  if (!is_initialized) {
    if (argc == nullptr && argv == nullptr) {
      MPI_Init(NULL, NULL);
    } else {
      MPI_Init(argc, argv);
    }
  }

  // store the processed argc and argv in memory for later use
  m_argc = argc;
  m_argv = argv;
}

sim_environment::~sim_environment() {
  destroy();

  int is_finalized = 0;
  MPI_Finalized(&is_finalized);

  if (!is_finalized) MPI_Finalize();
}

void
sim_environment::parse_options() {
  // Read command line arguments
  try {
    m_commandline_args.reset(
        new cxxopts::ParseResult(m_options->parse(*m_argc, *m_argv)));
    auto& result = *m_commandline_args;

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
sim_environment::destroy() {
  for (auto t = m_system_order.rbegin(); t != m_system_order.rend();
       ++t) {
    m_system_map[*t]->destroy();
  }
}

void
sim_environment::run() {
  uint32_t max_steps = m_params.get<int64_t>("max_steps", 1l);
  for (uint32_t step = 0; step < 1; step++) {
    Logger::print_info("=== Time step {}", step);
    for (auto& name : m_system_order) {
      m_system_map[name]->update(0.1, step);
    }
  }
}

}  // namespace Aperture
