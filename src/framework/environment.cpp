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

#include "environment.h"
#include "cxxopts.hpp"
#include "utils/timer.h"
#include <algorithm>
#include <iostream>
#include <mpi.h>

namespace Aperture {

sim_environment_impl::sim_environment_impl() : sim_environment_impl(nullptr, nullptr) {}

sim_environment_impl::sim_environment_impl(int* argc, char*** argv) {
  // Parse options
  m_options = std::unique_ptr<cxxopts::Options>(
      new cxxopts::Options("aperture", "Aperture PIC code"));
  m_options->add_options()("h,help", "Prints this help message.")(
      "c,config", "Configuration file for the simulation.",
      cxxopts::value<std::string>()->default_value("config.toml"))(
      "d,dry-run",
      "Only initialize, do not actualy run the simulation. Useful for looking "
      "at initialization stage problems.");

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
  m_params.parse(m_params.get_as<std::string>("config_file", "config.toml"));
}

sim_environment_impl::~sim_environment_impl() {
  int is_finalized = 0;
  MPI_Finalized(&is_finalized);

  if (!is_finalized) MPI_Finalize();
}

void
sim_environment_impl::parse_options(int argc, char** argv) {
  // Read command line arguments
  try {
    m_commandline_args.reset(
        new cxxopts::ParseResult(m_options->parse(argc, argv)));
    auto& result = *m_commandline_args;

    if (result["help"].as<bool>()) {
      std::cout << m_options->help() << std::endl;
      exit(0);
    }
    if (result["dry-run"].as<bool>()) {
      is_dry_run = true;
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
sim_environment_impl::init() {
  // Set log level independent of domain comm
  int log_level = (int)LogLevel::info;
  m_params.get_value("log_level", log_level);
  Logger::set_log_level((LogLevel)log_level);

  step = 0;
  time = 0.0;
  max_steps = m_params.get_as<int64_t>("max_steps", 1l);
  dt = m_params.get_as<double>("dt", 0.01);
  perf_interval = m_params.get_as<int64_t>("perf_interval", 10);

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
sim_environment_impl::update() {
  Logger::print_info("=== Time step {}, Time is {:.5f} ===", step, time);
  for (auto& name : m_system_order) {
    timer::stamp();
    m_system_map[name]->update(dt, step);
    float time_spent = timer::get_duration_since_stamp("us");
    if (step % perf_interval == 0 && time_spent > 10.0f)
      Logger::print_info("Time for {} is {:.2f}ms", name,
                         time_spent / 1000.0);
    // timer::show_duration_since_stamp(name, "us");
  }
  time += dt;
  step += 1;
}

void
sim_environment_impl::run() {
  if (is_dry_run) {
    Logger::print_info("This is a dry-run, exiting...");
    return;
  }

  Logger::print_debug("Max steps is: {}", max_steps);
  for (int n = 0; n < max_steps; n++) {
    update();
  }
}

}  // namespace Aperture
