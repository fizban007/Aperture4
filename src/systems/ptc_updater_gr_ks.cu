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

#include "core/math.hpp"
#include "data/curand_states.h"
#include "framework/config.h"
#include "helpers/ptc_update_helper.hpp"
#include "ptc_updater_gr_ks.h"
#include "utils/kernel_helper.hpp"
#include "utils/timer.h"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf>
ptc_updater_gr_ks_cu<Conf>::ptc_updater_gr_ks_cu(sim_environment& env,
                                                 const grid_ks_t<Conf>& grid,
                                                 const domain_comm<Conf>* comm)
    : ptc_updater_cu<Conf>(env, grid, comm), m_ks_grid(grid) {}

template <typename Conf>
void
ptc_updater_gr_ks_cu<Conf>::init() {
  ptc_updater_cu<Conf>::init();

  this->m_env.params().get_value("bh_spin", m_a);
  this->m_env.params().get_value("damping_length", m_damping_length);
}

template <typename Conf>
void
ptc_updater_gr_ks_cu<Conf>::register_data_components() {
  ptc_updater_cu<Conf>::register_data_components();
}

template <typename Conf>
void
ptc_updater_gr_ks_cu<Conf>::update(double dt, uint32_t step) {
  Logger::print_info("Pushing {} particles", this->ptc->number());
  timer::stamp("pusher");
  // First update particle momentum
  // push_default(dt);
  timer::show_duration_since_stamp("push", "ms", "pusher");

  timer::stamp("depositer");
  // Then move particles and deposit current
  // move_and_deposit(dt, step);
  timer::show_duration_since_stamp("deposit", "ms", "depositer");

  // Communicate deposited current and charge densities
  if (this->m_comm != nullptr) {
    this->m_comm->send_add_guard_cells(*(this->J));
    this->m_comm->send_guard_cells(*(this->J));
    // if ((step + 1) % m_data_interval == 0) {
    if (step % this->m_rho_interval == 0) {
      for (uint32_t i = 0; i < this->Rho.size(); i++) {
        this->m_comm->send_add_guard_cells(*(this->Rho[i]));
        this->m_comm->send_guard_cells(*(this->Rho[i]));
      }
    }
  }

  timer::stamp("filter");
  // Filter current
  // filter_current(m_filter_times, step);
  timer::show_duration_since_stamp("filter", "ms", "filter");

  // Send particles
  if (this->m_comm != nullptr) {
    this->m_comm->send_particles(*(this->ptc), this->m_grid);
  }

  // Also move photons if the data component exists
  if (this->ph != nullptr) {
    Logger::print_info("Moving {} photons", this->ph->number());
    // move_photons(dt, step);

    if (this->m_comm != nullptr) {
      this->m_comm->send_particles(*(this->ph), this->m_grid);
    }
  }

  // Clear guard cells
  this->clear_guard_cells();

  // sort at the given interval. Turn off sorting if m_sort_interval is 0
  if (this->m_sort_interval > 0 && (step % this->m_sort_interval) == 0) {
    this->sort_particles();
  }
}

template class ptc_updater_gr_ks_cu<Config<2>>;

}  // namespace Aperture
