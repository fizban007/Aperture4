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
ptc_updater_gr_ks_cu<Conf>::update_particles(double dt, uint32_t step) {
  value_t a = m_a;
  auto ptc_num = this->ptc->number();

  if (ptc_num > 0) {
    auto ptc_kernel = [a, ptc_num, dt, step] __device__(
                          auto ptc, auto B, auto D, auto J, auto Rho,
                          auto rho_interval) {
      auto &grid = dev_grid<Conf::dim>();
      auto ext = grid.extent();


    };

    kernel_launch(ptc_kernel, this->ptc->get_dev_ptrs(),
                  this->B->get_const_ptrs(), this->E->get_const_ptrs(),
                  this->J->get_ptrs(), this->m_rho_ptrs.dev_ptr(),
                  this->m_rho_interval);
  }
}

template <typename Conf>
void
ptc_updater_gr_ks_cu<Conf>::move_photons_2d(value_t dt, uint32_t step) {
  value_t a = m_a;
  auto ph_num = this->ph->number();

  if (ph_num > 0) {
    auto photon_kernel = [a, ph_num, dt, step] __device__(auto ph, auto rho_ph,
                                                          auto data_interval) {
      auto& grid = dev_grid<Conf::dim>();
      auto ext = grid.extent();

      for (size_t n : grid_stride_range(0, ph_num)) {
        uint32_t cell = ph.cell[n];
        if (cell == empty_cell) continue;
      }
    };

    kernel_launch(photon_kernel, this->ph->get_dev_ptrs(),
                  this->rho_ph->dev_ndptr(), this->m_data_interval);
  }
}

template <typename Conf>
void
ptc_updater_gr_ks_cu<Conf>::fill_multiplicity(int mult, value_t weight) {}

template class ptc_updater_gr_ks_cu<Config<2>>;

}  // namespace Aperture
