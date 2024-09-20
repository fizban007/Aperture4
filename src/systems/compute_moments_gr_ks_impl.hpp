/*
 * Copyright (c) 2024 Alex Chen.
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

#include "compute_moments_gr_ks.h"
#include "data/fields.h"
#include "framework/environment.h"
#include "systems/physics/metric_kerr_schild.hpp"

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
void
compute_moments_gr_ks<Conf, ExecPolicy>::init() {
  compute_moments<Conf, ExecPolicy>::init();
}

template <typename Conf, template <class> class ExecPolicy>
void
compute_moments_gr_ks<Conf, ExecPolicy>::register_data_components() {
  compute_moments<Conf, ExecPolicy>::register_data_components();
}

template <typename Conf, template <class> class ExecPolicy>
void
compute_moments_gr_ks<Conf, ExecPolicy>::update(double dt, uint32_t step) {
  if (step % this->m_fld_interval == 0) {
    this->init_components();

    bool first = this->m_compute_first_moments;
    bool second = this->m_compute_second_moments;
    size_t num = this->ptc->number();
    value_t qe = this->m_qe;
    value_t a = m_grid.a;
    // int num_species = m_num_species;
    // First compute particle moments
    ExecPolicy<Conf>::launch(
        [first, second, num, qe, a] LAMBDA(auto ptc, auto ptc_num, auto ptc_flux,
                                        auto T00, auto T0i, auto T11, auto T12,
                                        auto T13, auto T22, auto T23, auto T33,
                                        auto grid_ptrs) {
          auto& grid = ExecPolicy<Conf>::grid();
          auto ext = grid.extent();
          ExecPolicy<Conf>::loop(0, num, [&] LAMBDA(auto n) {
            uint32_t cell = ptc.cell[n];
            if (cell == empty_cell) {
              return;
            }

            typename Conf::idx_t idx = Conf::idx(cell, ext);
            auto pos = get_pos(idx, ext);
            int sp = get_ptc_type(ptc.flag[n]);
            value_t r = grid_ks_t<Conf>::radius(grid.coord(0, pos[0], ptc.x1[n]));
            value_t th = grid_ks_t<Conf>::theta(grid.coord(1, pos[1], ptc.x2[n]));
            value_t alpha = Metric_KS::alpha(a, r, th);
            value_t inv_sqrt_g = grid.cell_size() / grid_ptrs.Ad[2][idx] / alpha;
            value_t u_0 = Metric_KS::u_0(a, r, th, {ptc.p1[n], ptc.p2[n], ptc.p3[n]});
            if (first) {
              atomic_add(&ptc_num[sp][idx], qe * ptc.weight[n] * inv_sqrt_g * u_0 / ptc.E[n]);
              atomic_add(&ptc_flux[sp][0][idx], qe * ptc.weight[n] * inv_sqrt_g * ptc.p1[n] / ptc.E[n]);
              atomic_add(&ptc_flux[sp][1][idx], qe * ptc.weight[n] * inv_sqrt_g * ptc.p2[n] / ptc.E[n]);
              atomic_add(&ptc_flux[sp][2][idx], qe * ptc.weight[n] * inv_sqrt_g * ptc.p3[n] / ptc.E[n]);
            }
            if (second) {
              // T00
              atomic_add(&T00[sp][idx], qe * ptc.weight[n] * inv_sqrt_g * u_0 * u_0 / ptc.E[n]);
              // T0i
              atomic_add(&T0i[sp][0][idx], qe * ptc.weight[n] * inv_sqrt_g * u_0 * ptc.p1[n] / ptc.E[n]);
              atomic_add(&T0i[sp][1][idx], qe * ptc.weight[n] * inv_sqrt_g * u_0 * ptc.p2[n] / ptc.E[n]);
              atomic_add(&T0i[sp][2][idx], qe * ptc.weight[n] * inv_sqrt_g * u_0 * ptc.p3[n] / ptc.E[n]);
              // Tii
              atomic_add(&T11[sp][idx],
                         qe * ptc.weight[n] * inv_sqrt_g * ptc.p1[n] * ptc.p1[n] / ptc.E[n]);
              atomic_add(&T22[sp][idx],
                         qe * ptc.weight[n] * inv_sqrt_g * ptc.p2[n] * ptc.p2[n] / ptc.E[n]);
              atomic_add(&T33[sp][idx],
                         qe * ptc.weight[n] * inv_sqrt_g * ptc.p3[n] * ptc.p3[n] / ptc.E[n]);
              // Tij
              atomic_add(&T23[sp][idx],
                         qe * ptc.weight[n] * inv_sqrt_g * ptc.p2[n] * ptc.p3[n] / ptc.E[n]);
              atomic_add(&T13[sp][idx],
                         qe * ptc.weight[n] * inv_sqrt_g * ptc.p3[n] * ptc.p1[n] / ptc.E[n]);
              atomic_add(&T12[sp][idx],
                         qe * ptc.weight[n] * inv_sqrt_g * ptc.p1[n] * ptc.p2[n] / ptc.E[n]);
            }
          });
        },
        *(this->ptc), this->ptc_num, this->ptc_flux, this->T00, this->T0i,
        this->T11, this->T12, this->T13, this->T22, this->T23, this->T33,
        m_grid.get_grid_ptrs());
    ExecPolicy<Conf>::sync();
    // Then compute photon moments if applicable
    if (this->m_photon_data) {
      size_t ph_num = this->ph->number();
      if (ph_num > 0) {
        auto size = this->m_size;
        ExecPolicy<Conf>::launch(
            [first, second, ph_num, qe, a] LAMBDA(auto ph, auto ptc_num, auto ptc_flux,
                                            auto T00, auto T0i, auto T11,
                                            auto T12, auto T13, auto T22,
                                            auto T23, auto T33, auto grid_ptrs) {
              auto& grid = ExecPolicy<Conf>::grid();
              auto ext = grid.extent();
              ExecPolicy<Conf>::loop(0, ph_num, [&] LAMBDA(auto n) {
                uint32_t cell = ph.cell[n];
                if (cell == empty_cell) {
                  return;
                }

                typename Conf::idx_t idx = Conf::idx(cell, ext);
                auto pos = get_pos(idx, ext);
                value_t r = grid_ks_t<Conf>::radius(grid.coord(0, pos[0], ph.x1[n]));
                value_t th = grid_ks_t<Conf>::theta(grid.coord(1, pos[1], ph.x2[n]));
                value_t alpha = Metric_KS::alpha(a, r, th);
                value_t inv_sqrt_g = grid.cell_size() / grid_ptrs.Ad[2][idx] / alpha;
                // value_t inv_sqrt_g = grid.cell_size() / grid_ptrs.Ad[2][idx];
                value_t u_0 = Metric_KS::u_0(a, r, th, {ph.p1[n], ph.p2[n], ph.p3[n]}, true);
                if (first) {
                  atomic_add(&ptc_num[idx], qe * ph.weight[n] * inv_sqrt_g * u_0 / ph.E[n]);
                  atomic_add(&ptc_flux[0][idx], qe * ph.weight[n] * inv_sqrt_g * ph.p1[n] / ph.E[n]);
                  atomic_add(&ptc_flux[1][idx], qe * ph.weight[n] * inv_sqrt_g * ph.p2[n] / ph.E[n]);
                  atomic_add(&ptc_flux[2][idx], qe * ph.weight[n] * inv_sqrt_g * ph.p3[n] / ph.E[n]);
                }
                if (second) {
                  // T00
                  atomic_add(&T00[idx], qe * ph.weight[n] * inv_sqrt_g * u_0 * u_0 / ph.E[n]);
                  // T0i
                  atomic_add(&T0i[0][idx], qe * ph.weight[n] * inv_sqrt_g * ph.p1[n] * u_0 / ph.E[n]);
                  atomic_add(&T0i[1][idx], qe * ph.weight[n] * inv_sqrt_g * ph.p2[n] * u_0 / ph.E[n]);
                  atomic_add(&T0i[2][idx], qe * ph.weight[n] * inv_sqrt_g * ph.p3[n] * u_0 / ph.E[n]);
                  // Tii
                  atomic_add(&T11[idx], qe * ph.weight[n] * inv_sqrt_g * ph.p1[n] * ph.p1[n] / ph.E[n]);
                  atomic_add(&T22[idx], qe * ph.weight[n] * inv_sqrt_g * ph.p2[n] * ph.p2[n] / ph.E[n]);
                  atomic_add(&T33[idx], qe * ph.weight[n] * inv_sqrt_g * ph.p3[n] * ph.p3[n] / ph.E[n]);
                  // Tij
                  atomic_add(&T23[idx], qe * ph.weight[n] * inv_sqrt_g * ph.p2[n] * ph.p3[n] / ph.E[n]);
                  atomic_add(&T13[idx], qe * ph.weight[n] * inv_sqrt_g * ph.p3[n] * ph.p1[n] / ph.E[n]);
                  atomic_add(&T12[idx], qe * ph.weight[n] * inv_sqrt_g * ph.p1[n] * ph.p2[n] / ph.E[n]);
                }
              });
            },
            *(this->ph), this->ptc_num[size - 1], this->ptc_flux[size - 1], this->T00[size - 1],
            this->T0i[size - 1], this->T11[size - 1], this->T12[size - 1], this->T13[size - 1],
            this->T22[size - 1], this->T23[size - 1], this->T33[size - 1], m_grid.get_grid_ptrs());
        ExecPolicy<Conf>::sync();
      }
    }
  }
}

}  // namespace Aperture

