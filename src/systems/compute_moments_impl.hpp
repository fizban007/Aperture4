/*
 * Copyright (c) 2022 Alex Chen.
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

#include "compute_moments.h"
#include "framework/environment.h"

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
void
compute_moments<Conf, ExecPolicy>::init() {
  sim_env().get_data("particles", ptc);
  sim_env().get_data_optional("photons", ph);
  if (ph != nullptr) {
    m_photon_data = true;
  }

  sim_env().params().get_value("compute_first_moments",
                               m_compute_first_moments);
  sim_env().params().get_value("compute_second_moments",
                               m_compute_second_moments);
  sim_env().params().get_value("fld_output_interval", m_fld_interval);
  sim_env().params().get_value("num_species", m_num_species);
  m_size = m_num_species;
  if (m_photon_data) {
    m_size += 1;
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
compute_moments<Conf, ExecPolicy>::register_data_components() {
  if (m_compute_first_moments) {
    S.resize(m_size);
    for (int n = 0; n < m_size; n++) {
      if (n < m_num_species) {
        S.set(n, sim_env().register_data<field_t<4, Conf>>(
                     std::string("flux_") + ptc_type_name(n), m_grid,
                     ExecPolicy<Conf>::data_mem_type()));
      } else {
        S.set(n, sim_env().register_data<field_t<4, Conf>>(
                     std::string("flux_ph"), m_grid,
                     ExecPolicy<Conf>::data_mem_type()));
      }
    }
    S.copy_to_device();
  }
  if (m_compute_second_moments) {
    T.resize(m_size);
    for (int n = 0; n < m_size; n++) {
      if (n < m_num_species) {
        T.set(n, sim_env().register_data<field_t<10, Conf>>(
                     std::string("stress_") + ptc_type_name(n), m_grid,
                     ExecPolicy<Conf>::data_mem_type()));
      } else {
        T.set(n, sim_env().register_data<field_t<10, Conf>>(
                     std::string("stress_ph"), m_grid,
                     ExecPolicy<Conf>::data_mem_type()));
      }
    }
    T.copy_to_device();
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
compute_moments<Conf, ExecPolicy>::update(double dt, uint32_t step) {
  if (step % m_fld_interval == 0) {
    for (int n = 0; n < m_size; n++) {
      if (m_compute_first_moments) {
        S.init();
      }
      if (m_compute_second_moments) {
        T.init();
      }
    }

    bool first_moments = m_compute_first_moments;
    bool second_moments = m_compute_second_moments;
    size_t num = ptc->number();
    int num_species = m_num_species;
    // First compute particle moments
    ExecPolicy<Conf>::launch(
        [first_moments, second_moments, num] LAMBDA(auto ptc, auto S, auto T) {
          auto& grid = ExecPolicy<Conf>::grid();
          auto ext = grid.extent();
          ExecPolicy<Conf>::loop(0, num, [&] LAMBDA(auto n) {
            uint32_t cell = ptc.cell[n];
            if (cell == empty_cell) {
              return;
            }

            typename Conf::idx_t idx = Conf::idx(cell, ext);
            int sp = get_ptc_type(ptc.flag[n]);
            if (first_moments) {
              atomic_add(&S[sp][0][idx], ptc.weight[n]);
              atomic_add(&S[sp][1][idx], ptc.weight[n] * ptc.p1[n] / ptc.E[n]);
              atomic_add(&S[sp][2][idx], ptc.weight[n] * ptc.p2[n] / ptc.E[n]);
              atomic_add(&S[sp][3][idx], ptc.weight[n] * ptc.p3[n] / ptc.E[n]);
            }
            if (second_moments) {
              // T00
              atomic_add(&T[sp][0][idx], ptc.weight[n] * ptc.E[n]);
              // T0i
              atomic_add(&T[sp][1][idx], ptc.weight[n] * ptc.p1[n]);
              atomic_add(&T[sp][2][idx], ptc.weight[n] * ptc.p2[n]);
              atomic_add(&T[sp][3][idx], ptc.weight[n] * ptc.p3[n]);
              // Tii
              atomic_add(&T[sp][4][idx],
                         ptc.weight[n] * ptc.p1[n] * ptc.p1[n] / ptc.E[n]);
              atomic_add(&T[sp][5][idx],
                         ptc.weight[n] * ptc.p2[n] * ptc.p2[n] / ptc.E[n]);
              atomic_add(&T[sp][6][idx],
                         ptc.weight[n] * ptc.p3[n] * ptc.p3[n] / ptc.E[n]);
              // Tij
              atomic_add(&T[sp][7][idx],
                         ptc.weight[n] * ptc.p2[n] * ptc.p3[n] / ptc.E[n]);
              atomic_add(&T[sp][8][idx],
                         ptc.weight[n] * ptc.p3[n] * ptc.p1[n] / ptc.E[n]);
              atomic_add(&T[sp][9][idx],
                         ptc.weight[n] * ptc.p1[n] * ptc.p2[n] / ptc.E[n]);
            }
          });
        },
        ptc, S, T);
    ExecPolicy<Conf>::sync();
    // Then compute photon moments if applicable
    if (m_photon_data) {
      size_t ph_num = ph->number();
      ExecPolicy<Conf>::launch(
          [first_moments, second_moments, ph_num] LAMBDA(auto ph, auto S,
                                                         auto T) {
            auto& grid = ExecPolicy<Conf>::grid();
            auto ext = grid.extent();
            ExecPolicy<Conf>::loop(0, ph_num, [&] LAMBDA(auto n) {
              uint32_t cell = ph.cell[n];
              if (cell == empty_cell) {
                return;
              }

              typename Conf::idx_t idx = Conf::idx(cell, ext);
              if (first_moments) {
                atomic_add(&S[0][idx], ph.weight[n]);
                atomic_add(&S[1][idx], ph.weight[n] * ph.p1[n] / ph.E[n]);
                atomic_add(&S[2][idx], ph.weight[n] * ph.p2[n] / ph.E[n]);
                atomic_add(&S[3][idx], ph.weight[n] * ph.p3[n] / ph.E[n]);
              }
              if (second_moments) {
                // T00
                atomic_add(&T[0][idx], ph.weight[n] * ph.E[n]);
                // T0i
                atomic_add(&T[1][idx], ph.weight[n] * ph.p1[n]);
                atomic_add(&T[2][idx], ph.weight[n] * ph.p2[n]);
                atomic_add(&T[3][idx], ph.weight[n] * ph.p3[n]);
                // Tii
                atomic_add(&T[4][idx], ph.weight[n] * ph.p1[n] * ph.p1[n] / ph.E[n]);
                atomic_add(&T[5][idx], ph.weight[n] * ph.p2[n] * ph.p2[n] / ph.E[n]);
                atomic_add(&T[6][idx], ph.weight[n] * ph.p3[n] * ph.p3[n] / ph.E[n]);
                // Tij
                atomic_add(&T[7][idx], ph.weight[n] * ph.p2[n] * ph.p3[n] / ph.E[n]);
                atomic_add(&T[8][idx], ph.weight[n] * ph.p3[n] * ph.p1[n] / ph.E[n]);
                atomic_add(&T[9][idx], ph.weight[n] * ph.p1[n] * ph.p2[n] / ph.E[n]);
              }
            });
          },
          ph, S[m_size - 1], T[m_size - 1]);
      ExecPolicy<Conf>::sync();
    }
  }
}

}  // namespace Aperture
