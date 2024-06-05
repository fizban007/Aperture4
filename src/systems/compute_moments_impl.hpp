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
#include "data/fields.h"
#include "framework/environment.h"

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
void
compute_moments<Conf, ExecPolicy>::init() {
  sim_env().get_data("particles", ptc);
  sim_env().get_data_optional("photons", ph);
  if (ph != nullptr) {
    Logger::print_info("Photon data detected.");
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
  if (m_compute_first_moments) {
    ptc_num.resize(m_size);
    ptc_flux.resize(m_size);
    for (int n = 0; n < m_size; n++) {
      if (n < m_num_species) {
        // S.set(n, sim_env().register_data<field_t<4, Conf>>(
        //              std::string("flux_") + ptc_type_name(n), m_grid,
        //              ExecPolicy<Conf>::data_mem_type()));
        ptc_num.set(n,
                    sim_env().register_data<scalar_field<Conf>>(
                        std::string("num_") + ptc_type_name(n), m_grid,
                        field_type::cell_centered,
                        ExecPolicy<Conf>::data_mem_type()));
        ptc_flux.set(n,
                     sim_env().register_data<field_t<3, Conf>>(
                         std::string("flux_") + ptc_type_name(n), m_grid,
                         ExecPolicy<Conf>::data_mem_type()));
      } else {
        // S.set(n, sim_env().register_data<field_t<4, Conf>>(
        //              std::string("flux_ph"), m_grid,
        //              ExecPolicy<Conf>::data_mem_type()));
        ptc_num.set(n, sim_env().register_data<scalar_field<Conf>>(
                          std::string("num_ph"), m_grid,
                          field_type::cell_centered,
                          ExecPolicy<Conf>::data_mem_type()));
        ptc_flux.set(n, sim_env().register_data<field_t<3, Conf>>(
                           std::string("flux_ph"), m_grid,
                           ExecPolicy<Conf>::data_mem_type()));
      }
    }
    ptc_num.copy_to_device();
    ptc_flux.copy_to_device();
  }
  if (m_compute_second_moments) {
    T00.resize(m_size);
    T0i.resize(m_size);
    T11.resize(m_size);
    T12.resize(m_size);
    T13.resize(m_size);
    T22.resize(m_size);
    T23.resize(m_size);
    T33.resize(m_size);
    for (int n = 0; n < m_size; n++) {
      if (n < m_num_species) {
        // T.set(n, sim_env().register_data<field_t<10, Conf>>(
        //              std::string("stress_") + ptc_type_name(n), m_grid,
        //              ExecPolicy<Conf>::data_mem_type()));
        T00.set(n, sim_env().register_data<scalar_field<Conf>>(
                       std::string("stress_") + ptc_type_name(n) + "00", m_grid,
                       field_type::cell_centered,
                       ExecPolicy<Conf>::data_mem_type()));
        T0i.set(n, sim_env().register_data<field_t<3, Conf>>(
                        std::string("stress_") + ptc_type_name(n) + "0", m_grid,
                        ExecPolicy<Conf>::data_mem_type()));
        T11.set(n, sim_env().register_data<scalar_field<Conf>>(
                        std::string("stress_") + ptc_type_name(n) + "11", m_grid,
                        field_type::cell_centered,
                        ExecPolicy<Conf>::data_mem_type()));
        T12.set(n, sim_env().register_data<scalar_field<Conf>>(
                        std::string("stress_") + ptc_type_name(n) + "12", m_grid,
                        field_type::cell_centered,
                        ExecPolicy<Conf>::data_mem_type()));
        T13.set(n, sim_env().register_data<scalar_field<Conf>>(
                        std::string("stress_") + ptc_type_name(n) + "13", m_grid,
                        field_type::cell_centered,
                        ExecPolicy<Conf>::data_mem_type()));
        T22.set(n, sim_env().register_data<scalar_field<Conf>>(
                        std::string("stress_") + ptc_type_name(n) + "22", m_grid,
                        field_type::cell_centered,
                        ExecPolicy<Conf>::data_mem_type()));
        T23.set(n, sim_env().register_data<scalar_field<Conf>>(
                        std::string("stress_") + ptc_type_name(n) + "23", m_grid,
                        field_type::cell_centered,
                        ExecPolicy<Conf>::data_mem_type()));
        T33.set(n, sim_env().register_data<scalar_field<Conf>>(
                        std::string("stress_") + ptc_type_name(n) + "33", m_grid,
                        field_type::cell_centered,
                        ExecPolicy<Conf>::data_mem_type()));
      } else {
        // T.set(n, sim_env().register_data<field_t<10, Conf>>(
        //              std::string("stress_ph"), m_grid,
        //              ExecPolicy<Conf>::data_mem_type()));
        T00.set(n, sim_env().register_data<scalar_field<Conf>>(
                       std::string("stress_ph00"), m_grid,
                       field_type::cell_centered,
                       ExecPolicy<Conf>::data_mem_type()));
        T0i.set(n, sim_env().register_data<field_t<3, Conf>>(
                        std::string("stress_ph0"), m_grid,
                        ExecPolicy<Conf>::data_mem_type()));
        T11.set(n, sim_env().register_data<scalar_field<Conf>>(
                        std::string("stress_ph11"), m_grid,
                        field_type::cell_centered,
                        ExecPolicy<Conf>::data_mem_type()));
        T12.set(n, sim_env().register_data<scalar_field<Conf>>(
                        std::string("stress_ph12"), m_grid,
                        field_type::cell_centered,
                        ExecPolicy<Conf>::data_mem_type()));
        T13.set(n, sim_env().register_data<scalar_field<Conf>>(
                        std::string("stress_ph13"), m_grid,
                        field_type::cell_centered,
                        ExecPolicy<Conf>::data_mem_type()));
        T22.set(n, sim_env().register_data<scalar_field<Conf>>(
                        std::string("stress_ph22"), m_grid,
                        field_type::cell_centered,
                        ExecPolicy<Conf>::data_mem_type()));
        T23.set(n, sim_env().register_data<scalar_field<Conf>>(
                        std::string("stress_ph23"), m_grid,
                        field_type::cell_centered,
                        ExecPolicy<Conf>::data_mem_type()));
        T33.set(n, sim_env().register_data<scalar_field<Conf>>(
                        std::string("stress_ph33"), m_grid,
                        field_type::cell_centered,
                        ExecPolicy<Conf>::data_mem_type()));
      }
    }
    T00.copy_to_device();
    T0i.copy_to_device();
    T11.copy_to_device();
    T12.copy_to_device();
    T13.copy_to_device();
    T22.copy_to_device();
    T23.copy_to_device();
    T33.copy_to_device();
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
compute_moments<Conf, ExecPolicy>::register_data_components() {
}

template <typename Conf, template <class> class ExecPolicy>
void
compute_moments<Conf, ExecPolicy>::update(double dt, uint32_t step) {
  if (step % m_fld_interval == 0) {
    Logger::print_info("size is {}", m_size);
    for (int n = 0; n < m_size; n++) {
      if (m_compute_first_moments) {
        ptc_num.init();
        ptc_flux.init();
      }
      if (m_compute_second_moments) {
        T00.init();
        T0i.init();
        T11.init();
        T12.init();
        T13.init();
        T22.init();
        T23.init();
        T33.init();
      }
    }
    Logger::print_info("Initialized");

    bool first = m_compute_first_moments;
    bool second = m_compute_second_moments;
    size_t num = ptc->number();
    // int num_species = m_num_species;
    // First compute particle moments
    ExecPolicy<Conf>::launch(
        [first, second, num] LAMBDA(auto ptc, auto ptc_num, auto ptc_flux,
                                    auto T00, auto T0i, auto T11, auto T12,
                                    auto T13, auto T22, auto T23, auto T33) {
          auto& grid = ExecPolicy<Conf>::grid();
          auto ext = grid.extent();
          ExecPolicy<Conf>::loop(0, num, [&] LAMBDA(auto n) {
            uint32_t cell = ptc.cell[n];
            if (cell == empty_cell) {
              return;
            }

            typename Conf::idx_t idx = Conf::idx(cell, ext);
            int sp = get_ptc_type(ptc.flag[n]);
            if (first) {
              atomic_add(&ptc_num[sp][idx], ptc.weight[n]);
              atomic_add(&ptc_flux[sp][0][idx], ptc.weight[n] * ptc.p1[n] / ptc.E[n]);
              atomic_add(&ptc_flux[sp][1][idx], ptc.weight[n] * ptc.p2[n] / ptc.E[n]);
              atomic_add(&ptc_flux[sp][2][idx], ptc.weight[n] * ptc.p3[n] / ptc.E[n]);
            }
            if (second) {
              // T00
              atomic_add(&T00[sp][idx], ptc.weight[n] * ptc.E[n]);
              // T0i
              atomic_add(&T0i[sp][0][idx], ptc.weight[n] * ptc.p1[n]);
              atomic_add(&T0i[sp][1][idx], ptc.weight[n] * ptc.p2[n]);
              atomic_add(&T0i[sp][2][idx], ptc.weight[n] * ptc.p3[n]);
              // Tii
              atomic_add(&T11[sp][idx],
                         ptc.weight[n] * ptc.p1[n] * ptc.p1[n] / ptc.E[n]);
              atomic_add(&T22[sp][idx],
                         ptc.weight[n] * ptc.p2[n] * ptc.p2[n] / ptc.E[n]);
              atomic_add(&T33[sp][idx],
                         ptc.weight[n] * ptc.p3[n] * ptc.p3[n] / ptc.E[n]);
              // Tij
              atomic_add(&T23[sp][idx],
                         ptc.weight[n] * ptc.p2[n] * ptc.p3[n] / ptc.E[n]);
              atomic_add(&T13[sp][idx],
                         ptc.weight[n] * ptc.p3[n] * ptc.p1[n] / ptc.E[n]);
              atomic_add(&T12[sp][idx],
                         ptc.weight[n] * ptc.p1[n] * ptc.p2[n] / ptc.E[n]);
            }
          });
        },
        *ptc, ptc_num, ptc_flux, T00, T0i, T11, T12, T13, T22, T23, T33);
    ExecPolicy<Conf>::sync();
    // Then compute photon moments if applicable
    if (m_photon_data) {
      size_t ph_num = ph->number();
      if (ph_num > 0) {
        ExecPolicy<Conf>::launch(
            [first, second, ph_num] LAMBDA(auto ph, auto ptc_num, auto ptc_flux,
                                            auto T00, auto T0i, auto T11,
                                            auto T12, auto T13, auto T22,
                                            auto T23, auto T33) {
              auto& grid = ExecPolicy<Conf>::grid();
              auto ext = grid.extent();
              ExecPolicy<Conf>::loop(0, ph_num, [&] LAMBDA(auto n) {
                uint32_t cell = ph.cell[n];
                if (cell == empty_cell) {
                  return;
                }

                typename Conf::idx_t idx = Conf::idx(cell, ext);
                if (first) {
                  atomic_add(&ptc_num[idx], ph.weight[n]);
                  atomic_add(&ptc_flux[1][idx], ph.weight[n] * ph.p1[n] / ph.E[n]);
                  atomic_add(&ptc_flux[2][idx], ph.weight[n] * ph.p2[n] / ph.E[n]);
                  atomic_add(&ptc_flux[3][idx], ph.weight[n] * ph.p3[n] / ph.E[n]);
                }
                if (second) {
                  // T00
                  atomic_add(&T00[idx], ph.weight[n] * ph.E[n]);
                  // T0i
                  atomic_add(&T0i[1][idx], ph.weight[n] * ph.p1[n]);
                  atomic_add(&T0i[2][idx], ph.weight[n] * ph.p2[n]);
                  atomic_add(&T0i[3][idx], ph.weight[n] * ph.p3[n]);
                  // Tii
                  atomic_add(&T11[idx], ph.weight[n] * ph.p1[n] * ph.p1[n] / ph.E[n]);
                  atomic_add(&T22[idx], ph.weight[n] * ph.p2[n] * ph.p2[n] / ph.E[n]);
                  atomic_add(&T33[idx], ph.weight[n] * ph.p3[n] * ph.p3[n] / ph.E[n]);
                  // Tij
                  atomic_add(&T23[idx], ph.weight[n] * ph.p2[n] * ph.p3[n] / ph.E[n]);
                  atomic_add(&T13[idx], ph.weight[n] * ph.p3[n] * ph.p1[n] / ph.E[n]);
                  atomic_add(&T12[idx], ph.weight[n] * ph.p1[n] * ph.p2[n] / ph.E[n]);
                }
              });
            },
            *ph, ptc_num[m_size - 1], ptc_flux[m_size - 1], T00[m_size - 1],
            T0i[m_size - 1], T11[m_size - 1], T12[m_size - 1], T13[m_size - 1],
            T22[m_size - 1], T23[m_size - 1], T33[m_size - 1]);
        ExecPolicy<Conf>::sync();
      }
    }
  }
}

}  // namespace Aperture
