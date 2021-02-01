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

#include "ptc_updater.h"
#include "core/constant_mem_func.h"
#include "core/detail/multi_array_helpers.h"
#include "framework/config.h"
#include "helpers/ptc_update_helper.hpp"
#include "systems/physics/pushers.hpp"
#include "utils/double_buffer.h"
#include "utils/range.hpp"
#include "utils/timer.h"
#include "utils/util_functions.h"
#include <random>

namespace Aperture {

template <typename Conf>
void
ptc_updater<Conf>::init_charge_mass() {
  // Default values are 1.0
  float q_e = 1.0;
  float ion_mass = 1.0;
  sim_env().params().get_value("q_e", q_e);
  sim_env().params().get_value("ion_mass", ion_mass);

  for (int i = 0; i < (max_ptc_types); i++) {
    m_charges[i] = q_e;
    m_masses[i] = q_e;
  }
  m_charges[(int)PtcType::electron] *= -1.0;
  m_masses[(int)PtcType::ion] *= ion_mass;
  for (int i = 0; i < (max_ptc_types); i++) {
    m_q_over_m[i] = m_charges[i] / m_masses[i];
  }
}

template <typename Conf>
// ptc_updater<Conf>::ptc_updater(sim_environment& env, const grid_t<Conf>& grid,
ptc_updater<Conf>::ptc_updater(const grid_t<Conf>& grid,
                               const domain_comm<Conf>* comm)
    : m_grid(grid), m_comm(comm) {
  sim_env().params().get_value("fld_output_interval", m_data_interval);
  // By default, rho_interval is the same as field output interval
  m_rho_interval = m_data_interval;
  // Override if there is a specific rho_interval option specified
  sim_env().params().get_value("rho_interval", m_rho_interval);

  sim_env().params().get_value("sort_interval", m_sort_interval);
  sim_env().params().get_value("current_smoothing", m_filter_times);
  init_charge_mass();

  auto pusher = sim_env().params().template get_as<std::string>("pusher");

  if (pusher == "boris") {
    m_pusher = Pusher::boris;
  } else if (pusher == "vay") {
    m_pusher = Pusher::vay;
  } else if (pusher == "higuera") {
    m_pusher = Pusher::higuera;
  }
}

template <typename Conf>
void
ptc_updater<Conf>::init() {
  // Allocate the tmp array for current filtering
  jtmp = std::make_unique<typename Conf::multi_array_t>(m_grid.extent(),
                                                        MemType::host_only);

  sim_env().get_data_optional("photons", ph);
  sim_env().get_data_optional("Rho_ph", rho_ph);
}

template <typename Conf>
void
ptc_updater<Conf>::register_data_components() {
  size_t max_ptc_num = 10000;
  sim_env().params().get_value("max_ptc_num", max_ptc_num);

  ptc = sim_env().register_data<particle_data_t>("particles", max_ptc_num,
                                             MemType::host_only);

  E = sim_env().register_data<vector_field<Conf>>(
      "E", m_grid, field_type::edge_centered, MemType::host_only);
  B = sim_env().register_data<vector_field<Conf>>(
      "B", m_grid, field_type::face_centered, MemType::host_only);
  J = sim_env().register_data<vector_field<Conf>>(
      "J", m_grid, field_type::edge_centered, MemType::host_only);

  sim_env().params().get_value("num_species", m_num_species);
  Rho.resize(m_num_species);
  for (int i = 0; i < m_num_species; i++) {
    Rho[i] = sim_env().register_data<scalar_field<Conf>>(
        std::string("Rho_") + ptc_type_name(i), m_grid,
        field_type::vert_centered, MemType::host_only);
  }
}

template <typename Conf>
void
ptc_updater<Conf>::push_default(value_t dt) {
  // dispatch according to enum
  if (m_pusher == Pusher::boris) {
    auto pusher = pusher_impl_t<boris_pusher>{};
    push(dt, pusher);
  } else if (m_pusher == Pusher::vay) {
    auto pusher = pusher_impl_t<vay_pusher>{};
    push(dt, pusher);
  } else if (m_pusher == Pusher::higuera) {
    auto pusher = pusher_impl_t<higuera_pusher>{};
    push(dt, pusher);
  }
}

template <typename Conf>
void
ptc_updater<Conf>::update_particles(value_t dt, uint32_t step) {
  Logger::print_info("Pushing {} particles", ptc->number());
  timer::stamp("pusher");
  // First update particle momentum
  push_default(dt);
  timer::show_duration_since_stamp("push", "ms", "pusher");

  timer::stamp("depositer");
  // Then move particles and deposit current
  move_and_deposit(dt, step);
  timer::show_duration_since_stamp("deposit", "ms", "depositer");
}

template <typename Conf>
void
ptc_updater<Conf>::update(double dt, uint32_t step) {
  update_particles(dt, step);

  // Communicate deposited current and charge densities
  if (m_comm != nullptr) {
    m_comm->send_add_guard_cells(*J);
    m_comm->send_guard_cells(*J);
    // if ((step + 1) % m_data_interval == 0) {
    if (step % m_rho_interval == 0) {
      for (uint32_t i = 0; i < Rho.size(); i++) {
        m_comm->send_add_guard_cells(*(Rho[i]));
        m_comm->send_guard_cells(*(Rho[i]));
      }
    }
  }

  timer::stamp("filter");
  // Filter current
  filter_current(m_filter_times, step);
  timer::show_duration_since_stamp("filter", "ms", "filter");

  // Send particles
  if (m_comm != nullptr) {
    m_comm->send_particles(*ptc, m_grid);
  }

  // Also move photons if the data component exists
  if (ph != nullptr) {
    Logger::print_info("Moving {} photons", ph->number());
    move_photons(dt, step);

    if (m_comm != nullptr) {
      m_comm->send_particles(*ph, m_grid);
    }
  }

  // Clear guard cells
  clear_guard_cells();

  // sort at the given interval. Turn off sorting if m_sort_interval is 0
  if (m_sort_interval > 0 && (step % m_sort_interval) == 0) {
    sort_particles();
  }
}

template <typename Conf>
void
ptc_updater<Conf>::move_and_deposit(double dt, uint32_t step) {
  if constexpr (Conf::dim == 1)
    move_deposit_1d(dt, step);
  else if constexpr (Conf::dim == 2)
    move_deposit_2d(dt, step);
  else if constexpr (Conf::dim == 3)
    move_deposit_3d(dt, step);
}

template <typename Conf>
void
ptc_updater<Conf>::move_photons(double dt, uint32_t step) {
  if constexpr (Conf::dim == 1) {
    move_photons_1d(dt, step);
  } else if constexpr (Conf::dim == 2) {
    move_photons_2d(dt, step);
  } else if constexpr (Conf::dim == 3) {
    move_photons_3d(dt, step);
  }
}

template <typename Conf>
void
ptc_updater<Conf>::move_deposit_1d(value_t dt, uint32_t step) {
  auto num = ptc->number();
  if (num > 0) {
    auto ext = m_grid.extent();

    for (auto n : range(0, num)) {
      uint32_t cell = ptc->cell[n];
      if (cell == empty_cell) continue;
      auto idx = (*E)[0].idx_at(cell);
      auto pos = idx.get_pos();

      // step 1: Move particles
      auto x1 = ptc->x1[n], x2 = ptc->x2[n], x3 = ptc->x3[n];
      value_t v1 = ptc->p1[n], v2 = ptc->p2[n], v3 = ptc->p3[n],
              gamma = ptc->E[n];

      v1 /= gamma;
      v2 /= gamma;
      v3 /= gamma;

      auto new_x1 = x1 + (v1 * dt) * m_grid.inv_delta[0];
      int dc1 = std::floor(new_x1);
      pos[0] += dc1;
      ptc->x1[n] = new_x1 - (value_t)dc1;
      ptc->x2[n] = x2 + v2 * dt;
      ptc->x3[n] = x3 + v3 * dt;

      ptc->cell[n] = m_grid.get_idx(pos).linear;

      // step 2: Deposit current
      auto flag = ptc->flag[n];
      auto sp = get_ptc_type(flag);
      auto interp = spline_t{};
      if (check_flag(flag, PtcFlag::ignore_current)) continue;
      auto weight = m_charges[sp] * ptc->weight[n];

      int i_0 = (dc1 == -1 ? -spline_t::radius : 1 - spline_t::radius);
      int i_1 = (dc1 == 1 ? spline_t::radius + 1 : spline_t::radius);
      value_t djx = 0.0f;
      for (int i = i_0; i <= i_1; i++) {
        value_t sx0 = interp(-x1 + i);
        value_t sx1 = interp(-new_x1 + i);

        // j1 is movement in x1
        int offset = i + pos[0] - dc1;
        djx += sx1 - sx0;
        (*J)[0][offset] += -weight * djx;
        // Logger::print_debug("J0 is {}", (*J)[0][offset]);

        // j2 is simply v2 times rho at center
        value_t val1 = 0.5f * (sx0 + sx1);
        (*J)[1][offset] += weight * v2 * val1;

        // j3 is simply v3 times rho at center
        (*J)[2][offset] += weight * v3 * val1;

        // rho is deposited at the final position
        // if ((step + 1) % m_data_interval == 0) {
        if (step % m_rho_interval == 0) {
          (*Rho[sp])[0][offset] += weight * sx1;
        }
      }
    }
  }
}

template <typename Conf>
void
ptc_updater<Conf>::move_deposit_2d(value_t dt, uint32_t step) {
  if constexpr (Conf::dim >= 2) {
    auto num = ptc->number();
    if (num > 0) {
      auto ext = m_grid.extent();

      for (auto n : range(0, num)) {
        uint32_t cell = ptc->cell[n];
        if (cell == empty_cell) continue;
        auto idx = (*E)[0].idx_at(cell);
        auto pos = idx.get_pos();

        // step 1: Move particles
        auto x1 = ptc->x1[n], x2 = ptc->x2[n], x3 = ptc->x3[n];
        value_t v1 = ptc->p1[n], v2 = ptc->p2[n], v3 = ptc->p3[n],
                gamma = ptc->E[n];

        v1 /= gamma;
        v2 /= gamma;
        v3 /= gamma;

        auto new_x1 = x1 + (v1 * dt) * m_grid.inv_delta[0];
        int dc1 = std::floor(new_x1);
        pos[0] += dc1;
        ptc->x1[n] = new_x1 - (value_t)dc1;

        auto new_x2 = x2 + (v2 * dt) * m_grid.inv_delta[1];
        int dc2 = std::floor(new_x2);
        pos[1] += dc2;
        ptc->x2[n] = new_x2 - (value_t)dc2;
        ptc->x3[n] = x3 + v3 * dt;

        ptc->cell[n] = m_grid.get_idx(pos).linear;

        // step 2: Deposit current
        auto flag = ptc->flag[n];
        auto sp = get_ptc_type(flag);
        auto interp = spline_t{};
        if (check_flag(flag, PtcFlag::ignore_current)) continue;
        auto weight = m_charges[sp] * ptc->weight[n];

        int i_0 = (dc1 == -1 ? -spline_t::radius : 1 - spline_t::radius);
        int i_1 = (dc1 == 1 ? spline_t::radius + 1 : spline_t::radius);
        int j_0 = (dc2 == -1 ? -spline_t::radius : 1 - spline_t::radius);
        int j_1 = (dc2 == 1 ? spline_t::radius + 1 : spline_t::radius);
        value_t djy[2 * spline_t::radius + 1] = {};
        for (int j = j_0; j <= j_1; j++) {
          value_t sy0 = interp(-x2 + j);
          value_t sy1 = interp(-new_x2 + j);

          value_t djx = 0.0f;
          for (int i = i_0; i <= i_1; i++) {
            value_t sx0 = interp(-x1 + i);
            value_t sx1 = interp(-new_x1 + i);

            // j1 is movement in x1
            auto offset = idx.inc_x(i).inc_y(j);
            djx += movement2d(sy0, sy1, sx0, sx1);
            (*J)[0][offset] += -weight * djx;

            // j2 is movement in x2
            djy[i - i_0] += movement2d(sx0, sx1, sy0, sy1);
            (*J)[1][offset] += -weight * djy[i - i_0];
            // Logger::print_debug("J1 is {}", (*J)[1][offset]);

            // j3 is simply v3 times rho at center
            (*J)[2][offset] += weight * v3 * center2d(sx0, sx1, sy0, sy1);

            // rho is deposited at the final position
            // if ((step + 1) % m_data_interval == 0) {
            if (step % m_rho_interval == 0) {
              (*Rho[sp])[0][offset] += weight * sx1 * sy1;
            }
          }
        }
      }
    }
  }
}

template <typename Conf>
void
ptc_updater<Conf>::move_deposit_3d(value_t dt, uint32_t step) {
  if constexpr (Conf::dim == 3) {
    auto num = ptc->number();
    if (num > 0) {
      auto ext = m_grid.extent();

      for (auto n : range(0, num)) {
        uint32_t cell = ptc->cell[n];
        if (cell == empty_cell) continue;
        auto idx = (*E)[0].idx_at(cell);
        auto pos = idx.get_pos();

        // step 1: Move particles
        auto x1 = ptc->x1[n], x2 = ptc->x2[n], x3 = ptc->x3[n];
        value_t v1 = ptc->p1[n], v2 = ptc->p2[n], v3 = ptc->p3[n],
                gamma = ptc->E[n];

        v1 /= gamma;
        v2 /= gamma;
        v3 /= gamma;

        auto new_x1 = x1 + (v1 * dt) * m_grid.inv_delta[0];
        int dc1 = std::floor(new_x1);
        pos[0] += dc1;
        ptc->x1[n] = new_x1 - (value_t)dc1;

        auto new_x2 = x2 + (v2 * dt) * m_grid.inv_delta[1];
        int dc2 = std::floor(new_x2);
        pos[1] += dc2;
        ptc->x2[n] = new_x2 - (value_t)dc2;

        auto new_x3 = x3 + (v3 * dt) * m_grid.inv_delta[2];
        int dc3 = std::floor(new_x3);
        pos[2] += dc3;
        ptc->x3[n] = new_x3 - (value_t)dc3;

        ptc->cell[n] = m_grid.get_idx(pos).linear;

        // step 2: Deposit current
        auto flag = ptc->flag[n];
        auto sp = get_ptc_type(flag);
        auto interp = spline_t{};
        if (check_flag(flag, PtcFlag::ignore_current)) continue;
        auto weight = m_charges[sp] * ptc->weight[n];

        int i_0 = (dc1 == -1 ? -spline_t::radius : 1 - spline_t::radius);
        int i_1 = (dc1 == 1 ? spline_t::radius + 1 : spline_t::radius);
        int j_0 = (dc2 == -1 ? -spline_t::radius : 1 - spline_t::radius);
        int j_1 = (dc2 == 1 ? spline_t::radius + 1 : spline_t::radius);
        int k_0 = (dc3 == -1 ? -spline_t::radius : 1 - spline_t::radius);
        int k_1 = (dc3 == 1 ? spline_t::radius + 1 : spline_t::radius);

        value_t djz[2 * spline_t::radius + 1][2 * spline_t::radius + 1] = {};
        for (int k = k_0; k <= k_1; k++) {
          value_t sz0 = interp(-x3 + k);
          value_t sz1 = interp(-new_x3 + k);

          value_t djy[2 * spline_t::radius + 1] = {};
          for (int j = j_0; j <= j_1; j++) {
            value_t sy0 = interp(-x2 + j);
            value_t sy1 = interp(-new_x2 + j);

            value_t djx = 0.0f;
            for (int i = i_0; i <= i_1; i++) {
              value_t sx0 = interp(-x1 + i);
              value_t sx1 = interp(-new_x1 + i);

              // j1 is movement in x1
              auto offset = idx.inc_x(i).inc_y(j).inc_z(k);
              djx += movement3d(sy0, sy1, sz0, sz1, sx0, sx1);
              (*J)[0][offset] += -weight * djx;

              // j2 is movement in x2
              djy[i - i_0] += movement3d(sz0, sz1, sx0, sx1, sy0, sy1);
              (*J)[1][offset] += -weight * djy[i - i_0];
              // Logger::print_debug("J1 is {}", (*J)[1][offset]);

              // j3 is movement in x3
              djz[j - j_0][i - i_0] += movement3d(sx0, sx1, sy0, sy1, sz0, sz1);
              (*J)[2][offset] += -weight * djz[j - j_0][i - i_0];

              // rho is deposited at the final position
              if (step % m_rho_interval == 0) {
                (*Rho[sp])[0][offset] += weight * sx1 * sy1 * sz1;
              }
            }
          }
        }
      }
    }
  }
}

template <typename Conf>
void
ptc_updater<Conf>::move_photons_1d(value_t dt, uint32_t step) {}

template <typename Conf>
void
ptc_updater<Conf>::move_photons_2d(value_t dt, uint32_t step) {}

template <typename Conf>
void
ptc_updater<Conf>::move_photons_3d(value_t dt, uint32_t step) {}

template <typename Conf>
void
ptc_updater<Conf>::clear_guard_cells() {
  for (auto n : range(0, ptc->number())) {
    uint32_t cell = ptc->cell[n];
    if (cell == empty_cell) continue;
    auto idx = typename Conf::idx_t(cell, m_grid.extent());
    auto pos = idx.get_pos();

    if (!m_grid.is_in_bound(pos)) ptc->cell[n] = empty_cell;
  }

  if (ph != nullptr) {
    for (auto n : range(0, ph->number())) {
      uint32_t cell = ph->cell[n];
      if (cell == empty_cell) continue;
      auto idx = typename Conf::idx_t(cell, m_grid.extent());
      auto pos = idx.get_pos();

      if (!m_grid.is_in_bound(pos)) ph->cell[n] = empty_cell;
    }
  }
}

template <typename Conf>
void
ptc_updater<Conf>::sort_particles() {
  ptc->sort_by_cell_host(m_grid.extent().size());
  if (ph != nullptr) ph->sort_by_cell_host(m_grid.extent().size());
}

template <typename Conf>
void
ptc_updater<Conf>::fill_multiplicity(int mult, value_t weight) {
  auto num = ptc->number();
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (auto idx : range(m_grid.begin(), m_grid.end())) {
    auto pos = idx.get_pos();
    if (m_grid.is_in_bound(pos)) {
      for (int n = 0; n < mult; n++) {
        uint32_t offset = num + idx.linear * mult * 2 + n * 2;

        ptc->x1[offset] = ptc->x1[offset + 1] = dist(engine);
        ptc->x2[offset] = ptc->x2[offset + 1] = dist(engine);
        ptc->x3[offset] = ptc->x3[offset + 1] = dist(engine);
        ptc->p1[offset] = ptc->p1[offset + 1] = 0.0;
        ptc->p2[offset] = ptc->p2[offset + 1] = 0.0;
        ptc->p3[offset] = ptc->p3[offset + 1] = 0.0;
        ptc->E[offset] = ptc->E[offset + 1] = 1.0;
        ptc->cell[offset] = ptc->cell[offset + 1] = idx.linear;
        ptc->weight[offset] = ptc->weight[offset + 1] = weight;
        ptc->flag[offset] =
            set_ptc_type_flag(flag_or(PtcFlag::primary), PtcType::electron);
        ptc->flag[offset + 1] =
            set_ptc_type_flag(flag_or(PtcFlag::primary), PtcType::positron);
      }
    }
  }
  ptc->set_num(num + mult * 2 * m_grid.extent().size());
}

template <typename Conf>
void
ptc_updater<Conf>::filter_current(int n_times, uint32_t step) {
  Logger::print_info("Filtering current {} times", n_times);
  for (int n = 0; n < n_times; n++) {
    this->filter_field(*J, 0);
    this->filter_field(*J, 1);
    this->filter_field(*J, 2);

    if (m_comm != nullptr) m_comm->send_guard_cells(*J);

    // if ((step + 1) % m_data_interval == 0) {
    if (step % m_rho_interval == 0) {
      for (int sp = 0; sp < m_num_species; sp++) {
        this->filter_field(*Rho[sp]);
        if (m_comm != nullptr) m_comm->send_guard_cells(*Rho[sp]);
      }
    }
  }
}

template <typename Conf>
void
ptc_updater<Conf>::filter_field(vector_field<Conf>& f, int comp) {}

template <typename Conf>
void
ptc_updater<Conf>::filter_field(scalar_field<Conf>& f) {}

#include "ptc_updater_impl.hpp"

// template class ptc_updater<Config<1>>;
// template class ptc_updater<Config<2>>;
// template class ptc_updater<Config<3>>;
INSTANTIATE_WITH_CONFIG(ptc_updater);

}  // namespace Aperture
