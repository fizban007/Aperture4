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

#include "field_solver.h"
#include "framework/config.h"
#include "systems/helpers/finite_diff_helper.hpp"
#include "utils/timer.h"

namespace Aperture {

template <typename Conf>
using fd = finite_diff<Conf::dim, 2>;

template <typename Conf>
void
compute_e_update_explicit(vector_field<Conf>& result,
                          const vector_field<Conf>& b,
                          const vector_field<Conf>& j,
                          typename Conf::value_t dt) {
  auto& grid = result.grid();
  for (auto idx : result[0].indices()) {
    auto pos = idx.get_pos();
    if (grid.is_in_bound(pos)) {
      result[0][idx] +=
          dt *
          (fd<Conf>::curl0(b, idx, b.stagger_vec(), grid) - j[0][idx]);

      result[1][idx] +=
          dt *
          (fd<Conf>::curl1(b, idx, b.stagger_vec(), grid) - j[1][idx]);

      result[2][idx] +=
          dt *
          (fd<Conf>::curl2(b, idx, b.stagger_vec(), grid) - j[2][idx]);
    }
  }
}

template <typename Conf>
void
compute_b_update_explicit(vector_field<Conf>& result,
                          const vector_field<Conf>& e,
                          typename Conf::value_t dt) {
  auto& grid = result.grid();
  for (auto idx : result[0].indices()) {
    auto pos = idx.get_pos();
    if (grid.is_in_bound(pos)) {
      result[0][idx] +=
          dt * fd<Conf>::curl0(e, idx, e.stagger_vec(), grid);

      result[1][idx] +=
          dt * fd<Conf>::curl1(e, idx, e.stagger_vec(), grid);

      result[2][idx] +=
          dt * fd<Conf>::curl2(e, idx, e.stagger_vec(), grid);
    }
  }
}

template <typename Conf>
void
compute_divs(scalar_field<Conf>& divE, scalar_field<Conf>& divB,
             const vector_field<Conf>& e, const vector_field<Conf>& b,
             const bool is_boundary[Conf::dim * 2]) {
  auto& grid = divE.grid();
  for (auto idx : divE[0].indices()) {
    auto pos = idx.get_pos();
    if (grid.is_in_bound(pos)) {
      divE[idx] = fd<Conf>::div(e, idx, e.stagger_vec(), grid);
      divB[idx] = fd<Conf>::div(b, idx, b.stagger_vec(), grid);

      // TODO: Maybe check boundary and skip some cells?
    }
  }
}

template <typename Conf>
void
field_solver<Conf>::init() {
  sim_env().params().get_value("use_implicit", m_use_implicit);
  if (m_use_implicit) {
    sim_env().params().get_value("implicit_beta", m_beta);
    m_alpha = 1.0 - m_beta;
    this->init_impl_tmp_fields();
  }
  sim_env().params().get_value("fld_output_interval", m_data_interval);

  sim_env().params().get_value("update_e", m_update_e);
  sim_env().params().get_value("update_b", m_update_b);
}

template <typename Conf>
void
field_solver<Conf>::init_impl_tmp_fields() {
  m_tmp_b1 =
      std::make_unique<vector_field<Conf>>(this->m_grid, MemType::host_only);
  m_tmp_b2 =
      std::make_unique<vector_field<Conf>>(this->m_grid, MemType::host_only);
  m_bnew =
      std::make_unique<vector_field<Conf>>(this->m_grid, MemType::host_only);
}

template <typename Conf>
void
field_solver<Conf>::register_data_components() {
  register_data_impl(MemType::host_only);
}

template <typename Conf>
void
field_solver<Conf>::register_data_impl(MemType type) {
  // output fields, we don't directly use here
  Etotal = sim_env().register_data<vector_field<Conf>>(
      "E", m_grid, field_type::edge_centered, type);
  Btotal = sim_env().register_data<vector_field<Conf>>(
      "B", m_grid, field_type::face_centered, type);

  E = sim_env().register_data<vector_field<Conf>>("Edelta", m_grid,
                                              field_type::edge_centered, type);
  E->skip_output(true);
  E->include_in_snapshot(true);
  E0 = sim_env().register_data<vector_field<Conf>>("E0", m_grid,
                                               field_type::edge_centered, type);
  E0->skip_output(true);
  E0->include_in_snapshot(true);
  B = sim_env().register_data<vector_field<Conf>>("Bdelta", m_grid,
                                              field_type::face_centered, type);
  B->skip_output(true);
  B->include_in_snapshot(true);
  B0 = sim_env().register_data<vector_field<Conf>>("B0", m_grid,
                                               field_type::face_centered, type);
  B0->skip_output(true);
  B0->include_in_snapshot(true);
  J = sim_env().register_data<vector_field<Conf>>("J", m_grid,
                                              field_type::edge_centered, type);
  divB = sim_env().register_data<scalar_field<Conf>>(
      "divB", m_grid, field_type::cell_centered, type);
  divE = sim_env().register_data<scalar_field<Conf>>(
      "divE", m_grid, field_type::vert_centered, type);
  B_sqr = sim_env().register_data<scalar_field<Conf>>(
      "B_sqr", m_grid, field_type::vert_centered, type);
  E_sqr = sim_env().register_data<scalar_field<Conf>>(
      "E_sqr", m_grid, field_type::vert_centered, type);
  flux = sim_env().register_data<scalar_field<Conf>>(
      "flux", m_grid, field_type::vert_centered, type);
  // EdotB = sim_env().register_data<scalar_field<Conf>>("EdotB", m_grid,
  //                                                 field_type::vert_centered);
}

template <typename Conf>
void
field_solver<Conf>::update(double dt, uint32_t step) {
  double time = sim_env().get_time();
  if (m_use_implicit)
    this->update_semi_implicit(dt, m_alpha, m_beta, time);
  else
    this->update_explicit(dt, time);

  this->Etotal->copy_from(*(this->E0));
  this->Etotal->add_by(*(this->E));
  this->Btotal->copy_from(*(this->B0));
  this->Btotal->add_by(*(this->B));
}

template <typename Conf>
void
field_solver<Conf>::update_explicit(double dt, double time) {
  if (time < TINY) {
    compute_e_update_explicit(*(this->E), *(this->B), *(this->J), 0.5f * dt);
    if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));
  }

  compute_b_update_explicit(*(this->B), *(this->E), dt);

  // Communicate the new B values to guard cells
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->B));

  compute_e_update_explicit(*(this->E), *(this->B), *(this->J), dt);
  // Communicate the new E values to guard cells
  if (this->m_comm != nullptr) this->m_comm->send_guard_cells(*(this->E));

  // compute_divs();
}

template <typename Conf>
void
field_solver<Conf>::update_semi_implicit(double dt, double alpha, double beta,
                                         double time) {}

//// Explicit instantiation for 1d, 2d, and 3d
INSTANTIATE_WITH_CONFIG(field_solver);

}  // namespace Aperture
