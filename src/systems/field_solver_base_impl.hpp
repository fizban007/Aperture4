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

#include "field_solver_base.h"
#include "framework/config.h"
#include "systems/helpers/finite_diff_helper.hpp"
#include "utils/timer.h"

namespace Aperture {

template <typename Conf>
void
field_solver_base<Conf>::init() {
  sim_env().params().get_value("use_implicit", m_use_implicit);
  if (m_use_implicit) {
    sim_env().params().get_value("implicit_beta", m_beta);
    m_alpha = 1.0 - m_beta;
  }
  sim_env().params().get_value("fld_output_interval", m_data_interval);

  sim_env().params().get_value("update_e", m_update_e);
  sim_env().params().get_value("update_b", m_update_b);
  sim_env().params().get_value("compute_field_energies", m_compute_energies);
  sim_env().params().get_value("compute_field_divs", m_compute_divs);

  // for (int i = 0; i < Conf::dim * 2; i++) {
  //   if (m_damping[i] == true) {
  //     m_pml[i] = std::make_unique<pml_data<Conf>>((BoundaryPos)i,
  //     m_pml_length,
  //                                                 m_grid, m_comm);
  //   }
  // }
  this->Etotal->copy_from(*(this->E0));
  this->Etotal->add_by(*(this->E));
  this->Btotal->copy_from(*(this->B0));
  this->Btotal->add_by(*(this->B));
}

template <typename Conf>
void
field_solver_base<Conf>::register_data_components_impl(MemType type) {
  // MemType type = ExecPolicy<Conf>::data_mem_type();
  // output fields, we don't directly use here
  Etotal = sim_env().register_data<vector_field<Conf>>(
      "E", m_grid, field_type::edge_centered, type);
  Btotal = sim_env().register_data<vector_field<Conf>>(
      "B", m_grid, field_type::face_centered, type);

  // actual fields for computation, do not output. Do save in snapshots
  E = sim_env().register_data<vector_field<Conf>>(
      "Edelta", m_grid, field_type::edge_centered, type);
  E->skip_output(true);
  E->include_in_snapshot(true);
  E0 = sim_env().register_data<vector_field<Conf>>(
      "E0", m_grid, field_type::edge_centered, type);
  E0->skip_output(true);
  E0->include_in_snapshot(true);
  B = sim_env().register_data<vector_field<Conf>>(
      "Bdelta", m_grid, field_type::face_centered, type);
  B->skip_output(true);
  B->include_in_snapshot(true);
  B0 = sim_env().register_data<vector_field<Conf>>(
      "B0", m_grid, field_type::face_centered, type);
  B0->skip_output(true);
  B0->include_in_snapshot(true);
  J = sim_env().register_data<vector_field<Conf>>(
      "J", m_grid, field_type::edge_centered, type);
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
field_solver_base<Conf>::update(double dt, uint32_t step) {
  double time = sim_env().get_time();
  if (m_use_implicit)
    this->update_semi_implicit(dt, m_alpha, m_beta, time);
  else
    this->update_explicit(dt, time);

  this->Etotal->copy_from(*(this->E0));
  this->Etotal->add_by(*(this->E));
  this->Btotal->copy_from(*(this->B0));
  this->Btotal->add_by(*(this->B));

  if (step % this->m_data_interval == 0) {
    this->compute_flux();
    if (m_compute_divs) {
      this->compute_divs_e_b();
    }
    if (m_compute_energies) {
      this->compute_EB_sqr();
    }
  }
}

template <typename Conf>
void
field_solver_base<Conf>::update_explicit(double dt, double time) {}

template <typename Conf>
void
field_solver_base<Conf>::update_semi_implicit(double dt, double alpha,
                                              double beta, double time) {}

}  // namespace Aperture
