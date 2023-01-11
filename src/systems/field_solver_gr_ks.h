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

#pragma once

#include "data/fields.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/field_solver.h"
#include "systems/grid_ks.h"
#include "systems/policies/coord_policy_gr_ks_sph.hpp"
#include "utils/nonown_ptr.hpp"

namespace Aperture {

/// System that solves curved spacetime Maxwell equations in spherical
/// Kerr-Schild coordinates
// template <typename Conf>
// class field_solver_gr_ks_cu : public field_solver_cu<Conf> {
template <typename Conf, template <class> class ExecPolicy>
class field_solver<Conf, ExecPolicy, coord_policy_gr_ks_sph>
    : public field_solver_base<Conf> {
 public:
  static std::string name() { return "field_solver"; }
  using value_t = typename Conf::value_t;

  // field_solver_gr_ks_cu(sim_environment& env, const grid_ks_t<Conf>& grid,
  field_solver(const grid_ks_t<Conf>& grid,
               const domain_comm<Conf, ExecPolicy>* comm = nullptr);

  virtual ~field_solver();

  virtual void init() override;
  // virtual void update(double dt, uint32_t step) override;
  virtual void register_data_components() override;

  // void solve_tridiagonal();
  virtual void update_semi_implicit(double dt, double alpha, double beta, double time) override;
  virtual void update_explicit(double dt, double time) override;

  virtual void compute_divs_e_b() override;
  virtual void compute_flux() override;
  virtual void compute_EB_sqr() override;

  void iterate_predictor(double dt);

  // void horizon_boundary(vector_field<Conf>& D, vector_field<Conf>& B);
 private:
  float m_a = 0.99;  // BH spin parameter a
  float m_damping_coef = 0.001;
  int m_damping_length = 20;
  const grid_ks_t<Conf>& m_ks_grid;
  const domain_comm<Conf, ExecPolicy>* m_comm = nullptr;

  // typename Conf::multi_array_t m_tmp_th_field, m_tmp_prev_field, m_tmp_predictor;
  // buffer<typename Conf::value_t> m_tri_dl, m_tri_d, m_tri_du, sp_buffer;
  std::unique_ptr<vector_field<Conf>> m_prev_D, m_prev_B, m_new_D, m_new_B;
};

}  // namespace Aperture
