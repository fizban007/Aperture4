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

#ifndef _FIELD_SOLVER_GR_KS_H_
#define _FIELD_SOLVER_GR_KS_H_

#include "core/multi_array.hpp"
#include "data/fields.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/field_solver.h"
#include "systems/grid_ks.h"

namespace Aperture {

// System that solves curved spacetime Maxwell equations in Kerr-Schild
// coordinates
template <typename Conf>
class field_solver_gr_ks_cu : public field_solver_cu<Conf> {
 private:
  float m_a = 0.99;  // BH spin parameter a
  float m_damping_coef = 0.001;
  int m_damping_length = 20;
  const grid_ks_t<Conf>& m_ks_grid;

  typename Conf::multi_array_t m_tmp_th_field, m_tmp_prev_field, m_tmp_predictor;
  buffer<typename Conf::value_t> m_tri_dl, m_tri_d, m_tri_du, sp_buffer;

  vector_field<Conf> m_prev_D, m_prev_B, m_new_D, m_new_B;

  scalar_field<Conf>* flux;

 public:
  static std::string name() { return "field_solver"; }
  using value_t = typename Conf::value_t;

  field_solver_gr_ks_cu(sim_environment& env, const grid_ks_t<Conf>& grid,
                        const domain_comm<Conf>* comm = nullptr)
      : field_solver_cu<Conf>(env, grid, comm), m_ks_grid(grid) {}

  virtual ~field_solver_gr_ks_cu();

  void init() override;
  void update(double dt, uint32_t step) override;
  void register_data_components() override;

  // void solve_tridiagonal();
  void update_Br(vector_field<Conf>& B, const vector_field<Conf>& B0,
                 const vector_field<Conf>& D, const vector_field<Conf>& D0,
                 value_t dt);
  void update_Bth(vector_field<Conf>& B, const vector_field<Conf>& B0,
                  const vector_field<Conf>& D, const vector_field<Conf>& D0,
                  value_t dt);
  void update_Bph(vector_field<Conf>& B, const vector_field<Conf>& B0,
                  const vector_field<Conf>& D, const vector_field<Conf>& D0,
                  value_t dt);
  void update_Dr(vector_field<Conf>& D, const vector_field<Conf>& D0,
                 const vector_field<Conf>& B, const vector_field<Conf>& B0,
                 const vector_field<Conf>& J, value_t dt);
  void update_Dth(vector_field<Conf>& D, const vector_field<Conf>& D0,
                  const vector_field<Conf>& B, const vector_field<Conf>& B0,
                  const vector_field<Conf>& J, value_t dt);
  void update_Dph(vector_field<Conf>& D, const vector_field<Conf>& D0,
                  const vector_field<Conf>& B, const vector_field<Conf>& B0,
                  const vector_field<Conf>& J, value_t dt);

  void update_old(double dt, uint32_t step);

  void iterate_predictor(double dt);

  // void horizon_boundary(vector_field<Conf>& D, vector_field<Conf>& B);
};

}  // namespace Aperture

#endif  // _FIELD_SOLVER_GR_KS_H_
