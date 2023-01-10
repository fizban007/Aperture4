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
#include "systems/grid_sph.hpp"
#include "systems/policies/coord_policy_spherical.hpp"
#include "utils/nonown_ptr.hpp"
#include <memory>

namespace Aperture {

// System that updates Maxwell equations in Spherical
// coordinates
template <typename Conf, template <class> class ExecPolicy>
class field_solver<Conf, ExecPolicy, coord_policy_spherical>
    : public field_solver_base<Conf> {
 public:
  using value_t = typename Conf::value_t;

  field_solver(const grid_sph_t<Conf>& grid,
               const domain_comm<Conf, ExecPolicy>* comm = nullptr);

  virtual ~field_solver() {}

  virtual void init() override;
  virtual void register_data_components() override;
  virtual void update_explicit(double dt, double time) override;
  virtual void update_semi_implicit(double dt, double alpha, double beta,
                                    double time) override;

  virtual void compute_divs_e_b() override;
  virtual void compute_flux() override;
  virtual void compute_EB_sqr() override;

  void compute_b_update_explicit(double dt);
  void compute_e_update_explicit(double dt);

 protected:
  const grid_sph_t<Conf>& m_grid_sph;
  const domain_comm<Conf, ExecPolicy>* m_comm = nullptr;

  int m_damping_length = 64;
  value_t m_damping_coef = 0.003;

  // These are temporary fields required for the implicit update
  // std::unique_ptr<vector_field<Conf>> m_bnew;
  nonown_ptr<vector_field<Conf>> m_tmp_b1, m_tmp_b2, m_bnew;
};

// template <typename Conf>
// class field_solver_sph_cu : public field_solver_cu<Conf> {
//  private:
//   int m_damping_length = 64;
//   double m_damping_coef = 0.003;

//  public:
//   static std::string name() { return "field_solver"; }

//   using field_solver_cu<Conf>::field_solver_cu;

//   void init() override;
//   void update(double dt, uint32_t step) override;
//   void register_data_components() override;

//   void update_explicit(double dt, double time) override;
//   void update_semi_implicit(double dt, double alpha, double beta,
//                             double time) override;
// };

}  // namespace Aperture

