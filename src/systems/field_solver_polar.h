/*
 * Copyright (c) 2021 Alex Chen.
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
#include "systems/grid_polar.hpp"
#include "systems/policies/coord_policy_polar.hpp"
#include "utils/nonown_ptr.hpp"
#include <memory>

namespace Aperture {

// System that updates Maxwell equations using an explicit scheme in polar
// coordinates
template <typename Conf, template <class> class ExecPolicy>
class field_solver<Conf, ExecPolicy, coord_policy_polar>
    : public field_solver_base<Conf> {
 public:
  using value_t = typename Conf::value_t;

  field_solver(const grid_polar_t<Conf>& grid,
               const domain_comm<Conf, ExecPolicy>* comm = nullptr)
      : field_solver_base<Conf>(grid), m_grid_polar(grid), m_comm(comm) {}

  virtual ~field_solver() {}

  virtual void init() override;
  // virtual void update(double dt, uint32_t step) override;
  virtual void register_data_components() override;
  virtual void update_explicit(double dt, double time) override;
  virtual void update_semi_implicit(double dt, double alpha, double beta,
                                    double time) override;

  // virtual void compute_divs_e_b() override;
  // virtual void compute_flux() override;
  // virtual void compute_EB_sqr() override;

 private:
  const grid_polar_t<Conf>& m_grid_polar;
  const domain_comm<Conf, ExecPolicy>* m_comm = nullptr;

  int m_damping_length = 64;
  double m_damping_coef = 0.003;
};

}  // namespace Aperture

