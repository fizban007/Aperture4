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

#ifndef __FIELD_SOLVER_SPH_H_
#define __FIELD_SOLVER_SPH_H_

#include "data/fields.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/field_solver.h"
#include "systems/grid_curv.h"
#include "utils/nonown_ptr.hpp"
#include <memory>

namespace Aperture {

// System that updates Maxwell equations using an explicit scheme in Spherical
// coordinates
template <typename Conf>
class field_solver_sph_cu : public field_solver_cu<Conf> {
 private:
  int m_damping_length = 64;
  double m_damping_coef = 0.003;

  nonown_ptr<scalar_field<Conf>> flux;

 public:
  static std::string name() { return "field_solver"; }

  using field_solver_cu<Conf>::field_solver_cu;

  void init() override;
  void update(double dt, uint32_t step) override;
  void register_data_components() override;

  void update_explicit(double dt, double time) override;
  void update_semi_implicit(double dt, double alpha, double beta,
                            double time) override;
};

}  // namespace Aperture

#endif
