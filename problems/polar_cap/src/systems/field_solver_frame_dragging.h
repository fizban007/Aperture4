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

#ifndef _FIELD_SOLVER_FRAME_DRAGGING_H_
#define _FIELD_SOLVER_FRAME_DRAGGING_H_

#include "systems/field_solver.h"

namespace Aperture {

// System that updates Maxwell equations using an explicit scheme in
// Cartesian coordinates
template <typename Conf>
class field_solver_frame_dragging : public field_solver_cu<Conf> {
 public:
  static std::string name() { return "field_solver"; }

  // field_solver(sim_environment& env, const grid_t<Conf>& grid,
  using field_solver_cu<Conf>::field_solver_cu;

  virtual void init() override;
  virtual void update_explicit(double dt, double time) override;

 protected:
  double m_a0 = 0.1; // This is the effective angular momentum a for GR
  double m_beta0 = 0.4 * 0.2; // This is a normalization factor for the GR shift vector
  double m_Rstar = 10.0;
  double m_Rpc = 1.0;
  virtual void init_tmp_fields() override;
};
}

#endif  // _FIELD_SOLVER_FRAME_DRAGGING_H_
