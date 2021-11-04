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

#ifndef __BOUNDARY_CONDITION_H_
#define __BOUNDARY_CONDITION_H_

#include "data/fields.h"
#include "data/particle_data.h"
// #include "data/curand_states.h"
#include "data/rng_states.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/grid_curv.h"
#include <memory>

namespace Aperture {

template <typename Conf>
class boundary_condition : public system_t {
  using value_t = typename Conf::value_t;
 protected:
  const grid_curv_t<Conf>& m_grid;
  value_t m_rpert1 = 5.0, m_rpert2 = 10.0;
  value_t m_tp_start, m_tp_end, m_nT, m_dw0, m_weight;

  vector_field<Conf> *E, *B, *E0, *B0;
  particle_data_t *ptc;
  rng_states_t *rng_states;

  buffer<float> m_surface_n;

 public:
  static std::string name() { return "boundary_condition"; }

  boundary_condition(const grid_curv_t<Conf>& grid) :
      m_grid(grid) {}

  void init() override;
  void update(double dt, uint32_t step) override;
};

}

#endif // __BOUNDARY_CONDITION_H_
