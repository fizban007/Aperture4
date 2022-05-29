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
#include "data/rng_states.h"
// #include "data/curand_states.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/domain_comm.h"
#include "systems/grid.h"
#include "systems/policies.h"
#include "systems/ptc_injector_new.h"
#include "utils/nonown_ptr.hpp"
#include <memory>

namespace Aperture {

template <typename Conf>
class boundary_condition : public system_t {
 public:
  using value_t = typename Conf::value_t;
  static std::string name() { return "boundary_condition"; }

  boundary_condition(const grid_t<Conf>& grid, const domain_comm<Conf>* comm = nullptr);

  void init() override;
  void update(double dt, uint32_t step) override;

  void apply_rotating_boundary();
  void inject_plasma();

 protected:
  const grid_t<Conf>& m_grid;
  const domain_comm<Conf>* m_comm;
  value_t m_Bp = 1.0e3;
  value_t m_Rpc = 1.0;
  value_t m_Rstar = 10.0;

  nonown_ptr<vector_field<Conf>> E, B, E0, B0;
  nonown_ptr<particle_data_t> ptc;
  nonown_ptr<rng_states_t> rng_states;
  nonown_ptr<ptc_injector<Conf, exec_policy_cuda>> injector;
  // curand_states_t *rand_states;

  // std::unique_ptr<typename Conf::multi_array_t> m_dens_e1, m_dens_e2, m_dens_p1, m_dens_p2;
  buffer<typename Conf::ndptr_t> m_prev_E, m_prev_B;

};

}

#endif // __BOUNDARY_CONDITION_H_
