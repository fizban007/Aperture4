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
#include "data/particle_data.h"
// #include "data/curand_states.h"
#include "data/rng_states.h"
#include "data/data_array.hpp"
#include "framework/system.h"
#include "systems/grid_ks.h"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "utils/nonown_ptr.hpp"

namespace Aperture {

template <typename Conf>
class bh_injector : public system_t {
 public:
  using value_t = typename Conf::value_t;
  static std::string name() { return "bh_injector"; }

  bh_injector(const grid_ks_t<Conf>& grid)
      : m_grid(grid) {}
  virtual ~bh_injector() {}

  void init() override;
  void update(double dt, uint32_t step) override;
  void register_data_components() override;

 private:
  value_t m_inj_thr, m_sigma_thr, m_qe;
  const grid_ks_t<Conf>& m_grid;
  multi_array<int, Conf::dim> m_num_per_cell;
  multi_array<int, Conf::dim> m_cum_num_per_cell;

  // curand_states_t* m_rand_states;
  using exec_tag = typename exec_policy_dynamic<Conf>::exec_tag;
  nonown_ptr<rng_states_t<exec_tag>> m_rng_states;
  nonown_ptr<particle_data_t> ptc;
  nonown_ptr<vector_field<Conf>> B, D;
  nonown_ptr<scalar_field<Conf>> DdotB, Bmag, pair_injected;
  // std::vector<const scalar_field<Conf>*> Rho;
  data_array<scalar_field<Conf>> Rho;

  using rho_ptrs_t = buffer<typename Conf::ndptr_const_t>;
  rho_ptrs_t m_rho_ptrs;
};

}

