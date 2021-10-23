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

#ifndef _COMPUTE_LORENTZ_FACTOR_H_
#define _COMPUTE_LORENTZ_FACTOR_H_

#include "data/fields.h"
#include "data/particle_data.h"
#include "framework/system.h"
#include "systems/grid.h"
#include "utils/nonown_ptr.hpp"
#include <vector>

namespace Aperture {

template <typename Conf>
class compute_lorentz_factor : public system_t {
 public:
  static std::string name() { return "compute_lorentz_factor"; }

  // compute_lorentz_factor(sim_environment& env, const grid_t<Conf>& grid)
  compute_lorentz_factor(const grid_t<Conf>& grid)
      : m_grid(grid) {}
  virtual ~compute_lorentz_factor() {}

  virtual void register_data_components() override;
  virtual void init() override;
  virtual void update(double dt, uint32_t step) override;

 protected:
  const grid_t<Conf>& m_grid;
  int m_data_interval = 1;
  int m_num_species = 2;

  std::vector<nonown_ptr<scalar_field<Conf>>> gamma;
  std::vector<nonown_ptr<scalar_field<Conf>>> nums;
  std::vector<nonown_ptr<vector_field<Conf>>> avg_p;
  nonown_ptr<particle_data_t> ptc;

  virtual void register_data_impl(MemType type);
};

template <typename Conf>
class compute_lorentz_factor_cu : public compute_lorentz_factor<Conf> {
 public:
  typedef typename Conf::value_t value_t;
  static std::string name() { return "compute_lorentz_factor"; }

  compute_lorentz_factor_cu(const grid_t<Conf>& grid)
      : compute_lorentz_factor<Conf>(grid) {}
  virtual ~compute_lorentz_factor_cu() {}

  virtual void register_data_components() override;
  virtual void init() override;
  virtual void update(double dt, uint32_t step) override;

 protected:
  buffer<typename Conf::ndptr_t> m_gamma_ptrs;
  buffer<typename Conf::ndptr_t> m_nums_ptrs;
  buffer<vec_t<typename Conf::ndptr_t, 3>> m_avgp_ptrs;

  // std::vector<std::unique_ptr<scalar_field<Conf>>> m_nums;
};

}

#endif  // _COMPUTE_LORENTZ_FACTOR_H_
