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

#ifndef _GATHER_MOMENTUM_SPACE_H_
#define _GATHER_MOMENTUM_SPACE_H_

#include "core/multi_array.hpp"
#include "data/data_array.hpp"
#include "data/momentum_space.hpp"
#include "data/particle_data.h"
#include "data/phase_space.hpp"
#include "framework/system.h"
#include "systems/grid.h"
#include "systems/policies.h"
#include "utils/nonown_ptr.hpp"
#include <vector>

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
class gather_momentum_space : public system_t {
 public:
  static std::string name() { return "gather_momentum_space"; }

  gather_momentum_space(const grid_t<Conf>& grid) : m_grid(grid) {}
  virtual ~gather_momentum_space() {}

  virtual void register_data_components() override;
  virtual void init() override;
  virtual void update(double dt, uint32_t step) override;

 protected:
  const grid_t<Conf>& m_grid;
  int m_data_interval = 1;
  int m_downsample = 16;
  int m_num_species = 2;
  int m_num_bins[4] = {256, 256, 256, 256};
  float m_lim_lower[4] = {-1.0, -1.0, -1.0, 1.0};
  float m_lim_upper[4] = {1.0, 1.0, 1.0, 1.0e2};
  bool m_use_log_scale = false;

  // nonown_ptr<momentum_space<Conf>> momentum;
  data_array<phase_space<Conf, 3>> momenta;
  data_array<phase_space<Conf, 1>> energies;
  nonown_ptr<particle_data_t> ptc;
};

// template <typename Conf>
// class gather_momentum_space_cu : public gather_momentum_space<Conf> {
//  public:
//   typedef typename Conf::value_t value_t;
//   static std::string name() { return "gather_momentum_space"; }

//   gather_momentum_space_cu(const grid_t<Conf>& grid)
//       : gather_momentum_space<Conf>(grid) {}
//   virtual ~gather_momentum_space_cu() {}

//   // virtual void register_data_components() override;
//   // virtual void init() override;
//   virtual void update(double dt, uint32_t step) override;
// };

}  // namespace Aperture

#endif  // _GATHER_MOMENTUM_SPACE_H_
