/*
 * Copyright (c) 2024 Alex Chen.
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

#include "systems/compute_moments.h"
#include "data/fields.h"
#include "data/particle_data.h"
#include "data/data_array.hpp"
#include "framework/system.h"
#include "systems/grid_ks.h"
#include "utils/nonown_ptr.hpp"
#include <vector>

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
class compute_moments_gr_ks : public compute_moments<Conf, ExecPolicy> {
 public:
  typedef typename Conf::value_t value_t;
  static std::string name() { return "compute_moments_gr_ks"; }

  compute_moments_gr_ks(const grid_ks_t<Conf>& grid) : 
    compute_moments<Conf, ExecPolicy>(grid), m_grid(grid) {}
  virtual ~compute_moments_gr_ks() {}

  virtual void register_data_components() override;
  virtual void init() override;
  virtual void update(double dt, uint32_t step) override;

 private:
  const grid_ks_t<Conf>& m_grid; 
};

}
