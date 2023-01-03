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

#pragma once

#include "data/fields.h"
#include "data/particle_data.h"
#include "data/data_array.hpp"
#include "framework/system.h"
#include "systems/grid.h"
#include "utils/nonown_ptr.hpp"
#include <vector>

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
class compute_moments : public system_t {
 public:
  typedef typename Conf::value_t value_t;
  static std::string name() { return "compute_moments"; }

  compute_moments(const grid_t<Conf>& grid) : m_grid(grid) {}
  virtual ~compute_moments() {}

  virtual void register_data_components() override;
  virtual void init() override;
  virtual void update(double dt, uint32_t step) override;

 protected:
  const grid_t<Conf>& m_grid;
  bool m_compute_first_moments = true;
  bool m_compute_second_moments = true;
  bool m_photon_data = false;
  int m_num_species = 2;
  int m_size = 2;
  int m_fld_interval = 100;

  nonown_ptr<particles_t> ptc;
  nonown_ptr<photons_t> ph;

  // This is the first moment, aka particle flux
  data_array<field_t<4, Conf>> S;
  // This is the second moment, aka stress energy tensor. 0 component is T00,
  // 1-3 are T0i, 4-6 are Tii, and 7-9 are Tij
  data_array<field_t<10, Conf>> T;
};

}  // namespace Aperture
