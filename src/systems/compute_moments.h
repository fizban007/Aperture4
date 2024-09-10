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

  void init_components();

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
  data_array<scalar_field<Conf>> ptc_num;
  data_array<field_t<3, Conf>> ptc_flux;
  // This is the second moment, aka stress energy tensor. 0 component is T00,
  // 1-3 are T0i, 4-6 are Tii, and 7-9 are Tij
  data_array<scalar_field<Conf>> T00;
  data_array<field_t<3, Conf>> T0i;
  data_array<scalar_field<Conf>> T11;
  data_array<scalar_field<Conf>> T12;
  data_array<scalar_field<Conf>> T13;
  data_array<scalar_field<Conf>> T22;
  data_array<scalar_field<Conf>> T23;
  data_array<scalar_field<Conf>> T33;
};

}  // namespace Aperture
