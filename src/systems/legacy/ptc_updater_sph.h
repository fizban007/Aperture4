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

#ifndef _PTC_UPDATER_SPH_H_
#define _PTC_UPDATER_SPH_H_

#include "ptc_updater_old.h"
#include "systems/grid_sph.hpp"

namespace Aperture {

template <typename Conf>
class ptc_updater_old_sph_cu : public ptc_updater_old_cu<Conf> {
 public:
  typedef typename Conf::value_t value_t;
  static std::string name() { return "ptc_updater"; }

  using ptc_updater_old_cu<Conf>::ptc_updater_old_cu;

  void init() override;
  void register_data_components() override;

  virtual void move_deposit_2d(value_t dt, uint32_t step) override;
  virtual void move_photons_2d(value_t dt, uint32_t step) override;
  virtual void filter_field(vector_field<Conf>& f, int comp) override;
  virtual void filter_field(scalar_field<Conf>& f) override;
  virtual void fill_multiplicity(int mult, value_t weight = 1.0) override;

 protected:
  value_t m_compactness = 0.0;
  value_t m_omega = 0.0;
  int m_damping_length = 32;
  float m_r_cutoff = 10.0;

};


}

#endif  // _PTC_UPDATER_SPH_H_
