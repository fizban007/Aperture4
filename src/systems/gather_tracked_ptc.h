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
#include "data/tracked_ptc.h"
#include "framework/system.h"
#include "utils/nonown_ptr.hpp"

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
class gather_tracked_ptc : public system_t {
 public:
  static std::string name() { return "gather_tracked_ptc"; }

  gather_tracked_ptc(const grid_t<Conf>& grid) : m_grid(grid) {}

  virtual ~gather_tracked_ptc() {}

  virtual void register_data_components() override;
  virtual void init() override;
  virtual void update(double dt, uint32_t step) override;

  template <typename BufferType>
  void gather_tracked_ptc_index(const particles_base<BufferType>& ptc);

  template <typename T, typename Func>
  void gather_tracked_ptc_attr(buffer<T>& data, const buffer<uint32_t>& tracked_map,
                               size_t tracked_num, Func data_func);

  template <typename T, typename Func>
  void gather_tracked_ph_attr(buffer<T>& data, const buffer<uint32_t>& tracked_map,
                              size_t tracked_num, Func data_func);

 private:
  const grid_t<Conf>& m_grid;
  nonown_ptr<particle_data_t> ptc;
  nonown_ptr<photon_data_t> ph;
  nonown_ptr<tracked_particles_t> tracked_ptc;
  nonown_ptr<tracked_photons_t> tracked_ph;
  nonown_ptr<vector_field<Conf>> E, B;

  uint32_t m_ptc_output_interval = 1;
  size_t m_max_tracked = 0;
  buffer<uint32_t> m_tracked_num;
  buffer<uint32_t> m_tracked_map;
};

}  // namespace Aperture

