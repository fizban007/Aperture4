/*
 * Copyright (c) 2021 Alex Chen.
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

#ifndef _GR_INVERSE_COMPTON_H_
#define _GR_INVERSE_COMPTON_H_

#include "core/cuda_control.h"
#include "core/particle_structs.h"
#include "core/random.h"
#include "framework/environment.h"
#include "systems/grid.h"
#include "systems/physics/ic_scattering.hpp"

namespace Aperture {

template <typename Conf>
struct gr_ic_radiation_scheme {
  using value_t = typename Conf::value_t;

  const grid_t<Conf>& m_grid;
  ic_scatter_t m_ic_module;
  value_t m_a = 0.99;

  gr_ic_radiation_scheme(const grid_t<Conf>& grid) : m_grid(grid) {}

  void init();

  HOST_DEVICE bool check_emit_photon(ptc_ptrs& ptc, size_t tid, rng_t& rng);

  HOST_DEVICE void emit_photon(ptc_ptrs& ptc, size_t tid, ph_ptrs& ph,
                               size_t offset, rng_t& rng);

  HOST_DEVICE bool check_produce_pair(ph_ptrs& ph, size_t tid, rng_t& rng);

  HOST_DEVICE void produce_pair(ph_ptrs& ph, size_t tid, ptc_ptrs& ptc,
                                size_t offset, rng_t& rng);
};

}  // namespace Aperture

#endif  // _GR_INVERSE_COMPTON_H_
