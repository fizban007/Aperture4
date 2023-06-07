/*
 * Copyright (c) 2023 Alex Chen.
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

#include "core/gpu_error_check.h"
#include "core/gpu_translation_layer.h"
#include "core/particle_structs.h"
#include "core/random.h"
#include "data/fields.h"
#include "data/multi_array_data.hpp"
#include "data/phase_space.hpp"
#include "framework/environment.h"
#include "systems/grid.h"
#include "systems/inverse_compton.h"
#include "systems/physics/ic_scattering.hpp"
#include "systems/physics/spectra.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf>
struct resonant_scattering_scheme {
  using value_t = typename Conf::value_t;

  const grid_t<Conf> &m_grid;
  // photon spectrum parameters
  int m_num_bins = 256;
  float m_lim_lower = 1.0e-4;
  float m_lim_upper = 1.0e4;
  int m_downsample = 16;
  mutable ndptr<value_t, Conf::dim> m_loss;
  mutable ndptr<value_t, Conf::dim> m_loss_total;
  mutable ndptr<value_t, 3> m_angle_dist_ptr;
  ndptr<float, Conf::dim + 1> m_spec_ptr;
  int m_ph_nth = 32;
  int m_ph_nphi = 64;
  extent_t<3> m_ext_ph_dist;

  resonant_scattering_scheme(const grid_t<Conf> &grid) : m_grid(grid) {}

  HOST_DEVICE size_t emit_photon(const Grid<Conf::dim, value_t> &grid,
                                 const extent_t<Conf::dim> &ext, ptc_ptrs &ptc,
                                 size_t tid, ph_ptrs &ph, size_t ph_num,
                                 unsigned long long int *ph_pos,
                                 rand_state &state, value_t dt) {
    // 1. Calculate resonant scattering cross section for the particle momentum
    // and local B field

    // 2. Determine if the outgoing photon is going to be capable of pair
    // production. If yes, then we move on to photon emission. If not, we add
    // the signal to the angular distribution

    // 3. Process photon emission

    return 0;
  }

  HOST_DEVICE size_t produce_pair(const Grid<Conf::dim, value_t> &grid,
                                  const extent_t<Conf::dim> &ext, ph_ptrs &ph,
                                  size_t tid, ptc_ptrs &ptc, size_t ptc_num,
                                  unsigned long long int *ptc_pos,
                                  rand_state &state, value_t dt) {
    return 0;
  }
};

}  // namespace Aperture
