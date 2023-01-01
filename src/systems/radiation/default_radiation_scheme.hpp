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

#pragma once

#include "fixed_photon_path.hpp"
#include "photon_pair_creation.hpp"
#include "systems/grid.h"
#include "threshold_emission.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf>
// struct default_radiation_scheme : public fixed_photon_path<Conf>,
//                                   public photon_pair_creation<Conf>,
//                                   public threshold_emission {
struct default_radiation_scheme {
  using value_t = typename Conf::value_t;

  value_t E_s = 2.0f;
  value_t photon_path = 0.0f;
  value_t gamma_thr = 30.0f;
  int ph_per_scatter = 1;

  default_radiation_scheme(const grid_t<Conf>& grid) {}

  void init() {
    sim_env().params().get_value("photon_path", photon_path);
    sim_env().params().get_value("E_secondary", E_s);
    sim_env().params().get_value("gamma_thr", gamma_thr);
    sim_env().params().get_value("ph_per_scatter", ph_per_scatter);

    if (E_s * 2.0f * ph_per_scatter > gamma_thr) {
      throw(std::runtime_error(
          "Total energy of secondaries exceed gamma threshold!"));
    }
  }

  HOST_DEVICE size_t emit_photon(const Grid<Conf::dim, value_t>& grid,
                                 const extent_t<Conf::dim>& ext, ptc_ptrs& ptc,
                                 size_t tid, ph_ptrs& ph, size_t ph_num,
                                 unsigned long long int* ph_pos,
                                 rand_state& state, value_t dt) {
    value_t gamma = ptc.E[tid];
    if (gamma < gamma_thr) {
      return 0;  // 0 means no photon is produced
    }

    value_t p1 = ptc.p1[tid];
    value_t p2 = ptc.p2[tid];
    value_t p3 = ptc.p3[tid];
    // value_t gamma = sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
    value_t pi = std::sqrt(gamma * gamma - 1.0f);

    // value_t u = rng.uniform<value_t>();
    value_t Eph = E_s * 2.0f;
    value_t pf = std::sqrt(square(gamma - Eph * ph_per_scatter) - 1.0f);

    ptc.p1[tid] = p1 * pf / pi;
    ptc.p2[tid] = p2 * pf / pi;
    ptc.p3[tid] = p3 * pf / pi;
    ptc.E[tid] = gamma - Eph * ph_per_scatter;

    size_t offset = ph_num + atomic_add(ph_pos, ph_per_scatter);
    value_t lph = photon_path;
    // Add the new photo
    // printf("Eph is %f, path is %f\n", Eph, path);
    for (int i = 0; i < ph_per_scatter; i++) {
      value_t u = rng_uniform<value_t>(state);
      value_t path = lph * (0.9f + 0.2f * u);
      ph.x1[offset + i] = ptc.x1[tid];
      ph.x2[offset + i] = ptc.x2[tid];
      ph.x3[offset + i] = ptc.x3[tid];
      ph.p1[offset + i] = Eph * p1 / pi;
      ph.p2[offset + i] = Eph * p2 / pi;
      ph.p3[offset + i] = Eph * p3 / pi;
      ph.E[offset + i] = Eph;
      ph.weight[offset + i] = ptc.weight[tid];
      ph.path_left[offset + i] = path;
      ph.cell[offset + i] = ptc.cell[tid];
    }
    return offset;
  }

  HOST_DEVICE size_t produce_pair(const Grid<Conf::dim, value_t>& grid,
                                  const extent_t<Conf::dim>& ext, ph_ptrs& ph,
                                  size_t tid, ptc_ptrs& ptc, size_t ptc_num,
                                  unsigned long long int* ptc_pos,
                                  rand_state& state, value_t dt) {
    value_t path_left = ph.path_left[tid];
    if (path_left > 0.0f) {
      return 0;  // 0 means no pairs are produced
    }

    value_t p1 = ph.p1[tid];
    value_t p2 = ph.p2[tid];
    value_t p3 = ph.p3[tid];
    value_t Eph2 = p1 * p1 + p2 * p2 + p3 * p3;
    if (Eph2 < 4.01f) Eph2 = 4.01f;

    value_t ratio = math::sqrt(0.25f - 1.0f / Eph2);
    value_t gamma = math::sqrt(1.0f + ratio * ratio * Eph2);
    size_t offset = ptc_num + atomic_add(ptc_pos, 2);
    size_t offset_e = offset;
    size_t offset_p = offset + 1;

    ptc.x1[offset_e] = ptc.x1[offset_p] = ph.x1[tid];
    ptc.x2[offset_e] = ptc.x2[offset_p] = ph.x2[tid];
    ptc.x3[offset_e] = ptc.x3[offset_p] = ph.x3[tid];

    ptc.p1[offset_e] = ptc.p1[offset_p] = ratio * p1;
    ptc.p2[offset_e] = ptc.p2[offset_p] = ratio * p2;
    ptc.p3[offset_e] = ptc.p3[offset_p] = ratio * p3;
    ptc.E[offset_e] = ptc.E[offset_p] = gamma;

#ifndef NDEBUG
    assert(ptc.cell[offset_e] == empty_cell);
    assert(ptc.cell[offset_p] == empty_cell);
#endif
    ptc.weight[offset_e] = ptc.weight[offset_p] = ph.weight[tid];
    ptc.cell[offset_e] = ptc.cell[offset_p] = ph.cell[tid];
    ptc.flag[offset_e] =
        set_ptc_type_flag(flag_or(PtcFlag::secondary), PtcType::electron);
    ptc.flag[offset_p] =
        set_ptc_type_flag(flag_or(PtcFlag::secondary), PtcType::positron);

    return offset;
  }
};

}  // namespace Aperture
