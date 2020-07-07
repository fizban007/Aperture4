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

#ifndef __PHOTON_PAIR_CREATION_H_
#define __PHOTON_PAIR_CREATION_H_

#include "core/cuda_control.h"
#include "core/math.hpp"
#include "core/particle_structs.h"
#include "data/curand_states.h"
#include "utils/util_functions.h"

namespace Aperture {

template <typename value_t>
struct photon_pair_creation_t {
  HOST_DEVICE void produce_pair(ph_ptrs& ph, uint32_t tid, ptc_ptrs& ptc,
                                uint32_t offset, cuda_rng_t& rng) const {
    value_t p1 = ph.p1[tid];
    value_t p2 = ph.p2[tid];
    value_t p3 = ph.p3[tid];
    value_t Eph2 = p1 * p1 + p2 * p2 + p3 * p3;
    if (Eph2 < 4.01f) Eph2 = 4.01f;

    value_t ratio = math::sqrt(0.25f - 1.0f / Eph2);
    value_t gamma = math::sqrt(1.0f + ratio * ratio * Eph2);
    uint32_t offset_e = offset;
    uint32_t offset_p = offset + 1;

    ptc.x1[offset_e] = ptc.x1[offset_p] = ph.x1[tid];
    ptc.x2[offset_e] = ptc.x2[offset_p] = ph.x2[tid];
    ptc.x3[offset_e] = ptc.x3[offset_p] = ph.x3[tid];
    // printf("x1 = %f, x2 = %f, x3 = %f\n",
    // ptc.x1[offset_e],
    // ptc.x2[offset_e], ptc.x3[offset_e]);

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

    // Set this photon to be empty
    ph.cell[tid] = empty_cell;
  }
};

}  // namespace Aperture

#endif  // __PHOTON_PAIR_CREATION_H_
