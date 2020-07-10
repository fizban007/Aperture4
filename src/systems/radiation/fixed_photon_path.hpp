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

#ifndef __FIXED_PHOTON_PATH_H_
#define __FIXED_PHOTON_PATH_H_

#include "core/constant_mem.h"
#include "core/cuda_control.h"
#include "core/particle_structs.h"
#include "data/curand_states.h"

namespace Aperture {

template <typename Conf>
struct fixed_photon_path {
  float E_s = 2.0f;
  float photon_path = 0.0f;

  __device__ void emit_photon(ptc_ptrs& ptc, uint32_t tid, ph_ptrs& ph,
                              uint32_t offset, cuda_rng_t& rng) {
    using value_t = typename Conf::value_t;
    value_t p1 = ptc.p1[tid];
    value_t p2 = ptc.p2[tid];
    value_t p3 = ptc.p3[tid];
    // value_t gamma = sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
    value_t gamma = ptc.E[tid];
    value_t pi = std::sqrt(gamma * gamma - 1.0f);

    value_t u = rng();
    value_t Eph = 2.5f + u * (E_s - 1.0f) * 2.0f;
    value_t pf = std::sqrt(square(gamma - Eph) - 1.0f);

    ptc.p1[tid] = p1 * pf / pi;
    ptc.p2[tid] = p2 * pf / pi;
    ptc.p3[tid] = p3 * pf / pi;
    ptc.E[tid] = gamma - Eph;

    // auto c = ptc.cell[tid];
    // auto& grid = dev_grid<Conf::dim>();
    // auto idx = typename Conf::idx_t(c, grid.extent());
    // auto pos = idx.get_pos();
    // value_t theta = grid.pos<1>(pos[0], ptc.x2[tid]);
    // value_t lph = min(10.0f, (1.0f / std::sin(theta) - 1.0f) * photon_path);
    // If photon energy is too low, do not track it, but still
    // subtract its energy as done above
    // if (std::abs(Eph) < dev_params.E_ph_min) continue;
    // if (theta < 0.005f || theta > M_PI - 0.005f) return;

    value_t lph = photon_path;
    u = rng();
    // Add the new photo
    value_t path = lph * (0.9f + 0.2f * u);
    // if (path > dev_params.r_cutoff) return;
    // printf("Eph is %f, path is %f\n", Eph, path);
    ph.x1[offset] = ptc.x1[tid];
    ph.x2[offset] = ptc.x2[tid];
    ph.x3[offset] = ptc.x3[tid];
    ph.p1[offset] = Eph * p1 / pi;
    ph.p2[offset] = Eph * p2 / pi;
    ph.p3[offset] = Eph * p3 / pi;
    ph.E[offset] = Eph;
    ph.weight[offset] = ptc.weight[tid];
    ph.path_left[offset] = path;
    ph.cell[offset] = ptc.cell[tid];
  }

  __device__ bool check_produce_pair(ph_ptrs& ph, uint32_t tid,
                                     cuda_rng_t& rng) {
    return ph.path_left[tid] < 0.0f;
  }
};

}  // namespace Aperture

#endif  // __FIXED_PHOTON_PATH_H_
