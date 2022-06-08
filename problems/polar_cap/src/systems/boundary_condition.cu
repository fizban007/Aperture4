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

#include "boundary_condition.h"
#include "core/math.hpp"
#include "framework/config.h"
#include "systems/grid.h"
#include "systems/policies/exec_policy_cuda.hpp"
#include "systems/ptc_injector_cuda.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf>
boundary_condition<Conf>::boundary_condition(const grid_t<Conf> &grid,
                                             const domain_comm<Conf> *comm)
    : m_grid(grid), m_comm(comm) {
  using multi_array_t = typename Conf::multi_array_t;

  sim_env().params().get_value("Bp", m_Bp);
  sim_env().params().get_value("Rpc", m_Rpc);
  sim_env().params().get_value("R_star", m_Rstar);
  sim_env().params().get_value("N_inject", m_Ninject);

  if (m_Ninject > 0) {
    injector =
        sim_env().register_system<ptc_injector<Conf, exec_policy_cuda>>(grid);
  }
}

template <typename Conf> void boundary_condition<Conf>::init() {
  sim_env().get_data("Edelta", E);
  sim_env().get_data("E0", E0);
  sim_env().get_data("Bdelta", B);
  sim_env().get_data("B0", B0);
  // sim_env().get_data("rand_states", &rand_states);
  // sim_env().get_data("particles", ptc);
  // sim_env().get_data("rng_states", rng_states);
  // sim_env().get_data("Rho_e", rho_e);
  // sim_env().get_data("Rho_p", rho_p);
}

template <typename Conf>
void boundary_condition<Conf>::update(double dt, uint32_t step) {
  apply_rotating_boundary(dt * step);
  inject_plasma(step);
}

template <typename Conf> void boundary_condition<Conf>::apply_rotating_boundary(double time) {
  typedef typename Conf::idx_t idx_t;
  value_t Bp = m_Bp;
  value_t Rpc = m_Rpc;
  value_t Rstar = m_Rstar;
  // dimensionless rotation omega = (r_pc/R_*)^2 c/R_*
  value_t omega = square(Rpc / Rstar);
  value_t prof_t = (time < 1.0 ? time : 1.0);
  omega *= prof_t;

  bool is_bottom = true;
  // 4 is the lower z boundary
  if (m_comm != nullptr && m_comm->domain_info().is_boundary[4] != true) {
    is_bottom = false;
  }

  // Apply rotation boundary condition on lower Z boundary and remove E.B in the
  // closed zone
  if (is_bottom) {
    kernel_launch(
        [Bp, Rstar, Rpc, omega] __device__(auto e, auto b0) {
          auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
          auto ext = grid.extent();
          for (auto idx : grid_stride_range(Conf::begin(ext),
                                            Conf::end(ext))) {
            auto pos = get_pos(idx, ext);
            value_t x = grid.pos(0, pos[0], false);
            value_t y = grid.pos(1, pos[1], false);
            value_t z = grid.pos(2, pos[2], false);

            // Rotating conductor
            if (pos[2] <= grid.guard[2] + 1) {
              value_t r = math::sqrt(x * x + y * y);
              value_t r_frac = r / Rpc;
              value_t smooth_prof =
                  0.5f * (1.0f - tanh((r_frac - 1.1f) / 0.15f));
              // value_t smooth_prof = (r_frac > 1.0f ? 0.0f : 1.0f);
              value_t E0 = -b0[2][idx] * omega * r * smooth_prof;
              value_t Br = math::sqrt(square(b0[0][idx]) + square(b0[1][idx]));

              e[0][idx] = E0 * x / r;
              e[1][idx] = E0 * y / r;
              if (pos[2] <= grid.guard[2]) {
                e[2][idx] = Br * omega * r * smooth_prof;
              }
            }

            // Closed zone
            // z += Rstar;
            // value_t r = math::sqrt(x*x + y*y + z*z);
            // value_t r_max = r / (1.0f - square(z / r));
            // if (r_max / Rstar < 1.0 / omega) {
            //   e[0][idx] = 0.0f;
            //   e[1][idx] = 0.0f;
            //   e[2][idx] = 0.0f;
            // }
          }
        },
        E->get_ptrs(), B0->get_ptrs());
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();
  }

}

template <typename Conf> void boundary_condition<Conf>::inject_plasma(int step) {
  if (m_Ninject == 0) return;

  int inj_length = m_grid.guard[2];
  int n_inject = m_Ninject;
  value_t Bp = m_Bp;
  value_t Rpc = m_Rpc;
  value_t Rstar = m_Rstar;
  // dimensionless rotation omega = (r_pc/R_*)^2 c/R_*
  value_t omega = square(Rpc / Rstar);
  auto num = ptc->number();

  bool is_bottom = true;
  // 4 is the lower z boundary
  if (m_comm != nullptr && m_comm->domain_info().is_boundary[4] != true) {
    is_bottom = false;
  }

  if (is_bottom && step % 10 == 0) {
  // if (step < 1) {
    injector->inject(
        [inj_length, Rpc] __device__(auto &pos, auto &grid, auto &ext) {
          if (pos[2] == inj_length) {
            value_t x = grid.pos(0, pos[0], false);
            value_t y = grid.pos(1, pos[1], false);
            value_t r = math::sqrt(x * x + y * y);

            if (r < 1.0f * Rpc) {
              return true;
            }
          }
          return false;
        },
        [n_inject] __device__(auto &pos, auto &grid, auto &ext) {
          return 2 * n_inject;
        },
        [] __device__(auto &pos, auto &grid, auto &ext, rng_t &rng,
                      PtcType type) {
          // vec_t<value_t, 3> u_d =
          // rng.maxwell_juttner_drifting<value_t>(upstream_kT, 0.995f);

          auto p1 = rng.gaussian(0.1f);
          auto p2 = 0.0f;
          auto p3 = 0.0f;
          return vec_t<value_t, 3>(p1, p2, p3);
          // return rng.maxwell_juttner_3d(0.1f);
        },
        // [upstream_n] __device__(auto &pos, auto &grid, auto &ext) {
        [n_inject, Bp, omega, Rpc] __device__(auto &x_global) {
          value_t r = math::sqrt(square(x_global[0]) + square(x_global[1]));
          value_t r_frac = r / Rpc;
          value_t smooth_prof = 0.5f * (1.0f - tanh((r_frac - 1.1f) / 0.2f));
          return 2.0f * Bp * omega * smooth_prof / n_inject;
          // return 2.0f * Bp * omega / n_inject;
        });
  }
}

// Only instantiate this for 3D, otherwise lower boundary doesn't make sense
template class boundary_condition<Config<3>>;

} // namespace Aperture
