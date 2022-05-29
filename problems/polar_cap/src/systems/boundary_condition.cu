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

  Logger::print_debug("Boundary condition Bp is {}", m_Bp);
  // injector =
  //     sim_env().register_system<ptc_injector<Conf, exec_policy_cuda>>(grid);
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
  apply_rotating_boundary();
  inject_plasma();
}

template <typename Conf> void boundary_condition<Conf>::apply_rotating_boundary() {
  typedef typename Conf::idx_t idx_t;
  value_t Bp = m_Bp;
  value_t Rpc = m_Rpc;
  value_t Rstar = m_Rstar;
  // dimensionless rotation omega = (r_pc/R_*)^2 c/R_*
  value_t omega = square(Rpc / Rstar);

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
                  0.5f * (1.0f - tanh((r_frac - 1.0f) / 0.1f));
              // value_t smooth_prof = (r_frac > 1.0f ? 0.0f : 1.0f);
              value_t E0 = -b0[2][idx] * omega * r * smooth_prof;
              value_t Br = math::sqrt(square(b0[0][idx]) + square(b0[1][idx]));

              e[0][idx] = E0 * x / r;
              e[1][idx] = E0 * y / r;
              // e[2][idx] = Br * omega * r * smooth_prof;
            }

            // Closed zone
            // z += Rstar;
            // value_t r = math::sqrt(x*x + y*y + z*z);
            // value_t r_max = r / (1.0f - square(z / r));
            // if (r_max / Rstar < 0.8 / omega) {
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

template <typename Conf> void boundary_condition<Conf>::inject_plasma() {
  // int inj_length = m_grid.guard[2];
  // auto num = ptc->number();

  // auto ext_inj = extent_t<3>(m_grid.reduced_dim(0),
  //                            m_grid.reduced_dim(1),
  //                            inj_length);
  // injector->inject(
  //     [rho_e_ptr, rho_p_ptr, inj_length, upstream_n] __device__(auto &pos, auto &grid, auto &ext) {
  //       if (pos[1] < inj_length + grid.guard[1] ||
  //           pos[1] >= grid.dims[1] - grid.guard[1] - inj_length) {
  //         auto idx = Conf::idx(pos, ext);
  //         return rho_p_ptr[idx] - rho_e_ptr[idx] < 2.0f - 3.0f / upstream_n;
  //       }
  //       // else if (pos[1] >= grid.dims[1] - grid.guard[1] - inj_length) {
  //       //   index_t<2> pos_inj(pos[0] - grid.guard[0], pos[1] - grid.dims[1] +
  //       //                                                  grid.guard[1] +
  //       //                                                  inj_length);
  //       //   auto idx = Conf::idx(pos_inj, ext_inj);
  //       //   return dens_e2[idx] + dens_p2[idx] < 2.0f - 4.0f / upstream_n;
  //       // }
  //       return false;
  //     },
  //     [] __device__(auto &pos, auto &grid, auto &ext) { return 2; },
  //     [upstream_kT] __device__(auto &pos, auto &grid, auto &ext, rng_t &rng,
  //                              PtcType type) {
  //       vec_t<value_t, 3> u_d = rng.maxwell_juttner_drifting<value_t>(upstream_kT, 0.995f);

  //       auto p1 = u_d[0];
  //       auto p2 = u_d[1];
  //       auto p3 = u_d[2];
  //       return vec_t<value_t, 3>(p1, p2, p3);
  //       // auto p1 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //       // auto p2 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //       // auto p3 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //       // // value_t gamma = math::sqrt(1.0f + p1*p1 + p2*p2 + p3*p3);
  //       // // value_t beta = p1 / gamma;
  //       // // return vec_t<value_t, 3>(beta / math::sqrt(1.0f - beta*beta), 0.0f, 0.0f);
  //       // return vec_t<value_t, 3>(p1, p2, p3);
  //     },
  //     // [upstream_n] __device__(auto &pos, auto &grid, auto &ext) {
  //     [upstream_n] __device__(auto& x_global) {
  //       return 1.0 / upstream_n;
  //     });

}

// Only instantiate this for 3D, otherwise lower boundary doesn't make sense
template class boundary_condition<Config<3>>;

} // namespace Aperture
