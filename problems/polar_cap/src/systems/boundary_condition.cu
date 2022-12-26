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
#include "systems/ptc_updater_base.h"
#include "utils/interpolation.hpp"
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
  sim_env().params().get_value("q_e", m_qe);
  sim_env().params().get_value("inj_weight", m_inj_weight);
  sim_env().params().get_value("pml_length", m_pml_length);

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
  sim_env().get_data("particles", ptc);
  sim_env().get_data("rng_states", rng_states);
  // sim_env().get_data("Rho_e", rho_e);
  // sim_env().get_data("Rho_p", rho_p);
  for (int i = 0; i < 3; i++) {
    m_E_ptr[i] = E->dev_ndptr(i);
  }

  auto pusher = sim_env().get_system("ptc_updater");
  if (dynamic_cast<ptc_updater_new<Conf, exec_policy_cuda, coord_policy_cartesian>*>(pusher.get()) != nullptr) {
    m_is_gca = false;
  }
}

template <typename Conf>
void boundary_condition<Conf>::update(double dt, uint32_t step) {
  apply_rotating_boundary(dt * step);
  inject_plasma(step, dt * step);
}

template <typename Conf> void boundary_condition<Conf>::apply_rotating_boundary(double time) {
  typedef typename Conf::idx_t idx_t;
  value_t Bp = m_Bp;
  value_t Rpc = m_Rpc;
  value_t Rstar = m_Rstar;
  // dimensionless rotation omega = (r_pc/R_*)^2 c/R_*
  value_t omega = square(Rpc / Rstar);
  value_t spinup_time = 5.0f;
  value_t prof_t = (time < spinup_time ? time / spinup_time : 1.0);
  omega *= prof_t;

  bool is_bottom = true;
  // 4 is the lower z boundary
  if (m_comm != nullptr && m_comm->domain_info().is_boundary[4] != true) {
    is_bottom = false;
  }

  int boundary_cell = m_grid.guard[2] + 1;
  // Apply rotation boundary condition on lower Z boundary and remove E.B in the
  // closed zone
  if (is_bottom) {
    kernel_launch(
        [Bp, Rstar, Rpc, omega, boundary_cell] __device__(auto e, auto b, auto b0) {
          auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
          auto ext = grid.extent();
          for (auto idx : grid_stride_range(Conf::begin(ext),
                                            Conf::end(ext))) {
            auto pos = get_pos(idx, ext);
            value_t x = grid.coord(0, pos[0], false);
            value_t xs = grid.coord(0, pos[0], true);
            value_t y = grid.coord(1, pos[1], false);
            value_t ys = grid.coord(1, pos[1], true);
            value_t z = grid.coord(2, pos[2], false);

            // Rotating conductor
            if (pos[2] <= boundary_cell) {
              value_t rx = math::sqrt(xs * xs + y * y);
              value_t r_frac = rx / Rpc;
              value_t smooth_prof =
                  0.5f * (1.0f - tanh((r_frac - 1.1f) / 0.2f));
              value_t E0 = -b0[2][idx] * omega * rx * smooth_prof * Rpc / Rstar;
              e[1][idx] = E0 * y / rx;

              value_t ry = math::sqrt(x * x + ys * ys);
              r_frac = ry / Rpc;
              smooth_prof =
                  0.5f * (1.0f - tanh((r_frac - 1.1f) / 0.2f));
              // value_t smooth_prof = (r_frac > 1.0f ? 0.0f : 1.0f);
              E0 = -b0[2][idx] * omega * ry * smooth_prof * Rpc / Rstar;
              e[0][idx] = E0 * x / ry;

              // b[0][idx] = 0.0;
              // b[1][idx] = 0.0;
              b[2][idx] = 0.0;
              value_t rxy = math::sqrt(x * x + ys * ys);
              r_frac = rxy / Rpc;
              smooth_prof =
                  0.5f * (1.0f - tanh((r_frac - 1.1f) / 0.2f));
              if (pos[2] < boundary_cell) {
                value_t Br = math::sqrt(square(b0[0][idx]) + square(b0[1][idx]));
                e[2][idx] = Br * omega * math::sqrt(xs*xs + ys*ys) * smooth_prof;
                b[0][idx] = 0.0;
                b[1][idx] = 0.0;
              }
            }

            // Closed zone
            // z += Rstar;
            // value_t r = math::sqrt(x*x + y*y + z*z);
            // value_t r_max = r / (1.0f - square(z / r));
            // value_t r_frac = Rstar / r_max / omega;
            // value_t smooth_prof =
            //       0.5f * (1.0f - tanh((r_frac - 1.1f) / 0.2f));
            // if (r_max / Rstar < 1.0 / omega) {
            //   e[0][idx] *= smooth_prof;
            //   e[1][idx] *= smooth_prof;
            //   e[2][idx] *= smooth_prof;
            // }
          }
        },
        E->get_ptrs(), B->get_ptrs(), B0->get_ptrs());
    CudaSafeCall(cudaDeviceSynchronize());
    CudaCheckError();
  }

}

template <typename Conf> void boundary_condition<Conf>::inject_plasma(int step, double time) {
  if (m_Ninject == 0) return;
  // if (time > 1.5) return;

  // int inj_length = m_grid.guard[2] + 10;
  int inj_length = m_grid.guard[2] + 1;
  int n_inject = m_Ninject;
  value_t Bp = m_Bp;
  value_t Rpc = m_Rpc;
  value_t Rstar = m_Rstar;
  // dimensionless rotation omega = (r_pc/R_*)^2 c/R_*
  value_t omega = square(Rpc / Rstar);
  value_t qe = m_qe;
  auto E_ptr = m_E_ptr;

  bool is_bottom = true;
  // 4 is the lower z boundary
  if (m_comm != nullptr && m_comm->domain_info().is_boundary[4] != true) {
    is_bottom = false;
  }

  auto num = ptc->number();
  auto max_num = ptc->size();
  value_t inj_weight = m_inj_weight;
  bool is_gca = m_is_gca;

  if (is_bottom && step % 1 == 0 && time >= 0.0) {
    kernel_launch([n_inject, E_ptr, qe, num, max_num, inj_length, inj_weight, Rpc, is_gca]
                  __device__ (auto ptc, auto states) {
          auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
          auto ext = grid.extent();
          rng_t<exec_tags::device> rng(states);
          auto interp = interp_t<1, Conf::dim>{};

          for (auto n : grid_stride_range(0, ext[0] * ext[1])) {
            index_t<3> pos(n % ext[0], n / ext[0], inj_length);
            value_t coordx = grid.coord(0, pos[0], false);
            value_t coordy = grid.coord(1, pos[1], false);
            value_t r = math::sqrt(coordx * coordx + coordy * coordy);

            if (r > 1.7f * Rpc) {
              continue;
            }
            auto idx = Conf::idx(pos, ext);
            // if (pos[0] < grid.guard[0] || pos[0] >= grid.guard[0] + grid.N[0])
            //   continue;
            // if (pos[1] < grid.guard[1] || pos[1] >= grid.guard[1] + grid.N[1])
            //   continue;
            // if (math::abs(E_ptr[2][idx]) < 1.0e-3 * Bp)
            //   continue;

            for (int i = 0; i < n_inject; i++) {
              uint32_t offset_e = num + n_inject * n * 2 + i * 2;
              uint32_t offset_p = offset_e + 1;
              if (offset_e >= max_num || offset_p >= max_num) {
                break;
              }

              ptc.cell[offset_e] = ptc.cell[offset_p] = (uint32_t)idx.linear;
              auto x = vec_t<value_t, 3>(rng.uniform<value_t>(),
                                         rng.uniform<value_t>(),
                                         0.05f);
              ptc.x1[offset_e] = ptc.x1[offset_p] = x[0];
              ptc.x2[offset_e] = ptc.x2[offset_p] = x[1];
              ptc.x3[offset_e] = ptc.x3[offset_p] = x[2];

              // ptc.p1[offset_e] = rng.gaussian(0.1);
              // ptc.p1[offset_p] = rng.gaussian(0.1);
              ptc.p1[offset_e] = ptc.p1[offset_p] = 0.0f;
              ptc.p2[offset_e] = ptc.p2[offset_p] = 0.0f;
              ptc.p3[offset_e] = ptc.p3[offset_p] = 0.0f;
              ptc.E[offset_e] = ptc.E[offset_p] = 1.0f;

              auto Ez = interp(x, E_ptr[2], idx, ext, stagger_t(0b011));
              // auto Ez = 1.0f;
              ptc.weight[offset_e] = ptc.weight[offset_p] = inj_weight * math::abs(Ez) / n_inject / qe;
              ptc.flag[offset_e] = set_ptc_type_flag(0, PtcType::electron);
              ptc.flag[offset_p] = set_ptc_type_flag(0, PtcType::positron);
            }
          }
      }, ptc->get_dev_ptrs(), rng_states->states().dev_ptr());
    CudaSafeCall(cudaDeviceSynchronize());
    ptc->add_num(2 * n_inject * m_grid.dims[0] * m_grid.dims[1]);
  }
}

// Only instantiate this for 3D, otherwise lower boundary doesn't make sense
template class boundary_condition<Config<3>>;

} // namespace Aperture
