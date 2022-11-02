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

#include "field_solver_frame_dragging.h"
#include "core/constant_mem_func.h"
#include "framework/environment.h"
#include "framework/config.h"
#include "systems/helpers/finite_diff_helper.hpp"
#include "utils/kernel_helper.hpp"

namespace {

HOST_DEVICE float smooth_prof(float r_frac) {  
  return 0.5f * (1.0f - tanh((r_frac - 2.0f) / 0.2f));
  // return 1.0f;
}

}

namespace Aperture {

template <typename Conf>
void
field_solver_frame_dragging<Conf>::init() {
  field_solver<Conf>::init();

  sim_env().params().get_value("Rpc", m_Rpc);
  sim_env().params().get_value("R_star", m_Rstar);
  sim_env().params().get_value("beta0", m_beta0);
}

template <typename Conf>
void
field_solver_frame_dragging<Conf>::init_tmp_fields() {
  field_solver_cu<Conf>::init_tmp_fields();

  if (this->m_use_implicit == false) {
    this->m_enew = std::make_unique<vector_field<Conf>>(
        this->m_grid, field_type::edge_centered, MemType::device_only);
    this->m_enew->init();
  }
}

template <typename Conf>
void
field_solver_frame_dragging<Conf>::update_explicit(double dt, double time) {
  Logger::print_detail("Running explicit Cartesian solver with frame dragging!");
  using value_t = typename Conf::value_t;
  value_t beta0 = -m_beta0;
  value_t Rpc = m_Rpc;
  value_t Rstar = m_Rstar;
  value_t Rstar_over_RLC = square(Rpc / Rstar);

  kernel_launch(
      [beta0, Rpc, Rstar, Rstar_over_RLC] __device__(auto e, auto b0, auto e_out) {
        auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();

        for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
          auto pos = get_pos(idx, ext);
          value_t x = grid.pos(0, pos[0], false);
          value_t xs = grid.pos(0, pos[0], true);
          value_t y = grid.pos(1, pos[1], false);
          value_t ys = grid.pos(1, pos[1], true);
          value_t z = grid.pos(2, pos[2], false) + Rstar / Rpc;
          value_t zs = grid.pos(2, pos[2], true) + Rstar / Rpc;
          value_t r, beta, r_cyl;

          // Normally beta would have an additional factor of sinth, and r would
          // have a factor of sinth too, but they cancel each other, so we omit
          // both, in order to avoid sinth = 0 singularity

          // E[0] is staggered in y, z, but not in x
          r = math::sqrt(ys*ys + zs*zs + x*x);
          r_cyl = math::sqrt(x*x + ys*ys);
          beta = beta0 / square(r * Rpc / Rstar) * Rstar_over_RLC * smooth_prof(r_cyl / Rpc);
          e_out[0][idx] = e[0][idx] + beta * (x / r * b0[2][idx]);

          // E[1] is staggered in x, z, but not in y
          r = math::sqrt(y*y + zs*zs + xs*xs);
          r_cyl = math::sqrt(xs*xs + y*y);
          beta = beta0 / square(r * Rpc / Rstar) * Rstar_over_RLC * smooth_prof(r_cyl / Rpc);
          e_out[1][idx] = e[1][idx] + beta * (y / r * b0[2][idx]);

          // E[2] is staggered in x, y, but not in z
          r = math::sqrt(ys*ys + z*z + xs*xs);
          r_cyl = math::sqrt(xs*xs + ys*ys);
          beta = beta0 / square(r * Rpc / Rstar) * Rstar_over_RLC * smooth_prof(r_cyl / Rpc);
          e_out[2][idx] = e[2][idx] + beta * (-ys / r * b0[1][idx] - xs / r * b0[0][idx]);
        }
      },
      this->E->get_const_ptrs(), this->B0->get_const_ptrs(), this->m_enew->get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  this->compute_b_update_pml(*(this->B), *(this->m_enew), dt);
  this->compute_e_update_pml(*(this->E), *(this->B), *(this->J), dt);

  auto step = sim_env().get_step();
  if (step % this->m_data_interval == 0) {
    this->compute_divs_e_b();
  }

  CudaSafeCall(cudaDeviceSynchronize());
}



template class field_solver_frame_dragging<Config<3, float>>;
template class field_solver_frame_dragging<Config<3, double>>;

}
