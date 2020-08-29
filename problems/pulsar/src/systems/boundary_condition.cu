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

#include "boundary_condition.h"
#include "framework/config.h"
#include "systems/grid_sph.h"
#include "utils/kernel_helper.hpp"

namespace Aperture {

template <typename Conf>
void
boundary_condition<Conf>::init() {
  m_env.get_data("Edelta", &E);
  m_env.get_data("E0", &E0);
  m_env.get_data("Bdelta", &B);
  m_env.get_data("B0", &B0);

  m_env.params().get_value("omega", m_omega_0);
}

template <typename Conf>
void
boundary_condition<Conf>::update(double dt, uint32_t step) {
  auto ext = m_grid.extent();
  typedef typename Conf::idx_t idx_t;
  typedef typename Conf::value_t value_t;

  value_t time = m_env.get_time();
  value_t omega;
  value_t atm_time = 0.0;
  value_t sp_time = 10.0;
  // if (m_omega_t * time < 5000.0)
  if (time <= atm_time) {
    omega = 0.0;
  } else if (time <= atm_time + sp_time) {
    // omega = env.params().omega *
    //         square(std::sin(CONST_PI * 0.5 * (time / 10.0)));
    omega = m_omega_0 * ((time - atm_time) / sp_time);
  } else {
    omega = m_omega_0;
  }
  Logger::print_info("time is {}, Omega is {}", time, omega);

  kernel_launch([ext, time, omega] __device__ (auto e, auto b, auto e0, auto b0) {
      auto& grid = dev_grid<Conf::dim>();
      auto ext = grid.extent();

      // for (auto idx : grid_stride_range(Conf::begin(ext), Conf::end(ext))) {
      for (auto n : grid_stride_range(0, ext.size())) {
        auto idx = Conf::idx(n, ext);
        auto pos = idx.get_pos();
        value_t r = grid_sph_t<Conf>::radius(grid.template pos<0>(pos[0], false));
        value_t r_s = grid_sph_t<Conf>::radius(grid.template pos<0>(pos[0], true));
        value_t r_reduced = grid_sph_t<Conf>::radius(grid.template pos<0>(pos[0] - 1, false));
        // value_t theta = grid_sph_t<Conf>::radius(grid.template pos<1>(pos[1], false));
        // value_t theta_s = grid_sph_t<Conf>::radius(grid.template pos<1>(pos[1], true));

        if (r < 1.0f) {
          value_t theta_s = grid_sph_t<Conf>::theta(grid.template pos<1>(pos[1], true));
          e[0][idx] = omega * sin(theta_s) * r * b0[1][idx];
          b[1][idx] = 0.0f;
          b[2][idx] = 0.0f;
        }

        if (r_reduced < 1.0f) {
          value_t theta = grid_sph_t<Conf>::theta(grid.template pos<1>(pos[1], false));
          b[0][idx] = 0.0f;
          e[1][idx] = -omega * sin(theta) * r_s * b0[0][idx];
          e[2][idx] = 0.0f;
        }
      }
      // for (auto n1 : grid_stride_range(0, grid.dims[1])) {
      //   value_t theta = grid_sph_t<Conf>::theta(grid.template pos<1>(n1, false));
      //   value_t theta_s = grid_sph_t<Conf>::theta(grid.template pos<1>(n1, true));

      //   // For quantities that are not continuous across the surface
      //   for (int n0 = 0; n0 < grid.skirt[0] + 4; n0++) {
      //     auto idx = idx_t(index_t<2>(n0, n1), ext);
      //     value_t r = grid_sph_t<Conf>::radius(grid.template pos<0>(n0, false));
      //     e[0][idx] = omega * sin(theta_s) * r * b0[1][idx];
      //     b[1][idx] = 0.0;
      //     b[2][idx] = 0.0;
      //   }
      //   // For quantities that are continuous across the surface
      //   for (int n0 = 0; n0 < grid.skirt[0] + 5; n0++) {
      //     auto idx = idx_t(index_t<2>(n0, n1), ext);
      //     value_t r_s = grid_sph_t<Conf>::radius(grid.template pos<0>(n0, true));
      //     b[0][idx] = 0.0;
      //     e[1][idx] = -omega * sin(theta) * r_s * b0[0][idx];
      //     e[2][idx] = 0.0;
      //   }
      // }
    }, E->get_ptrs(), B->get_ptrs(), E0->get_ptrs(), B0->get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
}


template class boundary_condition<Config<2>>;

}
