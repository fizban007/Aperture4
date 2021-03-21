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
#include "core/math.hpp"
#include "framework/config.h"
#include "systems/grid.h"
#include "utils/kernel_helper.hpp"
#include "utils/util_functions.h"

namespace Aperture {

HOST_DEVICE Scalar
pml_sigma(Scalar x, Scalar xh, Scalar pmlscale, Scalar sig0) {
  if (x > xh)
    return sig0 * square((x - xh) / pmlscale);
  else
    return 0.0;
}

template <typename Conf>
boundary_condition<Conf>::boundary_condition(const grid_t<Conf>& grid)
    : m_grid(grid) {
  using multi_array_t = typename Conf::multi_array_t;

  sim_env().params().get_value("damping_coef", m_damping_coef);
  sim_env().params().get_value("Bp", m_Bp);
  sim_env().params().get_value("damping_length", m_damping_length);
  sim_env().params().get_value("pmllen", m_pmllen);
  sim_env().params().get_value("sigpml", m_sigpml);

  // m_prev_E1 = std::make_unique<multi_array_t>(
  //     extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  // m_prev_E2 = std::make_unique<multi_array_t>(
  //     extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  // m_prev_E3 = std::make_unique<multi_array_t>(
  //     extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  // m_prev_B1 = std::make_unique<multi_array_t>(
  //     extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  // m_prev_B2 = std::make_unique<multi_array_t>(
  //     extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  // m_prev_B3 = std::make_unique<multi_array_t>(
  //     extent(m_damping_length, m_grid.dims[1]), MemType::device_only);

  // m_prev_E1->assign_dev(0.0f);
  // m_prev_E2->assign_dev(0.0f);
  // m_prev_E3->assign_dev(0.0f);
  // m_prev_B1->assign_dev(0.0f);
  // m_prev_B2->assign_dev(0.0f);
  // m_prev_B3->assign_dev(0.0f);

  // m_prev_E.set_memtype(MemType::host_device);
  // m_prev_B.set_memtype(MemType::host_device);
  // m_prev_E.resize(3);
  // m_prev_B.resize(3);
  // m_prev_E[0] = m_prev_E1->dev_ptr();
  // m_prev_E[1] = m_prev_E2->dev_ptr();
  // m_prev_E[2] = m_prev_E3->dev_ptr();
  // m_prev_B[0] = m_prev_B1->dev_ptr();
  // m_prev_B[1] = m_prev_B2->dev_ptr();
  // m_prev_B[2] = m_prev_B3->dev_ptr();
  // m_prev_E.copy_to_device();
  // m_prev_B.copy_to_device();
}

template <typename Conf>
void
boundary_condition<Conf>::init() {
  sim_env().get_data("Edelta", E);
  sim_env().get_data("E0", E0);
  sim_env().get_data("Bdelta", B);
  sim_env().get_data("B0", B0);
  // sim_env().get_data("rand_states", &rand_states);
  sim_env().get_data("particles", ptc);
}

template <typename Conf>
void
boundary_condition<Conf>::update(double dt, uint32_t step) {
  typedef typename Conf::idx_t idx_t;
  typedef typename Conf::value_t value_t;

  value_t time = sim_env().get_time();
  value_t Bp = m_Bp;

  // Apply damping boundary condition on both Y boundaries
  kernel_launch(
      [Bp] __device__(auto e, auto b, auto prev_e, auto prev_b, auto damping_length,
                    auto damping_coef) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        // auto ext_damping = extent(damping_length, grid.dims[0]);
        for (auto n0 : grid_stride_range(0, grid.dims[0])) {
          // y = -y_max boundary
          for (int i = 0; i < damping_length; i++) {
            int n1 = i;
            auto idx = idx_t(index_t<2>(n0, n1), ext);
            value_t lambda =
                1.0f - damping_coef * cube((value_t)(damping_length - i) / (damping_length - 1));
            e[0][idx] *= lambda;
            e[1][idx] *= lambda;
            e[2][idx] *= lambda;
            b[0][idx] = lambda * (b[0][idx] + Bp) - Bp;
            b[1][idx] *= lambda;
            b[2][idx] *= lambda;
          }
          // y = y_max boundary
          for (int i = 0; i < damping_length; i++) {
            int n1 = grid.dims[0] - damping_length + i;
            auto idx = idx_t(index_t<2>(n0, n1), ext);
            value_t lambda =
                1.0f - damping_coef * cube((value_t)i / (damping_length - 1));
            e[0][idx] *= lambda;
            e[1][idx] *= lambda;
            e[2][idx] *= lambda;
            b[0][idx] = lambda * (b[0][idx] - Bp) + Bp;
            b[1][idx] *= lambda;
            b[2][idx] *= lambda;
          }
        }
      },
      E->get_ptrs(), B->get_ptrs(), m_prev_E.dev_ptr(), m_prev_B.dev_ptr(),
      m_damping_length, m_damping_coef);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template class boundary_condition<Config<2>>;

}  // namespace Aperture
