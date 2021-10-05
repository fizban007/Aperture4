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

struct wpert_cart_t {
  float tp_start, tp_end, nT, dw0, y_start, y_end, q_e;

  HD_INLINE wpert_cart_t(float tp_s, float tp_e, float nT_, float dw0_, float qe)
      : tp_start(tp_s), tp_end(tp_e), nT(nT_), dw0(dw0_), q_e(qe) {
    y_start = 0.5f;
    y_end = 4.5f;
  }

  HD_INLINE Scalar operator()(Scalar t, Scalar x, Scalar y) {
    if (t >= tp_start && t <= tp_end && y > y_start && y < y_end) {
      Scalar omega =
          dw0 *
          math::sin((t - tp_start) * 2.0f * M_PI * nT / (tp_end - tp_start)) *
          math::sin(M_PI * (y - y_start) / (y_end - y_start));
          // math::sin((t - tp_start) * 2.0f * M_PI * nT / (tp_end - tp_start));
      return omega;
    } else {
      return 0.0;
    }
  }

  HD_INLINE Scalar j_x(Scalar t, Scalar x, Scalar y, Scalar theta) {
    return 0.0;
  }

  HD_INLINE Scalar j_y(Scalar t, Scalar x, Scalar y, Scalar theta) {
    return 0.0;
  }
};

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
}

template <typename Conf>
void
boundary_condition<Conf>::init() {
  sim_env().get_data("Edelta", &E);
  sim_env().get_data("E0", &E0);
  sim_env().get_data("Bdelta", &B);
  sim_env().get_data("B0", &B0);

  sim_env().params().get_value("a0", m_a0);
  sim_env().params().get_value("omega", m_omega);
  sim_env().params().get_value("num_lambda", m_num_lambda);
}

template <typename Conf>
void
boundary_condition<Conf>::update(double dt, uint32_t step) {
  typedef typename Conf::idx_t idx_t;
  typedef typename Conf::value_t value_t;

  value_t time = sim_env().get_time();
  value_t a0 = m_a0;
  value_t omega = m_omega;
  value_t num_lambda = m_num_lambda;

  // Apply wave boundary condition on the left side
  kernel_launch(
      [time, a0, omega, num_lambda] __device__(auto e, auto b) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto n1 : grid_stride_range(0, grid.dims[1])) {
          for (int i = 0; i < grid.guard[0] + 1; i++) {
            int n0 = i;
            auto idx = idx_t(index_t<2>(n0, n1), ext);
            if (omega * time * 0.5 / M_PI < num_lambda) {
              e[1][idx] = a0 * omega * math::sin(omega * time);
            }
          }
        }
      },
      E->get_ptrs(), B->get_ptrs());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  // Apply damping boundary condition on the other side
  // kernel_launch(
  //     [] __device__(auto e, auto b) {
  //       auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
  //       auto ext = grid.extent();
  //       auto ext_damping = extent(damping_length, grid.dims[1]);
  //       for (auto n1 : grid_stride_range(0, grid.dims[1])) {
  //         for (int i = 0; i < damping_length; i++) {
  //           int n0 = grid.dims[0] - damping_length + i;
  //           auto idx = idx_t(index_t<2>(n0, n1), ext);
  //           value_t lambda =
  //               1.0f - damping_coef * cube((value_t)i / (damping_length - 1));
  //           e[0][idx] *= lambda;
  //           e[1][idx] *= lambda;
  //           e[2][idx] *= lambda;
  //           b[0][idx] *= lambda;
  //           b[1][idx] *= lambda;
  //           b[2][idx] *= lambda;
  //           if (n0 > grid.dims[0] - 8) {
  //             e[0][idx] = 0.0f;
  //             e[1][idx] = 0.0f;
  //             e[2][idx] = 0.0f;
  //             b[0][idx] = 0.0f;
  //             b[1][idx] = 0.0f;
  //             b[2][idx] = 0.0f;
  //           }
  //         }
  //       }
  //     },
  //     E->get_ptrs(), B->get_ptrs());
  // CudaSafeCall(cudaDeviceSynchronize());
  // CudaCheckError();

}

template class boundary_condition<Config<2>>;

}  // namespace Aperture
