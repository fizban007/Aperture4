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

#ifndef __FIELD_SOLVER_HELPER_CU_H_
#define __FIELD_SOLVER_HELPER_CU_H_

#include "core/cuda_control.h"
#include "data/fields.h"
#include "utils/kernel_helper.hpp"

namespace Aperture {

template <typename Conf>
void
add_alpha_beta_cu(vector_field<Conf>& result, const vector_field<Conf>& b1,
                  const vector_field<Conf>& b2, typename Conf::value_t alpha,
                  typename Conf::value_t beta) {
  auto ext = result.grid().extent();
  kernel_launch(
      [alpha, beta, ext] __device__(auto result, auto b1, auto b2) {
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = typename Conf::idx_t(n, ext);
          result[0][idx] = alpha * b1[0][idx] + beta * b2[0][idx];
          result[1][idx] = alpha * b1[1][idx] + beta * b2[1][idx];
          result[2][idx] = alpha * b1[2][idx] + beta * b2[2][idx];
        }
      },
      result.get_ptrs(), b1.get_ptrs(), b2.get_ptrs());
}


}

#endif // __FIELD_SOLVER_HELPER_CU_H_
