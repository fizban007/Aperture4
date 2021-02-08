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

#ifndef __EXEC_POLICY_OMP_SIMD_H_
#define __EXEC_POLICY_OMP_SIMD_H_

#include "core/simd.h"
#include "exec_policy_host.hpp"
#include <omp.h>

namespace Aperture {

template <typename Conf>
class exec_policy_openmp_simd : public exec_policy_host<Conf> {
 public:
  template <typename Func, typename Idx, typename... Args>
  static void loop(Idx begin, Idx end, const Func& f, Args&&... args) {
#pragma omp parallel for
    for (auto idx = begin; idx < end - simd::vec_width;
         idx += simd::vec_width) {
      f(idx, args...);
    }

    // int iterated = ((end - begin) / simd::vec_width) * simd::vec_width;
    // if (iterated != end - begin) {
    //   // Use normal loop to process leftovers
    //   exec_policy_host<Conf>::loop(begin + iterated, end, f, args...);
    // }
  }
};

// template <typename Conf>
// using exec_policy_host = singleton_holder<exec_policy_host_impl<Conf>>;

}  // namespace Aperture

#endif  // __EXEC_POLICY_OMP_SIMD_H_
