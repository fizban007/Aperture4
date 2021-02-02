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

#ifndef __EXEC_POLICY_CUDA_H_
#define __EXEC_POLICY_CUDA_H_

#include "core/data_adapter.h"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include "utils/singleton_holder.h"

namespace Aperture {

template <typename Conf>
class exec_policy_cuda_impl {
 public:
  exec_policy_cuda_impl() = default;

  exec_policy_cuda_impl(const exec_policy_cuda_impl<Conf>&) = delete;
  exec_policy_cuda_impl(exec_policy_cuda_impl<Conf>&&) = delete;
  exec_policy_cuda_impl<Conf>& operator=(const exec_policy_cuda_impl<Conf>&) =
      delete;
  exec_policy_cuda_impl<Conf>& operator=(exec_policy_cuda_impl<Conf>&&) =
      delete;

  template <typename Func, typename Idx, typename... Args>
  void loop(const Func& f, Idx begin, Idx end, Args&&... args) {
    kernel_launch(
        [begin, end, f] __device__(
            typename cuda_adapter<std::remove_reference_t<Args>>::type... args) {
          for (auto idx : grid_stride_range(begin, end)) {
            f(idx, args...);
          }
        },
        adapt_cuda(args)...);
  }
};

template <typename Conf>
using exec_policy_cuda = singleton_holder<exec_policy_cuda_impl<Conf>>;

}  // namespace Aperture

#endif  // __EXEC_POLICY_CUDA_H_
