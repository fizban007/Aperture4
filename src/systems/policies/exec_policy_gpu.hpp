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

#pragma once

#include "core/constant_mem.h"
#include "core/constant_mem_func.h"
#include "core/cuda_control.h"
#include "core/data_adapter.h"
#include "core/exec_tags.h"
#include "systems/grid.h"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include "utils/singleton_holder.h"

namespace Aperture {

template <typename Conf>
class exec_policy_gpu {
 public:
  using exec_tag = exec_tags::device;

  static void set_grid(const grid_t<Conf>& grid) {
    // init_dev_grid<Conf::dim, typename Conf::value_t>(grid);
  }

  template <typename Func, typename... Args>
  static void launch(const Func& f, Args&&... args) {
    kernel_launch(f, adapt(exec_tags::device{}, args)...);
    GpuCheckError();
  }

  template <typename Func, typename Idx, typename... Args>
  static __device__ void loop(Idx begin, type_identity_t<Idx> end,
                              const Func& f, Args&&... args) {
    for (auto idx : grid_stride_range(begin, end)) {
      f(idx, args...);
    }
  }

  static void sync() {
    GpuSafeCall(gpuDeviceSynchronize());
    GpuCheckError();
  }

  static __device__ const Grid<Conf::dim, typename Conf::value_t>& grid() {
#if defined(__CUDACC__) || defined(__HIPCC__)
    return dev_grid<Conf::dim, typename Conf::value_t>();
#else
    return Grid<Conf::dim, typename Conf::value_t>{};
#endif
  }

  static MemType data_mem_type() { return MemType::host_device; }
  static MemType tmp_mem_type() { return MemType::device_only; }
  static MemType debug_mem_type() { return MemType::device_managed; }
};

// template <typename Conf>
// using exec_policy_gpu = singleton_holder<exec_policy_gpu_impl<Conf>>;
// using exec_policy_gpu = singleton_holder<exec_policy_gpu_impl>;

}  // namespace Aperture
