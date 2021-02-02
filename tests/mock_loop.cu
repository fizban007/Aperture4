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

#include "core/data_adapter.h"
#include "core/multi_array.hpp"
#include "data/fields.h"
#include "framework/config.h"
#include "systems/policies/exec_policy_cuda.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include <iostream>
#include <type_traits>

using namespace Aperture;

template <>
struct cuda_adapter<int> {
  typedef float type;

  static inline type apply(int n) { return type(n) + 10.0f; }
};

template <>
struct cuda_adapter<int*> {
  typedef float type;

  static inline type apply(const int* n) { return type(*n) + 8.0f; }
};

// template <typename Func, typename... Args>
// void
// loop(const Func& f, size_t begin, size_t end, Args&&... args) {
// #ifdef __CUDACC__
//   kernel_launch(
//       [begin, end, f] __device__(
//           typename cuda_a<std::remove_reference_t<Args>>::type... args) {
//         for (auto idx : grid_stride_range(begin, end)) {
//           f(args...);
//         }
//       },
//       convert(args)...);
// #else

// #endif
// }

struct Wrapper {
  float a = 0.0;

  __device__ void operator()(idx_col_major_t<2> idx, float v) const {
    // printf("%f\n", v + a + idx.linear);
    auto ext = extent_t<2>(10, 10);
    auto pos = get_pos(idx, ext);
    printf("%d, %d\n", pos[0], pos[1]);
  }
};

int
main() {
  std::cout << sizeof(ndptr<float, 3>) << std::endl;
  std::cout << sizeof(buffer<float>) << std::endl;
  std::cout << sizeof(extent_t<3>) << std::endl;
  std::cout << sizeof(multi_array<float, 3>) << std::endl;
  std::cout << sizeof(field_t<3, Config<2, float>>) << std::endl;

  // idx_col_major_t<2>
  extent_t<2> ext(10, 10);

  int n = 8;
  auto w = Wrapper{3.0};
  exec_policy_cuda<Config<3>>::instance().loop(
      w, idx_col_major_t<2>(0, ext), idx_col_major_t<2>(ext.size(), ext), &n);
  cudaDeviceSynchronize();
  return 0;
}
