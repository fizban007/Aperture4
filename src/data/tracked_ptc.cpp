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

#include "tracked_ptc.h"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename BufferType>
tracked_ptc<BufferType>::tracked_ptc(size_t max_size, MemType type)
    : x1(max_size, type),
      x2(max_size, type),
      x3(max_size, type),
      p1(max_size, type),
      p2(max_size, type),
      p3(max_size, type),
      E(max_size, type),
#ifdef PARA_PERP
      work_para(max_size, type),
      work_perp(max_size, type),
#endif
      weight(max_size, type),
      flag(max_size, type),
      id(max_size, type) {
  m_size = max_size;
  m_number = 0;
}

template <typename BufferType>
tracked_ptc<BufferType>::~tracked_ptc() {}

// template <typename BufferType>
// void
// tracked_ptc<BufferType>::get_tracked_ptc_map(particles_base<BufferType> &data) {
//   // First obtain a map of the tracked indices
//   size_t number = data.number();
//   size_t max_tracked = m_size;
//   kernel_launch(
//       [number, max_tracked] LAMBDA(auto flags, auto cells, auto tracked_map,
//                                    auto tracked_num) {
//         for (auto n : grid_stride_range(0, number)) {
//           if (check_flag(flags[n], PtcFlag::tracked) &&
//               cells[n] != empty_cell) {
//             uint32_t nt = atomic_add(&tracked_num[0], 1);
//             if (nt < max_tracked) {
//               tracked_map[nt] = n;
//             }
//           }
//         }
//       },
//       data.flag.dev_ptr(), data.cell.dev_ptr(), m_tracked_map.dev_ptr(),
//       m_tracked_num.dev_ptr());
//   GpuSafeCall(gpuDeviceSynchronize());

//   // // Now fill the data array with the correct data
//   // kernel_launch(
//   //     [number, max_tracked] LAMBDA(auto ptc, auto tracked_map, auto tracked_num,
//   //                                  auto x1, auto x2, auto x3, auto p1, auto p2,
//   //                                  auto p3, auto E, auto weight, auto flag,
//   //                                  auto id) {
//   //       for (auto n : grid_stride_range(0, tracked_num[0])) {
//   //         x1[n] =
//   //       }
//   //     },
//   //     data.dev_ptrs(), m_tracked_map.dev_ptr(), m_tracked_num.dev_ptr(),
//   //     x1.dev_ptr(), x2.dev_ptr(), x3.dev_ptr(), p1.dev_ptr(), p2.dev_ptr(),
//   //     p3.dev_ptr(), E.dev_ptr(), weight.dev_ptr(), flag.dev_ptr(),
//   //     id.dev_ptr());
// }

template class tracked_ptc<ptc_buffer>;
template class tracked_ptc<ph_buffer>;

}  // namespace Aperture
