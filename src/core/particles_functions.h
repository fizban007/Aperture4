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

#ifndef _PARTICLES_FUNCTIONS_H_
#define _PARTICLES_FUNCTIONS_H_

#include "core/buffer.hpp"
#include "core/exec_tags.h"
#include "core/particles.h"
#include "utils/vec.hpp"

namespace Aperture {

template <typename BufferType>
void ptc_sort_by_cell(exec_tags::device, particles_base<BufferType>& ptc,
                      size_t max_cell);
template <typename BufferType>
void ptc_sort_by_cell(exec_tags::host, particles_base<BufferType>& ptc,
                      size_t max_cell);

template <typename BufferType>
void ptc_append(exec_tags::host, particles_base<BufferType>& ptc,
                const vec_t<Scalar, 3>& x, const vec_t<Scalar, 3>& p,
                uint32_t cell, Scalar weight = 1.0, uint32_t flag = 0);

template <typename BufferType>
void ptc_append(exec_tags::device, particles_base<BufferType>& ptc,
                const vec_t<Scalar, 3>& x, const vec_t<Scalar, 3>& p,
                uint32_t cell, Scalar weight = 1.0, uint32_t flag = 0);

template <typename BufferType, typename Conf>
void ptc_copy_to_comm_buffers(exec_tags::host, particles_base<BufferType>& ptc,
                              std::vector<buffer<typename BufferType::single_type>>& buffers,
                              buffer<typename BufferType::single_type*>& buf_ptrs,
                              buffer<int>& buf_nums,
                              const grid_t<Conf>& grid);
template <typename BufferType, typename Conf>
void ptc_copy_to_comm_buffers(exec_tags::device,
                              particles_base<BufferType>& ptc,
                              std::vector<buffer<typename BufferType::single_type>>& buffers,
                              buffer<typename BufferType::single_type*>& buf_ptrs,
                              buffer<int>& buf_nums,
                              const grid_t<Conf>& grid);

template <typename BufferType>
void ptc_copy_from_buffer(exec_tags::device, particles_base<BufferType>& ptc,
                          const buffer<typename BufferType::single_type>& buf,
                          int num, size_t dst_idx);
template <typename BufferType>
void ptc_copy_from_buffer(exec_tags::host, particles_base<BufferType>& ptc,
                          const buffer<typename BufferType::single_type>& buf,
                          int num, size_t dst_idx);

template <int Dim>
constexpr int
get_zone_offset() {
  return 0;
}

template <>
constexpr int
get_zone_offset<1>() {
  return 12;
}

template <>
constexpr int
get_zone_offset<2>() {
  return 9;
}



}  // namespace Aperture

#endif  // _PARTICLES_FUNCTIONS_H_
