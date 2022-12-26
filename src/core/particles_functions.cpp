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

#include "particles_functions.h"
#include "core/math.hpp"
#include "core/typedefs_and_constants.h"
#include "framework/config.h"
#include "utils/for_each_dual.hpp"
#include "utils/range.hpp"
#include "visit_struct/visit_struct.hpp"
#include <algorithm>
#include <numeric>

namespace Aperture {

template <typename BufferType>
void
ptc_rearrange_arrays(exec_tags::host, particles_base<BufferType>& data) {
  typename BufferType::single_type p_tmp;
  auto& m_index = data.index();
  for (size_t i = 0; i < data.number(); i++) {
    // -1 means LLONG_MAX for unsigned long int
    if (m_index[i] != (size_t)-1) {
      for_each_double(p_tmp, data.get_host_ptrs(),
                      [i](auto& x, auto& y) { x = y[i]; });
      for (size_t j = i;;) {
        if (m_index[j] != i) {
          // put(index[j], m_data[j]);
          data.swap(m_index[j], p_tmp);
          size_t id = m_index[j];
          m_index[j] = (size_t)-1;  // Mark as done
          j = id;
        } else {
          assign_ptc(data.get_host_ptrs(), i, p_tmp);
          m_index[j] = (size_t)-1;  // Mark as done
          break;
        }
      }
    }
  }
}

template <typename BufferType>
void
ptc_append(exec_tags::host, particles_base<BufferType>& ptc,
           const vec_t<Scalar, 3>& x, const vec_t<Scalar, 3>& p, uint32_t cell,
           Scalar weight, uint32_t flag) {
  if (ptc.number() == ptc.size()) return;
  auto m_number = ptc.number();
  ptc.x1[m_number] = x[0];
  ptc.x2[m_number] = x[1];
  ptc.x3[m_number] = x[2];
  ptc.p1[m_number] = p[0];
  ptc.p2[m_number] = p[1];
  ptc.p3[m_number] = p[2];
  ptc.E[m_number] = std::sqrt(1.0f + p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
  ptc.weight[m_number] = weight;
  ptc.cell[m_number] = cell;
  ptc.flag[m_number] = flag;
  // m_number += 1;
  ptc.set_num(ptc.number() + 1);
}

template void ptc_append(exec_tags::host, particles_base<ptc_buffer>& ptc,
                         const vec_t<Scalar, 3>& x, const vec_t<Scalar, 3>& p,
                         uint32_t cell, Scalar weight, uint32_t flag);
template void ptc_append(exec_tags::host, particles_base<ph_buffer>& ptc,
                         const vec_t<Scalar, 3>& x, const vec_t<Scalar, 3>& p,
                         uint32_t cell, Scalar weight, uint32_t flag);

template <typename BufferType>
void
ptc_sort_by_cell(exec_tags::host, particles_base<BufferType>& ptc,
                 size_t num_cells) {
  auto m_number = ptc.number();
  auto& m_partition = ptc.partition();
  auto& m_index = ptc.index();
  if (m_number > 0) {
    // Compute the number of cells and resize the partition array if
    // needed
    if (m_partition.size() != num_cells + 2) m_partition.resize(num_cells + 2);
    if (m_index.size() != ptc.size()) m_index.resize(ptc.size());

    std::fill(m_partition.begin(), m_partition.end(), 0);
    // Generate particle index from 0 up to the current number
    std::iota(m_index.host_ptr(), m_index.host_ptr() + m_number, 0);

    // Loop over the particle array to count how many particles in each
    // cell
    for (std::size_t i = 0; i < m_number; i++) {
      size_t cell_idx = 0;
      if (ptc.cell[i] == empty_cell)
        cell_idx = num_cells;
      else
        cell_idx = ptc.cell[i];
      // Right now m_index array saves the id of each particle in its
      // cell, and partitions array saves the number of particles in
      // each cell
      m_index[i] = m_partition[cell_idx + 1];
      m_partition[cell_idx + 1] += 1;
    }

    // Scan the array, now the array contains the starting index of each
    // zone in the main particle array
    for (uint32_t i = 1; i < num_cells + 2; i++) {
      m_partition[i] += m_partition[i - 1];
      // The last element means how many particles are empty
    }

    // Second pass through the particle array, get the real index
    for (size_t i = 0; i < m_number; i++) {
      size_t cell_idx = 0;
      if (ptc.cell[i] == empty_cell) {
        cell_idx = num_cells;
      } else {
        cell_idx = ptc.cell[i];
      }
      m_index[i] += m_partition[cell_idx];
    }

    // Rearrange the particles to reflect the partition
    // timer::show_duration_since_stamp("partition", "ms");
    ptc_rearrange_arrays(exec_tags::host{}, ptc);

    // num_cells is where the empty particles start, so we record this as
    // the new particle number
    if (m_partition[num_cells] != m_number) ptc.set_num(m_partition[num_cells]);
    Logger::print_info("Sorting complete, there are {} particles in the pool",
                       m_number);
  }
}

template void ptc_sort_by_cell(exec_tags::host, particles_base<ptc_buffer>& ptc,
                               size_t max_cell);
template void ptc_sort_by_cell(exec_tags::host, particles_base<ph_buffer>& ptc,
                               size_t max_cell);

template <typename BufferType>
void
ptc_copy_from_buffer(exec_tags::host, particles_base<BufferType>& ptc,
                     const buffer<typename BufferType::single_type>& buf,
                     int num, size_t dst_idx) {
  if (dst_idx + num > ptc.size()) num = ptc.size() - dst_idx;
  if (num > 0) {
    for (auto n : range(0, num)) {
      assign_ptc(ptc.get_host_ptrs(), dst_idx + n, buf[n]);
    }
  }
  if (dst_idx + num > ptc.number()) ptc.set_num(dst_idx + num);
}

template void ptc_copy_from_buffer(exec_tags::host,
                                   particles_base<ptc_buffer>& ptc,
                                   const buffer<single_ptc_t>& buf, int num,
                                   size_t dst_idx);
template void ptc_copy_from_buffer(exec_tags::host,
                                   particles_base<ph_buffer>& ptc,
                                   const buffer<single_ph_t>& buf, int num,
                                   size_t dst_idx);

// template <typename BufferType>
// void
// particles_base<BufferType>::copy_from_buffer(
//     exec_tags::host,
//     const buffer<single_type> &buf,
//     int num, size_t dst_idx) {
template <typename BufferType, typename Conf>
void
ptc_copy_to_comm_buffers(exec_tags::host, particles_base<BufferType>& ptc,
                         std::vector<buffer<typename BufferType::single_type>>& buffers,
                         buffer<typename BufferType::single_type*>& buf_ptrs,
                         buffer<int>& buf_nums, const grid_t<Conf>& grid) {}

template void ptc_copy_to_comm_buffers<ptc_buffer, Config<1>>(
    exec_tags::host, particles_base<ptc_buffer>& ptc,
    std::vector<buffer<single_ptc_t>>& buffers, buffer<single_ptc_t*>& buf_ptrs,
    buffer<int>& buf_nums, const grid_t<Config<1, Scalar>>& grid);

template void ptc_copy_to_comm_buffers<ptc_buffer, Config<2>>(
    exec_tags::host, particles_base<ptc_buffer>& ptc,
    std::vector<buffer<single_ptc_t>>& buffers, buffer<single_ptc_t*>& buf_ptrs,
    buffer<int>& buf_nums, const grid_t<Config<2, Scalar>>& grid);

template void ptc_copy_to_comm_buffers<ptc_buffer, Config<3>>(
    exec_tags::host, particles_base<ptc_buffer>& ptc,
    std::vector<buffer<single_ptc_t>>& buffers, buffer<single_ptc_t*>& buf_ptrs,
    buffer<int>& buf_nums, const grid_t<Config<3, Scalar>>& grid);

template void ptc_copy_to_comm_buffers<ph_buffer, Config<1>>(
    exec_tags::host, particles_base<ph_buffer>& ptc,
    std::vector<buffer<single_ph_t>>& buffers, buffer<single_ph_t*>& buf_ptrs,
    buffer<int>& buf_nums, const grid_t<Config<1, Scalar>>& grid);

template void ptc_copy_to_comm_buffers<ph_buffer, Config<2>>(
    exec_tags::host, particles_base<ph_buffer>& ptc,
    std::vector<buffer<single_ph_t>>& buffers, buffer<single_ph_t*>& buf_ptrs,
    buffer<int>& buf_nums, const grid_t<Config<2, Scalar>>& grid);

template void ptc_copy_to_comm_buffers<ph_buffer, Config<3>>(
    exec_tags::host, particles_base<ph_buffer>& ptc,
    std::vector<buffer<single_ph_t>>& buffers, buffer<single_ph_t*>& buf_ptrs,
    buffer<int>& buf_nums, const grid_t<Config<3, Scalar>>& grid);


}  // namespace Aperture
