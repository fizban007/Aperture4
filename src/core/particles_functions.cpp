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
#include "framework/environment.h"
#include "utils/for_each_dual.hpp"
#include "utils/range.hpp"
#include "visit_struct/visit_struct.hpp"
#include <algorithm>
#include <numeric>

namespace Aperture {

template <typename BufferType>
void
ptc_rearrange_arrays(exec_tags::host, particles_base<BufferType>& data,
                     size_t offset, size_t num) {
  typename BufferType::single_type p_tmp;
  auto& m_index = data.index();
  for (size_t i = 0; i < num; i++) {
    // -1 means LLONG_MAX for unsigned long int
    if (m_index[i] != (size_t)-1) {
      for_each_double(p_tmp, data.get_host_ptrs(),
                      [i, offset](auto& x, auto& y) { x = y[i + offset]; });
      for (size_t j = i;;) {
        if (m_index[j] != i + offset) {
          // put(index[j], m_data[j]);
          data.swap(m_index[j], p_tmp);
          size_t id = m_index[j];
          m_index[j] = (size_t)-1;  // Mark as done
          j = id - offset;
        } else {
          assign_ptc(data.get_host_ptrs(), i + offset, p_tmp);
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
  size_t rank = sim_env().get_rank();
  rank <<= 32;
  ptc.id[m_number] = rank + atomic_add(&ptc.ptc_id()[0], 1);
  // m_number += 1;
  ptc.set_num(ptc.number() + 1);
}

template void ptc_append(exec_tags::host, particles_base<ptc_buffer>& ptc,
                         const vec_t<Scalar, 3>& x, const vec_t<Scalar, 3>& p,
                         uint32_t cell, Scalar weight, uint32_t flag);
template void ptc_append(exec_tags::host, particles_base<ph_buffer>& ptc,
                         const vec_t<Scalar, 3>& x, const vec_t<Scalar, 3>& p,
                         uint32_t cell, Scalar weight, uint32_t flag);

template <typename BufferType, typename Conf>
void ptc_append_global(exec_tags::host, particles_base<BufferType>& ptc,
                       const grid_t<Conf>& grid,
                       const vec_t<Scalar, 3>& x_global, const vec_t<Scalar, 3>& p,
                       Scalar weight, uint32_t flag) {
  if (grid.is_in_bound(x_global)) {
    vec_t<Scalar, 3> x_rel;
    uint32_t cell;
    grid.from_x_global(x_global, x_rel, cell);
    ptc_append(exec_tags::host{}, ptc, x_rel, p, cell, weight, flag);
  }
}

template void ptc_append_global(exec_tags::host, particles_base<ptc_buffer>&,
                                const grid_t<Config<1>>&, const vec_t<Scalar, 3>&,
                                const vec_t<Scalar, 3>&, Scalar, uint32_t);
template void ptc_append_global(exec_tags::host, particles_base<ptc_buffer>&,
                                const grid_t<Config<2>>&, const vec_t<Scalar, 3>&,
                                const vec_t<Scalar, 3>&, Scalar, uint32_t);
template void ptc_append_global(exec_tags::host, particles_base<ptc_buffer>&,
                                const grid_t<Config<3>>&, const vec_t<Scalar, 3>&,
                                const vec_t<Scalar, 3>&, Scalar, uint32_t);
template void ptc_append_global(exec_tags::host, particles_base<ph_buffer>&,
                                const grid_t<Config<1>>&, const vec_t<Scalar, 3>&,
                                const vec_t<Scalar, 3>&, Scalar, uint32_t);
template void ptc_append_global(exec_tags::host, particles_base<ph_buffer>&,
                                const grid_t<Config<2>>&, const vec_t<Scalar, 3>&,
                                const vec_t<Scalar, 3>&, Scalar, uint32_t);
template void ptc_append_global(exec_tags::host, particles_base<ph_buffer>&,
                                const grid_t<Config<3>>&, const vec_t<Scalar, 3>&,
                                const vec_t<Scalar, 3>&, Scalar, uint32_t);

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
    if (ptc.segment_nums().size() != ptc.size() / ptc.sort_segment_size() + 1)
      ptc.segment_nums().resize(ptc.size() / ptc.sort_segment_size() + 1);
    if (m_index.size() != ptc.sort_segment_size())
      m_index.resize(ptc.sort_segment_size());

    size_t total_num = 0;

    // 1st: Go through the particle array segment by segment
    for (int n = 0; n < ptc.number() / ptc.sort_segment_size() + 1; n++) {
      size_t offset = n * ptc.sort_segment_size();
      // Fringe case of number being an exact multiple of segment_size
      if (offset == ptc.number()) {
        ptc.segment_nums()[n] = 0;
        continue;
      }

      std::fill(m_partition.begin(), m_partition.end(), 0);
      // Generate particle index from 0 up to the current number
      size_t sort_size =
          std::min(ptc.sort_segment_size(), ptc.number() - offset);
      std::iota(m_index.host_ptr(), m_index.host_ptr() + sort_size, 0);

      // Loop over the particle array to count how many particles in each
      // cell
      for (std::size_t i = 0; i < sort_size; i++) {
        size_t cell_idx = 0;
        if (ptc.cell[i + offset] == empty_cell)
          cell_idx = num_cells;
        else
          cell_idx = ptc.cell[i + offset];
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
      for (size_t i = 0; i < sort_size; i++) {
        size_t cell_idx = 0;
        if (ptc.cell[i + offset] == empty_cell) {
          cell_idx = num_cells;
        } else {
          cell_idx = ptc.cell[i + offset];
        }
        m_index[i] += m_partition[cell_idx] + offset;
      }

      // Rearrange the particles to reflect the partition
      // timer::show_duration_since_stamp("partition", "ms");
      ptc_rearrange_arrays(exec_tags::host{}, ptc, offset, sort_size);

      ptc.segment_nums()[n] = m_partition[num_cells];
      // if (n < ptc.number() / ptc.sort_segment_size()) {
      //   total_num += sort_size;
      // } else {
      //   total_num += m_partition[num_cells];
      // }
    }

    // 2nd: Defrag the whole particle array
    int last_segment = ptc.number() / ptc.sort_segment_size();
    for (int m = 0; m < last_segment; m++) {
      // Logger::print_info(
      //     "Filling segment {}, last_segment is {}, num_last is {}", m,
      //     last_segment, m_segment_nums[last_segment]);

      while (ptc.segment_nums()[m] < ptc.sort_segment_size()) {
        // deficit is how many "holes" do we have in this segment
        int deficit = ptc.sort_segment_size() - ptc.segment_nums()[m];
        // do not copy more particles than the number in the last segment
        int num_to_copy = std::min(deficit, ptc.segment_nums()[last_segment]);
        // calculate offsets
        size_t offset_from = last_segment * ptc.sort_segment_size() +
                             ptc.segment_nums()[last_segment] - num_to_copy;
        size_t offset_to = m * ptc.sort_segment_size() + ptc.segment_nums()[m];
        // Logger::print_info(
        //     "deficit is {}, num_to_copy is {}, offset_from is {}", deficit,
        //     num_to_copy, offset_from);

        // Copy the particles from the end of the last segment to the end of
        // this segment
        ptc.copy_from(ptc, num_to_copy, offset_from, offset_to);
        // Erase particles from the last segment
        ptc.erase(offset_from, num_to_copy);

        ptc.segment_nums()[m] += num_to_copy;
        ptc.segment_nums()[last_segment] -= num_to_copy;
        // Logger::print_info("Segment num is {}", m_segment_nums[m]);

        if (ptc.segment_nums()[last_segment] == 0) {
          last_segment -= 1;
          if (last_segment == m) break;
        }
      }
    }

    total_num = last_segment * ptc.sort_segment_size() +
                ptc.segment_nums()[last_segment];
    ptc.set_num(total_num);
    // // num_cells is where the empty particles start, so we record this as
    // // the new particle number
    // // if (m_partition[num_cells] != m_number) ptc.set_num(m_partition[num_cells]);
    // if (total_num != m_number) ptc.set_num(total_num);
    // Logger::print_info("Sorting complete, there are {} particles in the pool",
                       // total_num);
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

template <typename BufferType, typename Conf>
void
ptc_copy_to_comm_buffers(exec_tags::host, particles_base<BufferType>& ptc,
                         std::vector<buffer<typename BufferType::single_type>>& buffers,
                         buffer<typename BufferType::single_type*>& buf_ptrs,
                         buffer<int>& buf_nums, const grid_t<Conf>& grid) {
  if (ptc.number() > 0) {
    auto ext = grid.extent();
    buf_nums.assign(0);

    for (size_t n : range(0, ptc.number())) {
      // Loop over the particle array and copy particles to the correct
      // communication buffer
      uint32_t cell = ptc.cell[n];
      if (cell == empty_cell) continue;
      auto idx = Conf::idx(cell, ext);
      auto grid_pos = get_pos(idx, ext);
      int zone_offset = get_zone_offset<Conf::dim>();
      int zone_id = grid.find_zone(grid_pos);
      int zone = zone_id + zone_offset;
      if (zone == 13) continue; // Zone 13 is center, no need for communication
      assign_ptc(buffers[zone_id][buf_nums[zone_id]], ptc.get_host_ptrs(), n);
      int dz = (Conf::dim > 2 ? (zone / 9) - 1 : 0);
      int dy = (Conf::dim > 1 ? (zone / 3) % 3 - 1 : 0);
      int dx = zone % 3 - 1;
      buffers[zone_id][buf_nums[zone_id]].cell =
          idx.dec_z(dz * grid.reduced_dim(2))
              .dec_y(dy * grid.reduced_dim(1))
              .dec_x(dx * grid.reduced_dim(0))
              .linear;
      ptc.cell[n] = empty_cell;
      buf_nums[zone_id] += 1;
    }
  }
}

template void ptc_copy_to_comm_buffers<ptc_buffer, Config<1>>(
    exec_tags::host, particles_base<ptc_buffer>& ptc,
    std::vector<buffer<single_ptc_t>>& buffers, buffer<single_ptc_t*>& buf_ptrs,
    buffer<int>& buf_nums, const grid_t<Config<1>>& grid);

template void ptc_copy_to_comm_buffers<ptc_buffer, Config<2>>(
    exec_tags::host, particles_base<ptc_buffer>& ptc,
    std::vector<buffer<single_ptc_t>>& buffers, buffer<single_ptc_t*>& buf_ptrs,
    buffer<int>& buf_nums, const grid_t<Config<2>>& grid);

template void ptc_copy_to_comm_buffers<ptc_buffer, Config<3>>(
    exec_tags::host, particles_base<ptc_buffer>& ptc,
    std::vector<buffer<single_ptc_t>>& buffers, buffer<single_ptc_t*>& buf_ptrs,
    buffer<int>& buf_nums, const grid_t<Config<3>>& grid);

template void ptc_copy_to_comm_buffers<ph_buffer, Config<1>>(
    exec_tags::host, particles_base<ph_buffer>& ptc,
    std::vector<buffer<single_ph_t>>& buffers, buffer<single_ph_t*>& buf_ptrs,
    buffer<int>& buf_nums, const grid_t<Config<1>>& grid);

template void ptc_copy_to_comm_buffers<ph_buffer, Config<2>>(
    exec_tags::host, particles_base<ph_buffer>& ptc,
    std::vector<buffer<single_ph_t>>& buffers, buffer<single_ph_t*>& buf_ptrs,
    buffer<int>& buf_nums, const grid_t<Config<2>>& grid);

template void ptc_copy_to_comm_buffers<ph_buffer, Config<3>>(
    exec_tags::host, particles_base<ph_buffer>& ptc,
    std::vector<buffer<single_ph_t>>& buffers, buffer<single_ph_t*>& buf_ptrs,
    buffer<int>& buf_nums, const grid_t<Config<3>>& grid);


}  // namespace Aperture
