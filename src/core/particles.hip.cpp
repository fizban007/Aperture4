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

#include "core/cached_allocator.hpp"
#include "core/constant_mem.h"
#include "core/math.hpp"
#include "core/typedefs_and_constants.h"
#include "framework/config.h"
#include "particles_impl.hpp"
#include "utils/for_each_dual.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include "utils/timer.h"
#include "visit_struct/visit_struct.hpp"

#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/replace.h>
#include <thrust/sort.h>

namespace Aperture {

template <int Dim>
constexpr int
get_zone_offset() {
  return 0;
  // if constexpr (Dim == 1) {
  //   return 12;
  // } else if (Dim == 2) {
  //   return 9;
  // }
  // return 0;
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

template <typename Conf>
void
compute_target_buffers(const uint32_t* cells, size_t offset, size_t num,
                       buffer<int>& buffer_num, size_t* index) {
  kernel_launch(
      [offset, num] __device__(auto cells, auto buffer_num, auto index) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        // int zone_offset = 0;
        // if (Conf::dim == 2)
        //   zone_offset = 9;
        // else if (Conf::dim == 1)
        //   zone_offset = 12;
        int zone_offset = get_zone_offset<Conf::dim>();
        for (auto n : grid_stride_range(0, num)) {
          uint32_t cell = cells[n + offset];
          if (cell == empty_cell) continue;
          auto idx = Conf::idx(cell, ext);
          auto grid_pos = get_pos(idx, ext);
          size_t zone = grid.find_zone(grid_pos) + zone_offset;
          if (zone == 13) continue;
          size_t pos = atomicAdd(&buffer_num[zone], 1);
          // printf("pos is %lu, zone is %lu\n", pos, zone);
          // Zone is less than 32, so we can use 5 bits to represent this. The
          // rest of the bits go to encode the index of this particle in that
          // zone.
          int bitshift_width = (sizeof(size_t) * 8 - 5);
          index[n] = ((zone & 0b11111) << bitshift_width) + pos;
          // printf("pos is %lu, index is %lu, zone is %lu\n", pos, index[n], zone);
        }
      },
      cells, buffer_num.dev_ptr(), index);
  // GpuSafeCall(gpuDeviceSynchronize());
  GpuCheckError();
}

template <typename Conf, typename PtcPtrs, typename SinglePtc>
void
copy_component_to_buffer(PtcPtrs ptc_data, size_t offset, size_t num,
                         size_t* index, buffer<SinglePtc*>& ptc_buffers) {
  kernel_launch(
      [offset, num] __device__(auto ptc_data, auto index, auto ptc_buffers) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        int bitshift_width = (sizeof(size_t) * 8 - 5);
        int zone_offset = get_zone_offset<Conf::dim>();
        // loop through the particle array
        for (auto n : grid_stride_range(0, num)) {
          auto cell = ptc_data.cell[n + offset];
          size_t i = index[n];
          if (cell == empty_cell || i == size_t(-1)) continue;
          size_t zone = ((i >> bitshift_width) & 0b11111);
          if (zone == 13 || zone > 27) continue;
          size_t pos = i - (zone << bitshift_width);
          // printf("i is %lu, zone is %lu, zone_shifted is %lu, pos is %lu\n",
          //        i, zone, zone << bitshift_width, pos);
          // Copy the particle data from ptc_data[n] to ptc_buffers[zone][pos]
          // assign_ptc(ptc_buffers[zone - zone_offset], pos, ptc_data, n);
          assign_ptc(ptc_buffers[zone - zone_offset][pos], ptc_data, n + offset);
          // printf("pos is %lu, %u, %u\n", pos, ptc_buffers[zone -
          //                                                 zone_offset].cell[pos],
          //                                                 ptc_data.cell[n]);
          // printf("target zone is %lu\n", zone - zone_offset);
          // Compute particle cell delta
          int dz = (Conf::dim > 2 ? (zone / 9) - 1 : 0);
          int dy = (Conf::dim > 1 ? (zone / 3) % 3 - 1 : 0);
          int dx = zone % 3 - 1;
          auto idx = Conf::idx(cell, ext);
          // int dcell =
          //     -dz * grid.reduced_dim(2) * grid.dims[0] * grid.dims[1] -
          //     dy * grid.reduced_dim(1) * grid.dims[0] -
          //     dx * grid.reduced_dim(0);

          // TODO: This needs to be modified if the domain decomp is not uniform
          ptc_buffers[zone - zone_offset][pos].cell =
              idx.dec_z(dz * grid.reduced_dim(2))
                  .dec_y(dy * grid.reduced_dim(1))
                  .dec_x(dx * grid.reduced_dim(0))
                  .linear;
          // auto cell_after = ptc_buffers[zone - zone_offset][pos].cell;
          // printf("i is %lu, pos is %lu, cell is %u, cell after is (%d, %d), zone is %lu\n",
          //        i, pos, cell,
          //        cell_after % grid.dims[0],
          //        cell_after / grid.dims[0],
          //        zone - zone_offset);
          // Set the particle to empty
          ptc_data.cell[n + offset] = empty_cell;
        }
      },
      ptc_data, index, ptc_buffers.dev_ptr());
  GpuSafeCall(gpuDeviceSynchronize());
  GpuCheckError();
}

template <typename BufferType>
void
particles_base<BufferType>::resize_tmp_arrays() {
  if (m_index.size() != m_sort_segment_size ||
      m_tmp_data.size() != m_sort_segment_size) {
    m_index.resize(m_sort_segment_size);
    m_tmp_data.resize(m_sort_segment_size);
    m_segment_nums.set_memtype(MemType::host_device);
    m_segment_nums.resize(m_size / m_sort_segment_size + 1);
  }
}

template <typename BufferType>
void
particles_base<BufferType>::rearrange_arrays(const std::string& skip,
                                             size_t offset, size_t num) {
  auto ptc = typename BufferType::single_type{};
  for_each_double_with_name(
      m_dev_ptrs, ptc,
      [this, offset, num, &skip](const char* name, auto& x, auto& u) {
        typedef typename std::remove_reference<decltype(x)>::type x_type;
        auto ptr_index = thrust::device_pointer_cast(m_index.dev_ptr());
        if (std::strcmp(name, skip.c_str()) == 0) return;

        auto x_ptr = thrust::device_pointer_cast(x + offset);
        auto tmp_ptr = thrust::device_pointer_cast(
            reinterpret_cast<x_type>(m_tmp_data.dev_ptr()));
        thrust::gather(ptr_index, ptr_index + num, x_ptr, tmp_ptr);
        thrust::copy_n(tmp_ptr, num, x_ptr);
        GpuCheckError();
      });
}

template <typename BufferType>
void
particles_base<BufferType>::sort_by_cell_dev(size_t max_cell) {
  if (m_number > 0) {
    // Lazy resize the tmp arrays
    resize_tmp_arrays();
    m_segment_nums.assign_host(0);

    // 1st: Sort the particle array segment by segment
    for (int n = 0; n < m_number / m_sort_segment_size + 1; n++) {
      // Logger::print_info("Sorting segment {}", n);
      size_t offset = n * m_sort_segment_size;
      // Fringe case of m_number being an exact multiple of segment_size
      if (offset == m_number) {
        m_segment_nums[n] = 0;
        continue;
      }
      // Generate particle index array
      auto ptr_cell =
          thrust::device_pointer_cast(this->cell.dev_ptr() + offset);
      auto ptr_idx = thrust::device_pointer_cast(m_index.dev_ptr());

      // Sort the index array by key
      size_t sort_size = std::min(m_sort_segment_size, m_number - offset);
      thrust::counting_iterator<size_t> iter(0);
      thrust::copy_n(iter, sort_size, ptr_idx);

      // Logger::print_info("Sort_size is {}, offset is {}", sort_size, offset);
      thrust::sort_by_key(ptr_cell, ptr_cell + sort_size, ptr_idx);

      // Move the rest of particle array using the new index
      // Logger::print_info("Rearranging");
      rearrange_arrays("cell", offset, sort_size);

      // Update the new number of particles in each sorted segment
      m_segment_nums[n] =
          thrust::upper_bound(ptr_cell, ptr_cell + sort_size, empty_cell - 1) -
          ptr_cell;
      // Logger::print_info("segment[{}] has size {}", n, m_segment_nums[n]);
    }

    // 2nd: Defragment the particle array
    int last_segment = m_number / m_sort_segment_size;
    for (int m = 0; m < last_segment; m++) {
      // Logger::print_info(
      //     "Filling segment {}, last_segment is {}, num_last is {}", m,
      //     last_segment, m_segment_nums[last_segment]);

      while (m_segment_nums[m] < m_sort_segment_size) {
        // deficit is how many "holes" do we have in this segment
        int deficit = m_sort_segment_size - m_segment_nums[m];
        // do not copy more particles than the number in the last segment
        int num_to_copy = std::min(deficit, m_segment_nums[last_segment]);
        // calculate offsets
        size_t offset_from = last_segment * m_sort_segment_size +
                             m_segment_nums[last_segment] - num_to_copy;
        size_t offset_to = m * m_sort_segment_size + m_segment_nums[m];
        // Logger::print_info(
        //     "deficit is {}, num_to_copy is {}, offset_from is {}", deficit,
        //     num_to_copy, offset_from);

        // Copy the particles from the end of the last segment to the end of
        // this segment
        copy_from(*this, num_to_copy, offset_from, offset_to);
        // Erase particles from the last segment
        erase(offset_from, num_to_copy);

        m_segment_nums[m] += num_to_copy;
        m_segment_nums[last_segment] -= num_to_copy;
        // Logger::print_info("Segment num is {}", m_segment_nums[m]);

        if (m_segment_nums[last_segment] == 0) {
          last_segment -= 1;
          if (last_segment == m) break;
        }
      }
    }

    // Logger::print_info("Last segment size is {}",
    // m_segment_nums[last_segment]);
    m_number =
        last_segment * m_sort_segment_size + m_segment_nums[last_segment];

    gpuDeviceSynchronize();
    GpuCheckError();
  }
}

template <typename BufferType>
void
particles_base<BufferType>::append_dev(const vec_t<Scalar, 3>& x,
                                       const vec_t<Scalar, 3>& p, uint32_t cell,
                                       Scalar weight, uint32_t flag) {
  if (m_number == m_size) return;
  kernel_launch(
      {1, 1},
      [x, p, cell, weight, flag] __device__(auto ptrs, size_t pos) {
        ptrs.x1[pos] = x[0];
        ptrs.x2[pos] = x[1];
        ptrs.x3[pos] = x[2];
        ptrs.p1[pos] = p[0];
        ptrs.p2[pos] = p[1];
        ptrs.p3[pos] = p[2];
        ptrs.E[pos] =
            math::sqrt(1.0f + p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        ptrs.weight[pos] = weight;
        ptrs.cell[pos] = cell;
        ptrs.flag[pos] = flag;
      },
      m_dev_ptrs, m_number);
  GpuSafeCall(gpuDeviceSynchronize());
  m_number += 1;
}

// template <typename BufferType>
// template <typename Conf>
// void
// particles_base<BufferType>::copy_to_comm_buffers(
//     std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Conf>& grid) {
//   if (m_number > 0) {
//     // timer::stamp("compute_buffer");
//     // Lazy resize the tmp arrays
//     resize_tmp_arrays();
//     m_zone_buffer_num.assign_dev(0);

//     for (int n = 0; n < m_number / m_sort_segment_size + 1; n++) {
//       size_t offset = n * m_sort_segment_size;
//       // Logger::print_debug("offset is {}, n is {}", offset, n);

//       m_index.assign_dev(0, m_sort_segment_size, size_t(-1));
//       auto ptr_idx = thrust::device_pointer_cast(m_index.dev_ptr());
//       compute_target_buffers<Conf>(this->cell.dev_ptr(), offset,
//                                    m_sort_segment_size, m_zone_buffer_num,
//                                    m_index.dev_ptr());

//       copy_component_to_buffer<Conf>(m_dev_ptrs, offset, m_sort_segment_size,
//                                      m_index.dev_ptr(), buf_ptrs);
//     }

//     m_zone_buffer_num.copy_to_host();
//     // timer::show_duration_since_stamp("Computing target buffers", "ms",
//     // "compute_buffer");

//     int zone_offset = 0;
//     if (buffers.size() == 9)
//       zone_offset = 9;
//     else if (buffers.size() == 3)
//       zone_offset = 12;
//     for (unsigned int i = 0; i < buffers.size(); i++) {
//       // Logger::print_debug("zone {} buffer has {} ptc", i + zone_offset,
//       //                     m_zone_buffer_num[i + zone_offset]);
//       if (i + zone_offset == 13) continue;
//       buffers[i].set_num(m_zone_buffer_num[i + zone_offset]);
//     }
//     // timer::stamp("copy_to_buffer");
//     // for (unsigned int i = 0; i < buffers.size(); i++) {
//     //   if (buffers[i].number() > 0) {
//     //     buffers[i].copy_to_host();
//     //   }
//     // }
//     // if (buffers[7].number() > 0) {
//     //   buffers[7].copy_to_host();
//     //   Logger::print_debug("buffer[7] cell[0] is {}", buffers[7].cell[0]);
//     // }
//     // GpuSafeCall(gpuDeviceSynchronize());
//     // timer::show_duration_since_stamp("Copy to buffer", "ms",
//     // "copy_to_buffer");
//   }
// }

template <typename BufferType>
template <typename Conf>
void
particles_base<BufferType>::copy_to_comm_buffers(
    std::vector<buffer<single_type>>& buffers,
    buffer<single_type*>& buf_ptrs,
    buffer<int>& buf_nums,
    const grid_t<Conf>& grid) {
  if (m_number > 0) {
    // timer::stamp("compute_buffer");
    // Lazy resize the tmp arrays
    resize_tmp_arrays();
    m_zone_buffer_num.assign_dev(0);

    for (int n = 0; n < m_number / m_sort_segment_size + 1; n++) {
      size_t offset = n * m_sort_segment_size;
      // Logger::print_debug("offset is {}, n is {}", offset, n);

      m_index.assign_dev(0, m_sort_segment_size, size_t(-1));
      // auto ptr_idx = thrust::device_pointer_cast(m_index.dev_ptr());
      compute_target_buffers<Conf>(this->cell.dev_ptr(), offset,
                                   m_sort_segment_size, m_zone_buffer_num,
                                   m_index.dev_ptr());

      copy_component_to_buffer<Conf>(m_dev_ptrs, offset, m_sort_segment_size,
                                     m_index.dev_ptr(), buf_ptrs);
    }

    m_zone_buffer_num.copy_to_host();
    // timer::show_duration_since_stamp("Computing target buffers", "ms",
    // "compute_buffer");

    int zone_offset = get_zone_offset<Conf::dim>();
    for (unsigned int i = 0; i < buffers.size(); i++) {
      // Logger::print_debug("zone {} buffer has {} ptc", i + zone_offset,
      //                     m_zone_buffer_num[i + zone_offset]);
      if (i + zone_offset == 13) continue;
      // buffers[i].set_num(m_zone_buffer_num[i + zone_offset]);
      buf_nums[i] = m_zone_buffer_num[i + zone_offset];
    }
    // timer::stamp("copy_to_buffer");
    // for (unsigned int i = 0; i < buffers.size(); i++) {
    //   if (buffers[i].number() > 0) {
    //     buffers[i].copy_to_host();
    //   }
    // }
    // if (buffers[7].number() > 0) {
    //   buffers[7].copy_to_host();
    //   Logger::print_debug("buffer[7] cell[0] is {}", buffers[7].cell[0]);
    // }
    // GpuSafeCall(gpuDeviceSynchronize());
    // timer::show_duration_since_stamp("Copy to buffer", "ms",
    // "copy_to_buffer");
  }
}

template <typename BufferType>
void
particles_base<BufferType>::copy_from_buffer(const buffer<single_type> &buf,
                                             int num, size_t dst_idx) {
  if (dst_idx + num > m_size)
    num = m_size - dst_idx;
  if (num > 0) {
    kernel_launch([num, dst_idx] __device__(auto ptc, auto buf) {
        for (auto n : grid_stride_range(0, num)) {
            assign_ptc(ptc, dst_idx + n, buf[n]);
        }
        }, m_dev_ptrs, buf.dev_ptr());
    // GpuSafeCall(gpuDeviceSynchronize());
  }
  if (dst_idx + num > m_number)
    m_number = dst_idx + num;
}

// Explicit instantiation
template class particles_base<ptc_buffer>;
template void particles_base<ptc_buffer>::copy_to_comm_buffers(
    std::vector<buffer<single_type>>& buffers,
    buffer<single_type*>& buf_ptrs,
    buffer<int>& buf_nums,
    // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<1, float>>& grid);
template void particles_base<ptc_buffer>::copy_to_comm_buffers(
    std::vector<buffer<single_type>>& buffers,
    buffer<single_type*>& buf_ptrs,
    buffer<int>& buf_nums,
    // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<1, double>>& grid);
template void particles_base<ptc_buffer>::copy_to_comm_buffers(
    std::vector<buffer<single_type>>& buffers,
    buffer<single_type*>& buf_ptrs,
    buffer<int>& buf_nums,
    // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<2, float>>& grid);
template void particles_base<ptc_buffer>::copy_to_comm_buffers(
    std::vector<buffer<single_type>>& buffers,
    buffer<single_type*>& buf_ptrs,
    buffer<int>& buf_nums,
    // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<2, double>>& grid);
template void particles_base<ptc_buffer>::copy_to_comm_buffers(
    std::vector<buffer<single_type>>& buffers,
    buffer<single_type*>& buf_ptrs,
    buffer<int>& buf_nums,
    // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<3, float>>& grid);
template void particles_base<ptc_buffer>::copy_to_comm_buffers(
    std::vector<buffer<single_type>>& buffers,
    buffer<single_type*>& buf_ptrs,
    buffer<int>& buf_nums,
    // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<3, double>>& grid);

template class particles_base<ph_buffer>;
template void particles_base<ph_buffer>::copy_to_comm_buffers(
    std::vector<buffer<single_type>>& buffers,
    buffer<single_type*>& buf_ptrs,
    buffer<int>& buf_nums,
    // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<1, float>>& grid);
template void particles_base<ph_buffer>::copy_to_comm_buffers(
    std::vector<buffer<single_type>>& buffers,
    buffer<single_type*>& buf_ptrs,
    buffer<int>& buf_nums,
    // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<1, double>>& grid);
template void particles_base<ph_buffer>::copy_to_comm_buffers(
    std::vector<buffer<single_type>>& buffers,
    buffer<single_type*>& buf_ptrs,
    buffer<int>& buf_nums,
    // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<2, float>>& grid);
template void particles_base<ph_buffer>::copy_to_comm_buffers(
    std::vector<buffer<single_type>>& buffers,
    buffer<single_type*>& buf_ptrs,
    buffer<int>& buf_nums,
    // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<2, double>>& grid);
template void particles_base<ph_buffer>::copy_to_comm_buffers(
    std::vector<buffer<single_type>>& buffers,
    buffer<single_type*>& buf_ptrs,
    buffer<int>& buf_nums,
    // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<3, float>>& grid);
template void particles_base<ph_buffer>::copy_to_comm_buffers(
    std::vector<buffer<single_type>>& buffers,
    buffer<single_type*>& buf_ptrs,
    buffer<int>& buf_nums,
    // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<3, double>>& grid);

}  // namespace Aperture
