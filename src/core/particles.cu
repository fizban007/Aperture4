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

template <typename Conf>
void
compute_target_buffers(const uint32_t* cells, size_t num,
                       buffer<int>& buffer_num, size_t* index) {
  kernel_launch(
      [num] __device__(auto cells, auto buffer_num, auto index) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        int zone_offset = 0;
        if (Conf::dim == 2)
          zone_offset = 9;
        else if (Conf::dim == 1)
          zone_offset = 12;
        for (auto n : grid_stride_range(0, num)) {
          uint32_t cell = cells[n];
          if (cell == empty_cell) continue;
          auto idx = Conf::idx(cell, ext);
          auto grid_pos = idx.get_pos();
          size_t zone = grid.find_zone(grid_pos) + zone_offset;
          if (zone == 13) continue;
          size_t pos = atomicAdd(&buffer_num[zone], 1);
          // printf("pos is %lu, zone is %lu\n", pos, zone);
          // Zone is less than 32, so we can use 5 bits to represent this. The
          // rest of the bits go to encode the index of this particle in that
          // zone.
          index[n] = ((zone & 0b11111) << (sizeof(size_t) * 8 - 5)) + pos;
        }
      },
      cells, buffer_num.dev_ptr(), index);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename Conf, typename PtcPtrs>
void
copy_component_to_buffer(PtcPtrs ptc_data, size_t num, size_t* index,
                         buffer<PtcPtrs>& ptc_buffers) {
  kernel_launch(
      [num] __device__(auto ptc_data, auto index, auto ptc_buffers) {
        auto& grid = dev_grid<Conf::dim>();
        auto ext = grid.extent();
        int bitshift_width = (sizeof(size_t) * 8 - 5);
        int zone_offset = 0;
        if (Conf::dim == 2)
          zone_offset = 9;
        else if (Conf::dim == 1)
          zone_offset = 12;
        // loop through the particle array
        for (auto n : grid_stride_range(0, num)) {
          auto cell = ptc_data.cell[n];
          if (cell == empty_cell) continue;
          size_t i = index[n];
          size_t zone = ((i >> bitshift_width) & 0b11111);
          if (zone == 13 || zone > 27) continue;
          size_t pos = i - (zone << bitshift_width);
          // Copy the particle data from ptc_data[n] to ptc_buffers[zone][pos]
          assign_ptc(ptc_buffers[zone - zone_offset], pos, ptc_data, n);
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
          ptc_buffers[zone - zone_offset].cell[pos] =
              idx.dec_z(dz * grid.reduced_dim(2))
                  .dec_y(dy * grid.reduced_dim(1))
                  .dec_x(dx * grid.reduced_dim(0))
                  .linear;
          // printf("dc is %d, cell is %u, cell after is %u, zone is %lu\n", dcell,
          //        ptc_data.cell[n],
          //        ptc_buffers[zone - zone_offset].cell[pos],
          //        zone - zone_offset);
          // Set the particle to empty
          ptc_data.cell[n] = empty_cell;
        }
      },
      ptc_data, index, ptc_buffers.dev_ptr());
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

template <typename BufferType>
void
particles_base<BufferType>::rearrange_arrays(const std::string& skip) {
  const uint32_t padding = 100;
  auto ptc = typename BufferType::single_type{};
  for_each_double_with_name(
      m_dev_ptrs, ptc,
      [this, padding, &skip](const char* name, auto& x, auto& u) {
        typedef typename std::remove_reference<decltype(x)>::type x_type;
        auto ptr_index = thrust::device_pointer_cast(m_index.dev_ptr());
        if (std::strcmp(name, skip.c_str()) == 0) return;

        auto x_ptr = thrust::device_pointer_cast(x);
        auto tmp_ptr = thrust::device_pointer_cast(
            reinterpret_cast<x_type>(m_tmp_data.dev_ptr()));
        thrust::gather(ptr_index, ptr_index + m_number, x_ptr, tmp_ptr);
        thrust::copy_n(tmp_ptr, m_number, x_ptr);
        CudaCheckError();
      });
}

template <typename BufferType>
void
particles_base<BufferType>::sort_by_cell_dev(size_t max_cell) {
  if (m_number > 0) {
    // Lazy resize the tmp arrays
    if (m_index.size() != m_size || m_tmp_data.size() != m_size) {
      m_index.resize(m_size);
      m_tmp_data.resize(m_size);
    }

    // Generate particle index array
    auto ptr_cell = thrust::device_pointer_cast(this->cell.dev_ptr());
    auto ptr_idx = thrust::device_pointer_cast(m_index.dev_ptr());
    thrust::counting_iterator<size_t> iter(0);
    thrust::copy_n(iter, m_number, ptr_idx);

    // Sort the index array by key
    thrust::sort_by_key(ptr_cell, ptr_cell + m_number, ptr_idx);
    // cudaDeviceSynchronize();
    // Logger::print_debug("Finished sorting");

    // Move the rest of particle array using the new index
    rearrange_arrays("cell");

    // Update the new number of particles
    const int padding = 0;
    m_number = thrust::upper_bound(ptr_cell, ptr_cell + m_number + padding,
                                   empty_cell - 1) -
               ptr_cell;

    Logger::print_info("Sorting complete, there are {} particles in the pool",
                       m_number);
    cudaDeviceSynchronize();
    CudaCheckError();
  }
}

template <typename BufferType>
void
particles_base<BufferType>::append_dev(const vec_t<Pos_t, 3>& x,
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
        ptrs.E[pos] = math::sqrt(1.0f + p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        ptrs.weight[pos] = weight;
        ptrs.cell[pos] = cell;
        ptrs.flag[pos] = flag;
      },
      m_dev_ptrs, m_number);
  CudaSafeCall(cudaDeviceSynchronize());
  m_number += 1;
}

template <typename BufferType>
template <typename Conf>
void
particles_base<BufferType>::copy_to_comm_buffers(
    std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Conf>& grid) {
  if (m_number > 0) {
    // timer::stamp("compute_buffer");
    if (m_index.size() != m_size) m_index.resize(m_size);
    m_index.assign_dev(0, m_number, size_t(-1));
    auto ptr_idx = thrust::device_pointer_cast(m_index.dev_ptr());

    m_zone_buffer_num.assign_dev(0);
    compute_target_buffers<Conf>(this->cell.dev_ptr(), m_number,
                                 m_zone_buffer_num, m_index.dev_ptr());
    m_zone_buffer_num.copy_to_host();
    CudaSafeCall(cudaDeviceSynchronize());
    // timer::show_duration_since_stamp("Computing target buffers", "ms",
    // "compute_buffer");

    int zone_offset = 0;
    if (buffers.size() == 9)
      zone_offset = 9;
    else if (buffers.size() == 3)
      zone_offset = 12;
    for (unsigned int i = 0; i < buffers.size(); i++) {
      // Logger::print_debug("zone {} buffer has {} ptc", i + zone_offset,
      //                     m_zone_buffer_num[i + zone_offset]);
      if (i + zone_offset == 13) continue;
      buffers[i].set_num(m_zone_buffer_num[i + zone_offset]);
    }
    // timer::stamp("copy_to_buffer");
    copy_component_to_buffer<Conf>(m_dev_ptrs, m_number, m_index.dev_ptr(),
                                   buf_ptrs);
    // for (unsigned int i = 0; i < buffers.size(); i++) {
    //   if (buffers[i].number() > 0) {
    //     buffers[i].copy_to_host();
    //   }
    // }
    // if (buffers[7].number() > 0) {
    //   buffers[7].copy_to_host();
    //   Logger::print_debug("buffer[7] cell[0] is {}", buffers[7].cell[0]);
    // }
    CudaSafeCall(cudaDeviceSynchronize());
    // timer::show_duration_since_stamp("Copy to buffer", "ms",
    // "copy_to_buffer");
  }
}

// Explicit instantiation
template class particles_base<ptc_buffer>;
template void particles_base<ptc_buffer>::copy_to_comm_buffers(
    std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<1>>& grid);
template void particles_base<ptc_buffer>::copy_to_comm_buffers(
    std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<2>>& grid);
template void particles_base<ptc_buffer>::copy_to_comm_buffers(
    std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<3>>& grid);

template class particles_base<ph_buffer>;
template void particles_base<ph_buffer>::copy_to_comm_buffers(
    std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<1>>& grid);
template void particles_base<ph_buffer>::copy_to_comm_buffers(
    std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<2>>& grid);
template void particles_base<ph_buffer>::copy_to_comm_buffers(
    std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
    const grid_t<Config<3>>& grid);

}  // namespace Aperture
