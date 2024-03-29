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

#pragma once

#include "core/buffer.hpp"
#include "core/exec_tags.h"
// #include "data/tracked_ptc.h"
#include "particle_structs.h"
#include "systems/grid.h"
#include "utils/vec.hpp"
#include <vector>

namespace Aperture {

template <typename BufferType>
class particles_base : public BufferType {
 public:
  typedef BufferType base_type;
  typedef particles_base<BufferType> self_type;
  typedef typename BufferType::single_type single_type;
  typedef typename BufferType::ptrs_type ptrs_type;

  particles_base(MemType model = default_mem_type);
  particles_base(size_t size, MemType model = default_mem_type);
  particles_base(const self_type& other) = delete;
  particles_base(self_type&& other) = default;
  ~particles_base() {}

  self_type& operator=(const self_type& other) = delete;
  self_type& operator=(self_type&& other) = default;

  void set_memtype(MemType memtype);
  MemType mem_type() const { return m_mem_type; }

  void resize(size_t size);

  void copy_from(const self_type& other);
  void copy_from(const self_type& other, size_t num, size_t src_pos,
                 size_t dst_pos);

  void erase(size_t pos, size_t amount = 1);

  void init() { erase(0, m_size); }

  // void sort_by_cell(size_t max_cell);
  // void sort_by_cell(exec_tags::device, size_t max_cell);
  // void sort_by_cell(exec_tags::host, size_t max_cell);
  // void sort_by_cell_host(size_t max_cell);
  // void sort_by_cell_dev(size_t max_cell);

  // void append(exec_tags::host, const vec_t<Scalar, 3>& x, const vec_t<Scalar,
  // 3>& p,
  //             uint32_t cell, Scalar weight = 1.0, uint32_t flag = 0);

  // void append(exec_tags::device, const vec_t<Scalar, 3>& x, const
  // vec_t<Scalar, 3>& p,
  //             uint32_t cell, Scalar weight = 1.0, uint32_t flag = 0);

  void copy_to_host(bool all = true);
  void copy_to_device(bool all = true);
  void copy_to_host(gpuStream_t stream, bool all = true);
  void copy_to_device(gpuStream_t stream, bool all = true);

  // template <typename Conf>
  // void copy_to_comm_buffers(std::vector<self_type>& buffers,
  //                           buffer<ptrs_type>& buf_ptrs,
  //                           const grid_t<Conf>& grid);
  // template <typename Conf>
  // void copy_to_comm_buffers(exec_tags::device,
  //                           std::vector<buffer<single_type>>& buffers,
  //                           buffer<single_type*>& buf_ptrs,
  //                           buffer<int>& buf_nums, const grid_t<Conf>& grid);
  // template <typename Conf>
  // void copy_to_comm_buffers(exec_tags::host,
  //                           std::vector<buffer<single_type>>& buffers,
  //                           buffer<single_type*>& buf_ptrs,
  //                           buffer<int>& buf_nums, const grid_t<Conf>& grid);
  // void copy_from_buffer(exec_tags::device, const buffer<single_type>& buf,
  // int num,
  //                       size_t dst_idx);
  // void copy_from_buffer(exec_tags::host, const buffer<single_type>& buf, int
  // num,
  //                       size_t dst_idx);

  void swap(size_t pos, single_type& p);

  size_t size() const { return m_size; }
  size_t number() const { return m_number; }

  void set_num(size_t num) {
    // Can't set a number larger than maximum size
    m_number = std::min(num, m_size);
  }

  void set_segment_size(size_t s) { m_sort_segment_size = s; }

  void add_num(size_t num) { set_num(m_number + num); }

  typename BufferType::ptrs_type& get_host_ptrs() { return m_host_ptrs; }
  typename BufferType::ptrs_type& get_dev_ptrs() { return m_dev_ptrs; }
  buffer<uint32_t>& ptc_id() { return m_ptc_id; }

  size_t sort_segment_size() { return m_sort_segment_size; }
  buffer<size_t>& index() { return m_index; }
  buffer<double>& tmp_data() { return m_tmp_data; }
  buffer<int>& segment_nums() { return m_segment_nums; }
  buffer<int>& zone_buffer_num() { return m_zone_buffer_num; }
  std::vector<size_t>& partition() { return m_partition; }

 private:
  size_t m_size = 0;
  size_t m_number = 0;
  size_t m_sort_segment_size = 10000000;
  MemType m_mem_type;
  buffer<uint32_t> m_ptc_id;
  buffer<int> m_segment_nums;

  // Temporary data for sorting particles on device
  buffer<size_t> m_index;
  buffer<double> m_tmp_data;
  buffer<int> m_zone_buffer_num;
  // Temporary data for sorting particles on host
  std::vector<size_t> m_partition;
  // Map for tracked particles
  // buffer<uint32_t> m_tracked_map;
  // buffer<int> m_tracked_num;  // Current number of tracked particles, using
  // int because need to do atomicAdd
  // size_t m_max_tracked_num = 0;

  typename BufferType::ptrs_type m_host_ptrs;
  typename BufferType::ptrs_type m_dev_ptrs;

  void resize_tmp_arrays();
  // void rearrange_arrays(exec_tags::device, const std::string& skip, size_t
  // offset, size_t num); void rearrange_arrays(exec_tags::host);
};

using particles_t = particles_base<ptc_buffer>;
using photons_t = particles_base<ph_buffer>;

template <typename BufferType>
struct host_adapter<particles_base<BufferType>> {
  typedef typename BufferType::ptrs_type type;
  typedef typename BufferType::ptrs_type const_type;

  static inline const_type apply(const particles_base<BufferType>& array) {
    return array.get_host_ptrs();
  }
  static inline type apply(particles_base<BufferType>& array) {
    return array.get_host_ptrs();
  }
};

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)

template <typename BufferType>
struct gpu_adapter<particles_base<BufferType>> {
  typedef typename BufferType::ptrs_type type;
  typedef typename BufferType::ptrs_type const_type;

  static inline const_type apply(const particles_base<BufferType>& array) {
    return array.get_dev_ptrs();
  }
  static inline type apply(particles_base<BufferType>& array) {
    return array.get_dev_ptrs();
  }
};

#endif

}  // namespace Aperture

#include "particles_functions.h"
