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

#include "framework/config.h"
#include "particles.h"
#include "utils/for_each_dual.hpp"
#include "visit_struct/visit_struct.hpp"
#include <algorithm>
#include <numeric>

namespace Aperture {

template <typename BufferType>
particles_base<BufferType>::particles_base(MemType model) : m_mem_type(model) {
  set_memtype(m_mem_type);
}

template <typename BufferType>
particles_base<BufferType>::particles_base(size_t size, MemType model)
    : m_mem_type(model) {
  set_memtype(m_mem_type);
  resize(size);
  m_host_ptrs = this->host_ptrs();
  m_dev_ptrs = this->dev_ptrs();
}

template <typename BufferType>
void
particles_base<BufferType>::set_memtype(MemType memtype) {
  if (memtype == MemType::host_only) {
    m_zone_buffer_num.set_memtype(memtype);
  } else {
    m_zone_buffer_num.set_memtype(MemType::host_device);
  }
  visit_struct::for_each(
      *static_cast<base_type*>(this),
      [memtype](const char* name, auto& x) { x.set_memtype(memtype); });
}

template <typename BufferType>
void
particles_base<BufferType>::resize(size_t size) {
  visit_struct::for_each(*static_cast<base_type*>(this),
                         [size](const char* name, auto& x) { x.resize(size); });
  m_size = size;
  m_zone_buffer_num.resize(27);
  // Resize the particle id buffer to one, since it only holds a single number
  // identifying all the tracked particles
  m_ptc_id.resize(1);
  m_ptc_id.assign(0);
  m_ptc_id.assign_dev(0);
  if (m_mem_type == MemType::host_only || m_mem_type == MemType::host_device) {
    this->cell.assign_host(empty_cell);
  }
  if (m_mem_type == MemType::device_only || m_mem_type == MemType::device_managed) {
    this->cell.assign_dev(empty_cell);
  }
}

template <typename BufferType>
void
particles_base<BufferType>::copy_from(const self_type& other) {
  copy_from(other, other.number(), 0, 0);
}

template <typename BufferType>
void
particles_base<BufferType>::copy_from(const self_type& other, size_t num,
                                      size_t src_pos, size_t dst_pos) {
  if (dst_pos + num > m_size) num = m_size - dst_pos;
  visit_struct::for_each(
      *static_cast<base_type*>(this), *static_cast<const base_type*>(&other),
      [num, src_pos, dst_pos](const char* name, auto& u, auto& v) {
        u.copy_from(v, num, src_pos, dst_pos);
      });
  if (dst_pos + num > m_number) set_num(dst_pos + num);
}

template <typename BufferType>
void
particles_base<BufferType>::erase(size_t pos, size_t amount) {
  this->cell.assign(pos, pos + amount, empty_cell);
}

template <typename BufferType>
void
particles_base<BufferType>::swap(size_t pos, single_type& p) {
  single_type p_tmp;
  for_each_double(p_tmp, m_host_ptrs, [pos](auto& x, auto& y) { x = y[pos]; });
  assign_ptc(m_host_ptrs, pos, p);
  p = p_tmp;
}

template <typename BufferType>
void
particles_base<BufferType>::copy_to_host(bool all) {
  auto num = (all ? m_size : m_number);
  if (m_mem_type == MemType::host_device) {
    visit_struct::for_each(
        *static_cast<base_type*>(this),
        [num](const char* name, auto& x) { x.copy_to_host(0, num); });
        // [num](const char* name, auto& x) { x.copy_to_host(); });
  }
}

template <typename BufferType>
void
particles_base<BufferType>::copy_to_host(gpuStream_t stream, bool all) {
  auto num = (all ? m_size : m_number);
  if (m_mem_type == MemType::host_device)
    visit_struct::for_each(*static_cast<base_type*>(this),
                           [num, stream](const char* name, auto& x) {
                             x.copy_to_host(0, num, stream);
                             // x.copy_to_host(stream);
                           });
}

template <typename BufferType>
void
particles_base<BufferType>::copy_to_device(bool all) {
  auto num = (all ? m_size : m_number);
  if (m_mem_type == MemType::host_device)
    visit_struct::for_each(
        *static_cast<base_type*>(this),
        [num](const char* name, auto& x) { x.copy_to_device(0, num); });
        // [num](const char* name, auto& x) { x.copy_to_device(); });
}

template <typename BufferType>
void
particles_base<BufferType>::copy_to_device(gpuStream_t stream, bool all) {
  auto num = (all ? m_size : m_number);
  if (m_mem_type == MemType::host_device)
    visit_struct::for_each(*static_cast<base_type*>(this),
                           [num, stream](const char* name, auto& x) {
                             x.copy_to_device(0, num, stream);
                             // x.copy_to_device(stream);
                           });
}

// Explicit instantiation
template class particles_base<ptc_buffer>;
template class particles_base<ph_buffer>;

}  // namespace Aperture
