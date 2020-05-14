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
  m_zone_buffer_num.set_memtype(memtype);
  visit_struct::for_each(
      *dynamic_cast<base_type*>(this),
      [memtype](const char* name, auto& x) { x.set_memtype(memtype); });
}

template <typename BufferType>
void
particles_base<BufferType>::resize(size_t size) {
  visit_struct::for_each(*dynamic_cast<base_type*>(this),
                         [size](const char* name, auto& x) { x.resize(size); });
  m_size = size;
  m_zone_buffer_num.resize(27);
}

template <typename BufferType>
void
particles_base<BufferType>::copy_from(const self_type& other, size_t num,
                                      size_t src_pos, size_t dst_pos) {
  visit_struct::for_each(
      *dynamic_cast<base_type*>(this), *dynamic_cast<const base_type*>(&other),
      [num, src_pos, dst_pos](const char* name, auto& u, auto& v) {
        u.copy_from(v, num, src_pos, dst_pos);
      });
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
particles_base<BufferType>::rearrange_arrays_host() {
  typename BufferType::single_type p_tmp;
  for (size_t i = 0; i < m_number; i++) {
    // -1 means LLONG_MAX for unsigned long int
    if (m_index[i] != (size_t)-1) {
      for_each_double(p_tmp, m_host_ptrs, [i](auto& x, auto& y) { x = y[i]; });
      for (size_t j = i;;) {
        if (m_index[j] != i) {
          // put(index[j], m_data[j]);
          swap(m_index[j], p_tmp);
          size_t id = m_index[j];
          m_index[j] = (size_t)-1;  // Mark as done
          j = id;
        } else {
          assign_ptc(m_host_ptrs, i, p_tmp);
          m_index[j] = (size_t)-1;  // Mark as done
          break;
        }
      }
    }
  }
}

template <typename BufferType>
void
particles_base<BufferType>::append(const vec_t<Pos_t, 3>& x,
                                   const vec_t<Scalar, 3>& p, uint32_t cell,
                                   Scalar weight,
                                   uint32_t flag) {
  if (m_number == m_size) return;
  this->x1[m_number] = x[0];
  this->x2[m_number] = x[1];
  this->x3[m_number] = x[2];
  this->p1[m_number] = p[0];
  this->p2[m_number] = p[1];
  this->p3[m_number] = p[2];
  this->E[m_number] = std::sqrt(1.0f + p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
  this->weight[m_number] = weight;
  this->cell[m_number] = cell;
  this->flag[m_number] = flag;
  m_number += 1;
}

template <typename BufferType>
void
particles_base<BufferType>::sort_by_cell(size_t max_cell) {
  if (m_mem_type == MemType::host_only)
    sort_by_cell_host(max_cell);
  else
    sort_by_cell_dev(max_cell);
}

template <typename BufferType>
void
particles_base<BufferType>::sort_by_cell_host(size_t num_cells) {
  if (m_number > 0) {
    // Compute the number of cells and resize the partition array if
    // needed
    if (m_partition.size() != num_cells + 2) m_partition.resize(num_cells + 2);
    if (m_index.size() != m_size) m_index.resize(m_size);

    std::fill(m_partition.begin(), m_partition.end(), 0);
    // Generate particle index from 0 up to the current number
    std::iota(m_index.host_ptr(), m_index.host_ptr() + m_number, 0);

    // Loop over the particle array to count how many particles in each
    // cell
    for (std::size_t i = 0; i < m_number; i++) {
      size_t cell_idx = 0;
      if (this->cell[i] == empty_cell)
        cell_idx = num_cells;
      else
        cell_idx = this->cell[i];
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
      if (this->cell[i] == empty_cell) {
        cell_idx = num_cells;
      } else {
        cell_idx = this->cell[i];
      }
      m_index[i] += m_partition[cell_idx];
    }

    // Rearrange the particles to reflect the partition
    // timer::show_duration_since_stamp("partition", "ms");
    rearrange_arrays_host();

    // num_cells is where the empty particles start, so we record this as
    // the new particle number
    if (m_partition[num_cells] != m_number) set_num(m_partition[num_cells]);
    Logger::print_info("Sorting complete, there are {} particles in the pool",
                       m_number);
  }
}

template <typename BufferType>
void
particles_base<BufferType>::copy_to_host() {
  if (m_mem_type == MemType::host_device)
    visit_struct::for_each(*dynamic_cast<base_type*>(this),
                           [](const char* name, auto& x) { x.copy_to_host(); });
}

template <typename BufferType>
void
particles_base<BufferType>::copy_to_device() {
  if (m_mem_type == MemType::host_device)
    visit_struct::for_each(
        *dynamic_cast<base_type*>(this),
        [](const char* name, auto& x) { x.copy_to_device(); });
}

}  // namespace Aperture
