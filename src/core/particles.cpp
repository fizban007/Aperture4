#include "particles.h"
#include "utils/for_each_dual.hpp"
#include "visit_struct/visit_struct.hpp"
#include <algorithm>
#include <numeric>

namespace Aperture {

template <typename BufferType>
void
particles_base<BufferType>::obtain_ptrs() {
  m_ptrs = this->host_ptrs();
}

template <typename BufferType>
void
particles_base<BufferType>::swap(size_t pos, single_type &p) {
  single_type p_tmp;
  for_each_double(p_tmp, m_ptrs, [pos](auto& x, auto& y) {
                                   x = y[pos];
                                 });
  assign_ptc(m_ptrs, pos, p);
  p = p_tmp;
}

template <typename BufferType>
void
particles_base<BufferType>::rearrange_arrays(const std::string& skip) {
  typename BufferType::single_type p_tmp;
  for (size_t i = 0; i < m_number; i++) {
    // -1 means LLONG_MAX for unsigned long int
    if (m_index[i] != (size_t)-1) {
      for_each_double(p_tmp, m_ptrs, [i](auto& x, auto& y) {
                                       x = y[i];
                                     });
      for (size_t j = i;;) {
        if (m_index[j] != i) {
          // put(index[j], m_data[j]);
          swap(m_index[j], p_tmp);
          size_t id = m_index[j];
          m_index[j] = (size_t)-1;  // Mark as done
          j = id;
        } else {
          assign_ptc(m_ptrs, i, p_tmp);
          m_index[j] = (size_t)-1;  // Mark as done
          break;
        }
      }
    }
  }
}

template <typename BufferType>
void
particles_base<BufferType>::sort_by_cell(size_t num_cells) {
  if (m_number > 0) {
    // Compute the number of cells and resize the partition array if
    // needed
    if (m_partition.size() != num_cells + 2)
      m_partition.resize(num_cells + 2);
    if (m_index.size() != m_size)
      m_index.resize(m_size);

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
    rearrange_arrays("");

    // num_cells is where the empty particles start, so we record this as
    // the new particle number
    if (m_partition[num_cells] != m_number)
      set_num(m_partition[num_cells]);
    Logger::print_info("Sorting complete, there are {} particles in the pool", m_number);
  }
}

template <typename BufferType>
void
particles_base<BufferType>::append(const typename BufferType::single_type &p) {

}
// Explicit instantiation
template class particles_base<ptc_buffer<MemoryModel::host_only>>;
template class particles_base<ph_buffer<MemoryModel::host_only>>;

}
