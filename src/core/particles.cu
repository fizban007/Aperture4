#include "core/typedefs_and_constants.h"
#include "particles_impl.hpp"
#include "utils/for_each_dual.hpp"
#include "utils/kernel_helper.hpp"
#include "visit_struct/visit_struct.hpp"

#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/replace.h>
#include <thrust/sort.h>

namespace Aperture {

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
                                       Scalar weight,
                                       uint32_t flag) {
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
        ptrs.weight[pos] = weight;
        ptrs.cell[pos] = cell;
        ptrs.flag[pos] = flag;
      },
      m_dev_ptrs, m_number);
  CudaSafeCall(cudaDeviceSynchronize());
  m_number += 1;
}

// Explicit instantiation
template class particles_base<ptc_buffer>;
template class particles_base<ph_buffer>;

}  // namespace Aperture
