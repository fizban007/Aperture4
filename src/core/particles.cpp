#include "particles_impl.hpp"
#include "framework/config.h"

namespace Aperture {

template <typename BufferType>
void
particles_base<BufferType>::sort_by_cell_dev(size_t max_cell) {}

template <typename BufferType>
void
particles_base<BufferType>::rearrange_arrays(const std::string &skip) {}

template <typename BufferType>
void
particles_base<BufferType>::append_dev(const vec_t<Pos_t, 3> &x,
                                       const vec_t<Scalar, 3> &p, uint32_t cell,
                                       Scalar weight, uint32_t flag) {}

template <typename BufferType>
template <typename Conf>
void
particles_base<BufferType>::copy_to_comm_buffers(
    std::vector<self_type> &buffers, buffer<ptrs_type> &buf_ptrs,
    const grid_t<Conf>& grid) {}

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
