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
#include "particles_impl.hpp"

namespace Aperture {

// template <typename BufferType>
// void
// particles_base<BufferType>::sort_by_cell_dev(size_t max_cell) {}

// template <typename BufferType>
// void
// particles_base<BufferType>::rearrange_arrays(const std::string& skip,
//                                              size_t offset, size_t num) {}

// template <typename BufferType>
// void
// particles_base<BufferType>::append(exec_tags::device{}, const vec_t<Scalar, 3>& x,
//                                        const vec_t<Scalar, 3>& p, uint32_t cell,
//                                        Scalar weight, uint32_t flag) {}

// template <typename BufferType>
// void
// particles_base<BufferType>::resize_tmp_arrays() {}

// // Explicit instantiation
// template class particles_base<ptc_buffer>;
// template void particles_base<ptc_buffer>::copy_to_comm_buffers(
//     std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<1>>& grid);
// template void particles_base<ptc_buffer>::copy_to_comm_buffers(
//     std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<2>>& grid);
// template void particles_base<ptc_buffer>::copy_to_comm_buffers(
//     std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<3>>& grid);

// template class particles_base<ph_buffer>;
// template void particles_base<ph_buffer>::copy_to_comm_buffers(
//     std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<1>>& grid);
// template void particles_base<ph_buffer>::copy_to_comm_buffers(
//     std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<2>>& grid);
// template void particles_base<ph_buffer>::copy_to_comm_buffers(
//     std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<3>>& grid);

// Explicit instantiation
template class particles_base<ptc_buffer>;
// template void particles_base<ptc_buffer>::copy_to_comm_buffers(
//     exec_tags::host,
//     std::vector<buffer<single_type>>& buffers, buffer<single_type*>& buf_ptrs,
//     buffer<int>& buf_nums,
//     // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<1, float>>& grid);
// template void particles_base<ptc_buffer>::copy_to_comm_buffers(
//     exec_tags::host,
//     std::vector<buffer<single_type>>& buffers, buffer<single_type*>& buf_ptrs,
//     buffer<int>& buf_nums,
//     // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<1, double>>& grid);
// template void particles_base<ptc_buffer>::copy_to_comm_buffers(
//     exec_tags::host,
//     std::vector<buffer<single_type>>& buffers, buffer<single_type*>& buf_ptrs,
//     buffer<int>& buf_nums,
//     // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<2, float>>& grid);
// template void particles_base<ptc_buffer>::copy_to_comm_buffers(
//     exec_tags::host,
//     std::vector<buffer<single_type>>& buffers, buffer<single_type*>& buf_ptrs,
//     buffer<int>& buf_nums,
//     // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<2, double>>& grid);
// template void particles_base<ptc_buffer>::copy_to_comm_buffers(
//     exec_tags::host,
//     std::vector<buffer<single_type>>& buffers, buffer<single_type*>& buf_ptrs,
//     buffer<int>& buf_nums,
//     // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<3, float>>& grid);
// template void particles_base<ptc_buffer>::copy_to_comm_buffers(
//     exec_tags::host,
//     std::vector<buffer<single_type>>& buffers, buffer<single_type*>& buf_ptrs,
//     buffer<int>& buf_nums,
//     // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<3, double>>& grid);

template class particles_base<ph_buffer>;
// template void particles_base<ph_buffer>::copy_to_comm_buffers(
//     exec_tags::host,
//     std::vector<buffer<single_type>>& buffers, buffer<single_type*>& buf_ptrs,
//     buffer<int>& buf_nums,
//     // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<1, float>>& grid);
// template void particles_base<ph_buffer>::copy_to_comm_buffers(
//     exec_tags::host,
//     std::vector<buffer<single_type>>& buffers, buffer<single_type*>& buf_ptrs,
//     buffer<int>& buf_nums,
//     // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<1, double>>& grid);
// template void particles_base<ph_buffer>::copy_to_comm_buffers(
//     exec_tags::host,
//     std::vector<buffer<single_type>>& buffers, buffer<single_type*>& buf_ptrs,
//     buffer<int>& buf_nums,
//     // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<2, float>>& grid);
// template void particles_base<ph_buffer>::copy_to_comm_buffers(
//     exec_tags::host,
//     std::vector<buffer<single_type>>& buffers, buffer<single_type*>& buf_ptrs,
//     buffer<int>& buf_nums,
//     // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<2, double>>& grid);
// template void particles_base<ph_buffer>::copy_to_comm_buffers(
//     exec_tags::host,
//     std::vector<buffer<single_type>>& buffers, buffer<single_type*>& buf_ptrs,
//     buffer<int>& buf_nums,
//     // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<3, float>>& grid);
// template void particles_base<ph_buffer>::copy_to_comm_buffers(
//     exec_tags::host,
//     std::vector<buffer<single_type>>& buffers, buffer<single_type*>& buf_ptrs,
//     buffer<int>& buf_nums,
//     // std::vector<self_type>& buffers, buffer<ptrs_type>& buf_ptrs,
//     const grid_t<Config<3, double>>& grid);

}  // namespace Aperture
