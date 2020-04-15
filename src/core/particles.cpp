#include "particles_impl.hpp"

namespace Aperture {

template <typename BufferType>
void
particles_base<BufferType>::sort_by_cell_dev(size_t max_cell) {}

template <typename BufferType>
void
particles_base<BufferType>::rearrange_arrays(const std::string& skip) {}

// Explicit instantiation
template class particles_base<ptc_buffer>;
template class particles_base<ph_buffer>;

}
