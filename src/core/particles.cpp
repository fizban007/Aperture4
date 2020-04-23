#include "particles_impl.hpp"

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
                                       Scalar weight,
                                       uint32_t flag) {}

// Explicit instantiation
template class particles_base<ptc_buffer>;
template class particles_base<ph_buffer>;

}  // namespace Aperture
