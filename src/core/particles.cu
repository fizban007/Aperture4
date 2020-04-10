#include "core/typedefs_and_constants.h"
#include "particles.h"
#include "visit_struct/visit_struct.hpp"

namespace Aperture {

// Explicit instantiation
template class particles_base<ptc_buffer<MemoryModel::host_only>>;
template class particles_base<ptc_buffer<MemoryModel::host_device>>;
template class particles_base<ptc_buffer<MemoryModel::device_managed>>;
template class particles_base<ptc_buffer<MemoryModel::device_only>>;

template class particles_base<ph_buffer<MemoryModel::host_only>>;
template class particles_base<ph_buffer<MemoryModel::host_device>>;
template class particles_base<ph_buffer<MemoryModel::device_managed>>;
template class particles_base<ph_buffer<MemoryModel::device_only>>;

}  // namespace Aperture
