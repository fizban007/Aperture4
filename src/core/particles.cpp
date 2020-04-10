#include "particles.h"
#include "visit_struct/visit_struct.hpp"

namespace Aperture {

// Explicit instantiation
template class particles_base<ptc_buffer<MemoryModel::host_only>>;
template class particles_base<ph_buffer<MemoryModel::host_only>>;

}
