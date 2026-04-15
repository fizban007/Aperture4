// GPU-side instantiation of prismatic_mesh_metric::compute_metric.
// The explicit instantiations in prismatic_mesh_metric.cpp would
// produce host-only template code when compiled as a regular TU;
// including the impl header from a HIP/CUDA TU (this file) produces
// the device-side kernel code for each metric type.

#include "systems/prismatic/prismatic_mesh_metric_impl.hpp"

namespace Aperture {

template void prismatic_mesh_metric::compute_metric<flat_spherical_metric>(
    const flat_spherical_metric&);
template void prismatic_mesh_metric::compute_metric<ks_spherical_metric>(
    const ks_spherical_metric&);

}  // namespace Aperture
