#include "particle_data.h"
#include "framework/environment.hpp"
#include "framework/config.h"

namespace Aperture {

template <MemoryModel Model>
particle_data_t<Model>::particle_data_t() :
    particles_t<Model>() {}

template <MemoryModel Model>
particle_data_t<Model>::particle_data_t(size_t size) :
    particles_t<Model>(size) {}

template <MemoryModel Model>
photon_data_t<Model>::photon_data_t() :
    photons_t<Model>() {}

template <MemoryModel Model>
photon_data_t<Model>::photon_data_t(size_t size) :
    photons_t<Model>(size) {}


#ifdef CUDA_ENABLED
template class particle_data_t<MemoryModel::host_device>;
template class particle_data_t<MemoryModel::device_only>;
template class particle_data_t<MemoryModel::device_managed>;
template class photon_data_t<MemoryModel::host_device>;
template class photon_data_t<MemoryModel::device_only>;
template class photon_data_t<MemoryModel::device_managed>;
#else
template class particle_data_t<MemoryModel::host_only>;
template class photon_data_t<MemoryModel::host_only>;
#endif


}
