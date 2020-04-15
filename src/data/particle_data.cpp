#include "particle_data.h"
#include "framework/environment.hpp"
#include "framework/config.h"

namespace Aperture {

particle_data_t::particle_data_t(MemoryModel model) :
    particles_t(model) {}

particle_data_t::particle_data_t(size_t size, MemoryModel model) :
    particles_t(size, model) {}

photon_data_t::photon_data_t(MemoryModel model) :
    photons_t(model) {}

photon_data_t::photon_data_t(size_t size, MemoryModel model) :
    photons_t(size, model) {}


}
