#include "particle_data.h"
#include "framework/environment.hpp"
#include "framework/config.h"

namespace Aperture {

particle_data_t::particle_data_t(MemType model) :
    particles_t(model) {}

particle_data_t::particle_data_t(size_t size, MemType model) :
    particles_t(size, model) {}

photon_data_t::photon_data_t(MemType model) :
    photons_t(model) {}

photon_data_t::photon_data_t(size_t size, MemType model) :
    photons_t(size, model) {}


}
