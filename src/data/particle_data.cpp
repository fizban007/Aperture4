#include "particle_data.h"
#include "framework/environment.hpp"
#include "framework/config.h"

namespace Aperture {

particle_data_t::particle_data_t(MemType model) :
    particles_t(model) {}

particle_data_t::particle_data_t(size_t size, MemType model) :
    particles_t(size, model) {}

void
particle_data_t::init() {
  particles_t::init();
}

photon_data_t::photon_data_t(MemType model) :
    photons_t(model) {}

photon_data_t::photon_data_t(size_t size, MemType model) :
    photons_t(size, model) {}

void
photon_data_t::init() {
  photons_t::init();
}


}
