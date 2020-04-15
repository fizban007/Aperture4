#ifndef __PARTICLE_DATA_H_
#define __PARTICLE_DATA_H_

#include "framework/data.h"
#include "core/particles.h"

namespace Aperture {

class particle_data_t : public data_t, public particles_t {
 public:
  particle_data_t(MemType model = default_mem_type);
  particle_data_t(size_t size, MemType = default_mem_type);
};

class photon_data_t : public data_t, public photons_t {
 public:
  photon_data_t(MemType model = default_mem_type);
  photon_data_t(size_t size, MemType = default_mem_type);
};

}

#endif
