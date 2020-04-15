#ifndef __PARTICLE_DATA_H_
#define __PARTICLE_DATA_H_

#include "framework/data.h"
#include "core/particles.h"

namespace Aperture {

class particle_data_t : public data_t, public particles_t {
 public:
  particle_data_t(MemoryModel model = default_memory_model);
  particle_data_t(size_t size, MemoryModel = default_memory_model);
};

class photon_data_t : public data_t, public photons_t {
 public:
  photon_data_t(MemoryModel model = default_memory_model);
  photon_data_t(size_t size, MemoryModel = default_memory_model);
};

}

#endif
