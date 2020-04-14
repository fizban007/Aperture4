#ifndef __PARTICLE_DATA_H_
#define __PARTICLE_DATA_H_

#include "framework/data.h"
#include "core/particles.h"

namespace Aperture {

template <MemoryModel Model>
class particle_data_t : public data_t, public particles_t<Model> {
 public:
  particle_data_t();
  particle_data_t(size_t size);
};

template <MemoryModel Model>
class photon_data_t : public data_t, public photons_t<Model> {
 public:
  photon_data_t();
  photon_data_t(size_t size);
};

}

#endif
