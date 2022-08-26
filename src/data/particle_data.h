/*
 * Copyright (c) 2020 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __PARTICLE_DATA_H_
#define __PARTICLE_DATA_H_

#include "framework/data.h"
#include "core/particles.h"

namespace Aperture {

class particle_data_t : public data_t, public particles_t {
 public:
  particle_data_t(MemType model = default_mem_type);
  particle_data_t(size_t size, MemType = default_mem_type);

  void init() override;
};

class photon_data_t : public data_t, public photons_t {
 public:
  photon_data_t(MemType model = default_mem_type);
  photon_data_t(size_t size, MemType = default_mem_type);

  void init() override;
};

template <>
struct host_adapter<particle_data_t> {
  typedef ptc_ptrs type;
  typedef ptc_ptrs const_type;

  static inline type apply(particle_data_t& array) {
    return array.get_host_ptrs();
  }
};

template <>
struct host_adapter<photon_data_t> {
  typedef ph_ptrs type;
  typedef ph_ptrs const_type;

  static inline type apply(photon_data_t& array) {
    return array.get_host_ptrs();
  }
};

#ifdef CUDA_ENABLED

template <>
struct gpu_adapter<particle_data_t> {
  typedef ptc_ptrs type;
  typedef ptc_ptrs const_type;

  static inline type apply(particle_data_t& array) {
    return array.get_dev_ptrs();
  }
};

template <>
struct gpu_adapter<photon_data_t> {
  typedef ph_ptrs type;
  typedef ph_ptrs const_type;

  static inline type apply(photon_data_t& array) {
    return array.get_dev_ptrs();
  }
};

#endif

}

#endif
