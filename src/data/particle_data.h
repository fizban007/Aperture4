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

}

#endif
