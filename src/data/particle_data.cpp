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

#include "particle_data.h"
#include "framework/environment.h"
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
