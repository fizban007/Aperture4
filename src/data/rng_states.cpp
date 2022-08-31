/*
 * Copyright (c) 2021 Alex Chen.
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

#include "rng_states.h"

#ifndef CUDA_ENABLED

namespace Aperture {

rng_states_t::rng_states_t(uint64_t seed) {
  m_states = new rand_state;
  *m_states = rand_state(seed);

  m_states_host = new char[sizeof(rand_state)];
}

rng_states_t::~rng_states_t() {
  delete m_states;
  delete[] m_states_host;
}

void
rng_states_t::copy_to_host() {
  memcpy(m_states_host, m_states, sizeof(rand_state));
}

void
rng_states_t::copy_to_device() {
  memcpy(m_states, m_states_host, sizeof(rand_state));
}

void
rng_states_t::init() {}

}

#endif
