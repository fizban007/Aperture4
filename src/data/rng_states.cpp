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

// #ifndef CUDA_ENABLED

namespace Aperture {

template <>
rng_states_t<exec_tags::host>::rng_states_t(uint64_t seed) {
  m_states.set_memtype(MemType::host_only);
  m_states.resize(1);
  m_size = 1;
  m_initial_seed = seed;
}

template <>
rng_states_t<exec_tags::host>::~rng_states_t() {}

template <>
void
rng_states_t<exec_tags::host>::init() {
  int rank = sim_env().get_rank();
  m_states[0].init(m_initial_seed);
  for (int i = 0; i < rank; i++) {
    m_states[0].jump();
  }
}

}

// #endif
