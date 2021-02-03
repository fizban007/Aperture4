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

namespace {

uint64_t split_mix_64 (uint64_t x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}

}

namespace Aperture {

rng_states_t::rng_states_t(uint64_t seed) {
  m_states.resize(1);

  uint64_t seeds[4];
  seeds[0] = split_mix_64(seed);
  seeds[1] = split_mix_64(seeds[0]);
  seeds[2] = split_mix_64(seeds[1]);
  seeds[3] = split_mix_64(seeds[2]);

  m_states[0] = rand_state(seeds);
}

void
rng_states_t::init() {}

}

#endif
