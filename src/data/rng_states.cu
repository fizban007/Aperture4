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

#include "data/rng_states.h"
#include "utils/kernel_helper.hpp"

namespace Aperture {

rng_states_t::rng_states_t(uint64_t seed) {
  m_states.set_memtype(MemType::host_device);
  m_states.resize(size_t(block_num * thread_num));

  m_initial_seed = seed;
}

void
rng_states_t::init() {
  kernel_launch(
      {block_num, thread_num},
      [] __device__(auto states, auto seed) {
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init(seed, id, 0, &states[id]);
      },
      m_states.dev_ptr(), m_initial_seed);
  CudaCheckError();
  CudaSafeCall(cudaDeviceSynchronize());
}

}
