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

#include "curand_states.h"
#include "framework/environment.h"
#include "utils/kernel_helper.hpp"

namespace Aperture {

curand_states_t::curand_states_t(size_t size, int seed) {
  m_states.set_memtype(MemType::host_device);
  m_states.resize(std::max(size, size_t(512 * 1024)));
  Logger::print_info("Resized the random state to {}", m_states.size());
  m_init_seed = seed;
}

void
curand_states_t::init() {
  kernel_launch(
      {512, 1024},
      [] __device__(auto states, int seed) {
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init(seed, id, 0, &states[id]);
      },
      m_states.dev_ptr(), m_init_seed);
  CudaSafeCall(cudaDeviceSynchronize());
}

}  // namespace Aperture
