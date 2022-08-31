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
#include <hip/hip_runtime.h>

namespace Aperture {

rng_states_t::rng_states_t(uint64_t seed) {
  m_size = size_t(block_num * thread_num);
  GpuSafeCall(gpuMalloc(&m_states, sizeof(rand_state) * m_size));
  m_initial_seed = seed;
  m_states_host = new char[sizeof(rand_state) * m_size];
}

rng_states_t::~rng_states_t() {
  GpuSafeCall(gpuFree(m_states));
  delete[] m_states_host;
}

void
rng_states_t::copy_to_host() {
  GpuSafeCall(gpuMemcpy(m_states_host, m_states, m_size * sizeof(rand_state),
                        gpuMemcpyDeviceToHost));
}

void
rng_states_t::copy_to_device() {
  GpuSafeCall(gpuMemcpy(m_states, m_states_host, m_size * sizeof(rand_state),
                        gpuMemcpyHostToDevice));
}

void
rng_states_t::init() {
  kernel_exec_policy p{block_num, thread_num};
  kernel_launch(
      p,
      [] __device__(auto states, auto seed) {
        int id = threadIdx.x + blockIdx.x * blockDim.x;
#if defined(CUDA_ENABLED)
        curand_init(seed, id, 0, &states[id]);
#elif defined(HIP_ENABLED)
        rocrand_init(seed, id, 0, &states[id]);
#endif
      },
      m_states, m_initial_seed);
  GpuCheckError();
  GpuSafeCall(gpuDeviceSynchronize());
}

}  // namespace Aperture
