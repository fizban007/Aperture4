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
#include "framework/environment.h"
#include "utils/kernel_helper.hpp"
// #include <hip/hip_runtime.h>

namespace Aperture {

template <>
rng_states_t<ExecDev>::rng_states_t(uint64_t seed) {
  m_size = size_t(block_num * thread_num);
  m_initial_seed = seed;
  // GpuSafeCall(gpuMalloc(&m_states, sizeof(rand_state) * m_size));
  // m_states_host = new char[sizeof(rand_state) * m_size];
  m_states.set_memtype(MemType::host_device);
  m_states.resize(m_size);
}

template <>
rng_states_t<ExecDev>::~rng_states_t() {}

template <>
void
rng_states_t<ExecDev>::init() {
  // kernel_exec_policy p{block_num, thread_num};
  int rank = sim_env().get_rank();
  uint32_t size = m_size;

  kernel_launch(
      {block_num, thread_num},
      [rank, size] __device__(auto states, auto seed) {
        int id = threadIdx.x + blockIdx.x * blockDim.x;
// #if defined(CUDA_ENABLED)
//         curand_init(seed, id, 0, &states[id]);
// #elif defined(HIP_ENABLED)
//         rocrand_init(seed, id, 0, &states[id]);
// #endif
        states[id].init(seed + id + rank * size);
        states[id].jump();
        // Advance the state by 2^128 * id times
        // for (uint64_t i = 0; i < id; i++) {
        //   states[id].jump();
        // }
      },
      m_states.dev_ptr(), m_initial_seed);
  GpuCheckError();
  GpuSafeCall(gpuDeviceSynchronize());
}

}  // namespace Aperture
