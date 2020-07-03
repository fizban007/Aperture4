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

#ifndef _CURAND_STATES_H_
#define _CURAND_STATES_H_

#ifdef CUDA_ENABLED
#include "framework/data.h"
#include "core/buffer.hpp"
#include "core/cuda_control.h"

#include <curand_kernel.h>

namespace Aperture {

// Helper struct to plug into kernels
struct cuda_rng_t {
  HOST_DEVICE cuda_rng_t(curandState* state) : m_state(state) {
    m_local_state = *state;
  }
  HOST_DEVICE ~cuda_rng_t() {
    *m_state = m_local_state;
  }

  // Generates a device random number between 0.0 and 1.0
  __device__ __forceinline__ float operator()() {
    return curand_uniform(&m_local_state);
  }

  curandState* m_state;
  curandState m_local_state;
};

class curand_states_t : public data_t {
 private:
  buffer<curandState> m_states;
  int m_init_seed = 1234;
  int m_rand_state_size = sizeof(curandState);
  int m_block_num = 512;
  int m_thread_num = 1024;

 public:
  curand_states_t(size_t size, int seed);
  void init() override;

  inline curandState* states() { return m_states.dev_ptr(); }
  inline void* states_host() { return m_states.host_ptr(); }
  inline int block_num() const { return m_block_num; }
  inline int thread_num() const { return m_thread_num; }
};

}

#else

namespace Aperture {

class curand_states_t : public data_t {};

}

#endif

#endif  // _CURAND_STATES_H_
