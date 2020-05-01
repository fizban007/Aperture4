#ifndef _CURAND_STATES_H_
#define _CURAND_STATES_H_

#ifdef CUDA_ENABLED
#include "framework/data.h"
#include "utils/buffer.h"
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
